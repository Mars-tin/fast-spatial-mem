#
# This file is derived from code released under the MIT License.
#
# Original copyright:
# Copyright (c) 2025 Tianyuan Zhang, Hao Tan
#
# Modifications copyright:
# Copyright (c) 2026 Martin Ziqiao Ma
#
# This file is distributed under the licensing terms provided in the
# repository LICENSE and NOTICE files.
#
import collections
import math
from typing import Optional

import torch
from torch import nn

import torch.nn.functional as F
from einops import rearrange

TTTOperator = collections.namedtuple("TTTOperator", ["start", "end", "update", "apply"])

@torch.compile
def inv_softplus(x):
    y = x + math.log(-math.expm1(-x))
    return y

@torch.compile
def silu_backprop(dy: torch.Tensor, x: torch.Tensor):
    """
    Args:
        dy: [b, d, l], gradient of the outer loss wrt the y
        x: [b, d, l], input of the silu activation
    outs:
        dx: [b, d, l], gradient of the outer loss wrt the x
        dx = dy * sigma * (1 + x * (1 - sigma))
    """
    sigma = torch.sigmoid(x)
    dx = dy * sigma * (1 + x * (1 - sigma))
    return dx


@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    modified from https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py#L49
    Major change: G is [b, d, d] rather than [d, d]
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    Args:
        G: [b, d, d]
        steps: int
    Returns:
        X: [b, d, d]
    """
    assert len(G.shape) == 3
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.transpose(1, 2)
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X


@torch.compile
def fast_weight_swish_glu_weight_norm_mini_batch_apply(
    # Fast weights
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    # Sequence tensors
    q: torch.Tensor,   # [B*H, L, d]
    k: torch.Tensor,   # [B*H, L, d]
    v: torch.Tensor,   # [B*H, L, d]
    # Per-token learning rates (broadcast on feature dim)
    lr0: torch.Tensor, # [B*H, L, 1]
    lr1: torch.Tensor, # [B*H, L, 1]
    lr2: torch.Tensor, # [B*H, L, 1]
    # TTT schedule (stable list of TTTOperator)
    ttt_ua_order: list,
    # Newton–Schulz steps (Muon)
    muon_update_steps: int = 0,
    # Chunking controls
    chunk_size: int = 0,                       # 0/None => no chunking (full segment)
    apply_before_update: bool = False,         # True => apply then update (shifted block-causal)
    skip_update_on_last_subchunk: bool = False,# True => last subchunk apply-only
    # EWC flags/tensors (optional, tensor-only)
    ewc_enabled: bool = False,
    lambda_ewc: float = 0.0,
    fisher_alpha: float = 0.95,
    fisher_mode: int = 1,   # 0: abs(update), 1: sq(update)
    w0_star: Optional[torch.Tensor] = None,
    w1_star: Optional[torch.Tensor] = None,
    w2_star: Optional[torch.Tensor] = None,
    F0: Optional[torch.Tensor] = None,
    F1: Optional[torch.Tensor] = None,
    F2: Optional[torch.Tensor] = None,
    # Re-anchoring policy inside the forward
    # 0: global/online (no mid-forward re-anchoring)
    # 1: streaming (per-chunk copy)
    # 2: ema (per-chunk EMA with anchor_beta)
    anchor_mode: int = 0,
    anchor_beta: float = 0.95,
):
    """
    Returns:
      output: [B*H, L_total, d]
      w0, w1, w2: final fast weights after the last operator segment
    """
    # --- Safety: ensure anchors/Fishers are detached constants wrt autograd graph ---
    if w0_star is not None:
        w0_star = w0_star.detach()
        w1_star = w1_star.detach()
        w2_star = w2_star.detach()
    if F0 is not None:
        F0 = F0.detach()
        F1 = F1.detach()
        F2 = F2.detach()

    # Pre-norms for channel-wise renorming (detach to avoid grads through norms)
    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)
    w2_norm = w2.detach().norm(dim=1, keepdim=True)

    output_chunks = []
    do_chunk = (chunk_size is not None) and (chunk_size > 0)
    do_anchor = (anchor_mode != 0) and (w0_star is not None)
    one_minus_beta = 1.0 - anchor_beta

    for start, end, update, apply in ttt_ua_order:
        w0_now, w1_now, w2_now = w0, w1, w2

        n_chunks = (end - start + chunk_size - 1) // chunk_size if do_chunk else 1
        for ch in range(n_chunks):
            s = start + ch * chunk_size if do_chunk else start
            e = min(end, s + chunk_size) if do_chunk else end
            last_sub = (ch == n_chunks - 1)

            if update:
                # ---- Optional: apply BEFORE update (shifted block-causal) ----
                if apply and apply_before_update:
                    qi = q[:, s:e, :]
                    oi = (F.silu(qi @ w0_now, inplace=False) * (qi @ w2_now)) @ w1_now
                    output_chunks.append(oi)

                # ---- Compute per-subchunk grads/updates ----
                ki, vi = k[:, s:e, :], v[:, s:e, :]
                lr0i, lr1i, lr2i = lr0[:, s:e, :], lr1[:, s:e, :], lr2[:, s:e, :]

                gate_before_act = ki @ w0_now
                hidden_before_mul = ki @ w2_now
                hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

                dhidden = vi @ w1_now.transpose(-1, -2)
                dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
                dgate = dhidden * hidden_before_mul
                dgate_before_act = silu_backprop(dgate, gate_before_act)

                w1_grad = zeropower_via_newtonschulz5(
                    (hidden * lr1i).transpose(-1, -2) @ vi, muon_update_steps
                )
                w0_grad = zeropower_via_newtonschulz5(
                    (ki * lr0i).transpose(-1, -2) @ dgate_before_act, muon_update_steps
                )
                w2_grad = zeropower_via_newtonschulz5(
                    (ki * lr2i).transpose(-1, -2) @ dhidden_before_mul, muon_update_steps
                )

                # ---- Skip weight update for the last subchunk if requested ----
                if not (skip_update_on_last_subchunk and last_sub):

                    # LaCT fast-weight update (out-of-place arithmetic)
                    w0_now = w0_now + w0_grad
                    w1_now = w1_now + w1_grad
                    w2_now = w2_now + w2_grad

                    # ---- (Optional) EWC Fisher update + proximal shrinkage ----
                    if ewc_enabled and (w0_star is not None) and (F0 is not None):
                        
                        if fisher_mode == 0:
                            # MAS-like: importance ~ |Δθ|
                            s0 = w0_grad.detach().abs().to(dtype=torch.float32)
                            s1 = w1_grad.detach().abs().to(dtype=torch.float32)
                            s2 = w2_grad.detach().abs().to(dtype=torch.float32)
                        elif fisher_mode == 1:
                            # EWC-like: importance ~ (Δθ)^2
                            s0 = (w0_grad.detach() * w0_grad.detach()).to(dtype=torch.float32)
                            s1 = (w1_grad.detach() * w1_grad.detach()).to(dtype=torch.float32)
                            s2 = (w2_grad.detach() * w2_grad.detach()).to(dtype=torch.float32)
                        elif fisher_mode == 2:
                            # SI-style: importance ~ |g ⊙ (w_now - anchor)|
                            # If we set prev_w0 = w0_now (|g ⊙ Δθ|), this is identical to EWC squared updates
                            delta0 = (w0_now - w0_star).detach()
                            delta1 = (w1_now - w1_star).detach()
                            delta2 = (w2_now - w2_star).detach()
                            s0 = (w0_grad.detach() * delta0).abs().to(dtype=torch.float32)
                            s1 = (w1_grad.detach() * delta1).abs().to(dtype=torch.float32)
                            s2 = (w2_grad.detach() * delta2).abs().to(dtype=torch.float32)
                        else:
                            raise NotImplementedError(f"Unknown fisher_mode: {fisher_mode}")

                        # If Fisher lacks batch dim, mean-reduce over batch
                        if s0.shape != F0.shape and s0.dim() >= 1 and F0.dim() >= 1 and s0.shape[0] != F0.shape[0]:
                            s0 = s0.mean(dim=0)
                            s1 = s1.mean(dim=0)
                            s2 = s2.mean(dim=0)

                        F0.mul_(fisher_alpha).add_(s0, alpha=(1.0 - fisher_alpha))
                        F1.mul_(fisher_alpha).add_(s1, alpha=(1.0 - fisher_alpha))
                        F2.mul_(fisher_alpha).add_(s2, alpha=(1.0 - fisher_alpha))

                        # Proximal shrink (out-of-place)
                        w0_now = w0_now - lambda_ewc * F0 * (w0_now - w0_star)
                        w1_now = w1_now - lambda_ewc * F1 * (w1_now - w1_star)
                        w2_now = w2_now - lambda_ewc * F2 * (w2_now - w2_star)

                    # Renormalize channels (out-of-place)
                    w0_now = w0_now / (w0_now.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
                    w1_now = w1_now / (w1_now.norm(dim=1, keepdim=True) + 1e-5) * w1_norm
                    w2_now = w2_now / (w2_now.norm(dim=1, keepdim=True) + 1e-5) * w2_norm

                    # ---- Per-chunk re-anchoring (streaming/ema) ----
                    if do_anchor:

                        # only mutate anchors in-place if shapes match [B*H, ...]
                        same_shape = (
                            (w0_star.shape == w0_now.shape) and
                            (w1_star.shape == w1_now.shape) and
                            (w2_star.shape == w2_now.shape)
                        )
                        if same_shape:
                            if anchor_mode == 1:
                                # streaming-copy
                                w0_star.copy_(w0_now)
                                w1_star.copy_(w1_now)
                                w2_star.copy_(w2_now)
                            elif anchor_mode == 2:
                                # streaming-ema
                                w0_star.mul_(anchor_beta).add_(w0_now, alpha=one_minus_beta)
                                w1_star.mul_(anchor_beta).add_(w1_now, alpha=one_minus_beta)
                                w2_star.mul_(anchor_beta).add_(w2_now, alpha=one_minus_beta)
                        # else: shapes not compatible (likely per-head anchors) -> skip silently

                # ---- Apply AFTER update (default) ----
                if apply and not apply_before_update:
                    qi = q[:, s:e, :]
                    oi = (F.silu(qi @ w0_now, inplace=False) * (qi @ w2_now)) @ w1_now
                    output_chunks.append(oi)

            else:
                # ---- No update — just apply ----
                if apply:
                    qi = q[:, s:e, :]
                    oi = (F.silu(qi @ w0_now, inplace=False) * (qi @ w2_now)) @ w1_now
                    output_chunks.append(oi)

        # Commit segment fast weights
        w0, w1, w2 = w0_now, w1_now, w2_now

    output = torch.cat(output_chunks, dim=1) if len(output_chunks) > 0 else torch.zeros_like(q)
    return output, w0, w1, w2


class FastWeightGluMLPMultihead(nn.Module):
    """
    On init of fast_weight:

    Let's start with the magnitude of the value.
    value_proj is initialized with uniform distribution with range [-1.0/sqrt(d), 1.0/sqrt(d)]
        x is layernormed. So during init, value is unit norm total (not per head, per head is 1.0/sqrt(num_head))
        After silu, value is around norm of 2.7 per head.  (why? seems wired)

    Then for the fast weight, assume initial lr = 0.
    Then with l2_norm of q,k, input is unit normed.
    if w0 is initialized with kaiming, relu(w0 @ q) is unit normed.
    Then w1 is initialized with kaiming, so w1 @ relu(w0 @ q) is of norm sqrt(2) per head
    Since I compute total norm, it is sqrt(2) * sqrt(num_head), which is around 2.7 for dim=512, num_head=4.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        inter_multi: int = 1,
        bias: bool = False,
        base_lr=0.01,
        muon_update_steps=0,
        chunk_size=0,
        apply_before_update=False,
        skip_update_on_last_subchunk=False
    ):
        super().__init__()
        self.dim = dim
        assert dim % head_dim == 0
        self.num_heads = dim // head_dim
        self.muon_update_steps = muon_update_steps
        self.chunk_size = chunk_size
        self.apply_before_update = apply_before_update
        self.skip_update_on_last_subchunk = skip_update_on_last_subchunk

        d_in = d_out = head_dim
        d_h = int(head_dim * inter_multi)

        gain = math.sqrt(2)  # for relu activations
        self.w0 = nn.Parameter(
            torch.randn(self.num_heads, d_in, d_h) * gain / math.sqrt(d_in)
        )  # [d_h * num_heads,  d_in]
        self.w1 = nn.Parameter(
            torch.randn(self.num_heads, d_h, d_out) * gain / math.sqrt(d_h)
        )  # [d_in * num_heads,  d_h]
        self.w2 = nn.Parameter(
            torch.randn(self.num_heads, d_in, d_h) * gain / math.sqrt(d_in)
        )  # [d_h * num_heads,  d_in]

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.c_proj = nn.Linear(dim, dim, bias=bias)

        self.lr_dim = self.num_heads
        self.lr_fc = nn.Linear(dim, self.lr_dim * 3)
        self.base_lr_inv = inv_softplus(base_lr)

        self.o_norm = torch.nn.RMSNorm(head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x: torch.Tensor, info={}, *args):
        """
        x: (b, l, d)
        """
        qkv = F.silu(self.to_qkv(x), inplace=True)  # Silu - Linear
        q, k, v = rearrange(
            qkv, "b l (qkv h d) -> qkv (b h) l d",
            qkv=3, h=self.num_heads
        )
        q = q / (q.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)
        k = k / (k.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)

        with torch.autocast(device_type="cuda", enabled=False):
            lr = self.lr_fc(x.float())  # [b, l, lr_dim]

        lr = torch.nn.functional.softplus(lr.float() + self.base_lr_inv)
        lr0, lr1, lr2 = rearrange(
            lr, "b l (lrs h d) -> lrs (b h) l d",
            lrs=3, h=self.num_heads
        )

        if "w0" in info:
            assert "w1" in info and "w2" in info
            w0 = info["w0"]
            w1 = info["w1"]
            w2 = info["w2"]
        else:
            w0 = self.w0.repeat(x.shape[0], 1, 1)
            w1 = self.w1.repeat(x.shape[0], 1, 1)
            w2 = self.w2.repeat(x.shape[0], 1, 1)

        # EWC flags/tensors in info (optional)
        ewc_enabled = bool(info.get("ewc_enabled", False))
        lambda_ewc = float(info.get("lambda_ewc", 0.0))
        fisher_alpha = float(info.get("fisher_alpha", 0.95))
        fisher_mode = int(info.get("fisher_mode", 0))

        w0_star = info.get("w0_star", None)
        w1_star = info.get("w1_star", None)
        w2_star = info.get("w2_star", None)
        F0 = info.get("F0", None)
        F1 = info.get("F1", None)
        F2 = info.get("F2", None)

        anchor_mode = getattr(self, "anchor_mode", 0)
        anchor_beta = getattr(self, "anchor_beta", 0)

        output, w0, w1, w2 = fast_weight_swish_glu_weight_norm_mini_batch_apply(
            # TTT parameters
            w0, w1, w2, q, k, v, lr0, lr1, lr2, info["ttt_op_order"],
            muon_update_steps=self.muon_update_steps,
            # Chunking controls
            chunk_size=self.chunk_size,
            apply_before_update=self.apply_before_update,
            skip_update_on_last_subchunk=self.skip_update_on_last_subchunk,
            # EWC flags/tensors (optional, tensor-only)
            ewc_enabled=ewc_enabled,
            lambda_ewc=lambda_ewc,
            fisher_alpha=fisher_alpha,
            fisher_mode=fisher_mode,
            w0_star=w0_star, w1_star=w1_star, w2_star=w2_star,
            F0=F0, F1=F1, F2=F2,
            # Re-anchoring policy
            anchor_mode=anchor_mode,
            anchor_beta=anchor_beta
        )

        output = self.o_norm(output)
        output = rearrange(
            output, "(b h) l d -> b l (h d)", h=self.num_heads, b=x.shape[0]
        )

        output = self.c_proj(output)
        return output, {"w0": w0, "w1": w1, "w2": w2}

    def extra_repr(self) -> str:
        return (f"w0 shape: {self.w0.shape}, w1 shape: {self.w1.shape}, w2 shape: {self.w2.shape}, "
                f"Chunk size: {self.chunk_size}, "
                f"Muon update steps: {self.muon_update_steps}, "
                f"Base lr: {math.log(1 + math.exp(self.base_lr_inv))}, ")
