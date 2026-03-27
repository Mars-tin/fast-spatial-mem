#
# This file is derived from code originally released by the DINO authors
# under the Apache License, Version 2.0.
#
# It also incorporates modifications originating from code released by
# Tianyuan Zhang and Hao Tan, with attribution retained here.
#
# Original work:
# Copyright (c) the DINO authors
#
# Subsequent modifications:
# Copyright (c) 2025 Tianyuan Zhang, Hao Tan
#
# Further modifications:
# Copyright (c) 2026 Martin Ziqiao Ma
#
# This file is distributed under the licensing terms provided in the
# repository LICENSE and NOTICE files.
#
from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.nn import functional as F


def get_class_by_name(name: str):
    """Import and return a class from its fully qualified name.

    Args:
        name (str): Fully qualified class name, e.g. "package.module.ClassName".

    Returns:
        type: The class object resolved from the given name.
    """
    module_name, class_name = name.rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


class SelfAttention(nn.Module):
    """
    Self-attention layer
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L68-L92
    """

    def __init__(
        self,
        dim,
        head_dim,
        use_qk_norm=True,
        causal=False,
        bias=False,
    ):
        super().__init__()
        assert dim % head_dim == 0
        self.dim = dim
        self.head_dim = head_dim

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.c_proj = nn.Linear(dim, dim, bias=bias)
        self.use_qk_norm = use_qk_norm

        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(head_dim)
            self.k_norm = nn.RMSNorm(head_dim)

        self.causal = causal

    def forward(self, x, *args):
        """
        x: (b, l, d)
        """
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, "b l (qkv nh dh) -> qkv b nh l dh", qkv=3, dh=self.head_dim)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        x = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        x = rearrange(x, "b nh l dh -> b l (nh dh)")

        x = self.c_proj(x)
        return x, {}


class MLP(nn.Module):

    def __init__(self, dim, inter_multi=4, bias=False):
        super().__init__()
        intermediate_dim = int(dim * inter_multi)
        self.c_fc = nn.Linear(dim, intermediate_dim, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(intermediate_dim, dim, bias=bias)

    def forward(self, x, *args):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x, {}


class Block(nn.Module):
    def __init__(self, dim, bias, block_config):
        super().__init__()
        module_list = []
        self.length_dim_list = []

        for _, module_config in enumerate(block_config):
            CLASS = get_class_by_name(module_config["type"])
            module = nn.ModuleDict(
                {
                    "ln": LayerNorm(dim, bias=bias),
                    "f": CLASS(dim=dim, bias=bias, **module_config["params"]),
                }
            )

            module_list.append(module)
            self.length_dim_list.append(module_config.get("length_dim", "vl"))

        self.module_list = nn.ModuleList(module_list)

    def forward(self, x, info):
        results = {}
        for module, length_dim in zip(self.module_list, self.length_dim_list):
            residual = x
            x = module["ln"](x)

            if length_dim == "l":
                b, vl, d = x.shape
                l = info["num_img_tokens"]
                x = x.reshape(b * (vl // l), l, d)
                x, result = module["f"](x, info)
                x = x.reshape(b, vl, d)
            else:
                x, result = module["f"](x, info)

            x = residual + x
            results.update(result)
        return x, results
