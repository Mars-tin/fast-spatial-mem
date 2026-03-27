#
#  Apache 2.0 License License
#  Copyright (c) 2026 Martin Ziqiao Ma
#
import os
import random
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist


TIMEOUT = timedelta(hours=1)
MAGIC_SEED = 42


def is_dist_launched() -> bool:
    """Return whether distributed training appears to be launched.

    This checks the ``WORLD_SIZE`` environment variable and treats values
    greater than 1 as indicating a distributed run.

    Returns:
        bool: True if ``WORLD_SIZE`` is greater than 1, otherwise False.
    """
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_ddp(backend: str = "nccl", timeout: timedelta = TIMEOUT):
    """Initialize distributed training and return runtime process metadata.

    This function supports both distributed and single-process execution. When
    distributed execution is not launched, it falls back to a single-process
    setup and returns default rank/device information. When distributed
    execution is enabled, it initializes the process group, resolves rank and
    device metadata from environment variables, seeds all random number
    generators with a rank-dependent seed, and returns a dictionary describing
    the current process.

    Args:
        backend: Distributed backend passed to ``dist.init_process_group``.
        timeout: Timeout for process group initialization.

    Returns:
        dict: Runtime metadata including rank, local rank, world size, device,
        primary-process status, and whether distributed mode is active.
    """
    if not is_dist_launched():
        # single-process fallback
        torch.cuda.set_device(0 if torch.cuda.is_available() else "cpu")
        return {
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "local_world_size": 1,
            "node_rank": 0,
            "device": torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu"),
            "is_primary": True,
            "distributed": False,
        }

    # Init pg
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=timeout,
    )

    # Ranks & sizes (robust fallbacks)
    rank = int(os.environ.get("RANK", dist.get_rank()))
    world_size = int(os.environ.get("WORLD_SIZE", dist.get_world_size()))
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
    node_rank = int(os.environ.get("NODE_RANK", os.environ.get("GROUP_RANK", rank // max(1, local_world_size))))

    # Device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        # nccl requires CUDA; if you need CPU dist, switch backend to gloo
        device = torch.device("cpu")

    # Seeding (rank-different)
    seed = MAGIC_SEED + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # optional determinism knobs:
    # torch.use_deterministic_algorithms(False)
    # torch.backends.cudnn.benchmark = True

    is_primary = (rank % local_world_size == 0)  # or rank == 0 for global primary

    if is_primary:
        print(f"[DDP] world_size={world_size}, nodes≈{world_size//local_world_size}, "
              f"gpus/node={local_world_size}, backend={backend}")

    dist.barrier()
    return {
        "seed": seed,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "local_world_size": local_world_size,
        "node_rank": node_rank,
        "device": device,
        "is_primary": is_primary,
        "distributed": True,
    }


def cleanup_ddp():
    """Synchronize and tear down the distributed process group if initialized.

    This function is a no-op when distributed execution is unavailable or has
    not been initialized.

    Returns:
        None.
    """
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def make_worker_init_fn(base_seed: int):
    """Create a DataLoader worker initialization function with rank-aware seeding.

    The returned function seeds NumPy, Python ``random``, and PyTorch using a
    deterministic combination of the base seed, distributed rank, and worker
    ID so that workers across ranks receive different seeds.

    Args:
        base_seed: Base random seed used to derive per-worker seeds.

    Returns:
        Callable: Worker initialization function for a DataLoader.
    """
    def _init_fn(worker_id: int):
        ws = dist.get_world_size() if dist.is_initialized() else 1
        rk = dist.get_rank() if dist.is_initialized() else 0
        seed = base_seed + rk * 1000 + worker_id
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
    return _init_fn


def apply_linear_scaling(config, ctx, disable_env="DISABLE_LINEAR_SCALING"):
    """Apply linear learning-rate scaling based on distributed world size.

    This scales ``config.training.lr`` proportionally to the current world
    size relative to ``config.training.base_world_size``, and inversely scales
    ``config.training.warmup_steps``. Scaling can be disabled via an
    environment variable.

    Args:
        config: Config object containing ``training.lr``,
            ``training.warmup_steps``, and optionally
            ``training.base_world_size``.
        ctx: Runtime context dictionary, typically returned by ``setup_ddp()``,
            containing at least ``world_size`` and optionally ``is_primary``.
        disable_env: Name of the environment variable used to disable scaling.

    Returns:
        float: Applied scaling factor.
    """
    world_size = ctx.get("world_size", 1)

    base_world_size = int(config.training.get("base_world_size", 8))
    disable = os.environ.get(disable_env, "0") == "1"
    
    if disable or base_world_size <= 0:
        return 1.0

    scale = world_size / float(base_world_size)

    # Apply scaling
    config.training.lr *= scale
    config.training.warmup_steps = max(1, int(int(config.training.warmup_steps) / max(1.0, scale)))

    if ctx.get("is_primary", False):
        print(f"[DDP] Scaled training hyperparams → "
              f"lr={config.training.lr:.6f}, warmup={config.training.warmup_steps}, "
              f"scale={scale:.2f} (world_size={world_size})")

    return scale
