import os
import torch
import torch.distributed as dist
import random
import numpy as np


def setup_ddp():
    if not torch.cuda.is_available():
        raise RuntimeError("cuda is not available for DDP")

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")
    is_master = local_rank == 0

    return local_rank, device, is_master


def cleanup_ddp():
    dist.destroy_process_group()


def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
