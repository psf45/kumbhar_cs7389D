import os
import socket
import torch
import torch.distributed as dist

def get_env_int(name, default):
    val = os.environ.get(name, None)
    return int(val) if val is not None else default

def init_distributed():
    world_size = get_env_int("WORLD_SIZE", 1)
    rank = get_env_int("RANK", 0)
    local_rank = get_env_int("LOCAL_RANK", 0)

    distributed = world_size > 1

    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    return {
        "distributed": distributed,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
    }

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0