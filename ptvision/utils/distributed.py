import os

import torch
import torch.distributed as dist


def is_distributed():
    return get_world_size() > 1


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    return local_rank


def get_node_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    rank = get_rank()
    ngpus_per_node = torch.cuda.device_count()
    node_rank = rank // ngpus_per_node
    return node_rank


def is_master():
    return get_rank() == 0


def is_local_master():
    return get_local_rank() == 0


def print_master(*args, local=True, **kwargs):
    do_print = (
        is_local_master() if local else
        is_master()
    )
    if do_print:
        print(*args, **kwargs)
