"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist(devices=None):
    if dist.is_initialized():
        return

    # torchrun かどうかを判定
    using_torchrun = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    # ★ torchrun でないときだけ可視 GPU を制限する
    if not using_torchrun and devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))

    if using_torchrun:                     # ===== torchrun =====
        rank        = int(os.environ["RANK"])
        world_size  = int(os.environ["WORLD_SIZE"])
        local_rank  = int(os.environ.get("LOCAL_RANK", 0))
        backend     = "nccl" if th.cuda.is_available() else "gloo"
        if th.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            th.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend,
                                init_method="env://",
                                world_size=world_size,
                                rank=rank)
    else:                                  # ===== 1 プロセス実行 =====
        backend = "nccl" if th.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend,
                                init_method="file:///tmp/ddp_init",
                                world_size=1, rank=0)
        
def dev(device_number=0):
    """
    Get the device to use for torch.distributed.
    """
    if isinstance(device_number, (list, tuple)):  # multiple devices specified
        return [dev(k) for k in device_number]    # recursive call
    if th.cuda.is_available():
        device_count = th.cuda.device_count()
        if device_count == 1:
            return th.device(f"cuda")
        else:
            if device_number < device_count:  # if we specify multiple devices, we have to be specific
                return th.device(f'cuda:{device_number}')
            else:
                raise ValueError(f'requested device number {device_number} (0-indexed) but only {device_count} devices available')
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    #print('mpicommworldgetrank', MPI.COMM_WORLD.Get_rank())
    mpigetrank=0
   # if MPI.COMM_WORLD.Get_rank() == 0:
    if mpigetrank==0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
   # data = MPI.COMM_WORLD.bcast(data)
  #  print('mpibacst', MPI.COMM_WORLD.bcast(data))
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    for p in params:
        with th.no_grad():
            dist.broadcast(p, src=0)
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    #for p in params:
    #    with th.no_grad():
    #        dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()

def get_rank() -> int:
    return 0 if not dist.is_initialized() else dist.get_rank()

def get_world_size() -> int:
    return 1 if not dist.is_initialized() else dist.get_world_size()