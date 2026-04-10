import torch
from lmtracer.core.logger import lmtracer_log
from lmtracer.cuda import globaltimer

def sync_worker_gpu_time():
    rank = torch.distributed.get_rank()

    torch.distributed.barrier()
    torch.cuda.synchronize()
    
    if rank == 0:
        t0_gpu = globaltimer.read_reg().to(torch.float64)
    else:
        t0_gpu = torch.zeros(1, dtype=torch.float64, device="cuda")

    torch.distributed.broadcast(t0_gpu, src=0)

    torch.cuda.synchronize()
    ti_gpu = globaltimer.read_reg().cuda()
    offset = int(t0_gpu.item() - ti_gpu.item())

    return offset