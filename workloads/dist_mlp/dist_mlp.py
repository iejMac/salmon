import os
import time

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity


from salmon.computations.torch import MLP
from salmon.dist.ddp import DistributedDataParallel
from salmon.dist.utils import print_rank_n


if __name__ == "__main__":
    torch.manual_seed(0)
    dist.init_process_group(backend="nccl")  
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
 
    torch.cuda.set_device(local_rank)

    bs, dim = 2048, 1024
    n_layers = 10
    mlp_kwargs = {
        "dim": dim,
        "mlp_ratio": 4,
        "bias": True,
    }
    n_iterations = 20
    backend="salmon"
    run_name=f"{backend}_buckets_50mb"
    # run_name = backend
    print_rank_n(f"running DDP[{backend}] at {run_name}")

    x = torch.randn((bs, dim))
    model = nn.Sequential(*[MLP(**mlp_kwargs) for _ in range(n_layers)]).cuda()

    if backend == "pytorch":
        ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    elif backend == "salmon":
        ddp_model = DistributedDataParallel(model, world_size=world_size, bucket_cap_mb=50.0)
    optimizer = torch.optim.AdamW(ddp_model.parameters())

    bs_r = x.shape[0] // world_size
    x = x[rank * bs_r:(rank + 1) * bs_r].cuda() * 10.0

    t0 = time.time()
    with profile(activities=[
            ProfilerActivity.CPU, 
            ProfilerActivity.CUDA], 
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=n_iterations - 3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/{run_name}'),
            record_shapes=True, profile_memory=True, with_stack=True) as prof:
        for i in range(n_iterations):  # fake training
            prof.step()
            optimizer.zero_grad()
            y = ddp_model(x)
            loss = F.mse_loss(y, x)
            loss.backward()
            optimizer.step()
        dist.barrier()
    tf = time.time()
    print_rank_n(f"time/sample: {(tf-t0)/n_iterations}")

    dist.destroy_process_group()
    print(f"rank[{rank}] done")