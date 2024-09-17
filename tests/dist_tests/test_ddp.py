import os
import time

import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from salmon.dist.ddp import DistributedDataParallel
from salmon.dist.utils import print_rank_n


def setup_distributed_environment(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)

def grad_step(x, model, optimizer):
    y = model(x)
    loss = y.log().sum()
    loss.backward()
    optimizer.step()
    return y


def run_local(x, sd, device):
    # init
    model = nn.Linear(8, 2).to(device)
    model.load_state_dict(sd)
    optimizer = optim.AdamW(model.parameters())
    # prepare data
    x = x.to(device)
    # compute
    y = grad_step(x, model, optimizer)
    g_w, g_b = model.weight.data, model.bias.data

    outputs ={
        'y': y.detach().cpu(),
        'g_w': g_w.detach().cpu(),
        'g_b': g_b.detach().cpu(),
    }
    return outputs

def run_pytorch_ddp(rank, world_size, x, sd, queue, device):
    # dist
    setup_distributed_environment(rank, world_size)
    if device == "cuda":
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"

    # init
    model = nn.Linear(8, 2).to(device)
    if rank == 0:  # DDP should sync weights from rank 0 to all in constructro
        model.load_state_dict(sd)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    optimizer = optim.AdamW(ddp_model.parameters())

    # prepare data
    bs_per_rank = x.shape[0] // world_size
    x = x[rank * bs_per_rank:(rank+1) * bs_per_rank].to(device)

    # compute
    y = grad_step(x, ddp_model, optimizer)
    g_w, g_b = ddp_model.module.weight.data, ddp_model.module.bias.data

    # gather outputs
    y_all = [torch.zeros_like(y) for r in range(world_size)]
    dist.all_gather(y_all, y)
    y = torch.cat(y_all)

    dist.destroy_process_group()
    if rank == 0:  # save result
        output = {
            'y': y.detach().cpu(),
            'g_w': g_w.detach().cpu(),
            'g_b': g_b.detach().cpu(),
        }
        queue.put(output)
    return

def run_salmon_ddp(rank, world_size, x, sd, queue, device):
    # dist
    setup_distributed_environment(rank, world_size)
    if device == "cuda":
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"

    # init
    model = nn.Linear(8, 2).to(device)
    if rank == 0:  # DDP should sync weights from rank 0 to all in constructro
        model.load_state_dict(sd)
    ddp_model = DistributedDataParallel(model)
    optimizer = optim.AdamW(ddp_model.parameters())

    # prepare data
    bs_per_rank = x.shape[0] // world_size
    x = x[rank * bs_per_rank:(rank+1) * bs_per_rank].to(device)
    # compute
    y = grad_step(x, ddp_model, optimizer)
    g_w, g_b = ddp_model.module.weight.data, ddp_model.module.bias.data

    # gather outputs
    y_all = [torch.zeros_like(y) for r in range(world_size)]
    dist.all_gather(y_all, y)
    y = torch.cat(y_all)

    dist.destroy_process_group()
    if rank == 0:  # save result
        output = {
            'y': y.detach().cpu(),
            'g_w': g_w.detach().cpu(),
            'g_b': g_b.detach().cpu(),
        }
        queue.put(output)
    return


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(4, 8, requires_grad=False) * 100
    sd = nn.Linear(8, 2).state_dict()
    device = "cuda"
    world_size = 2

    # local computation
    out_l = run_local(x, sd, device)
    print('local pt: ', out_l)

    # pytorch ddp computation
    ctx = mp.get_context("spawn")
    queue = ctx.Manager().Queue(10000)
    mp.spawn(
        run_pytorch_ddp,
        args=(world_size, x, sd, queue, device),
        nprocs=world_size,
        join=True
    )
    out_pt = queue.get()
    print('ddp pt: ', out_pt)

    # salmon ddp computation
    ctx = mp.get_context("spawn")
    queue = ctx.Manager().Queue(10000)
    mp.spawn(
        run_salmon_ddp,
        args=(world_size, x, sd, queue, device),
        nprocs=world_size,
        join=True
    )
    out_slmn = queue.get()
    print('ddp slmn: ', out_slmn)

    for k in ['y', 'g_w', 'g_b']:
        assert torch.allclose(out_l[k], out_pt[k]), (k, (out_l[k] - out_pt[k]).abs().mean())
        assert torch.allclose(out_l[k], out_slmn[k]), (k, (out_l[k] - out_slmn[k]).abs().mean())
    print("passed")
