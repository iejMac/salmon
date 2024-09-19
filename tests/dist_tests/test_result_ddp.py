import os
import time

import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from salmon.computations.torch import MLP

from salmon.dist.ddp import DistributedDataParallel
from salmon.dist.utils import print_rank_n


def setup_distributed_environment(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)

def grad_step(x, model, optimizer):
    y = model(x)
    loss = F.mse_loss(y, x)
    loss.backward()
    optimizer.step()
    return y


def run_local(x, model_kwargs, sd, device, n=1):
    # init
    model = MLP(**model_kwargs).cuda()
    model.load_state_dict(sd)
    optimizer = optim.AdamW(model.parameters())
    # prepare data
    x = x.to(device)
    # compute
    t0 = time.time()
    for _ in range(n):
        y = grad_step(x, model, optimizer)
    tf = time.time()
    g_w, g_b = model.c_fc.weight.data, model.c_fc.bias.data

    outputs ={
        'y': y.detach().cpu(),
        'g_w': g_w.detach().cpu(),
        'g_b': g_b.detach().cpu(),
        "t_avg": (tf-t0)/n,
    }
    return outputs

def run_pytorch_ddp(rank, world_size, x, model_kwargs, sd, queue, device, n=1):
    # dist
    setup_distributed_environment(rank, world_size)
    if device == "cuda":
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"

    # init
    model = MLP(**model_kwargs).cuda()
    if rank == 0:  # DDP should sync weights from rank 0 to all in constructro
        model.load_state_dict(sd)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    optimizer = optim.AdamW(ddp_model.parameters())

    # prepare data
    bs_per_rank = x.shape[0] // world_size
    x = x[rank * bs_per_rank:(rank+1) * bs_per_rank].to(device)

    # compute
    t0 = time.time()
    for _ in range(n):
        y = grad_step(x, ddp_model, optimizer)
    tf = time.time()
    g_w, g_b = ddp_model.module.c_fc.weight.data, ddp_model.module.c_fc.bias.data

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
            't_avg': (tf-t0)/n,
        }
        queue.put(output)
    return

def run_salmon_ddp(rank, world_size, x, model_kwargs, sd, queue, device, n=1):
    # dist
    setup_distributed_environment(rank, world_size)
    if device == "cuda":
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"

    # init
    model = MLP(**model_kwargs).cuda()
    if rank == 0:  # DDP should sync weights from rank 0 to all in constructro
        model.load_state_dict(sd)
    ddp_model = DistributedDataParallel(model, world_size=world_size)
    optimizer = optim.AdamW(ddp_model.parameters())

    # prepare data
    bs_per_rank = x.shape[0] // world_size
    x = x[rank * bs_per_rank:(rank+1) * bs_per_rank].to(device)
    # compute
    t0 = time.time()
    for _ in range(n):
        y = grad_step(x, ddp_model, optimizer)
    tf = time.time()
    g_w, g_b = ddp_model.module.c_fc.weight.data, ddp_model.module.c_fc.bias.data

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
            't_avg': (tf-t0)/n,
        }
        queue.put(output)
    return


if __name__ == "__main__":
    torch.manual_seed(0)
    bs, dim = 16, 64
    model_kwargs = {'dim': dim, 'mlp_ratio': 4, 'bias': True}
    world_size = 8
    n_iterations = 8

    x = torch.randn(bs, dim, requires_grad=False) * 100
    sd = MLP(**model_kwargs).state_dict()
    device = "cuda"

    # local computation
    out_l = run_local(x, model_kwargs, sd, device, n=n_iterations)

    # pytorch ddp computation
    ctx = mp.get_context("spawn")
    queue = ctx.Manager().Queue(10000)
    mp.spawn(
        run_pytorch_ddp,
        args=(world_size, x, model_kwargs, sd, queue, device, n_iterations),
        nprocs=world_size,
        join=True
    )
    out_pt = queue.get()

    # salmon ddp computation
    ctx = mp.get_context("spawn")
    queue = ctx.Manager().Queue(10000)
    mp.spawn(
        run_salmon_ddp,
        args=(world_size, x, model_kwargs, sd, queue, device, n_iterations),
        nprocs=world_size,
        join=True
    )
    out_slmn = queue.get()

    print('local pt: ', out_l['t_avg'])
    print('ddp pt: ', out_pt['t_avg'])
    print('ddp slmn: ', out_slmn['t_avg'])

    for k in ['y', 'g_w', 'g_b']:
        print(out_l[k])
        print(out_pt[k])
        print(out_slmn[k])

        try:
            assert torch.allclose(out_l[k], out_pt[k], atol=1e-3), (k, (out_l[k] - out_pt[k]).abs().mean())
        except Exception as e:
            print(str(e))
        assert torch.allclose(out_l[k], out_slmn[k], atol=1e-3), (k, (out_l[k] - out_slmn[k]).abs().mean())
    print("passed")
