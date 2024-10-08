import time
import math

import torch
import torch.nn as nn
import torch.distributed as dist

from salmon.computations.torch import MLP
from salmon.dist.reducer import Reducer
from salmon.dist.utils import print_rank_n


class DistributedDataParallel(nn.Module):
    def __init__(self, module, bucket_cap_mb=50.0):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.bucket_cap_mb = bucket_cap_mb

        # TODO: maybe just replicate the module on each DP rank from a mesh?
        sd = self.module.state_dict()
        for k in sd.keys():  # ensure all initial replicas are the same
            dist.broadcast(sd[k], 0)  # by default async_op = False
        self.module.load_state_dict(sd)

        # reducer handles gradient bucketing and asynchronousa all_reduce
        self.reducer = Reducer(self.module.named_parameters(), world_size=self.world_size, bucket_cap_mb=self.bucket_cap_mb)

    def forward(self, *args, **kwargs):
        # TODO: find_unused_parameters. This will hang if some params don't become ready
        self.reducer.reset_state()  # prepare for bwd
        out = self.module(*args, **kwargs)
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    dist.init_process_group(backend="nccl", init_method="env://")  
    rank, world_size = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)

    X = torch.randn((2, 8)).cuda()
    model = MLP(8, 4, bias=True).cuda()
    # X = torch.randn((4, 1024)).cuda()
    # model = nn.Sequential(*[MLP(1024, 4, bias=True) for _ in range(10)]).cuda()

    ddp_model = DistributedDataParallel(model, bucket_cap_mb=15.0)
    optimizer = torch.optim.AdamW(ddp_model.parameters())

    bs_r = X.shape[0] // world_size
    x = X[rank * bs_r:(rank + 1) * bs_r].cuda() * 100.0

    for i in range(4):
        y = ddp_model(x)
        loss = y.mean()
        loss.backward()
        optimizer.step()

    Y = ddp_model(X)

    print_rank_n(Y.sum(), rank=rank)
    # print_rank_n(f"rank {rank}/{world_size}: \nw: {ddp_model.module.c_fc.weight.data[:1, :5]}\nb: {ddp_model.module.c_fc.bias.data[:5]}\ny: {y}")
    dist.destroy_process_group()
    print("done")
