import time

import torch
import torch.nn as nn
import torch.distributed as dist


class DistributedDataParallel(nn.Module):
    """
    - passes through forward
    - all_reduce gradients during backward
    - since we do this in backward, the optimizer doesn't need to know about DDP at all

    TODO:
    - backward hooks
    """
    def __init__(self, module):
        super().__init__()
        self.module = module

        sd = self.state_dict()
        for k in sd.keys():
            dist.broadcast(sd[k], 0)  # ensure all initial replicas are the same
        # TODO: do we need barrier here?
        # dist.barrier()

        for p in self.parameters():
            p.register_hook(lambda g: dist.all_reduce(g))  # when gradient is computed, sync

    def forward(self, *args, **kwargs):
        out = self.module(*args, **kwargs)
        return out

if __name__ == "__main__":
    torch.manual_seed(0)
    dist.init_process_group(backend="nccl", init_method="env://")  
    rank, world_size = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)

    if rank == 1: # HACK: remove
        time.sleep(2)

    x = torch.randn((2, 8))
    model = nn.Linear(8, 4).cuda()
    ddp_model = DistributedDataParallel(model)
    optimizer = torch.optim.AdamW(ddp_model.parameters())

    bs_r = x.shape[0] // world_size
    x = x[rank * bs_r:(rank + 1) * bs_r].cuda()
    y = ddp_model(x)
    loss = y.log().sum()
    loss.backward()
    optimizer.step()

    print(f"rank {rank}/{world_size}: \nw: {ddp_model.module.weight.data}\nb: {ddp_model.module.bias.data}")
    dist.destroy_process_group()
    print("done")
