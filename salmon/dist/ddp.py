import time

import torch
import torch.nn as nn
import torch.distributed as dist

from salmon.computations.torch import MLP


class DistributedDataParallel(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

        sd = self.state_dict()
        for k in sd.keys():  # ensure all initial replicas are the same
            dist.broadcast(sd[k], 0)  # by default async_op = False

        # TODO: implement buckets
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

    x = torch.randn((4, 8))
    model = MLP(8, 4, bias=True).cuda()
    ddp_model = DistributedDataParallel(model)
    optimizer = torch.optim.AdamW(ddp_model.parameters())

    bs_r = x.shape[0] // world_size
    x = x[rank * bs_r:(rank + 1) * bs_r].cuda()
    y = ddp_model(x)
    loss = y.log().sum()
    loss.backward()
    optimizer.step()


    time.sleep(rank)
    print(f"rank {rank}: ", y)
    # print(f"rank {rank}/{world_size}: \nw: {ddp_model.module.weight.data}\nb: {ddp_model.module.bias.data}")
    dist.destroy_process_group()
    print("done")


"""
DistributedDataParallel notes

data parallel (DP) training. 
- model weights and optimizer replica on each device,
- synchronization happens during backward pass. optimizer does not need to be aware this is happening

TODO:
reducer: 
- backward hook highlights the current param as "ready for sync" in reducer
- when all grads in one bucket are ready, launch async all_reduce
- when all buckets ready, barrier so all all_reduce can finish
- when everything done, write output to param.grad

- important to invoke all_reduce in the same order so just do it in bucket index order

why is last step not done by default???
why can't you just do async all_reduce 
"""