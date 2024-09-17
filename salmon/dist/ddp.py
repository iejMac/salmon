import time
import math

import torch
import torch.nn as nn
import torch.distributed as dist

from salmon.computations.torch import MLP
from salmon.dist.utils import print_rank_n


class DistributedDataParallel(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

        sd = self.module.state_dict()
        for k in sd.keys():  # ensure all initial replicas are the same
            dist.broadcast(sd[k], 0)  # by default async_op = False

        self.reducer = Reducer(self.module.named_parameters(), num_buckets=2)

        # TODO: implement buckets
        # for p in self.parameters():  # reducer with num_buckets = num_tensors
        #     p.register_hook(lambda g: dist.all_reduce(g))  # when gradient is computed, sync

    def forward(self, *args, **kwargs):
        out = self.module(*args, **kwargs)
        return out


class Reducer:
    """
    I want both buckets and params to access the same bools
    
    p1 p2 p3 p4 p5
    |  |  |  |  |
    F  F  T  T  T
    |  |  |     |
    ----  -------
     B1     B2

    The point being, the autograd hooks set p.ready = True
    And then we just check and(p2b[p])
    """
    def __init__(self, named_parameters, num_buckets=1):
        self.p_ready = {}

        # track readiness
        for k, p in named_parameters:
            p_mb = p.data.numel() * p.data.element_size() / (1024 ** 2)  # parameter size in MB
            self.p_ready[k] = False
            is_ready = lambda g, k=k: self.ready_up(k)
            p.register_hook(is_ready)

        # create buckets
        # TODO: do based on p_mb, for now just uniformly
        p_names = list(self.p_ready.keys())
        p_names.reverse()  # reverse order since thats usually order of grad computation
        bucket_size = len(p_names) // num_buckets
        self.buckets = [p_names[i * bucket_size:(i + 1) * bucket_size] for i in range(num_buckets)]
        self.b_ready = [False for _ in range(num_buckets)]

        self.p2b = {}  # useful mapping
        for i, b in enumerate(self.buckets):
            for kp in b:
                self.p2b[kp] = i

    def reset(self):
        for k, p in self.p_ready:
            self.p_ready[k] = False

    def ready_up(self, k):
        # TODO: do python dicts support async writes?
        self.p_ready.update({k: True})  # set param

        # TODO: do we need to worry about race conditions here?
        # what if two parameters set p_ready at the same time?
        i = self.p2b[k]
        b_i_ready = all([self.p_read[k] for k in self.blocks[i]])
        if b_i_ready:
            self.b_ready[i] = b_i_ready
            self.sync_block(i)

        if all(self.b_ready):
            dist.barrier()  # wait for all params to reduce

    def sync_block(self, i):
        for k in self.blocks[i]:
            # TODO: trigger all_reduce for grad...
            pass



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

    print(ddp_model.reducer.p_ready)

    # time.sleep(rank)
    # print(f"rank {rank}/{world_size}: \nw: {ddp_model.module.c_fc.weight.data[:10]}\nb: {ddp_model.module.c_fc.bias.data}")
    dist.destroy_process_group()
    print("done")


"""
DistributedDataParallel notes

data parallel (DP) training. 
- model weights and optimizer replica on each device,
- synchronization happens during backward pass.
- optimizer does not need to be aware this is happening

TODO reducer: 
- backward hook highlights the current param as "ready for sync" in reducer
- when all grads in one bucket are ready, launch async all_reduce
- when all buckets ready, barrier so all all_reduce can finish
- when everything done, write output to param.grad

- important to invoke all_reduce in the same order so just do it in bucket index order

Qs:
- why do we need to copy in last step? why not just reduce into params.grad?
- why can't you just do async all_reduce 
"""