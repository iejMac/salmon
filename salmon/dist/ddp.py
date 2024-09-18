import time
import math

import torch
import torch.nn as nn
import torch.distributed as dist

from salmon.computations.torch import MLP
from salmon.dist.utils import print_rank_n


class DistributedDataParallel(nn.Module):
    def __init__(self, module, num_buckets=2, world_size=None):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size() if world_size is None else world_size

        # TODO: maybe just replicate the module on each DP rank from a mesh?
        sd = self.module.state_dict()
        for k in sd.keys():  # ensure all initial replicas are the same
            dist.broadcast(sd[k], 0)  # by default async_op = False
        self.reducer = Reducer(self.module.named_parameters(), world_size=world_size, num_buckets=num_buckets)

    def forward(self, *args, **kwargs):
        # TODO: find_unused_parameters. This will hang if some params don't become ready
        self.reducer.unready_params()  # prepare for bwd
        out = self.module(*args, **kwargs)
        return out


class Reducer:
    def __init__(self, named_parameters, world_size, num_buckets=1):
        self.num_buckets = num_buckets
        self.world_size = world_size
        self.p_ready = {}
        self.p_ref = {}

        # track readiness
        for k, p in named_parameters:
            p_mb = p.data.numel() * p.data.element_size() / (1024 ** 2)  # parameter size in MB
            self.p_ready[k] = False
            self.p_ref[k] = p
            is_ready = lambda g, k=k: self.ready_param(k)
            p.register_post_accumulate_grad_hook(is_ready)

        # create buckets
        # TODO: do based on p_mb, for now just uniformly
        p_names = list(self.p_ready.keys())
        p_names.reverse()  # reverse order since thats usually order of grad computation
        bucket_size = len(p_names) // num_buckets
        self.buckets = [p_names[i * bucket_size:(i + 1) * bucket_size] for i in range(num_buckets)]
        self.b_work = [[] for _ in range(num_buckets)]

        self.p2b = {}  # useful mapping
        for i, b in enumerate(self.buckets):
            for kp in b:
                self.p2b[kp] = i

    def ready_param(self, k):
        self.p_ready.update({k: True})  # set param

        # TODO: do we need to worry about race conditions here?
        # what if two parameters set p_ready at the same time?
        i = self.p2b[k]
        b_i_ready = all([self.p_ready[k] for k in self.buckets[i]])
        if b_i_ready:
            for k in self.buckets[i]:  # sync bucket
                work_k = dist.all_reduce(self.p_ref[k].grad, async_op=True)  # async all_reduce for bucket
                self.b_work[i].append(work_k)

        if all(self.b_work):  # wait for all params to reduce
            for bucket_works in self.b_work:
                for work in bucket_works:
                    work.wait()  # block until all async ops are done
            for k, p in self.p_ref.items(): # TODO: there must be a cleaner way of doing this, this can be done in parallel etc.
                p.grad /= self.world_size

    def unready_params(self):
        self.b_work = [[] for _ in range(self.num_buckets)]
        for k in self.p_ready:
            self.p_ready[k] = False


if __name__ == "__main__":
    torch.manual_seed(0)
    dist.init_process_group(backend="nccl", init_method="env://")  
    rank, world_size = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)

    x = torch.randn((4, 8))
    model = MLP(8, 4, bias=True).cuda()
    ddp_model = DistributedDataParallel(model, num_buckets=2)
    optimizer = torch.optim.AdamW(ddp_model.parameters())

    bs_r = x.shape[0] // world_size
    x = x[rank * bs_r:(rank + 1) * bs_r].cuda() * 100.0

    for i in range(4):
        y = ddp_model(x)
        loss = y.log().mean()
        loss.backward()
        optimizer.step()

    print_rank_n(f"rank {rank}/{world_size}: \nw: {ddp_model.module.c_fc.weight.data[:1, :5]}\nb: {ddp_model.module.c_fc.bias.data[:5]}\ny: {y}")
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