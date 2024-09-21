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
        self.module.load_state_dict(sd)
        self.reducer = Reducer(self.module.named_parameters(), world_size=world_size, num_buckets=num_buckets)

    def forward(self, *args, **kwargs):
        # TODO: find_unused_parameters. This will hang if some params don't become ready
        self.reducer.reset()  # prepare for bwd
        out = self.module(*args, **kwargs)
        return out


class Reducer:
    def __init__(self, named_parameters, world_size, num_buckets=1):
        self.num_buckets = num_buckets
        self.world_size = world_size

        self.p_ref = {
            "param": {},
            "ready": {},
            "grad_buffer": {},
            "bucket": {},
            "work": {},
        }

        # track readiness
        for k, p in named_parameters:
            p_mb = p.data.numel() * p.data.element_size() / (1024 ** 2)  # parameter size in MB
            self.p_ref["param"][k] = p
            self.p_ref["ready"][k] = False
            self.p_ref["grad_buffer"][k] = torch.zeros_like(p.data)
            is_ready = lambda g, k=k: self.ready_param(k)
            p.register_post_accumulate_grad_hook(is_ready)

        # create buckets
        p_names = list(self.p_ref["ready"].keys())
        p_names.reverse()  # reverse order since thats usually order of grad computation
        bucket_size = len(p_names) // num_buckets
        # TODO: do based on p_mb, for now just uniformly
        self.buckets = [p_names[i * bucket_size:(i + 1) * bucket_size] for i in range(num_buckets)]
        for i, b in enumerate(self.buckets):
            for kp in b:
                self.p_ref["bucket"][kp] = i

    def ready_param(self, k):
        self.p_ref["ready"].update({k: True})  # set param

        b_i = self.p_ref["bucket"][k]
        b_i_ready = all([self.p_ref["ready"][k] for k in self.buckets[b_i]])
        if b_i_ready:
            for k in self.buckets[b_i]:  # sync bucket
                # https://github.com/pytorch/pytorch/blob/687e5cf8c5a511494c41bf061653e45c3c521bf0/torch/nn/parallel/distributed.py#L940
                self.p_ref["param"][k].grad.div_(self.world_size)
                work_k = dist.all_reduce(self.p_ref["param"][k].grad, async_op=True)  # async all_reduce for bucket
                # self.p_ref["grad_buffer"][k].copy_(self.p_ref["param"][k].grad)  # TODO: is this too slow? didn't see anythign like this in PT DDP
                # work_k = dist.all_reduce(self.p_ref["grad_buffer"][k], async_op=True)
                self.p_ref["work"][k] = work_k

        if len(self.p_ref["work"]) == len(self.p_ref["param"]):
            for bucket in self.buckets:
                for k in bucket:
                    self.p_ref["work"][k].wait()
                    # self.p_ref["param"][k].grad.copy_(self.p_ref["grad_buffer"][k])

    def reset(self):
        self.p_ref["work"] = {}
        for k in self.p_ref["ready"].keys():
            self.p_ref["ready"][k] = False
            self.p_ref["grad_buffer"][k].zero_()


if __name__ == "__main__":
    torch.manual_seed(0)
    dist.init_process_group(backend="nccl", init_method="env://")  
    rank, world_size = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)

    x = torch.randn((4, 8))
    model = MLP(8, 4, bias=True).cuda()
    ddp_model = DistributedDataParallel(model, num_buckets=2, world_size=world_size)
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