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
            "bucket_slice": {},
        }

        # track readiness
        for k, p in named_parameters:
            p_mb = p.data.numel() * p.data.element_size() / (1024 ** 2)  # parameter size in MB
            self.p_ref["param"][k] = p
            self.p_ref["ready"][k] = False
            is_ready = lambda g, k=k: self.ready_param(k)
            p.register_post_accumulate_grad_hook(is_ready)

        # create buckets
        p_names = list(self.p_ref["ready"].keys())
        p_names.reverse()  # reverse order since thats usually order of grad computation
        bucket_size = len(p_names) // num_buckets
        # TODO: do based on p_mb, for now just uniformly
        self.buckets = [p_names[i * bucket_size:(i + 1) * bucket_size] for i in range(num_buckets)]
        self.bucket_buffers = []
        for i, b in enumerate(self.buckets):
            buffer_size = 0
            for kp in b:
                self.p_ref["bucket"][kp] = i
                n_weights = self.p_ref["param"][kp].numel()
                start, end = buffer_size, buffer_size + n_weights
                self.p_ref["bucket_slice"][kp] = slice(start, end)
                buffer_size += n_weights
                device, dtype = self.p_ref["param"][kp].device, self.p_ref["param"][kp].dtype  # assumption that all have the same device and dtype
            self.bucket_buffers.append(torch.zeros((buffer_size,), dtype=dtype, device=device))
        self.bucket_work = []

    def ready_param(self, k):
        self.p_ref["ready"].update({k: True})  # set param

        b_i = self.p_ref["bucket"][k]
        p_sl = self.p_ref["bucket_slice"][k]
        self.bucket_buffers[b_i][p_sl].copy_(self.p_ref["param"][k].grad.view(-1))  # copy the flattened grad into buffer
        b_i_ready = all([self.p_ref["ready"][k] for k in self.buckets[b_i]])
        if b_i_ready:
            self.bucket_buffers[b_i].div_(self.world_size)
            work_b = dist.all_reduce(self.bucket_buffers[b_i], async_op=True)
            self.bucket_work.append(work_b)

        if len(self.bucket_work) == len(self.buckets):
            for b_i, bw in enumerate(self.bucket_work):
                bw.wait()
                for k in self.buckets[b_i]:
                    b_i = self.p_ref["bucket"][k]
                    p_sl = self.p_ref["bucket_slice"][k]
                    g_sh = self.p_ref["param"][k].grad.shape
                    self.p_ref["param"][k].grad.copy_(self.bucket_buffers[b_i][p_sl].view(g_sh))

    def reset(self):
        self.bucket_work = []
        for buf in self.bucket_buffers:
            buf.zero_()
        for k in self.p_ref["ready"].keys():
            self.p_ref["ready"][k] = False


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

    for i in range(2):
        y = ddp_model(x)
        loss = y.mean()
        loss.backward()
        optimizer.step()

    print_rank_n(f"rank {rank}/{world_size}: \nw: {ddp_model.module.c_fc.weight.data[:1, :5]}\nb: {ddp_model.module.c_fc.bias.data[:5]}\ny: {y}")
    dist.destroy_process_group()
    print("done")