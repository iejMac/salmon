import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Shard, Replicate, distribute_tensor, distribute_module, init_device_mesh

from salmon.computations.torch import MLP
from salmon.dist.utils import print_rank_n


def _get_submodules(root_module):
    modules = []
    visited_modules = set()
    def dfs(module):
        # TODO: if already wrapped
        # if module is not root_module and _get_module_fsdp_state(module) is not None:
        #     return
        visited_modules.add(module)
        for submodule in module.children():
            if submodule not in visited_modules:
                dfs(submodule)
        modules.append(module)
    dfs(root_module)
    return modules


class FSDPModule(nn.Module):
    def __init__(self, root_module, device_mesh):
        super().__init__()
        self.root_module = root_module
        self.device_mesh = device_mesh

        self.submodules = _get_submodules(self.root_module)
        self._shard()

    def _shard(self):
        for mod in self.submodules:
            for n, p in mod.named_parameters(recurse=False):
                fsdp_placements = [Shard(0)]
                d_param = torch.nn.Parameter( # shard params on dim 0
                    distribute_tensor(p.data, self.device_mesh, fsdp_placements)
                )
                mod.register_parameter(n, d_param)

    def _unshard(self):
        for mod in self.submodules:
            for n, p in mod.named_parameters(recurse=False):
                d_param = torch.nn.Parameter( # shard params on dim 0
                    p.data.full_tensor()
                )
                mod.register_parameter(n, d_param)

    def forward(self, *args, **kwargs):
        self._unshard()
        out = self.root_module(*args, **kwargs)
        self._shard()
        return out


if __name__ == "__main__":
    device_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("sh",))
    ws = dist.get_world_size()
    rank = dist.get_rank()

    shard_mesh = device_mesh["sh"]

    x = torch.randn(2, 8).cuda()

    mlp = MLP(8, 4, bias=True)
    fsdp_mlp = FSDPModule(mlp, shard_mesh)

    for n, p in fsdp_mlp.root_module.named_parameters():
        print_rank_n(n, type(p.data))
        print_rank_n(p.data.shape)
        print_rank_n(p.data.to_local().shape)


    # we need to call in the param group.
    # wrapping needed!
    y = fsdp_mlp(x)

    print(y.shape)



"""
NOTE:
- If we do sharding properly and the wrapped FSDP only contains cur device shard 
  then optimizer should be fine since we just input dist_model.params()

TODO:
- model sharding
    - type:
        - normal
        - hybrid (DDP between nodes, FSDP within)
    - wrapping policy
        - user defines which layers should be wrapped in FSDP
    - max size shards
        - get device size, shard uniformly max-size shards 

- computation/communication in fwd/bwd


REFS:
- FSDP2 debugging: https://x.com/HamelHusain/status/1800315287574847701


FSDP2:
- you create an FSDPParamGroup for each layer, there you also not that this layer has been wrapped
- then you run fully_shard on the full model (not sure why?)

fully_shard creates FDSPParamGroup

FSDPParamGroup creates FSDPParam

FSDPParam does DTensor stuff? 

What is FSDPState?


"""

'''
if __name__ == "__main__":
    torch.manual_seed(0)
    dist.init_process_group(backend="nccl", init_method="env://")  
    rank, world_size = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)

    X = torch.randn((2, 8)).cuda()
    model = MLP(8, 4, bias=True).cuda()
    # X = torch.randn((4, 1024)).cuda()
    # model = torch.nn.Sequential(*[MLP(1024, 4, bias=True) for _ in range(10)]).cuda()

    dist_model = FSDP(model)
    optimizer = torch.optim.AdamW(dist_model.parameters())

    bs_r = X.shape[0] // world_size
    x = X[rank * bs_r:(rank + 1) * bs_r].cuda() * 100.0

    for i in range(4):
        y = dist_model(x)
        loss = y.mean()
        loss.backward()
        optimizer.step()

    Y = dist_model(X)

    print_rank_n(Y.sum(), rank=rank)
    dist.destroy_process_group()
    print("done")
'''