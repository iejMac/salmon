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
def _get_managed_states(modules):
    params = []
    buffers = []
    # Track visited parameters/buffers to avoid visiting shared parameters and
    # buffers multiple times
    visited_params = set()
    visited_buffers = set()
    for module in modules:
        for param in module.parameters(recurse=False):
            if param not in visited_params:
                params.append(param)
                visited_params.add(param)
        for buffer in module.buffers(recurse=False):
            if buffer not in visited_buffers:
                buffers.append(buffer)
                visited_buffers.add(buffer)
    return params, buffers




def fsdp_shard(tensor, mesh):
    placements = [Shard(0)]
    return distribute_tensor(tensor, mesh, placements)


def fully_shard(module, device_mesh):

    submodules = _get_submodules(module)
    params, buffers = _get_managed_states(submodules)
    print_rank_n([p.data.shape for p in params])

    # for n, p in module.named_parameters():
    #     d_param = torch.nn.Parameter(
    #         fsdp_shard(p.data, device_mesh)
    #     )
    #     module.register_parameter(n, d_param)

    # for n, p in module.named_parameters():
    #     print(n, type(p.data))

    return module



if __name__ == "__main__":
    device_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("sh",))
    rank = dist.get_rank()

    shard_mesh = device_mesh["sh"]

    mlp = MLP(8, 4, bias=True)
    fully_shard(mlp, shard_mesh)

    # print(type(mlp.c_fc.weight.data))
    # print_rank_n(mlp.c_fc.weight.data)




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