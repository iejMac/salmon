import time

from typing import Any, List

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Shard, Replicate, distribute_tensor, distribute_module, init_device_mesh
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta
from torch.distributed.device_mesh import DeviceMesh
from torch.utils._pytree import tree_flatten

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


class FSDPParam:
    def __init__(self, param: torch.nn.Parameter, param_name: str, ref_module: torch.nn.Module):
        self.ref_module = ref_module  # param belongs to this module
        self.param_name = param_name  # param has this name in ref_module

    def _init_sharded_param(self, device_mesh):
        param = getattr(self.ref_module, self.param_name)

        self.cached_data = torch.clone(param.data)
        temp = torch.zeros_like(param.data)

        d_param = torch.nn.Parameter(temp)

        # fsdp_placements = [Shard(0)]

        # d_param = torch.nn.Parameter(
        #     DTensor.from_local(param.data, device_mesh).redistribute(device_mesh, fsdp_placements)
        #     # DTensor.from_local(param.data, device_mesh)
        # )
        # d_param.requires_grad = param.requires_grad
        # self.ref_module.register_parameter(self.param_name, d_param)
        setattr(self.ref_module, self.param_name, d_param)

    def _shard(self, device_mesh: DeviceMesh):
        pass

    def _unshard(self):
        param = getattr(self.ref_module, self.param_name)
        d_param = torch.nn.Parameter(self.cached_data)
        # d_param.requires_grad = param.requires_grad
        # self.ref_module.register_parameter(self.param_name, d_param)
        setattr(self.ref_module, self.param_name, d_param)


class FSDPParamGroup:
    def __init__(self, module: torch.nn.Module, device_mesh: DeviceMesh):
        self.device_mesh = device_mesh

        submodules = _get_submodules(module)
        param_info = [(p, n, mod) for mod in submodules for n, p in mod.named_parameters(recurse=False)]
        self.param_group = [FSDPParam(*p_info) for p_info in param_info]

        for p in self.param_group:  # init sharding
            p._init_sharded_param(self.device_mesh)

        self.state = None

    def _pre_forward(self):
        for fp in self.param_group:
            fp._unshard()  # all-gather weights
            # TODO: make async
        # TODO: wait
        self.state = "forward"

    def _post_forward(self):
        for fp in self.param_group:
            fp._shard(self.device_mesh)  # re-shard
            # NOTE: we don't need to send anything, just delete what you don't need
            # not sure if normal re-distribute will do this or send

    def _pre_backward(self, *unused: Any):
        if self.state == "backward":  # unshard already called
            return
        self.state = "backward"
        for fp in self.param_group:
            fp._unshard()  # all-gather weights

    def _post_backward(self):
        # TODO: reduce-scatter gradients

        # TODO: release
        for fp in self.param_group:
            fp._shard(self.device_mesh)  # re-shard
            # NOTE: we don't need to send anything, just delete what you don't need
            # not sure if normal re-distribute will do this or send


class FSDPModule(torch.nn.Module):
    def __init__(self, root_module, device_mesh):
        super().__init__()
        self.root_module = root_module
        self.device_mesh = device_mesh

        self.fsdp_param_group = FSDPParamGroup(self.root_module, self.device_mesh)

    def _post_forward(self, output):
        self.fsdp_param_group._post_forward()
        # NOTE: if backward called on any output tensor, prepare for backward in cur layer
        flat_outputs, _ = tree_flatten(output)
        tensors = [t for t in flat_outputs if (torch.is_tensor(t) and t.requires_grad)]
        if tensors:
            for tensor in tensors:
                tensor.register_hook(self._pre_backward)
        return output

    def _pre_backward(self, grad: torch.Tensor):
        self.fsdp_param_group._pre_backward()
        return grad

    def forward(self, *args, **kwargs):
        self.fsdp_param_group._pre_forward()
        out = self.root_module(*args, **kwargs)
        out = self._post_forward(out)
        return out


"""
I think the way this is done is by resizing param tensors

all_gather weights into some joint big tensor buffer

If unsharded_param doesn't exist, create it
If exists allocate_storage, and copy in gathered weights
To free, just resize to 0

sharded_param stores DTensor I think
to create sharded_param use all_gather output

They are hotswapping though so idk why this breaks...
"""



if __name__ == "__main__":
    device_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("sh",))
    ws = dist.get_world_size()
    rank = dist.get_rank()

    shard_mesh = device_mesh["sh"]

    x = torch.randn(2, 8).cuda()

    # TESTING
    class FakeModule(torch.nn.Module):
        def __init__(self, w_dat):
            super().__init__()
            self.W = torch.nn.Parameter(w_dat)
        def forward(self, x):
            return x @ self.W

    w_dat = torch.randn(8, 4).cuda()
    mod = FakeModule(w_dat)

    sh_dat = torch.zeros_like(w_dat)
    sharded_param = torch.nn.Parameter(sh_dat)
    ush_dat = torch.ones_like(w_dat)
    unsharded_param = torch.nn.Parameter(ush_dat)

    # swap in unsharded
    setattr(mod, "W", unsharded_param)

    y = mod(x)  # fwd graph set?

    # swap in sharded
    setattr(mod, "W", sharded_param)
    # save dat
    cch_dat = torch.clone(unsharded_param.data)
    unsharded_param.data.resize_(0)

    # recover dims before bwd
    # unsharded_param.data.resize_(ush_dat.size())
    # unsharded_param.data.copy_(cch_dat)
    setattr(mod, "W", unsharded_param)

    y.sum().backward()

    print_rank_n(unsharded_param.grad)


    quit()
    # TESTING

    mlp = MLP(8, 4, bias=True).cuda()
    fsdp_mlp = FSDPModule(mlp, shard_mesh)

    for i in range(1):
        y = fsdp_mlp(x)
        print_rank_n(y.shape)
        # print_rank_n(y)

        loss = y.mean()
        loss.backward()

        print_rank_n(type(fsdp_mlp.root_module.c_fc.weight.data))  # None
        print_rank_n(fsdp_mlp.root_module.c_fc.weight.grad)  # None
        # print_rank_n(type(fsdp_mlp.root_module.c_proj.weight.data))  # None
        # print_rank_n(fsdp_mlp.root_module.c_proj.weight.grad)  # None