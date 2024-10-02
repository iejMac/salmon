import time

from typing import Any, List

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Shard, Replicate, distribute_tensor, distribute_module, init_device_mesh
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
        self.dtype = param.dtype
        self.requires_grad = param.requires_grad
        self.ref_module = ref_module  # param belongs to this module
        self.param_name = param_name  # param has this name in ref_module

    # TODO: something here is breaking the computational graph...
    def _shard(self, device_mesh: DeviceMesh):
        fsdp_placements = [Shard(0)]
        param = getattr(self.ref_module, self.param_name)
        d_param = torch.nn.Parameter(
            DTensor.from_local(param.data, device_mesh).redistribute(device_mesh, fsdp_placements)
        )
        d_param.requires_grad = self.requires_grad
        # TODO: fast/unsafe/(evil) register
        self.ref_module.register_parameter(self.param_name, d_param)

    def _unshard(self):
        param = getattr(self.ref_module, self.param_name)
        d_param = torch.nn.Parameter(
            param.data.full_tensor()
        )
        d_param.requires_grad = self.requires_grad
        self.ref_module.register_parameter(self.param_name, d_param)


class FSDPParamGroup:
    def __init__(self, module: torch.nn.Module, device_mesh: DeviceMesh):
        self.device_mesh = device_mesh

        submodules = _get_submodules(module)
        param_info = [(p, n, mod) for mod in submodules for n, p in mod.named_parameters(recurse=False)]
        self.param_group = [FSDPParam(*p_info) for p_info in param_info]

        for p in self.param_group:  # init sharding
            p._shard(self.device_mesh)

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
        print_rank_n(grad)
        return grad

    def forward(self, *args, **kwargs):
        self.fsdp_param_group._pre_forward()
        out = self.root_module(*args, **kwargs)
        out = self._post_forward(out)
        return out


if __name__ == "__main__":
    device_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("sh",))
    ws = dist.get_world_size()
    rank = dist.get_rank()

    shard_mesh = device_mesh["sh"]

    x = torch.randn(2, 8).cuda()

    mlp = MLP(8, 4, bias=True)
    fsdp_mlp = FSDPModule(mlp, shard_mesh)

    for i in range(1):
        y = fsdp_mlp(x)
        print_rank_n(y.shape)
        # print_rank_n(y)

        loss = y.mean()
        loss.backward()

        print_rank_n(type(fsdp_mlp.root_module.c_fc.weight.data))  # None
        print_rank_n(fsdp_mlp.root_module.c_fc.weight.grad)  # None
        print_rank_n(fsdp_mlp.root_module.c_fc.weight.requires_grad)  # None