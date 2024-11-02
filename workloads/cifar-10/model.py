import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dims, bias=False):
        super(MLP, self).__init__()
        self.n_layers = len(dims) - 1
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=bias) for i in range(len(dims) - 1)])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.n_layers - 1:
                x = F.relu(x)
        return x


def standard_parametrization(mlp, init_prefactor=2**0.5):
    print('standard p')
    for layer in mlp.layers:
        fan_in = layer.weight.data.shape[0]
        sigma_l = (1/fan_in**0.5)
        torch.nn.init.normal_(layer.weight, mean=0.0, std=init_prefactor * sigma_l)
    return list(mlp.parameters())

def spectral_parameterization(mlp, init_prefactor=2**0.5):
    print('spectral p')
    lr_scale_groups = {}
    for layer in mlp.layers:
        fan_in, fan_out = layer.weight.data.shape
        sigma_l = (1/fan_in**0.5) * min(1, (fan_out/fan_in)**0.5)
        torch.nn.init.normal_(layer.weight, mean=0.0, std=init_prefactor * sigma_l)

        lr_scale = fan_out/fan_in
        lr_scale_groups[lr_scale] = lr_scale_groups.get(lr_scale, []) + [layer.weight]
    optim_groups = [{'params': params, 'lr_scale': lr_scale} for lr_scale, params in lr_scale_groups.items()]
    return optim_groups



if __name__ == "__main__":
    import copy
    # mod = MLP([2, 4, 8])
    mod = MLP([2, 3])
    mod_c = copy.deepcopy(mod)

    print(mod.layers[0].weight.data)
    print(mod_c.layers[0].weight.data)

    sp_p = standard_parametrization(mod)
    spec_p = spectral_parameterization(mod_c)

    print(sp_p[0].data)
    print(spec_p[0]['params'][0].data)

    quit()
    x = torch.randn(2)
    y = mod(x)
    print(y.shape)