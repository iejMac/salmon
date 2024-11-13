import torch
import torch.nn.functional as F


def unstable_parametrization(mlp, lr_prefactor=0.1, std_prefactor=2**0.5):
    lr_scale_groups = {}
    for i, layer in enumerate(mlp.layers):
        fan_out, fan_in = layer.weight.data.shape

        # abc-param format from [Everett et al.]
        if i == 0: # embedding
            n = fan_out
            # NOTE: instability here
            a, b, c = 0.0, 0.0, -0.5 # stable version a + b == 0
            # a, b, c = 0.0, -0.25, -0.5 # unstable version a + b < 0
            # a, b, c = -0.5, 0.0, -0.5 # unstable version a + b < 0
        elif i < mlp.n_layers - 1: # hidden
            n = fan_out
            # a, b, c = 0.0, 0.5, 0.5
            a, b, c = 0.0, 0.0, 0.0
        else: # readout
            n = fan_in
            a, b, c = 0.0, 0.5, 1.0

        l_mult = n ** -a
        var_l = n ** (-2*b)
        lr_scale = n ** -c

        lr_scale_groups[lr_scale] = lr_scale_groups.get(lr_scale, []) + [layer.weight]
        mlp.layer_multipliers[i] = l_mult
        torch.nn.init.normal_(layer.weight, mean=0.0, std=std_prefactor * (var_l ** 0.5))
    optim_groups = [{'params': params, 'lr': lr_prefactor * lr_scale} for lr_scale, params in lr_scale_groups.items()]
    return optim_groups

def standard_parametrization(mlp, lr_prefactor=0.1, std_prefactor=2**0.5):
    lr_scale_groups = {}
    for i, layer in enumerate(mlp.layers):
        fan_out, fan_in = layer.weight.data.shape

        # abc-param format from [Everett et al.]
        if i == 0: # embedding
            n = fan_out
            a, b, c = 0.0, 0.0, -0.5 # sgd full alignment
        elif i < mlp.n_layers - 1: # hidden
            n = fan_out
            a, b, c = 0.0, 0.5, 0.5 # sgd full alignment
        else: # readout
            n = fan_in
            a, b, c = 0.0, 0.5, 1.0 # sgd full alignment

        l_mult = n ** -a
        var_l = n ** (-2*b)
        lr_scale = n ** -c

        lr_scale_groups[lr_scale] = lr_scale_groups.get(lr_scale, []) + [layer.weight]
        mlp.layer_multipliers[i] = l_mult
        torch.nn.init.normal_(layer.weight, mean=0.0, std=std_prefactor * (var_l ** 0.5))
    optim_groups = [{'params': params, 'lr': lr_prefactor * lr_scale} for lr_scale, params in lr_scale_groups.items()]
    return optim_groups

def mu_parametrization(mlp, lr_prefactor=0.1, std_prefactor=2**0.5):
    lr_scale_groups = {}
    for i, layer in enumerate(mlp.layers):
        fan_out, fan_in = layer.weight.data.shape

        # abc-param format from [Everett et al.]
        if i == 0: # embedding
            n = fan_out
            a, b, c = -0.5, 0.5, 0.0
        elif i < mlp.n_layers - 1: # hidden
            n = fan_out
            a, b, c = 0.0, 0.5, 0.0
        else: # readout
            n = fan_in
            a, b, c = 0.5, 0.5, 0.0

        l_mult = n ** -a
        var_l = n ** (-2*b)
        lr_scale = n ** -c

        lr_scale_groups[lr_scale] = lr_scale_groups.get(lr_scale, []) + [layer.weight]
        mlp.layer_multipliers[i] = l_mult
        torch.nn.init.normal_(layer.weight, mean=0.0, std=std_prefactor * (var_l ** 0.5))
    optim_groups = [{'params': params, 'lr': lr_prefactor * lr_scale} for lr_scale, params in lr_scale_groups.items()]
    return optim_groups


def abc_parametrization(mlp, n, al, bl, cl, lr_prefactor=0.1, std_prefactor=2**0.5):
    lr_scale_groups = {}
    for i, layer in enumerate(mlp.layers):
        a, b, c = al[i], bl[i], cl[i]

        l_mult = n ** -a
        var_l = n ** (-2*b)
        lr_scale = n ** -c

        lr_scale_groups[lr_scale] = lr_scale_groups.get(lr_scale, []) + [layer.weight]
        mlp.layer_multipliers[i] = l_mult
        torch.nn.init.normal_(layer.weight, mean=0.0, std=std_prefactor * (var_l ** 0.5))
    optim_groups = [{'params': params, 'lr': lr_prefactor * lr_scale} for lr_scale, params in lr_scale_groups.items()]
    return optim_groups