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


# def abc_parametrization(mlp, n, al, bl, cl, lr_prefactor=0.1, std_prefactor=2**0.5):
#     lr_scale_groups = {}
#     for i, layer in enumerate(mlp.layers):
#         a, b, c = al[i], bl[i], cl[i]
# 
#         l_mult = n ** -a
#         var_l = n ** (-2*b)
#         lr_scale = n ** -c
# 
#         lr_scale_groups[lr_scale] = lr_scale_groups.get(lr_scale, []) + [layer.weight]
#         mlp.layer_multipliers[i] = l_mult
#         torch.nn.init.normal_(layer.weight, mean=0.0, std=std_prefactor * (var_l ** 0.5))
#     optim_groups = [{'params': params, 'lr': lr_prefactor * lr_scale} for lr_scale, params in lr_scale_groups.items()]
#     return optim_groups

# new group for each param (so dynamic setting is easier)
def abc_parametrization(mlp, n, al, bl, cl, lr_prefactor=0.1, std_prefactor=2**0.5):
    lr_scale_groups = []
    for i, layer in enumerate(mlp.layers):
        a, b, c = al[i], bl[i], cl[i]

        l_mult = n ** -a
        var_l = n ** (-2*b)
        lr_scale = n ** -c

        # lr_scale_groups[lr_scale] = lr_scale_groups.get(lr_scale, []) + [layer.weight]
        lr_scale_groups.append((lr_scale, layer.weight))
        mlp.layer_multipliers[i] = l_mult
        torch.nn.init.normal_(layer.weight, mean=0.0, std=std_prefactor * (var_l ** 0.5))
    optim_groups = [{'params': params, 'lr': lr_prefactor * lr_scale} for lr_scale, params in lr_scale_groups]
    return optim_groups



from solver import find_c_adam, find_c_sgd

def maximal_lr_scheduler(optimizer, n, al, bl, lr_prefactor=0.1):
    def _compute_cl(alpha_l, omega_l, u_l):
        # Find maximal lr exponents
        if isinstance(optimizer, torch.optim.AdamW):
            solver = find_c_adam
        elif isinstance(optimizer, torch.optim.SGD):
            solver = find_c_sgd
        else:
            raise ValueError(f"Unsupported optimizer: {type(optimizer)}")
        cl, rl = solver(a=al, b=bl, alpha=alpha_l, u=u_l, omega=omega_l)
        return cl

    def _lr_adjuster(alpha_l, u_l, omega_l):
        # Compute c_l based on measured alignment
        cl = _compute_cl(alpha_l=alpha_l, omega_l=omega_l, u_l=u_l)
        # Dynamically adjust learning rates for each parameter group 
        for i, (param_group, c) in enumerate(zip(optimizer.param_groups, cl)):
            lr_scale = n ** -c
            param_group['lr'] = lr_prefactor * lr_scale
        return [param_group['lr'] for param_group in optimizer.param_groups]
    return _lr_adjuster    
