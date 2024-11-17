import numpy as np

from .config import Config

# Jascha setup (https://arxiv.org/abs/2402.06184):
N_FRAC = 16
def mlp1h():
    from model import MLP
    return Config(
        obj=MLP,
        params={
            "dims": [N_FRAC, N_FRAC, 1],
            "bias": False,
        },
    )
N_FREE_PARAMS = N_FRAC * N_FRAC + N_FRAC * 1

def mean_field_parametrization():
    from parametrization import abc_parametrization
    # NOTE: overfit to mlp1h model config
    al, bl, cl = [0.0, 1.0], [0.0, 0.0], [-1.0, -1.0] # mean field
    return Config(
        obj=abc_parametrization,
        params={
            "al": al,
            "bl": bl,
            "cl": cl,
        }
    )

def jascha_grid(resolution):
    c1_grid = np.linspace(-1.5, -0.5, num=resolution)
    c2_grid = np.linspace(-1.5, -0.5, num=resolution)

    for c1_id, c1 in enumerate(c1_grid):
        for c2_id, c2 in enumerate(c2_grid):
            exp_id = c1_id * len(c2_grid) + c2_id
            def mean_field_parametrization_with_diff_cl():
                mf_cfg = mean_field_parametrization()
                mf_cfg['cl'] = [c1, c2]
                return mf_cfg

            param_args = (mlp1h, sgd_frac, training_frac, mean_field_parametrization_with_diff_cl)

            run_name = f"mfp_c1_{c1:.14f}_c2_{c2:.14f}"
            yield exp_id, run_name, param_args



# New setup:
# N_FRAC = 16
# def mlp2h():
#     from model import MLP
#     return Config(
#         obj=MLP,
#         params={
#             "dims": [N_FRAC, N_FRAC, N_FRAC, 1],
#             "bias": False,
#         },
#     )
# N_FREE_PARAMS = N_FRAC * N_FRAC + N_FRAC * N_FRAC + N_FRAC * 1

# def maximal_update_parametrization():
#     from parametrization import abc_parametrization
#     # NOTE: overfit to mlp1h model config
#     al, bl, cl = [0.0, 1.0], [0.0, 0.0], [-1.0, -1.0] # mean field
#     return Config(
#         obj=abc_parametrization,
#         params={
#             "al": al,
#             "bl": bl,
#             "cl": cl,
#         }
#     )


# Common:
def sgd_frac():
    from torch.optim import SGD
    return Config(
        obj=SGD,
        params={
            "lr": 1e-2,
        },
    )

def training_frac():
    from fractal import train
    N_STEPS = 1000
    return Config(
        obj=train,
        params={
            "seed": 0,
            "batch_size": N_FREE_PARAMS,  # full batch GD for nonlinear networks
            "n_train_steps": N_STEPS,
            "log_freq": 1,
        },
    )