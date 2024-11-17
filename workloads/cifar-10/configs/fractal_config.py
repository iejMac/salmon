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

def jascha_data():
    from data import SyntheticNormalDataset
    return Config(
        obj=SyntheticNormalDataset,
        params={
            "resample": False,
            "signal_strength": 0.0,
        }
    )

def jascha_grid():
    resolution = 4
    c1_grid = np.linspace(-1.5, -0.5, num=resolution)
    c2_grid = np.linspace(-1.5, -0.5, num=resolution)

    for c1_id, c1 in enumerate(c1_grid):
        for c2_id, c2 in enumerate(c2_grid):
            exp_id = c1_id * len(c2_grid) + c2_id
            def mean_field_parametrization_with_diff_cl():
                mf_cfg = mean_field_parametrization()
                mf_cfg['cl'] = [c1, c2]
                return mf_cfg

            param_args = (training_frac, mlp1h, sgd_frac, mean_field_parametrization_with_diff_cl, jascha_data)

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

def jascha_grid_w_eps():
    sp_resolution = 4
    t_resolution = 11

    # Define the grids for c1, c2, and eps
    c1_grid = np.linspace(-1.5, -0.5, num=sp_resolution)
    c2_grid = np.linspace(-1.5, -0.5, num=sp_resolution)
    eps_grid = np.linspace(0.0, 1.0, num=t_resolution)  # Generates [0.0, 0.1, ..., 1.0]

    # Iterate over all combinations of c1, c2, and eps
    for c1_id, c1 in enumerate(c1_grid):
        for c2_id, c2 in enumerate(c2_grid):
            for eps_id, eps in enumerate(eps_grid):
                # Compute a unique experiment ID
                exp_id = (c1_id * len(c2_grid) + c2_id) * len(eps_grid) + eps_id

                # Define a parametrization function that includes c1, c2, and eps
                def mean_field_parametrization_with_diff_cl_eps():
                    mf_cfg = mean_field_parametrization()
                    mf_cfg['cl'] = [c1, c2]
                    return mf_cfg

                def jascha_data_w_signal():
                    data_config = jascha_data()
                    data_config["signal_strength"] = eps
                    return data_config

                # Prepare the parameter arguments tuple
                param_args = (training_frac, mlp1h, sgd_frac, mean_field_parametrization_with_diff_cl_eps, jascha_data_w_signal)

                # Create a unique run name that includes c1, c2, and eps
                run_name = f"mfp_c1_{c1:.14f}_c2_{c2:.14f}_eps_{eps:.3f}"

                # Yield the experiment configuration
                yield exp_id, run_name, param_args

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
    N_STEPS = 3000
    return Config(
        obj=train,
        params={
            "seed": 0,
            "batch_size": N_FREE_PARAMS,  # full batch GD for nonlinear networks
            "n_train_steps": N_STEPS,
            "log_freq": 1,
        },
    )
