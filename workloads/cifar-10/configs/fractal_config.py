import numpy as np

from .config import Config


# Jascha setup (https://arxiv.org/abs/2402.06184):
N_FRAC_1 = 16
def mlp1h():
    from model import MLP
    return Config(
        obj=MLP,
        params={
            "dims": [N_FRAC_1, N_FRAC_1, 1],
            "bias": False,
        },
    )
N_FREE_PARAMS_1 = N_FRAC_1 * N_FRAC_1 + N_FRAC_1 * 1

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
            "dataset_size": N_FREE_PARAMS_1,
            "batch_size": N_FREE_PARAMS_1,
            "width": N_FRAC_1,
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
DATA_DIM=2
N_FRAC_2 = 64
def mlp2h():
    from model import MLP
    return Config(
        obj=MLP,
        params={
            "dims": [DATA_DIM, N_FRAC_2, N_FRAC_2, 1],
            "bias": False,
        },
    )
N_FREE_PARAMS_2 = (DATA_DIM * N_FRAC_2) + (N_FRAC_2 * N_FRAC_2) + (N_FRAC_2 * 1)

def maximal_update_parametrization():
    from parametrization import abc_parametrization
    # NOTE: overfit to mlp2h model config
    al, bl, cl = [-0.5, 0.0, 0.5], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0] # mean field
    return Config(
        obj=abc_parametrization,
        params={
            "al": al,
            "bl": bl,
            "cl": cl,
        }
    )

def a3b3_data():
    from data import SyntheticNormalDataset
    return Config(
        obj=SyntheticNormalDataset,
        params={
            "dataset_size": N_FREE_PARAMS_2,
            "batch_size": N_FREE_PARAMS_2,
            "width": DATA_DIM,
            "resample": False,
            "signal_strength": 0.5,
        }
    )


def mup_a3b3_grid():
    resolution = 16
    a3_grid = np.linspace(-0.5, 1.5, num=resolution)  # Adjust the range as needed
    b3_grid = np.linspace(-0.5, 1.5, num=resolution)  # Adjust the range as needed

    for a3_id, a3 in enumerate(a3_grid):
        for b3_id, b3 in enumerate(b3_grid):
            exp_id = a3_id * len(b3_grid) + b3_id

            def mup_w_diff_a3b3():
                cfg = maximal_update_parametrization()
                cfg['al'][-1] = a3
                cfg['bl'][-1] = b3
                return cfg

            param_args = (training_frac, mlp2h, sgd_frac, mup_w_diff_a3b3, a3b3_data)

            run_name = f"mup_a3_{a3:.14f}_b3_{b3:.14f}"
            yield exp_id, run_name, param_args

def mup_a3b3_eps_grid():
    sp_resolution = 4
    t_resolution = 11

    a3_grid = np.linspace(-0.5, 1.5, num=sp_resolution)
    b3_grid = np.linspace(-0.5, 1.5, num=sp_resolution)
    eps_grid = np.linspace(0.0, 1.0, num=t_resolution)

    for a3_id, a3 in enumerate(a3_grid):
        for b3_id, b3 in enumerate(b3_grid):
            for eps_id, eps in enumerate(eps_grid):
                exp_id = (a3_id * len(b3_grid) + b3_id) * len(eps_grid) + eps_id

                def mup_w_diff_a3b3_eps(a3=a3, b3=b3):
                    cfg = maximal_update_parametrization()
                    cfg['al'][-1] = a3
                    cfg['bl'][-1] = b3
                    return cfg

                def a3b3_data_w_signal(eps=eps):
                    data_cfg = a3b3_data()
                    data_cfg["signal_strength"] = eps
                    return data_cfg

                param_args = (training_frac, mlp2h, sgd_frac, mup_w_diff_a3b3_eps, a3b3_data_w_signal)
                run_name = f"mup_a3_{a3:.14f}_b3_{b3:.14f}_eps_{eps:.3f}"
                yield exp_id, run_name, param_args


# Common:
def sgd_frac():
    from torch.optim import SGD
    return Config(
        obj=SGD,
        params={
            "lr": 1e-1,
        },
    )

def training_frac():
    from fractal import train
    N_STEPS = 1500
    return Config(
        obj=train,
        params={
            "seed": 0,
            "n_train_steps": N_STEPS,
            "log_freq": 1,
        },
    )
