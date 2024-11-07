"""general config class."""

class Config:
    def __init__(self, obj, params={}):
        self.obj = obj
        self.init_params = params

    def get_params(self):
        # Return common JSON-serializable types (for checkpointing)
        return {k: v for k, v in self.init_params.items() if isinstance(v, (int, float, str, list, dict, bool, type(None)))}

    def __getitem__(self, key):
        return self.init_params[key]
    def __setitem__(self, key, value):
        self.init_params[key] = value

    def build(self, **more_params):
        params = {**self.init_params, **more_params}
        return self.obj(**params)

# CIFAR-10 training:
def mlp_base():
    from model import MLP
    return Config(
        obj=MLP,
        params={
            "dims": [3*32*32, 10, 10, 10],
            "bias": False,
        },
    )

def adamw_base():
    from torch.optim import AdamW
    return Config(
        obj=AdamW,
        params={
            "lr": 1e-3,
        },
    )

def sgd_base():
    from torch.optim import SGD
    return Config(
        obj=SGD,
        params={
            "lr": 1e-3,
        },
    )

def training_base():
    from train import train
    N_STEPS = 3000
    return Config(
        obj=train,
        params={
            "seed": 0,
            "batch_size": 32,
            "n_train_steps": N_STEPS,
            "n_eval_steps": 32,
            "eval_freq": N_STEPS // 100,
            "log_freq": N_STEPS // 500,
        },
    )

# Fractals:
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

def sgd_frac():
    from torch.optim import SGD
    return Config(
        obj=SGD,
        params={
            "lr": 1e-2,
        },
    )

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

def training_frac():
    from fractal import train
    N_STEPS = 100 # 500 # 1000
    return Config(
        obj=train,
        params={
            "seed": 0,
            "batch_size": N_FRAC ** 2 + N_FRAC,  # full batch GD for nonlinear networks
            "n_train_steps": N_STEPS,
            "log_freq": 1,
        },
    )

if __name__ == "__main__":
    pass