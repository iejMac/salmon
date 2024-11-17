from .config import Config

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