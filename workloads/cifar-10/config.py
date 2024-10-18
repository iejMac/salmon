"""general config class."""

class Config:
    def __init__(self, obj, params={}):
        self.obj = obj
        self.init_params = params

    def __getitem__(self, key):
        return self.init_params[key]

    def __setitem__(self, key, value):
        self.init_params[key] = value

    def build(self, **more_params):
        params = {**self.init_params, **more_params}
        return self.obj(**params)


def mlp_base():
    from model import MLP
    return Config(
        obj=MLP,
        params={
            "dims": [3*32*32, 10, 10],
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

def training_base():
    from train import train
    N_STEPS = 5000
    return Config(
        obj=train,
        params={
            "seed": 0,
            "batch_size": 128,
            "n_train_steps": N_STEPS,
            "n_eval_steps": 32,
            "eval_freq": N_STEPS // 100,
            "log_freq": N_STEPS // 500,
        },
    )


if __name__ == "__main__":
    pass