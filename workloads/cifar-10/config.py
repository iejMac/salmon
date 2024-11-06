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

if __name__ == "__main__":
    pass