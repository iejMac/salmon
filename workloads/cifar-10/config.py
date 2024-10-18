

class Config:
    def __init__(self, obj, params={}):
        self.obj = obj
        self.init_params = params

    def build(self, **more_params):
        params = {**self.init_params, **more_params}
        return self.obj(**params)


def base_model():
    from model import CNN
    return Config(
        obj=CNN,
        params={
            "n_layers": 3,
            "hidden_channels": 128,
        },
    )

def base_optimizer():
    from torch.optim import AdamW
    return Config(
        obj=AdamW,
        params={
            "lr": 1e-3,
            "weight_decay": 1e-2,
        },
    )

def base_training():
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