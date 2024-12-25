import os
import csv
import datetime
import json

import torch
import torch.nn.functional as F
import numpy as np

from configs.cifar_config import mlp_base, adamw_base, sgd_base, training_base
from data import CIFAR10Dataset

from parametrization import standard_parametrization, mu_parametrization


class CSVLogger:
    def __init__(self, fieldnames=None, log_dir="./logs", overwrite=True):
        os.makedirs(log_dir, exist_ok=True)
        self.file = open(os.path.join(log_dir, f"{'metrics' if overwrite else datetime.datetime.now().strftime('metrics_%Y%m%d_%H%M%S')}.csv"), 'w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()

    def log(self, step, **metrics):
        row = {'step': step, **metrics}
        self.writer.writerow(row)
        self.file.flush()
        metrics = ', '.join([f"{k}: {v}" for k, v in metrics.items()]) # log to stdout:
        print(f"step: {step}; {metrics}")


def evaluate(model, test_dataloader, n_steps):
    was_train = model.train
    model.eval()

    correct, total = 0, 0
    loss = 0.0
    with torch.no_grad():
        s = 0
        for X, y in test_dataloader:
            if s >= n_steps:
                break
            y_hat = model(X)

            loss += F.cross_entropy(y_hat, y)
            y_pred = torch.argmax(y_hat, dim=-1)
            correct += (y_pred == y).sum()
            total += y_pred.shape[0]
            s += 1
    stats = {
        f"accuracy": correct.item() / total,
        f"loss": loss.item() / n_steps,
    }

    if was_train:
        model.train()
    return stats


def train(
        model_config, optimizer_config,
        batch_size, 
        n_train_steps, n_eval_steps,
        eval_freq, log_freq,
        seed=0,
        run_dir="./runs", data_dir="./data",
    ):
    torch.manual_seed(seed)

    worker_id = int(os.environ.get("WORKER_ID", 0))
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    logger = CSVLogger(fieldnames=["step", "train_loss", "train_acc", "val_loss", "val_acc"], log_dir=run_dir, overwrite=False)

    train_loader = CIFAR10Dataset(batch_size=batch_size, train=True, device=device, root=data_dir)
    test_loader = CIFAR10Dataset(batch_size=batch_size, train=False, device=device, root=data_dir)

    model = model_config().build()
    model = model.to(device)

    opt_cfg = optimizer_config()
    # params = standard_parametrization(model, lr_prefactor=opt_cfg['lr'], std_prefactor=1.0)
    params = mu_parametrization(model, lr_prefactor=opt_cfg['lr'], std_prefactor=1.0)
    opt = opt_cfg.build(params=params)

    s = 0
    for X, y in train_loader:
        if s >= n_train_steps:
            break
        opt.zero_grad()
        y_hat = model(X)

        loss = F.cross_entropy(y_hat, y)
        if s % log_freq == 0:
            print(f"step {s}; loss = {loss.item()}")

        loss.backward()
        opt.step()

        if s % eval_freq == 0:
            stats_train = evaluate(model, train_loader, n_steps=n_eval_steps)
            stats_val = evaluate(model, test_loader, n_steps=n_eval_steps)
            logger.log(step=s, train_loss=stats_train["loss"], train_acc=stats_train["accuracy"], val_loss=stats_val["loss"], val_acc=stats_val["accuracy"])
        s += 1

    stats_train = evaluate(model, train_loader, n_steps=n_eval_steps)
    stats_val = evaluate(model, test_loader, n_steps=n_eval_steps)
    logger.log(step=s, train_loss=stats_train["loss"], train_acc=stats_train["accuracy"], val_loss=stats_val["loss"], val_acc=stats_val["accuracy"])


def main(run_name, model_config, optimizer_config, training_config):
    run_dir = os.path.join("./runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    configs = {
        "training": training_config,
        "model": model_config,
        "optimizer": optimizer_config
    }
    for config_name, config in configs.items():
        config_path = os.path.join(run_dir, f"{config_name}_config.json")
        with open(config_path, "w") as f:
            json.dump(config().get_params(), f, indent=4)

    training_config().build(
        model_config=model_config, 
        optimizer_config=optimizer_config,
        run_dir=os.path.join("./runs", run_name),
    )


if __name__ == "__main__":
    worker_id = int(os.environ.get("WORKER_ID", 0))
    n_workers = int(os.environ.get("N_WORKERS", 1))

    widths = [256, 512, 1024, 2048]
    lrs = np.power(10, np.linspace(-2.0, -1.0, num=16)).tolist()
    N_REPETITIONS = 4  # for confidence intervals

    for w_id, w in enumerate(widths):
        for lr_id, lr in enumerate(lrs):
            exp_id = w_id * len(lrs) + lr_id
            if exp_id % n_workers == worker_id:
                def mlp_w():
                    model_config = mlp_base()
                    model_config["dims"][1] = w
                    model_config["dims"][2] = w
                    return model_config
                def sgd_lr():
                    opt_config = sgd_base()
                    opt_config["lr"] = lr
                    return opt_config

                for r in range(N_REPETITIONS):  # overwrite=False in CSVLogger so timestamp included
                    def seeded_training_base():
                        training_config = training_base()
                        training_config["seed"] = training_config["seed"] + r
                        return training_config

                    main(f"mlp_3layer_hiddenw_{w}_lr_{lr}", mlp_w, sgd_lr, seeded_training_base)