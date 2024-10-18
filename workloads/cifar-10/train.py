import os
import datetime
import csv

import torch
import torch.nn.functional as F

from model import CNN
from config import base_training, base_model, base_optimizer
from data import CIFAR10Dataset


class CSVLogger:
    def __init__(self, run_name=None, fieldnames=None, log_dir="./logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.file = open(os.path.join(log_dir, f"{run_name or datetime.now().strftime('run_%Y%m%d_%H%M%S')}.csv"), 'w', newline='')
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
        run_name,
        model_config, optimizer_config,
        batch_size, 
        n_train_steps, n_eval_steps,
        eval_freq, log_freq,
        seed=0,
        run_dir="./runs", data_dir="./data",
    ):
    torch.manual_seed(seed)

    run_name = "test_cifar"
    logger = CSVLogger(run_name=run_name, fieldnames=["step", "train_loss", "train_acc", "val_loss", "val_acc"], log_dir=run_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = CIFAR10Dataset(batch_size=batch_size, train=True, device=device, root=data_dir)
    test_loader = CIFAR10Dataset(batch_size=batch_size, train=False, device=device, root=data_dir)

    # model = CNN(n_layers=3, hidden_channels=128)
    model = model_config().build()
    model = model.to(device)
    print(f"model has {sum([p.numel() for p in model.parameters()])} parameters")

    # opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    opt = optimizer_config().build(params=model.parameters())

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


if __name__ == "__main__":
    run_name = "test_cifar"
    model_config = base_model
    optimizer_config = base_optimizer

    # train()
    base_training().build(run_name=run_name, model_config=model_config, optimizer_config=optimizer_config)