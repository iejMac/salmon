import os
import copy
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from data import SyntheticNormalDataset
from parametrization import abc_parametrization

torch.set_default_dtype(torch.float64)

class BinaryLogger:
    def __init__(self, log_dir, n_steps, n_layers):
        self.log_dir = log_dir
        self.n_steps = n_steps
        self.losses = np.zeros(n_steps, dtype=np.float64)
        self.rLs = np.zeros(n_steps, dtype=np.float64)
        self.Als = np.zeros((n_steps, n_layers, 4), dtype=np.float64)  # A_cum, A_alpha, A_omega, A_u
    
    def log(self, step, loss, rL, Al):
        self.losses[step] = loss
        self.rLs[step] = rL
        self.Als[step, :] = Al

    def save(self):
        np.save(os.path.join(self.log_dir, 'losses.npy'), self.losses)
        np.save(os.path.join(self.log_dir, 'rLs.npy'), self.rLs)
        np.save(os.path.join(self.log_dir, 'Als.npy'), self.Als)

def train(
        model_config, optimizer_config, parametrization_config,
        batch_size, 
        n_train_steps,
        log_freq,
        seed=0,
        run_dir="./runs", data_dir="./data",
    ):
    torch.manual_seed(seed)

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")


    model = model_config().build()
    model = model.to(device)

    opt_cfg = optimizer_config()

    width = model_config()["dims"][1]  # Feature dimension (width of hidden layer)
    params = parametrization_config().build(mlp=model, n=width, lr_prefactor=opt_cfg['lr'], std_prefactor=1.0)

    opt = opt_cfg.build(params=params)

    train_loader = SyntheticNormalDataset(
        dataset_size=batch_size, batch_size=batch_size, width=width, resample=False, # full batch GD
        signal_strength=0.0,
        device=device,
    )

    # Set up a fixed measurement batch for logging
    measurement_X, _ = next(iter(train_loader))

    # Use BinaryLogger instead of CSVLogger for faster writes
    logger = BinaryLogger(run_dir, n_steps=n_train_steps, n_layers=model.n_layers)

    log_sample_size = 32
    trace = {}
    def trace_layer(name):
        def hook(model, input, output):
            trace[name] = {
                "input": input[0][:log_sample_size].detach(),
                "output": output[:log_sample_size].detach(),
                "weight": model.weight.data.detach(),
            }
        return hook
    for l_id, l in enumerate(model.layers):
        l.register_forward_hook(trace_layer(f'lin_{l_id}'))

    # Compute initial features
    with torch.no_grad():
        model(measurement_X)
    zL_init = trace[f'lin_{model.n_layers - 1}']["input"].clone()
    trace_init = copy.deepcopy(trace)

    s = 0
    diverged = False
    for X, y in train_loader:
        if s >= n_train_steps or diverged:
            break
        opt.zero_grad()
        y_hat = model(X)

        loss = F.mse_loss(y_hat, y)
        loss_item = loss.item()

        # Check for divergence
        if not np.isfinite(loss_item):  # Detect inf/nan
            loss_item = np.inf
            diverged = True  # Set flag to stop further computation
            logger.losses[s:] = loss_item  # Fill remaining steps with inf
            logger.log_norm_delta_features[s:] = np.nan
            print("Exiting early due to divergence...")
            break

        if s % log_freq == 0:
            with torch.no_grad():
                # compute feature_learning metric rL
                zL_current = trace[f'lin_{model.n_layers - 1}']["input"]
                delta_zL = zL_current - zL_init
                norm_delta_zL = delta_zL.norm().item()
                if norm_delta_zL > 0:
                    rL = np.log(norm_delta_zL) / np.log(width)
                else:
                    rL = -np.inf if s > 0 else 0  # (first step its exactly 0)

                # compute alignment metric
                Al = []
                for l_id in range(model.n_layers):
                    rms_norm = lambda x: torch.sqrt(torch.mean(x ** 2))
                    z, w, o = trace[f"lin_{l_id}"]["input"], trace[f"lin_{l_id}"]["weight"], trace[f"lin_{l_id}"]["output"]
                    z_n, w_n, o_n = rms_norm(z), rms_norm(w), rms_norm(o)

                    z_init, w_init = trace_init[f"lin_{l_id}"]["input"], trace_init[f"lin_{l_id}"]["weight"]
                    dw, dz = w - w_init, z - z_init
                    dw_n, dz_n = rms_norm(dw), rms_norm(dz)

                    # I. cumulative alignment
                    # NOTE: this is supposed to be log_{fan_in}, for now its always always fan_in = width
                    A_cum = (torch.log(o_n) - torch.log(z_n * w_n)) / torch.log(torch.tensor(width))
                    if dw_n + dz_n != 0.0:
                        # II. alpha alignment
                        o = z @ dw.T
                        o_n = rms_norm(o)
                        A_alpha = (torch.log(o_n) - torch.log(z_n * dw_n)) / torch.log(torch.tensor(width))
                        if l_id > 0:  # by definition z = x therefore dz = 0
                            # III. omega alignment
                            o = dz @ w.T
                            o_n = rms_norm(o)
                            A_omega = (torch.log(o_n) - torch.log(dz_n * w_n)) / torch.log(torch.tensor(width))
                            # IV. u alignment
                            o = dz @ dw.T
                            o_n = rms_norm(o)
                            A_u = (torch.log(o_n) - torch.log(dz_n * dw_n)) / torch.log(torch.tensor(width))
                        else:
                            temp = -torch.inf if s > 0 else torch.tensor(0)
                            A_omega, A_u = temp, temp
                    else:
                        temp = -torch.inf if s > 0 else torch.tensor(0)
                        A_alpha, A_omega, A_u = temp, temp, temp

                    Al.append(torch.tensor([A_cum, A_alpha, A_omega, A_u]))
                Al = torch.stack(Al).detach().cpu().numpy()
            logger.log(s, loss_item, rL, Al)  # Log each step's loss and log_norm

        loss.backward()
        opt.step()
        s += 1

    # Save the final arrays to binary format
    logger.save()

def main(run_name, model_config, optimizer_config, training_config, parametrization_config):
    run_dir = os.path.join("/app/maciej/junk/fractal/runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    configs = {
        # "training": training_config,
        # "model": model_config,
        # "optimizer": optimizer_config,
        "parametrization": parametrization_config,
    }
    for config_name, config in configs.items():
        config_path = os.path.join(run_dir, f"{config_name}_config.json")
        with open(config_path, "w") as f:
            json.dump(config().get_params(), f, indent=4)

    training_config().build(
        model_config=model_config, 
        optimizer_config=optimizer_config,
        parametrization_config=parametrization_config,
        run_dir=run_dir,
    )

from configs.fractal_config import jascha_grid

if __name__ == "__main__":
    RESOLUTION = 4
    worker_id = int(os.environ.get("WORKER_ID", 0))
    n_workers = int(os.environ.get("N_WORKERS", 1))

    grid = jascha_grid
    for exp_id, run_name, param_args in grid(resolution=RESOLUTION):
        if exp_id % n_workers == worker_id:
            t0 = time.time()
            main(run_name, *param_args)
            tf = time.time()
            print(f"Experiment {run_name} (id={exp_id}) completed in {tf - t0:.2f} seconds by worker {worker_id}.")