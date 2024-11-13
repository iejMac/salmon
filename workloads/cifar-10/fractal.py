import os
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from config import mlp1h, sgd_frac, training_frac, mean_field_parametrization
from data import SyntheticNormalDataset
from parametrization import abc_parametrization

torch.set_default_dtype(torch.float64)  # Set precision to float64

class BinaryLogger:
    def __init__(self, log_dir, n_steps):
        self.log_dir = log_dir
        self.n_steps = n_steps
        self.losses = np.zeros(n_steps, dtype=np.float64)  # Pre-allocate space for losses
        self.log_norm_delta_features = np.zeros(n_steps, dtype=np.float64)
    
    def log(self, step, loss, log_norm_delta_features):
        self.losses[step] = loss
        self.log_norm_delta_features[step] = log_norm_delta_features

    def save(self):
        np.save(os.path.join(self.log_dir, 'losses.npy'), self.losses)
        np.save(os.path.join(self.log_dir, 'log_norm_delta_features.npy'), self.log_norm_delta_features)

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

    # Use BinaryLogger instead of CSVLogger for faster writes
    logger = BinaryLogger(run_dir, n_steps=n_train_steps)

    model = model_config().build()
    model = model.to(device)

    opt_cfg = optimizer_config()

    n = model_config()["dims"][1]  # Feature dimension (width of hidden layer)
    params = parametrization_config().build(mlp=model, n=n, lr_prefactor=opt_cfg['lr'], std_prefactor=1.0)

    opt = opt_cfg.build(params=params)

    train_loader = SyntheticNormalDataset(batch_size=batch_size, n=n, device=device)

    # Set up a fixed measurement batch for logging
    measurement_X, _ = next(iter(train_loader))

    # Set up a hook to capture activations before the last layer
    features = {}

    def get_activation(name):
        def hook(model, input, output):
            features[name] = input[0].detach()  # input[0] is the input to the last layer
        return hook

    # Register hook on the last layer to capture its input (features)
    model.layers[-1].register_forward_hook(get_activation('features'))

    # Compute initial features
    with torch.no_grad():
        model(measurement_X)
        features_init = features['features'].clone()

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
            # Compute log of norm of delta_features with base n
            with torch.no_grad():
                model(measurement_X)
                features_current = features['features']
                delta_features = features_current - features_init
                norm_delta_features = delta_features.norm().item()
                if norm_delta_features > 0:
                    log_norm_delta_features = np.log(norm_delta_features) / np.log(n)
                else:
                    log_norm_delta_features = -np.inf if s > 0 else 0  # (first step its exactly 0)

            logger.log(s, loss_item, log_norm_delta_features)  # Log each step's loss and log_norm

        loss.backward()
        opt.step()
        s += 1

    # Save the final arrays to binary format
    logger.save()

def main(run_name, model_config, optimizer_config, training_config, parametrization_config):
    # run_dir = os.path.join("./runs", run_name)
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

from config import mlp1h, sgd_frac, training_frac, mean_field_parametrization

if __name__ == "__main__":
    # Set grid resolution for adjustments
    RESOLUTION = 8  # Define how many steps per axis in the grid

    # Get worker ID and number of workers from environment variables
    worker_id = int(os.environ.get("WORKER_ID", 0))
    n_workers = int(os.environ.get("N_WORKERS", 1))

    # zoom0
    # c1_grid = np.linspace(-1.5, 0.5, num=RESOLUTION)
    # c2_grid = np.linspace(-3.4, -1.4, num=RESOLUTION)
    # zoom1
    # c1_grid = np.linspace(-0.65, -0.15, num=RESOLUTION)
    # c2_grid = np.linspace(-2.65, -2.15, num=RESOLUTION)
    # zoom2
    # c1_grid = np.linspace(-0.3, -0.2, num=RESOLUTION)
    # c2_grid = np.linspace(-2.55, -2.45, num=RESOLUTION)
    # zoom3
    # c1_grid = np.linspace(-0.255, -0.245, num=RESOLUTION)
    # c2_grid = np.linspace(-2.495, -2.485, num=RESOLUTION)
    # zoom4
    # c1_grid = np.linspace(-0.2501, -0.2499, num=RESOLUTION)
    # c2_grid = np.linspace(-2.4902, -2.4900, num=RESOLUTION)

    # zoom0 tanh
    # c1_grid = np.linspace(-0.5, 0.5, num=RESOLUTION)
    # c2_grid = np.linspace(-4.0, -3.0, num=RESOLUTION)
    # zoom1 tanh
    # c1_grid = np.linspace(-0.30, -0.15, num=RESOLUTION)
    # c2_grid = np.linspace(-3.55, -3.4, num=RESOLUTION)
    # zoom2 tanh
    c1_grid = np.linspace(-0.30, -0.22, num=RESOLUTION)
    c2_grid = np.linspace(-3.50, -3.47, num=RESOLUTION)

    # Loop over the grid, assigning tasks based on `exp_id`
    for c1_id, c1 in enumerate(c1_grid):
        for c2_id, c2 in enumerate(c2_grid):
            # Calculate unique experiment ID based on grid indices
            exp_id = c1_id * len(c2_grid) + c2_id
            if exp_id % n_workers == worker_id:
                # Define parametrization configuration with the modified `cl` values
                def mean_field_parametrization_with_diff_cl():
                    mf_cfg = mean_field_parametrization()
                    mf_cfg['cl'] = [c1, c2]
                    return mf_cfg

                # Create a unique run name for each experiment (careful about precision!)
                run_name = f"frac_test_c1_{c1:.14f}_c2_{c2:.14f}"

                # Run the experiment and measure execution time
                t0 = time.time()
                main(run_name, mlp1h, sgd_frac, training_frac, mean_field_parametrization_with_diff_cl)
                tf = time.time()
                print(f"Experiment {run_name} (id={exp_id}) completed in {tf - t0:.2f} seconds by worker {worker_id}.")
