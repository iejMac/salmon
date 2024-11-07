import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing

# Parameters
runs_dir = "./runs"
visualizations_dir = "./figures"
exp_name = "mf_c0.0,2.0_R256"
output_image = os.path.join(visualizations_dir, f"{exp_name}.png")
output_array = os.path.join(visualizations_dir, f"{exp_name}.npy")
max_val = 1e6

def load_parametrization_config(run_dir):
    """Load the parametrization config to retrieve `cl` values."""
    config_path = os.path.join(run_dir, "parametrization_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config["cl"]

def load_metrics(run_dir, metric_filename):
    """Load the binary metric file to get the training metric values."""
    metrics_path = os.path.join(run_dir, metric_filename)
    if os.path.exists(metrics_path):
        # Use memory mapping for large files
        return np.load(metrics_path, mmap_mode='r')
    return None

def convergence_measure(metrics, metric_type, max_val=1e6):
    """Calculate convergence measure to determine divergence or convergence."""
    # Convert memory-mapped array to regular array only for the required calculations
    if hasattr(metrics, 'compute'): 
        metrics = np.array(metrics)

    if metric_type == "losses":
        metrics = np.nan_to_num(metrics, nan=max_val, posinf=max_val, neginf=-max_val)
        metrics = metrics / (metrics[0] if metrics[0] != 0 else 1)
        metrics = np.clip(metrics, -max_val, max_val)
        converged = np.mean(metrics[-20:]) < 1
        return -np.sum(metrics) if converged else np.sum(1 / metrics)
    else:  # log_norm_delta_features
        valid_metrics = metrics[np.isfinite(metrics)]
        mean_exponent = valid_metrics.mean()
        return -mean_exponent

def process_single_run(args):
    """Process a single run directory - used for parallel processing."""
    run_dir, metric_filename = args
    if not os.path.isdir(run_dir):
        return None

    try:
        cl = load_parametrization_config(run_dir)
        metrics = load_metrics(run_dir, metric_filename)
        
        if metrics is not None:
            metric_type = metric_filename[:-4]
            convergence_value = convergence_measure(metrics, metric_type, max_val=max_val)
            return (cl[0], cl[1], convergence_value)
    except Exception as e:
        print(f"Error processing {run_dir}: {str(e)}")
        return None

def collect_run_data_parallel(runs_dir, metric_filename):
    """Collects data for each run in parallel."""
    run_dirs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir)]
    args = [(run_dir, metric_filename) for run_dir in run_dirs]
    
    # Use number of CPU cores minus 1 to avoid overloading the system
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_single_run, args))
    
    # Filter out None results and convert to list
    return [r for r in results if r is not None]

def cdf_img(x, x_ref, buffer=0.25):
    """Rescale x relative to x_ref, emphasizing intensity with a buffer."""
    x_flat = x_ref.ravel()
    u = np.sort(x_flat[~np.isnan(x_flat)])
    
    # Vectorized operations for better performance
    num_neg = np.sum(u < 0)
    num_nonneg = len(u) - num_neg
    
    v = np.concatenate([
        np.linspace(-1, -buffer, num_neg),
        np.linspace(buffer, 1, num_nonneg)
    ])
    
    return -np.interp(x, u, v)

def create_square_grid(run_data):
    """Create interpolated square grid from run data."""
    data = np.array(run_data)
    cl1_vals, cl2_vals, convergence_values = data.T
    
    grid_size = max(len(set(cl1_vals)), len(set(cl2_vals)))
    
    # Create grid points using vectorized operations
    cl1_grid = np.linspace(cl1_vals.min(), cl1_vals.max(), grid_size)
    cl2_grid = np.linspace(cl2_vals.min(), cl2_vals.max(), grid_size)
    grid_cl1, grid_cl2 = np.meshgrid(cl1_grid, cl2_grid)
    
    grid_convergence = griddata(
        (cl1_vals, cl2_vals),
        convergence_values,
        (grid_cl1, grid_cl2),
        method='linear'
    )
    
    return grid_cl1, grid_cl2, grid_convergence

def create_convergence_images(run_data_losses, run_data_features, output_image, output_array):
    """Create and save visualization with both metrics."""
    os.makedirs(visualizations_dir, exist_ok=True)

    # Process both metrics in parallel using ThreadPoolExecutor
    with ProcessPoolExecutor(max_workers=2) as executor:
        future_losses = executor.submit(create_square_grid, run_data_losses)
        future_features = executor.submit(create_square_grid, run_data_features)
        
        grid_cl1, grid_cl2, grid_convergence_losses = future_losses.result()
        _, _, grid_convergence_features = future_features.result()

    # Transform and handle NaNs
    rescaled_values_losses = np.nan_to_num(cdf_img(grid_convergence_losses, grid_convergence_losses), nan=0.0)
    rescaled_values_features = np.nan_to_num(cdf_img(grid_convergence_features, grid_convergence_features), nan=0.0)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    norm = Normalize(vmin=-1, vmax=1)

    for ax, data, title in zip(
        axes,
        [rescaled_values_losses, rescaled_values_features],
        ["convergence", "feature_learning"]
    ):
        im = ax.imshow(
            data,
            origin='lower',
            extent=(grid_cl1.min(), grid_cl1.max(), grid_cl2.min(), grid_cl2.max()),
            cmap='Spectral',
            norm=norm,
            interpolation='nearest',
            aspect='equal'
        )
        ax.set_xlabel("cl1")
        ax.set_ylabel("cl2")
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    
    # Save arrays in parallel
    with ProcessPoolExecutor(max_workers=2) as executor:
        executor.submit(np.save, output_array.replace('.npy', '_losses.npy'), rescaled_values_losses)
        executor.submit(np.save, output_array.replace('.npy', '_features.npy'), rescaled_values_features)
    
    print(f"Saved convergence visualization to {output_image}")

if __name__ == "__main__":
    # Collect data for both metrics in parallel
    with ProcessPoolExecutor(max_workers=2) as executor:
        future_losses = executor.submit(collect_run_data_parallel, runs_dir, "losses.npy")
        future_features = executor.submit(collect_run_data_parallel, runs_dir, "log_norm_delta_features.npy")
        
        run_data_losses = future_losses.result()
        run_data_features = future_features.result()

    create_convergence_images(run_data_losses, run_data_features, output_image, output_array)