import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Parameters
runs_dir = "/app/maciej/junk/fractal/runs"
visualizations_dir = "./figures"
exp_name = "testing"
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
    if metric_type == "losses":
        metrics = np.nan_to_num(metrics, nan=max_val, posinf=max_val, neginf=-max_val)
        first_metric = metrics[0] if metrics[0] != 0 else 1
        metrics = metrics / first_metric
        metrics = np.clip(metrics, -max_val, max_val)
        converged = np.mean(metrics[-20:]) < 1
        total_sum = np.sum(metrics)
        reciprocal_sum = np.sum(1 / metrics)
        return -total_sum if converged else reciprocal_sum
    else:  # log_norm_delta_features
        valid_metrics = metrics[np.isfinite(metrics)]
        mean_exponent = valid_metrics.mean()
        return -mean_exponent

def process_single_run(args):
    """Process a single run directory."""
    run_dir, metric_filename = args
    if not os.path.isdir(run_dir):
        return None

    try:
        cl = load_parametrization_config(run_dir)
        metrics = load_metrics(run_dir, metric_filename)
        
        if metrics is not None:
            metric_type = metric_filename[:-4]
            convergence_value = convergence_measure(metrics, metric_type, max_val=max_val)
            logging.debug(f"Processed {run_dir} for {metric_type}")
            return (metric_type, cl[0], cl[1], convergence_value)
        else:
            logging.warning(f"Metrics file {metric_filename} not found in {run_dir}")
    except Exception as e:
        logging.error(f"Error processing {run_dir}: {str(e)}")
        return None

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
    cl1_vals, cl2_vals, convergence_values = data[:, 0], data[:, 1], data[:, 2]

    grid_size_cl1 = len(np.unique(cl1_vals))
    grid_size_cl2 = len(np.unique(cl2_vals))
    grid_size = max(grid_size_cl1, grid_size_cl2)

    logging.info(f"Creating grid of size {grid_size}x{grid_size}")

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

    logging.info("Creating convergence images")

    # Process metrics sequentially to avoid unnecessary parallelism
    logging.info("Creating grid for losses")
    grid_cl1, grid_cl2, grid_convergence_losses = create_square_grid(run_data_losses)
    logging.info("Creating grid for features")
    _, _, grid_convergence_features = create_square_grid(run_data_features)

    # Transform and handle NaNs
    logging.info("Rescaling loss values")
    rescaled_values_losses = np.nan_to_num(cdf_img(grid_convergence_losses, grid_convergence_losses), nan=0.0)
    logging.info("Rescaling feature values")
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
            # aspect='equal'
            aspect='auto'
        )
        ax.set_xlabel("cl1")
        ax.set_ylabel("cl2")
        ax.set_title(title)

    plt.tight_layout()
    logging.info(f"Saving image to {output_image}")
    plt.savefig(output_image, dpi=300, bbox_inches='tight')

    # Save arrays sequentially to avoid unnecessary parallelism
    logging.info(f"Saving rescaled losses array to {output_array.replace('.npy', '_losses.npy')}")
    np.save(output_array.replace('.npy', '_losses.npy'), rescaled_values_losses)
    logging.info(f"Saving rescaled features array to {output_array.replace('.npy', '_features.npy')}")
    np.save(output_array.replace('.npy', '_features.npy'), rescaled_values_features)

    logging.info(f"Saved convergence visualization to {output_image}")

if __name__ == "__main__":
    logging.info("Starting data collection")
    # Collect run directories using os.scandir for better performance
    run_dirs = []
    logging.info("Scanning run directories")
    with os.scandir(runs_dir) as it:
        for entry in it:
            if entry.is_dir():
                run_dirs.append(entry.path)
    logging.info(f"Found {len(run_dirs)} run directories")

    metric_filenames = ["losses.npy", "log_norm_delta_features.npy"]
    tasks = ((run_dir, metric_filename) for metric_filename in metric_filenames for run_dir in run_dirs)

    total_tasks = len(run_dirs) * len(metric_filenames)
    logging.info(f"Total tasks to process: {total_tasks}")

    # Use number of CPU cores minus 1 to avoid overloading the system
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    logging.info(f"Using {num_workers} worker processes")

    # Process tasks in batches to avoid overloading the system
    batch_size = 10000  # Adjust based on your system's capacity
    tasks_list = list(tasks)  # Convert generator to list
    num_batches = (total_tasks + batch_size - 1) // batch_size
    logging.info(f"Processing tasks in {num_batches} batches of up to {batch_size} tasks each")

    run_data_dict = {'losses': [], 'log_norm_delta_features': []}

    for i in range(num_batches):
        batch_tasks = tasks_list[i * batch_size:(i + 1) * batch_size]
        logging.info(f"Processing batch {i + 1}/{num_batches} with {len(batch_tasks)} tasks")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_single_run, batch_tasks))

        # Separate results by metric
        for result in results:
            if result is not None:
                metric_type, cl1, cl2, convergence_value = result
                run_data_dict[metric_type].append((cl1, cl2, convergence_value))

    run_data_losses = run_data_dict['losses']
    run_data_features = run_data_dict['log_norm_delta_features']

    logging.info(f"Collected data for {len(run_data_losses)} runs for losses")
    logging.info(f"Collected data for {len(run_data_features)} runs for features")

    # Create convergence images without unnecessary parallelism
    create_convergence_images(run_data_losses, run_data_features, output_image, output_array)
    logging.info("Finished processing")
