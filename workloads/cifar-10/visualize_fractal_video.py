import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Parameters
runs_dir = "/app/maciej/junk/fractal/runs_tanh_zoom0"
visualizations_dir = "./figures"
exp_name = "mf_tanh_c_zoom0_R512"
output_gif = os.path.join(visualizations_dir, f"{exp_name}.gif")  # Output GIF file
max_val = 1e6
min_step = 20  # Starting step for frames
n_steps = 500  # Total number of steps

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
        return np.load(metrics_path, mmap_mode='r')  # Use memory mapping
    return None

def process_single_run(run_dir):
    """Process a single run directory to load cl values and metrics."""
    try:
        cl = load_parametrization_config(run_dir)
        losses = load_metrics(run_dir, "losses.npy")
        features = load_metrics(run_dir, "log_norm_delta_features.npy")

        if losses is not None and features is not None:
            # Convert memory-mapped arrays to regular arrays
            losses = losses[:n_steps]  # Ensure we only load up to n_steps
            features = features[:n_steps]

            return (cl[0], cl[1], losses, features)
        else:
            logging.warning(f"Metrics not found for {run_dir}")
            return None
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

def create_convergence_animation(cl1_array, cl2_array, losses_array, features_array, output_gif):
    """Create and save a convergence animation over time for both metrics."""
    os.makedirs(visualizations_dir, exist_ok=True)
    n_runs, n_steps_total = losses_array.shape
    logging.info(f"Creating convergence animation over time from step {min_step} to {n_steps}")

    # Precompute necessary quantities for losses
    first_metric_losses = losses_array[:, 0]
    scaled_losses_array = losses_array / first_metric_losses[:, np.newaxis]
    scaled_losses_array = np.nan_to_num(scaled_losses_array, nan=max_val, posinf=max_val, neginf=-max_val)
    scaled_losses_array = np.clip(scaled_losses_array, -max_val, max_val)
    cumsum_scaled_losses = np.cumsum(scaled_losses_array, axis=1)
    reciprocal_scaled_losses = 1 / scaled_losses_array
    reciprocal_scaled_losses = np.nan_to_num(reciprocal_scaled_losses, nan=0.0, posinf=0.0, neginf=0.0)
    cumsum_reciprocal_scaled_losses = np.cumsum(reciprocal_scaled_losses, axis=1)

    # Precompute necessary quantities for features
    features_array = np.nan_to_num(features_array, nan=0.0)
    cumsum_features = np.cumsum(features_array, axis=1)
    count_features = np.cumsum(np.isfinite(features_array), axis=1)

    # Prepare grid indices
    cl1_unique = np.sort(np.unique(cl1_array))
    cl2_unique = np.sort(np.unique(cl2_array))
    cl1_idx = np.searchsorted(cl1_unique, cl1_array)
    cl2_idx = np.searchsorted(cl2_unique, cl2_array)
    grid_shape = (len(cl2_unique), len(cl1_unique))  # (rows, cols)

    # Initialize the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    norm = Normalize(vmin=-1, vmax=1)

    cl1_min, cl1_max = cl1_unique.min(), cl1_unique.max()
    cl2_min, cl2_max = cl2_unique.min(), cl2_unique.max()

    # Initialize images
    im_losses = axes[0].imshow(
        np.zeros(grid_shape),
        origin='lower',
        extent=(cl1_min, cl1_max, cl2_min, cl2_max),
        cmap='Spectral',
        norm=norm,
        interpolation='nearest',
        aspect='auto'
    )
    axes[0].set_xlabel("cl1")
    axes[0].set_ylabel("cl2")
    axes[0].set_title("Convergence (Losses)")

    im_features = axes[1].imshow(
        np.zeros(grid_shape),
        origin='lower',
        extent=(cl1_min, cl1_max, cl2_min, cl2_max),
        cmap='Spectral',
        norm=norm,
        interpolation='nearest',
        aspect='auto'
    )
    axes[1].set_xlabel("cl1")
    axes[1].set_ylabel("cl2")
    axes[1].set_title("Feature Learning")

    plt.tight_layout()

    # Number of frames
    n_frames = n_steps - min_step + 1

    # Collect frames
    frames = []

    for frame_idx in range(n_frames):
        step = min_step + frame_idx
        logging.info(f"Processing frame {frame_idx + 1}/{n_frames} at step {step}")

        # Losses convergence calculation
        total_sum = cumsum_scaled_losses[:, step - 1]
        reciprocal_sum = cumsum_reciprocal_scaled_losses[:, step - 1]

        # Handle last 20 steps for convergence check
        if step >= 20:
            sum_last20 = cumsum_scaled_losses[:, step - 1] - cumsum_scaled_losses[:, step - 21]
            mean_last20 = sum_last20 / 20
        else:
            sum_last20 = cumsum_scaled_losses[:, step - 1]
            mean_last20 = sum_last20 / step

        converged = mean_last20 < 1
        convergence_values_losses = np.where(converged, -total_sum, reciprocal_sum)

        # Features convergence calculation
        sum_features = cumsum_features[:, step - 1]
        count_valid = count_features[:, step - 1]
        mean_exponent = np.divide(sum_features, count_valid, out=np.zeros_like(sum_features), where=count_valid!=0)
        convergence_values_features = -mean_exponent

        # Create grids for visualization
        convergence_grid_losses = np.full(grid_shape, np.nan)
        convergence_grid_features = np.full(grid_shape, np.nan)
        convergence_grid_losses[cl2_idx, cl1_idx] = convergence_values_losses
        convergence_grid_features[cl2_idx, cl1_idx] = convergence_values_features

        # Rescale for visualization
        rescaled_values_losses = np.nan_to_num(cdf_img(convergence_grid_losses, convergence_grid_losses), nan=0.0)
        rescaled_values_features = np.nan_to_num(cdf_img(convergence_grid_features, convergence_grid_features), nan=0.0)

        # Update images
        im_losses.set_data(rescaled_values_losses)
        axes[0].set_title(f"Convergence (Losses) up to Step {step}")

        im_features.set_data(rescaled_values_features)
        axes[1].set_title(f"Feature Learning up to Step {step}")

        # Draw the canvas and convert to image array
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

    plt.close(fig)  # Close the figure to free memory

    # Save frames as an animated GIF
    logging.info(f"Saving animation to {output_gif}")
    import imageio
    imageio.mimsave(output_gif, frames, fps=10)

def main():
    logging.info("Starting data collection")
    run_dirs = []
    logging.info("Scanning run directories")
    with os.scandir(runs_dir) as it:
        for entry in it:
            if entry.is_dir():
                run_dirs.append(entry.path)
    logging.info(f"Found {len(run_dirs)} run directories")

    # Prepare arrays to hold data
    cl1_list = []
    cl2_list = []
    losses_list = []
    features_list = []

    # Use number of CPU cores minus 1 to avoid overloading the system
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    logging.info(f"Using {num_workers} worker processes")

    # Process runs in batches to manage memory usage
    batch_size = 10000  # Adjust based on your system's capacity
    total_runs = len(run_dirs)
    num_batches = (total_runs + batch_size - 1) // batch_size
    logging.info(f"Processing runs in {num_batches} batches of up to {batch_size} runs each")

    for batch_idx in range(num_batches):
        batch_run_dirs = run_dirs[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        logging.info(f"Processing batch {batch_idx + 1}/{num_batches} with {len(batch_run_dirs)} runs")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_single_run, batch_run_dirs))

        # Collect results
        for result in results:
            if result is not None:
                cl1, cl2, losses, features = result
                cl1_list.append(cl1)
                cl2_list.append(cl2)
                losses_list.append(losses)
                features_list.append(features)

    # Convert lists to arrays
    cl1_array = np.array(cl1_list)
    cl2_array = np.array(cl2_list)
    losses_array = np.vstack(losses_list)
    features_array = np.vstack(features_list)

    logging.info(f"Collected data for {losses_array.shape[0]} runs")

    # Create convergence animation over time
    create_convergence_animation(cl1_array, cl2_array, losses_array, features_array, output_gif)

    logging.info("Finished processing")

if __name__ == "__main__":
    main()
