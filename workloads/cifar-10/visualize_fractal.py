import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata

# Parameters
runs_dir = "./runs"  # Directory where runs are saved
visualizations_dir = "./figures"  # Directory to save visualizations
output_image = os.path.join(visualizations_dir, "convergence_visualization.png")
output_array = os.path.join(visualizations_dir, "convergence_grid.npy")
max_val = 1e6  # Threshold for divergence

def load_parametrization_config(run_dir):
    """Load the parametrization config to retrieve `cl` values."""
    config_path = os.path.join(run_dir, "parametrization_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config["cl"]

def load_metrics(run_dir):
    """Load the binary loss file to get the training loss values."""
    metrics_path = os.path.join(run_dir, "losses.npy")
    if os.path.exists(metrics_path):
        losses = np.load(metrics_path)
        return losses
    return None

def convergence_measure(losses, max_val=1e6):
    """Calculate convergence measure to determine divergence or convergence."""
    losses = np.nan_to_num(losses, nan=max_val, posinf=max_val, neginf=max_val)
    losses /= losses[0] if losses[0] != 0 else 1
    losses = np.clip(losses, -max_val, max_val)

    converged = np.mean(losses[-20:]) < 1
    return -np.sum(losses) if converged else np.sum(1 / losses)

def collect_run_data(runs_dir):
    """Collects data for each run and organizes by cl1 and cl2 values."""
    run_data = []

    for run_name in os.listdir(runs_dir):
        run_dir = os.path.join(runs_dir, run_name)
        if not os.path.isdir(run_dir):
            continue

        cl = load_parametrization_config(run_dir)
        cl1, cl2 = cl[0], cl[1]

        losses = load_metrics(run_dir)
        if losses is not None:
            convergence_value = convergence_measure(losses, max_val=max_val)
            run_data.append((cl1, cl2, convergence_value))

    return run_data

def extract_edges(X):
    """Detect edges as sign changes, identifying regions of convergence and divergence."""
    Y = np.stack((X[1:,1:], X[:-1,1:], X[1:,:-1], X[:-1,:-1]), axis=-1)
    Z = np.sign(np.max(Y, axis=-1) * np.min(Y, axis=-1))
    return Z < 0  # Edges occur where there's a sign change

def cdf_img(x, x_ref, buffer=0.25):
    """Rescale x relative to x_ref, emphasizing intensity with a buffer."""
    x_flat = x_ref.ravel()
    u = np.sort(x_flat[~np.isnan(x_flat)])
    num_neg = np.sum(u < 0)
    num_nonneg = u.shape[0] - num_neg

    # Adjust buffer to create more intensity contrast
    v = np.concatenate(
        (np.linspace(-1, -buffer, num_neg), np.linspace(buffer, 1, num_nonneg)),
        axis=0
    )
    y = np.interp(x, u, v)
    return -y  # Flip sign for enhanced visual contrast

def create_square_grid(run_data):
    cl1_vals = np.array([d[0] for d in run_data])
    cl2_vals = np.array([d[1] for d in run_data])
    convergence_values = np.array([d[2] for d in run_data])

    # Define the grid dimensions
    grid_size = max(len(set(cl1_vals)), len(set(cl2_vals)))

    # Create a square grid
    cl1_grid = np.linspace(cl1_vals.min(), cl1_vals.max(), grid_size)
    cl2_grid = np.linspace(cl2_vals.min(), cl2_vals.max(), grid_size)
    grid_cl1, grid_cl2 = np.meshgrid(cl1_grid, cl2_grid)

    # Interpolate the convergence values onto the square grid
    grid_convergence = griddata(
        (cl1_vals, cl2_vals),
        convergence_values,
        (grid_cl1, grid_cl2),
        method='linear'
    )

    return grid_cl1, grid_cl2, grid_convergence

def create_convergence_image(run_data, output_image, output_array):
    """Create and save a high-contrast image with edge detection for convergence visualization."""
    # Create the visualizations directory if it doesn't exist
    if not os.path.exists(visualizations_dir):
        os.makedirs(visualizations_dir)

    # Create a square grid and interpolate the data
    grid_cl1, grid_cl2, grid_convergence = create_square_grid(run_data)

    # Apply cdf_img transformation
    rescaled_values = cdf_img(grid_convergence, grid_convergence)

    # Handle NaNs that may result from interpolation
    rescaled_values = np.nan_to_num(rescaled_values, nan=0.0)

    # Apply edge detection
    edge_map = extract_edges(rescaled_values)

    # Use Normalize to apply high-contrast color mapping with `Spectral` colormap
    norm = Normalize(vmin=-1, vmax=1)

    # Plot with edge overlay and save
    plt.figure(figsize=(6, 6))  # Set figure size to be square
    plt.imshow(
        rescaled_values,
        origin='lower',
        extent=(grid_cl1.min(), grid_cl1.max(), grid_cl2.min(), grid_cl2.max()),
        cmap='Spectral',
        norm=norm,
        interpolation='nearest',
        aspect='equal'  # Set aspect ratio to 'equal'
    )
    # plt.colorbar(label="Convergence Metric")

    # Overlay edges in black to emphasize boundaries
    # plt.imshow(
    #     edge_map,
    #     origin='lower',
    #     extent=(grid_cl1.min(), grid_cl1.max(), grid_cl2.min(), grid_cl2.max()),
    #     cmap='gray',
    #     alpha=0.5,
    #     interpolation='nearest',
    #     aspect='equal'
    # )

    plt.xlabel("cl1")
    plt.ylabel("cl2")
    plt.savefig(output_image, dpi=300)
    plt.show()
    np.save(output_array, rescaled_values)
    print(f"saved convergence visualization to {output_image}")

if __name__ == "__main__":
    run_data = collect_run_data(runs_dir)
    create_convergence_image(run_data, output_image, output_array)
