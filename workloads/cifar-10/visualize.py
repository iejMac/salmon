import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Helper to read JSON configs
def read_json_config(file_path, key):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config[key]

# Read all metrics_{timestamp}.csv files and aggregate metrics
def read_metrics_from_files(metrics_files):
    all_metrics = {key: [] for key in ['train_loss', 'train_acc', 'val_loss', 'val_acc']}
    
    for file in metrics_files:
        with open(file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key in all_metrics.keys():
                    all_metrics[key].append(float(row[key]))

    # Return arrays for all metrics
    return {key: np.array(values) for key, values in all_metrics.items()}

# Plot the curves with shading for standard deviation
def plot_performance_curves(lr_values, metric_means, metric_stds, widths, metric_name, ax, y_max=None, minimize_metric=True):
    cmap = cm.get_cmap('viridis_r', len(np.unique(widths)))  # Reverse colormap
    unique_widths = np.unique(widths)

    for i, width in enumerate(unique_widths):
        mask = widths == width
        lr_subset, mean_subset, std_subset = lr_values[mask], metric_means[mask], metric_stds[mask]

        ax.plot(lr_subset, mean_subset, color=cmap(i), linewidth=2)
        ax.fill_between(lr_subset, mean_subset - std_subset, mean_subset + std_subset, color=cmap(i), alpha=0.2)

        # Mark optimal point
        optimal_idx = np.argmin(mean_subset) if minimize_metric else np.argmax(mean_subset)
        ax.scatter(lr_subset[optimal_idx], mean_subset[optimal_idx], color=cmap(i), edgecolor='black', zorder=5)

    ax.set_title(f'{metric_name} vs Learning Rate')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel(metric_name)
    ax.set_xscale('log')
    if y_max: ax.set_ylim([0, y_max])

# Main function to aggregate and plot metrics
def main(run_dir_root, output_image_path):
    widths, learning_rates = [], []
    metrics = {key: {'means': [], 'stds': []} for key in ['train_loss', 'train_acc', 'val_loss', 'val_acc']}

    # Loop over runs and gather data
    for run_dir in [os.path.join(run_dir_root, d) for d in os.listdir(run_dir_root) if os.path.isdir(os.path.join(run_dir_root, d))]:
        model_config_path = os.path.join(run_dir, 'model_config.json')
        optimizer_config_path = os.path.join(run_dir, 'optimizer_config.json')
        metrics_files = [os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.startswith('metrics_') and f.endswith('.csv')]

        if not metrics_files: continue

        # Read width and learning rate
        widths.append(read_json_config(model_config_path, 'dims')[1])
        learning_rates.append(read_json_config(optimizer_config_path, 'lr'))

        # Read metrics and compute mean and std
        run_metrics = read_metrics_from_files(metrics_files)
        for key in metrics.keys():
            metrics[key]['means'].append(np.mean(run_metrics[key]))
            metrics[key]['stds'].append(np.std(run_metrics[key]))

    # Convert lists to NumPy arrays for easier manipulation
    widths, learning_rates = np.array(widths), np.array(learning_rates)
    for key in metrics.keys():
        metrics[key]['means'] = np.array(metrics[key]['means'])
        metrics[key]['stds'] = np.array(metrics[key]['stds'])

    # Plot results
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plot_params = [
        ('train_loss', 'Train Loss', axs[0, 0], 3, True),
        ('train_acc', 'Train Accuracy', axs[0, 1], None, False),
        ('val_loss', 'Validation Loss', axs[1, 0], 3, True),
        ('val_acc', 'Validation Accuracy', axs[1, 1], None, False)
    ]
    for key, title, ax, y_max, minimize in plot_params:
        plot_performance_curves(learning_rates, metrics[key]['means'], metrics[key]['stds'], widths, title, ax, y_max, minimize)

    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.close()

if __name__ == "__main__":
    run_dir_root = "./runs"  # Path to the root directory with run directories
    output_image_path = "./lr_vs_performance_curves_with_shading.png"  # Path to save the curve plots
    main(run_dir_root, output_image_path)
