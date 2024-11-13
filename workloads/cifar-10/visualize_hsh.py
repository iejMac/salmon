import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict

# Helper to read JSON configs
def read_json_config(file_path, key):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config[key]

def read_metrics_from_files(metrics_files, widths, learning_rates):
    grouped_metrics = defaultdict(lambda: {key: [] for key in ['train_loss', 'train_acc', 'val_loss', 'val_acc']})

    # Only accumulate final step values per (width, lr) combination
    for file, width, lr in zip(metrics_files, widths, learning_rates):
        with open(file, 'r') as f:
            reader = list(csv.DictReader(f))
            if reader:  # Ensure the file isn't empty
                final_row = reader[-1]  # Take only the last row
                for key in grouped_metrics[(width, lr)]:
                    grouped_metrics[(width, lr)][key].append(float(final_row[key]))

    # Calculate mean and std per (width, lr) combination using final step values
    all_means, all_stds = defaultdict(list), defaultdict(list)
    for (width, lr), metrics in grouped_metrics.items():
        for key, values in metrics.items():
            all_means[key].append((width, lr, np.mean(values)))
            all_stds[key].append((width, lr, np.std(values)))

    return all_means, all_stds


# Plot the curves with shading for standard deviation
def plot_performance_curves(metric_means, metric_stds, metric_name, ax, y_max=None, minimize_metric=True):
    cmap = cm.get_cmap('viridis_r', len(set(width for width, lr, mean in metric_means)))  # Reverse colormap

    for i, width in enumerate(sorted(set(width for width, lr, mean in metric_means))):
        # Filter for the current width and sort by learning rate to ensure consistent plotting
        lr_subset = sorted([lr for (w, lr, _) in metric_means if w == width])
        mean_subset = [mean for (w, lr, mean) in sorted(metric_means, key=lambda x: x[1]) if w == width]
        std_subset = [std for (w, lr, std) in sorted(metric_stds, key=lambda x: x[1]) if w == width]

        ax.plot(lr_subset, mean_subset, color=cmap(i), linewidth=2, label=f'Width {width}')
        ax.fill_between(lr_subset, np.array(mean_subset) - np.array(std_subset), np.array(mean_subset) + np.array(std_subset), color=cmap(i), alpha=0.2)

        # Mark optimal point
        optimal_idx = np.argmin(mean_subset) if minimize_metric else np.argmax(mean_subset)
        ax.scatter(lr_subset[optimal_idx], mean_subset[optimal_idx], color=cmap(i), edgecolor='black', zorder=5)

    ax.set_title(f'{metric_name} vs Learning Rate')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel(metric_name)
    ax.set_xscale('log')
    if y_max: ax.set_ylim([0, y_max])
    # ax.legend()

# Main function to aggregate and plot metrics
def main(run_dir_root, output_image_path):
    widths, learning_rates = [], []
    metrics_files = []

    # Loop over runs and gather data
    for run_dir in [os.path.join(run_dir_root, d) for d in os.listdir(run_dir_root) if os.path.isdir(os.path.join(run_dir_root, d))]:
        model_config_path = os.path.join(run_dir, 'model_config.json')
        optimizer_config_path = os.path.join(run_dir, 'optimizer_config.json')
        metrics_files_in_run = [os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.startswith('metrics_') and f.endswith('.csv')]

        if not metrics_files_in_run:
            continue

        # Append width and learning rate for each metric file
        width = read_json_config(model_config_path, 'dims')[1]
        lr = read_json_config(optimizer_config_path, 'lr')
        widths.extend([width] * len(metrics_files_in_run))
        learning_rates.extend([lr] * len(metrics_files_in_run))
        metrics_files.extend(metrics_files_in_run)

    # Read metrics and compute means and stds grouped by (width, lr)
    all_means, all_stds = read_metrics_from_files(metrics_files, widths, learning_rates)

    # Plot results
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plot_params = [
        ('train_loss', 'Train Loss', axs[0, 0], None, True),
        ('train_acc', 'Train Accuracy', axs[0, 1], None, False),
        ('val_loss', 'Validation Loss', axs[1, 0], None, True),
        ('val_acc', 'Validation Accuracy', axs[1, 1], None, False)
    ]
    for key, title, ax, y_max, minimize in plot_params:
        plot_performance_curves(all_means[key], all_stds[key], title, ax, y_max, minimize)

    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.close()

if __name__ == "__main__":
    run_dir_root = "./runs"  # Path to the root directory with run directories
    output_image_path = "./hparam_shift.png"  # Path to save the curve plots
    main(run_dir_root, output_image_path)
