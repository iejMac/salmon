#!/usr/bin/env python3

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

###############################################################################
#                              CONFIGURATION
###############################################################################
BASE_PATH = "/home/maciej/code/salmon/workloads/cifar-10/runs"
run_dirs = [os.path.join(BASE_PATH, d) for d in os.listdir(BASE_PATH)]
param1_name = "al.2"       # e.g., "al.2"
param2_name = "bl.2"       # e.g., "bl.2"
output_filename = "overlayed_losses.png"

###############################################################################
#                        HELPER FUNCTIONS
###############################################################################
def extract_parameter_values(config_path: str, param1: str, param2: str) -> Tuple[float, float]:
    """
    Load 'parametrization_config.json' and return the values of the specified two parameters.
    This function handles nested dicts/lists of the form 'a.b' or 'a.2'.
    Returns (param1_value, param2_value) or (None, None) if keys are missing or file not found.
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        return (None, None)
    except json.JSONDecodeError:
        print(f"JSON parsing error in {config_path}")
        return (None, None)

    def get_nested_value(cfg, dotted_key: str):
        keys = dotted_key.split('.')
        val = cfg
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            elif isinstance(val, list):
                try:
                    idx = int(k)
                    val = val[idx]
                except (ValueError, IndexError):
                    return None
            else:
                return None
        return val

    val1 = get_nested_value(config, param1)
    val2 = get_nested_value(config, param2)
    return (val1, val2)

def collect_losses_for_grid(run_dirs: List[str], param1: str, param2: str) -> Dict[Tuple[float, float], List[Tuple[str, np.ndarray]]]:
    """
    Collect losses across multiple run directories, keyed by (param1_value, param2_value).
    
    We store a list of (run_label, losses_array) so that we can later distinguish
    which run each curve came from.
    
    Returns a dict:
      {
         (p1_val, p2_val): [
             (run_label_1, losses_array_1),
             (run_label_2, losses_array_2),
             ...
         ],
         ...
      }
    """
    all_losses_dict = {}

    # Create a color cycle or just rely on Matplotlib's default cycle if you prefer.
    # For a consistent legend, we will track run_dir -> label -> color mapping later.
    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            print(f"Skipping invalid run_dir: {run_dir}")
            continue

        run_label = os.path.basename(run_dir)  # e.g., "run1" if path ends with "/run1"

        # Each run_dir is expected to have subdirectories for each param combo
        subdirs = [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))]
        for subdir in subdirs:
            subdir_path = os.path.join(run_dir, subdir)
            config_path = os.path.join(subdir_path, "parametrization_config.json")
            if not os.path.exists(config_path):
                continue

            (p1_val, p2_val) = extract_parameter_values(config_path, param1, param2)
            if p1_val is None or p2_val is None:
                # If the parameter is missing or config is broken, skip
                continue

            # Attempt to load losses
            losses_path = os.path.join(subdir_path, "losses.npy")
            try:
                losses = np.load(losses_path)
            except FileNotFoundError:
                print(f"File not found: {losses_path} (skipping)")
                continue
            except Exception as e:
                print(f"Error loading {losses_path}: {str(e)} (skipping)")
                continue

            key = (p1_val, p2_val)
            if key not in all_losses_dict:
                all_losses_dict[key] = []
            # Store (run_label, the losses array)
            all_losses_dict[key].append((run_label, losses))
    
    return all_losses_dict

def plot_overlayed_losses(
    all_losses_dict: Dict[Tuple[float, float], List[Tuple[str, np.ndarray]]],
    param1: str,
    param2: str,
    output_filename: str
):
    """
    Generate a grid of subplots where rows are unique sorted param1 values
    and columns are unique sorted param2 values. In each cell, overlay all
    losses from the multiple run directories that share that (param1, param2).
    
    A single legend is constructed at the top that maps run_label -> color.
    """
    if not all_losses_dict:
        print("No valid data found to plot.")
        return

    # Gather unique, sorted parameter values for param1 and param2
    param1_values = sorted(set(k[0] for k in all_losses_dict.keys()))
    param2_values = sorted(set(k[1] for k in all_losses_dict.keys()))

    n_rows = len(param1_values)
    n_cols = len(param2_values)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), sharex=False, sharey=False)
    
    # If there's only one row or one column, axes might be 1D. Make sure it's always 2D.
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([ax for ax in axes]).reshape(n_rows, 1)

    # Initialize all subplots with a "No data" note
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Loss")
            ax.grid(alpha=0.3)

    # We also want to track each run_label -> color for a single legend.
    # Let's define a color cycle or rely on the built-in color cycle from matplotlib.
    # We'll accumulate used run_labels in a set/dict so we can build a single legend at the end.
    run_label_to_color = {}
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']  # default color cycle
    next_color_idx = 0

    # Fill data for each valid param combination
    for (p1_val, p2_val), list_of_runs_losses in all_losses_dict.items():
        row = param1_values.index(p1_val)
        col = param2_values.index(p2_val)
        ax = axes[row, col]

        # Clear the "No data" placeholder
        ax.clear()

        # Overlay each run's loss curve
        for (run_label, losses_array) in list_of_runs_losses:
            # If we haven't assigned a color to this run_label yet, do so now
            if run_label not in run_label_to_color:
                run_label_to_color[run_label] = color_cycle[next_color_idx % len(color_cycle)]
                next_color_idx += 1

            ax.plot(losses_array,
                    label=run_label,   # We will handle legend in a single global place
                    color=run_label_to_color[run_label])

        # Title
        ax.set_title(f"{param1}={p1_val}, {param2}={p2_val}", fontsize=10)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)

    # Build a single legend for all run_labels
    legend_handles = []
    for run_label, c in run_label_to_color.items():
        # Create a fake line for the legend
        line = plt.Line2D([0], [0], color=c, label=run_label)
        legend_handles.append(line)

    # Place the legend outside or inside as you prefer:
    # Here, let's put it up top:
    fig.legend(
        handles=legend_handles,
        loc='upper center',
        ncol=len(legend_handles),
        bbox_to_anchor=(0.5, 1.02),
        fontsize=8
    )

    # Make layout a bit nicer
    fig.tight_layout()
    print(f"Saving figure to {output_filename}")
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

###############################################################################
#                                 MAIN
###############################################################################
def main():
    # 1. Collect data from all run directories
    all_losses_dict = collect_losses_for_grid(run_dirs, param1_name, param2_name)

    # 2. Plot data with a single legend that distinguishes the runs by folder name
    plot_overlayed_losses(all_losses_dict, param1_name, param2_name, output_filename)

if __name__ == "__main__":
    main()