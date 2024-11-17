import os
import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.lines import Line2D

# Enable LaTeX-like rendering (Option 1: Built-in math rendering)
plt.rcParams.update({
    "text.usetex": False,  # Set to True if you have a LaTeX installation and prefer full LaTeX rendering
    "font.family": "serif",
    "mathtext.fontset": "dejavusans",
})

# Directory path
root_dir = '/app/maciej/junk/fractal/runs'

# Function to load parameters from JSON config
def load_params_from_json(subdir_path):
    config_path = os.path.join(subdir_path, 'parametrization_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['cl'][0], config['cl'][1]

# Define line styles for components
COMPONENT_STYLES = {
    'Cumulative': '-',      # Solid line for Cumulative
    'Alpha': '--',          # Dashed line for Alpha
    'Omega': ':',           # Dotted line for Omega
    'U': '-.'               # Dash-dot line for U
}

# Define LaTeX labels for components
COMPONENT_LABELS = {
    'Cumulative': r'A',      # Plain text for Cumulative
    'Alpha': r'$\alpha$',             # Greek letter alpha
    'Omega': r'$\omega$',             # Greek letter omega
    'U': r'$\mathcal{u}$'             # Calligraphic U
}

# Define distinct colors for layers
LAYER_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']  # Color-blind friendly palette

# List all subdirectories
subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

# Extract unique c1 and c2 values from JSON configs
c1_c2_pairs = []
for subdir in subdirs:
    full_path = os.path.join(root_dir, subdir)
    c1, c2 = load_params_from_json(full_path)
    c1_c2_pairs.append((c1, c2))

# Get unique sorted values for c1 and c2
c1_values = sorted(set(pair[0] for pair in c1_c2_pairs))
c2_values = sorted(set(pair[1] for pair in c1_c2_pairs))

# Define grid size based on unique values of c1 and c2
N = len(c1_values)
assert N == len(c2_values), "Expected equal numbers of unique c1 and c2 values for a square grid."

# Create a mapping from c1 and c2 values to grid position
pos_map = {(c1, c2): (i, j) for i, c1 in enumerate(c1_values) for j, c2 in enumerate(c2_values)}

# Initialize the N x N grid of subplots with slightly smaller figure size
fig, axes = plt.subplots(N, N, figsize=(13, 12))
fig.suptitle('Alignment Metrics across Experiments', fontsize=16, y=0.95)

used_layer_indices = set()  # Track which layers are actually used

# Plot alignment metrics for each subdirectory
for subdir in subdirs:
    full_path = os.path.join(root_dir, subdir)
    
    # Load c1 and c2 from JSON
    c1, c2 = load_params_from_json(full_path)

    # Determine subplot position
    row, col = pos_map[(c1, c2)]
    ax = axes[row, col]

    # Load alignment_metrics.npy
    alignment_metrics = np.load(os.path.join(full_path, 'Als.npy'))
    n_steps, n_layers, n_components = alignment_metrics.shape

    # Load losses.npy and calculate average of the last 20 entries
    losses = np.load(os.path.join(full_path, 'losses.npy'))
    avg_last_20_losses = np.mean(losses[-20:])

    # Plot each layer and component
    for layer in range(n_layers):
        used_layer_indices.add(layer)  # Mark this layer as used
        for comp_idx, comp_name in enumerate(COMPONENT_STYLES.keys()):
            line_style = COMPONENT_STYLES[comp_name]
            ax.plot(alignment_metrics[:, layer, comp_idx], 
                    line_style,
                    color=LAYER_COLORS[layer],
                    alpha=0.7 if comp_idx > 0 else 1.0)  # Main result line is more prominent

    # Set plot title and labels
    ax.set_ylim(bottom=0.0, top=1.0)
    ax.set_title(f'c1={c1:.2f}, c2={c2:.2f}\nLoss(Last 20)={avg_last_20_losses:.4f}', fontsize=8)
    ax.tick_params(axis='both', which='both', labelsize=6)
    ax.grid(True, alpha=0.3)

# Create legends only for used layers and components
# 1. Layer Legend (using colors)
used_layer_legend_elements = [
    Line2D([0], [0], color=LAYER_COLORS[i], label=f'Layer {i}')
    for i in sorted(used_layer_indices)
]

# 2. Component Legend (using line styles and LaTeX labels)
component_legend_elements = [
    Line2D([0], [0], color='black', linestyle=COMPONENT_STYLES[comp_name], label=COMPONENT_LABELS[comp_name])
    for comp_name in COMPONENT_STYLES.keys()
]

# Add both legends at the bottom of the figure
layer_legend = fig.legend(handles=used_layer_legend_elements,
                          loc='lower center',
                          bbox_to_anchor=(0.3, 0.0),  # Adjust position as needed
                          ncol=len(used_layer_legend_elements),
                          title="Layers",
                          fontsize=8,
                          frameon=True)

component_legend = fig.legend(handles=component_legend_elements,
                              loc='lower center',
                              bbox_to_anchor=(0.7, 0.0),  # Adjust position as needed
                              ncol=len(component_legend_elements),
                              title="Components",
                              fontsize=8,
                              frameon=True)

# Adjust layout to provide space for the legends at the bottom
plt.tight_layout(rect=[0, 0.04, 1, 0.95])  # Reduced bottom margin from 0.12 to 0.08

# Save and close the figure
plt.savefig("plot.png", bbox_inches='tight', dpi=300)
plt.close(fig)
