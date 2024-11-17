import os
import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.lines import Line2D
from collections import defaultdict
from typing import List, Dict, Tuple

# Enable LaTeX-like rendering
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "dejavusans",
})

# Styling constants
COMPONENT_STYLES = {
    'Cumulative': '-',      # Solid line for Cumulative
    'Alpha': '--',          # Dashed line for Alpha
    'Omega': ':',           # Dotted line for Omega
    'U': '-.'               # Dash-dot line for U
}

COMPONENT_LABELS = {
    'Cumulative': r'A',
    'Alpha': r'$\alpha$',
    'Omega': r'$\omega$',
    'U': r'$\mathcal{u}$'
}

LAYER_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']

# Figure settings to ensure even dimensions
FIG_WIDTH_INCHES = 13
FIG_HEIGHT_INCHES = 12
DPI = 300

# Calculate pixel dimensions
pixel_width = int(FIG_WIDTH_INCHES * DPI)
pixel_height = int(FIG_HEIGHT_INCHES * DPI)

# Ensure even dimensions
pixel_width += pixel_width % 2
pixel_height += pixel_height % 2

# Recalculate figsize based on even pixel dimensions
FIG_WIDTH_INCHES = pixel_width / DPI
FIG_HEIGHT_INCHES = pixel_height / DPI

def load_params_from_json(subdir_path: str) -> Tuple[float, float]:
    """Load parameters from parametrization config."""
    config_path = os.path.join(subdir_path, 'parametrization_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['cl'][0], config['cl'][1]

def load_data_config(subdir_path: str) -> Dict:
    """Load the data configuration."""
    config_path = os.path.join(subdir_path, 'data_config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def create_single_plot(directories: List[str], output_path: str, signal_strength: float):
    """Create a single plot for a group of directories."""
    # Extract c1 and c2 values
    c1_c2_pairs = []
    for dir_path in directories:
        c1, c2 = load_params_from_json(dir_path)
        c1_c2_pairs.append((c1, c2))

    # Get unique sorted values
    c1_values = sorted(set(pair[0] for pair in c1_c2_pairs))
    c2_values = sorted(set(pair[1] for pair in c1_c2_pairs))

    # Define grid size
    N = len(c1_values)
    assert N == len(c2_values), "Expected equal numbers of unique c1 and c2 values for a square grid."

    # Create position mapping
    pos_map = {(c1, c2): (i, j) for i, c1 in enumerate(c1_values) for j, c2 in enumerate(c2_values)}

    # Create figure with even dimensions
    fig, axes = plt.subplots(N, N, figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES), dpi=DPI)
    fig.suptitle(f'Alignment Metrics (signal_strength={signal_strength:.4f})', fontsize=16, y=0.95)

    used_layer_indices = set()

    # Plot for each directory
    for dir_path in directories:
        # Load parameters
        c1, c2 = load_params_from_json(dir_path)
        row, col = pos_map[(c1, c2)]
        ax = axes[row, col]

        # Load metrics
        alignment_metrics = np.load(os.path.join(dir_path, 'Als.npy'))
        losses = np.load(os.path.join(dir_path, 'losses.npy'))
        avg_last_20_losses = np.mean(losses[-20:])

        n_steps, n_layers, n_components = alignment_metrics.shape

        # Create plots
        for layer in range(n_layers):
            used_layer_indices.add(layer)
            for comp_idx, comp_name in enumerate(COMPONENT_STYLES.keys()):
                line_style = COMPONENT_STYLES[comp_name]
                ax.plot(alignment_metrics[:, layer, comp_idx], 
                        line_style,
                        color=LAYER_COLORS[layer],
                        alpha=0.7 if comp_idx > 0 else 1.0)

        # Configure subplot
        ax.set_ylim(bottom=0.0, top=1.0)
        ax.set_title(f'c1={c1:.2f}, c2={c2:.2f}\nLoss(Last 20)={avg_last_20_losses:.4f}', fontsize=8)
        ax.tick_params(axis='both', which='both', labelsize=6)
        ax.grid(True, alpha=0.3)

    # Create legends
    used_layer_legend_elements = [
        Line2D([0], [0], color=LAYER_COLORS[i], label=f'Layer {i}')
        for i in sorted(used_layer_indices)
    ]

    component_legend_elements = [
        Line2D([0], [0], color='black', linestyle=COMPONENT_STYLES[comp_name], label=COMPONENT_LABELS[comp_name])
        for comp_name in COMPONENT_STYLES.keys()
    ]

    # Add legends
    # To avoid overlapping legends, use a single legend with two columns
    # Alternatively, adjust positions as needed
    fig.legend(handles=used_layer_legend_elements + component_legend_elements,
               loc='lower center',
               bbox_to_anchor=(0.5, 0.0),
               ncol=len(used_layer_legend_elements) + len(component_legend_elements),
               title="Layers and Components",
               fontsize=8,
               frameon=True)

    # Save plot
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust rect to make space for legend
    plt.savefig(output_path, bbox_inches='tight', dpi=DPI)
    plt.close(fig)

def group_directories_by_signal_strength(root_dir: str) -> List[Tuple[float, List[str]]]:
    """Group directories by signal strength and return sorted groups."""
    strength_groups = defaultdict(list)
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for subdir in subdirs:
        full_path = os.path.join(root_dir, subdir)
        try:
            data_config = load_data_config(full_path)
            signal_strength = float(data_config['signal_strength'])
            strength_groups[signal_strength].append(full_path)
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"Skipping {subdir}: {str(e)}")
            continue
    
    return sorted(strength_groups.items(), key=lambda x: x[0])

def save_frames(root_dir: str = '/app/maciej/junk/fractal/runs'):
    """Save frames grouped by signal strength."""
    print("Starting frame generation process...")
    
    # Create output directory
    output_dir = os.path.join('figures', 'alignment_evolution')
    os.makedirs(output_dir, exist_ok=True)
    
    # Group directories
    sorted_groups = group_directories_by_signal_strength(root_dir)
    
    if not sorted_groups:
        raise ValueError("No valid directories found with signal strength data")
    
    # Generate frames
    for frame_idx, (signal_strength, directories) in enumerate(sorted_groups):
        print(f"Processing signal strength {signal_strength:.4f} ({frame_idx + 1}/{len(sorted_groups)})")
        
        frame_path = os.path.join(output_dir, f'frame_{frame_idx:04d}.png')
        create_single_plot(directories, frame_path, signal_strength)
    
    # Create ffmpeg shell script
    create_ffmpeg_script(output_dir, len(sorted_groups))
    
    print(f"Frames saved to {output_dir}")
    print(f"Total frames generated: {len(sorted_groups)}")

def create_ffmpeg_script(output_dir: str, num_frames: int):
    """Create a shell script with the ffmpeg command to generate a video."""
    script_path = os.path.join(output_dir, 'create_video.sh')
    video_output = 'alignment_evolution.mp4'
    
    # FFmpeg command with scaling filter to ensure even dimensions
    ffmpeg_command = (
        f"#!/bin/bash\n\n"
        f"ffmpeg -framerate 1 -i figures/alignment_evolution/frame_%04d.png -vf \"scale=ceil(iw/2)*2:ceil(ih/2)*2\" "
        f"-c:v libx264 -r 30 -pix_fmt yuv420p {video_output}\n"
    )
    
    with open(script_path, 'w') as script_file:
        script_file.write(ffmpeg_command)
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    print(f"FFmpeg script created at: {script_path}")
    print(f"To generate the video, navigate to {output_dir} and run ./create_video.sh")

if __name__ == "__main__":
    # Set this path as needed
    ROOT_DIR = '/app/maciej/junk/fractal/runs'
    
    save_frames(ROOT_DIR)
