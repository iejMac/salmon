import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, List
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from model import MLP

# Define a synthetic dataset with learnable patterns
class SyntheticDataset(Dataset):
    def __init__(self, n_samples=1000, input_dim=16, num_classes=10, noise_level=0.1):
        self.n_samples = n_samples
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.noise_level = noise_level
        
        # Create a unique prototype vector for each class
        self.prototypes = torch.randn(num_classes, input_dim)
        
        # Generate data samples as noisy versions of class prototypes
        self.data = []
        self.labels = []
        for i in range(n_samples):
            label = torch.randint(0, num_classes, (1,)).item()
            prototype = self.prototypes[label]
            noisy_sample = prototype + noise_level * torch.randn(input_dim)
            self.data.append(noisy_sample)
            self.labels.append(label)
        
        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Coordinate check function
def coordinate_check(
    parameterization_fns: List[Callable],
    titles: List[str],
    step_list: List[int],
    learning_rate: float,
    dataset_type: str = "synthetic",  # 'synthetic' or 'cifar10'
    width_grid: List[int] = [16, 32, 64, 128, 256],
    base_width: int = 16,  # Adjusted for synthetic dataset; 3*32*32 for CIFAR10
    trials: int = 5,
    n_layers: int = 4,
    batch_size: int = 8,
    data_dir: str = "./data"
):
    """For each step, check that activation coordinate size does not depend on width."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Choose dataset based on dataset_type parameter
    if dataset_type == "synthetic":
        synthetic_data = SyntheticDataset(n_samples=1000, input_dim=base_width, num_classes=base_width, noise_level=0.1)
        train_loader = DataLoader(synthetic_data, batch_size=batch_size, shuffle=True)
    elif dataset_type == "cifar10":
        # Adjust base width for CIFAR10 images (3 channels, 32x32 pixels)
        base_width = 3 * 32 * 32
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        cifar_data = CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        train_loader = DataLoader(cifar_data, batch_size=batch_size, shuffle=True)
    else:
        raise ValueError("Invalid dataset_type. Choose 'synthetic' or 'cifar10'.")

    # Set up a grid of subplots to accommodate all parameterizations and steps
    num_parametrizations = len(parameterization_fns)
    fig, axes = plt.subplots(len(step_list), num_parametrizations, figsize=(6 * num_parametrizations, 4 * len(step_list)),
                             sharex=True, sharey=True, squeeze=False)

    for col, (param_fn, title) in enumerate(zip(parameterization_fns, titles)):
        coord_check_data = []

        for width in width_grid:
            for trial in range(trials):
                model = MLP([base_width] + [width] * n_layers + [10]).to(device)
                params = param_fn(model, lr_prefactor=learning_rate, std_prefactor=1.0)  # Apply parameterization
                optimizer = torch.optim.SGD(params, lr=learning_rate)

                # Register forward hooks to log activations
                layer_activations = {}
                def store_activations(layer_name):
                    def hook_fn(module, input, output):
                        layer_activations[layer_name] = output
                    return hook_fn

                for i, layer in enumerate(model.layers):
                    layer.register_forward_hook(store_activations(f'layers.{i}'))


                for step, (X, y) in enumerate(train_loader):
                    if step > max(step_list):
                        break

                    X, y = X.to(device), y.to(device)
                    optimizer.zero_grad()
                    # Flatten CIFAR-10 images for MLP if CIFAR-10 is used
                    if dataset_type == "cifar10":
                        X = X.view(X.size(0), -1)

                    output = model(X)

                    # Log activation size (L1 norm) for each specified step
                    if step in step_list:
                        for layer_name, activation in layer_activations.items():
                            layer_num = int(layer_name.split('.')[1])  # Extract layer index from name

                            coordinate_size = activation.abs().mean().item()

                            result = pd.Series({
                                'width': width,
                                'trial': trial,
                                'step': step,
                                'layer_num': layer_num,
                                'l1_norm': coordinate_size
                            })
                            coord_check_data.append(result)

                    # Backpropagation step
                    loss = F.cross_entropy(output, y)
                    loss.backward()
                    optimizer.step()

        coord_check_data = pd.DataFrame(coord_check_data)
        cmap = LinearSegmentedColormap.from_list('blue_to_grey', ['blue', 'grey'])
        colors = cmap(np.linspace(0, 1, len(model.layers)))

        for row, step in enumerate(step_list):
            ax = axes[row, col]
            for layer_num in range(len(model.layers)):
                df = coord_check_data[(coord_check_data['step'] == step) & (coord_check_data['layer_num'] == layer_num)]
                median = df.groupby('width')['l1_norm'].median()
                std = df.groupby('width')['l1_norm'].std()

                # Plot layer curve with confidence intervals
                ax.plot(median.index, median, label=f'Layer {layer_num}', color=colors[layer_num], linewidth=1.3, alpha=0.7, marker='o', markersize=3)
                ax.fill_between(median.index, median - std, median + std, color=colors[layer_num], alpha=0.075)

            ax.set_xscale('log')
            ax.set_yscale('log')
            if row == len(step_list) - 1:
                ax.set_xlabel('Width')
            if col == 0:
                ax.set_ylabel('Coordinate Size (Average L1 Norm)')
            ax.set_title(f'{title}, Step {step}')
            # ax.legend(title='Layer', loc=2)
            ax.grid()

    plt.tight_layout()
    plt.savefig('coordinate_check_multiple_parametrizations.png')
    plt.show()

if __name__ == "__main__":
    from parametrization import standard_parametrization, spectral_parametrization, mu_parametrization, unstable_parametrization

    coordinate_check(
        parameterization_fns=[standard_parametrization, mu_parametrization],
        titles=["Standard Parameterization", "Mu Parametrization"],
        step_list=[0, 20, 40, 60],
        learning_rate=0.1,
        dataset_type="cifar10",  # Choose between 'synthetic' and 'cifar10'
        width_grid=[512, 1024, 2048, 4096, 8192, 16384],
        base_width=8,  # Adjusted base width for synthetic; will be overridden for CIFAR-10
        trials=2,
        n_layers=3,
        batch_size=8,
        data_dir="./data"
    )
