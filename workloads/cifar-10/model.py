import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, n_layers=3, hidden_channels=64):
        super(CNN, self).__init__()

        assert n_layers >= 3, "n_layers must be at least 3 (input layer, hidden layers, and output layer)"

        layers = [nn.Conv2d(in_channels=3, out_channels=hidden_channels, kernel_size=3, padding=1)]
        for _ in range(n_layers - 2):
            layers.append(nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1))
        layers.append(nn.Conv2d(in_channels=hidden_channels, out_channels=10, kernel_size=3, padding=1))
        self.layers = nn.ModuleList(layers)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)  # Flatten to (batch_size, num_classes)
        return x