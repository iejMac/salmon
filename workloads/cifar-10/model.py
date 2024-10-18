import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dims, bias=False):
        super(MLP, self).__init__()
        self.n_layers = len(dims) - 1
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=bias) for i in range(len(dims) - 1)])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.n_layers - 1:
                x = F.relu(x)
        return x

if __name__ == "__main__":
    mod = MLP([2, 4, 8])
    x = torch.randn(2)
    y = mod(x)