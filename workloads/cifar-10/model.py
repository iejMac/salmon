import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dims, bias=False):
        super(MLP, self).__init__()
        self.n_layers = len(dims) - 1
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=bias) for i in range(self.n_layers)])
        self.layer_multipliers = [1.0 for i in range(self.n_layers)]

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)

            x *= self.layer_multipliers[i]

            if i < self.n_layers - 1:
                # x = F.relu(x)
                x = F.tanh(x)
        return x


if __name__ == "__main__":
    mod = MLP([2, 3])
    x = torch.randn(2)
    y = mod(x)
    print(y.shape)