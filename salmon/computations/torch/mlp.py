from torch import nn


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4, dropout=0.0, bias=False):
        super().__init__()
        self.c_fc    = nn.Linear(dim, int(mlp_ratio * dim), bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(int(mlp_ratio * dim), dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x