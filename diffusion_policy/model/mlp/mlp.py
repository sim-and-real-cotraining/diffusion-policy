import torch
import torch.nn as nn

# Note this is the same as the binary classification MLP
# But is separated out for clarity

class MLP(nn.Module):
    def __init__(self, input_dim, layer_sizes, output_dim=1):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim
        for size in layer_sizes:
            layers.append(nn.Linear(in_dim, size))
            layers.append(nn.ReLU())
            in_dim = size
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        return x