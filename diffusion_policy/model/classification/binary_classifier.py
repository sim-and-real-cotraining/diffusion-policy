import torch
import torch.nn as nn

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, layer_sizes):
        super(BinaryClassifier, self).__init__()
        layers = []
        in_dim = input_dim
        for size in layer_sizes:
            layers.append(nn.Linear(in_dim, size))
            layers.append(nn.ReLU())
            in_dim = size
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        return x