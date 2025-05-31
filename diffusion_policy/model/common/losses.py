import torch
from torch import nn

# MMD implementation modified from
# https://github.com/yiftachbeer/mmd_loss_pytorch

class RBF(nn.Module):

    def __init__(self, bandwidths=[0.25, 0.5, 1.0, 2.0, 4.0], base_bandwidth=None, device='cpu'):
        super().__init__()
        if base_bandwidth is not None:
            self.base_bandwidth = torch.tensor(base_bandwidth, device=device)
        else:
            self.base_bandwidth = None
        self.bandwidths = torch.tensor(bandwidths, device=device)

    # Base bandwidth used to normalize the kernels across different input data scales
    def get_base_bandwidth(self, L2_distances):
        if self.base_bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.base_bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        base_bandwidth = self.get_base_bandwidth(L2_distances)
        return torch.exp(-L2_distances[None, ...] / (base_bandwidth * self.bandwidths)[:, None, None]).mean(dim=0)


class MmdLoss(nn.Module):

    def __init__(self, bandwidths, base_bandwidth=None, device='cpu'):
        super().__init__()
        self.kernel = RBF(bandwidths, base_bandwidth, device=device)
        self.device = device

    def forward(self, X, Y):
        # Ensure inputs are on same device as kernel
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY
    
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mmd_loss = MmdLoss(bandwidths=[0.25, 0.5, 1.0, 2.0, 4.0], device=device)

    # Ensure that MMD = 0 when X = Y
    X = torch.randn(10, 10, device=device)
    Y = X.clone()
    assert mmd_loss(X, Y) == 0

    # Ensure that MMD is positive when X != Y
    X = torch.randn(10, 10, device=device)
    while torch.allclose(X, Y):
        Y = torch.randn(10, 10, device=device)
    loss = mmd_loss(X, Y)
    assert loss > 0

    # Ensure that MMD is symmetric
    assert torch.allclose(mmd_loss(X, Y), mmd_loss(Y, X))

    # ensure that MMD is invariant to translation
    X_translated = X + 1
    Y_translated = Y + 1
    assert torch.allclose(mmd_loss(X_translated, Y_translated), loss)

    # Ensure that MMD is invariant to scaling
    X_scaled = X * 2
    Y_scaled = Y * 2
    assert torch.allclose(mmd_loss(X_scaled, Y_scaled), loss)

    # Ensure that MMD grows with the distance between X and Y
    Y_far = X + 10
    loss_far = mmd_loss(X, Y_far)
    assert loss_far > loss

    # Ensure that MMD loss works with different number of samples
    X_small = X[:5]
    loss_small = mmd_loss(X_small, Y)

    # Ensure MMD works when one of the inputs is nothing
    loss_empty = mmd_loss(torch.empty(0, 10, device=device), Y)
    print(loss_empty)

    print(loss)