"""Graph autoencoder: a graph convolutional network (GCN) trained to
reconstruct healthy sensor readings, so unusually large reconstruction
error signals damage. See TowerWatch_guideline.md Sec 6.3.

Plain PyTorch, dense adjacency matmul, no PyTorch Geometric dependency
(TowerWatch_guideline.md Sec 1: "avoid as a hard dependency").
"""

import torch
from torch import nn


class GCNLayer(nn.Module):
    """One graph-convolution layer: mix each node's features with its
    graph neighbours' via `a_hat`, then a per-node linear projection.
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_nodes, d_in), a_hat: (n_nodes, n_nodes)
        return self.lin(a_hat @ x)


class GCNAutoencoder(nn.Module):
    """Encoder: X -> GCN -> ReLU -> GCN -> Z. Decoder: Z -> GCN -> ReLU -> GCN -> X_hat.

    Trained with MSE(X, X_hat) on healthy windows only; reconstruction
    error on unseen windows is the anomaly signal.
    """

    def __init__(self, n_features: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.enc1 = GCNLayer(n_features, hidden_dim)
        self.enc2 = GCNLayer(hidden_dim, latent_dim)
        self.dec1 = GCNLayer(latent_dim, hidden_dim)
        self.dec2 = GCNLayer(hidden_dim, n_features)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        z = self.relu(self.enc1(x, a_hat))
        z = self.enc2(z, a_hat)
        h = self.relu(self.dec1(z, a_hat))
        return self.dec2(h, a_hat)


def per_node_error(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Mean squared error per node, averaged over features: (batch, n_nodes)."""
    return ((x - x_hat) ** 2).mean(dim=-1)


def global_anomaly_score(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Detection score per window: mean node reconstruction error across
    the whole graph, Sec 6.3. Shape: (batch,).
    """
    return per_node_error(x, x_hat).mean(dim=-1)
