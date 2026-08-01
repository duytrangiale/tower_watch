"""A per-sensor autoencoder: the opposite end of the spectrum from the
graph autoencoder. Each sensor gets its own independent set of weights and
sees only its own 15 features, no mixing with any other sensor at any
layer. Built to test whether the graph autoencoder's neighbour-mixing is
helping or hurting localisation (see DAY_3.md, "Richer architectures":
a sparser graph already measured slightly better raw-error detection than
the denser one, this removes the graph entirely as the natural next
step). Idea from the SHM literature (Jiang et al. 2021; Giglioni et al.
2022, reviewed in Eltouny, Gomaa & Liang 2023), where training one
autoencoder per sensor is a standard, well-used approach specifically for
localisation.

Matches GCNAutoencoder's forward(x, prop_matrix) signature so it is a
drop-in replacement everywhere that expects one, even though prop_matrix
is accepted and ignored, there is no graph here.
"""

import torch
from torch import nn


class PerSensorLayer(nn.Module):
    """Like GCNLayer, but with an independent weight matrix per sensor and
    no neighbour mixing: sensor i's output depends only on sensor i's own
    input.
    """

    def __init__(self, n_nodes: int, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_nodes, d_in, d_out))
        self.bias = nn.Parameter(torch.empty(n_nodes, d_out))
        bound = 1.0 / (d_in ** 0.5)  # matches nn.Linear's default init
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, prop_matrix=None) -> torch.Tensor:
        # x: (batch, n_nodes, d_in) -> (batch, n_nodes, d_out)
        return torch.einsum("bni,nio->bno", x, self.weight) + self.bias


class PerSensorAutoencoder(nn.Module):
    """Encoder: X -> PerSensorLayer -> ReLU -> PerSensorLayer -> Z.
    Decoder: Z -> PerSensorLayer -> ReLU -> PerSensorLayer -> X_hat.
    Same 4-layer shape as GCNAutoencoder, trained the same way (healthy
    windows only, MSE loss), the only difference is no cross-sensor mixing.
    """

    def __init__(self, n_nodes: int, n_features: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.enc1 = PerSensorLayer(n_nodes, n_features, hidden_dim)
        self.enc2 = PerSensorLayer(n_nodes, hidden_dim, latent_dim)
        self.dec1 = PerSensorLayer(n_nodes, latent_dim, hidden_dim)
        self.dec2 = PerSensorLayer(n_nodes, hidden_dim, n_features)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, prop_matrix=None) -> torch.Tensor:
        z = self.relu(self.enc1(x, prop_matrix))
        z = self.enc2(z, prop_matrix)
        h = self.relu(self.dec1(z, prop_matrix))
        return self.dec2(h, prop_matrix)
