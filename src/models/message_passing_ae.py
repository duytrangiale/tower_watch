"""A richer message-passing autoencoder, built to test a specific
hypothesis raised by the Day 3 diagnosis (DAY_3.md): the plain GCN mixes
each node with its neighbours through the *same* linear transform (via
the A+I self-loop baked into a_hat), which may dilute a node's own
distinctive signal into its neighbours' average. This layer instead uses
a separate learned transform for a node's own features versus its
neighbours', with the neighbour side aggregated either by a plain mean
(classic message passing) or by learned attention (a graph attention
network, GAT): each node learns how much weight to give each neighbour,
rather than using the fixed structural weights from src/graph/build.py.

Plain PyTorch, no PyTorch Geometric dependency, consistent with
src/models/gcn_ae.py.
"""

import torch
from torch import nn


class MessagePassingLayer(nn.Module):
    """out_i = self_transform(x_i) + aggregate_{j in neighbours(i)}(neighbour_transform(x_j))

    `adjacency` must NOT include self-loops: self-information is carried
    by the dedicated self-transform instead, so it is never diluted by
    whatever weight the neighbour aggregation would otherwise give it.
    """

    def __init__(self, d_in: int, d_out: int, use_attention: bool = False):
        super().__init__()
        self.self_lin = nn.Linear(d_in, d_out)
        self.neighbor_lin = nn.Linear(d_in, d_out)
        self.use_attention = use_attention
        if use_attention:
            self.attn = nn.Linear(2 * d_out, 1)
            self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_nodes, d_in), adjacency: (n_nodes, n_nodes), no self-loops
        neighbor_h = self.neighbor_lin(x)  # (batch, n_nodes, d_out)

        if self.use_attention:
            n_nodes = neighbor_h.shape[1]
            h_i = neighbor_h.unsqueeze(2).expand(-1, -1, n_nodes, -1)
            h_j = neighbor_h.unsqueeze(1).expand(-1, n_nodes, -1, -1)
            e = self.leaky_relu(self.attn(torch.cat([h_i, h_j], dim=-1)).squeeze(-1))
            e = e.masked_fill(adjacency == 0, float("-inf"))
            alpha = torch.softmax(e, dim=-1)  # normalised over each node's neighbours
            aggregated = alpha @ neighbor_h
        else:
            degree = adjacency.sum(dim=-1, keepdim=True).clamp(min=1)
            aggregated = (adjacency @ neighbor_h) / degree  # plain mean over neighbours

        return self.self_lin(x) + aggregated


class MessagePassingAutoencoder(nn.Module):
    """Same encoder/decoder shape as GCNAutoencoder (Sec 6.3), built from
    MessagePassingLayer instead of GCNLayer, for a like-for-like comparison.
    """

    def __init__(self, n_features: int, hidden_dim: int, latent_dim: int, use_attention: bool = False):
        super().__init__()
        self.enc1 = MessagePassingLayer(n_features, hidden_dim, use_attention)
        self.enc2 = MessagePassingLayer(hidden_dim, latent_dim, use_attention)
        self.dec1 = MessagePassingLayer(latent_dim, hidden_dim, use_attention)
        self.dec2 = MessagePassingLayer(hidden_dim, n_features, use_attention)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        z = self.relu(self.enc1(x, adjacency))
        z = self.enc2(z, adjacency)
        h = self.relu(self.dec1(z, adjacency))
        return self.dec2(h, adjacency)
