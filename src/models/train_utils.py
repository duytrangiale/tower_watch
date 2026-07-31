"""Shared training loop for every autoencoder variant in this project
(GCNAutoencoder, MessagePassingAutoencoder), so all of them are trained
the same, honest way: full-batch gradient descent with early stopping on
a held-out healthy validation set, rather than a fixed number of epochs.
Training loss alone cannot show overfitting (it can keep dropping as a
model starts memorising the training set); tracking a separate
validation set is what makes that visible.
"""

import numpy as np
import torch
from torch import nn


def train_autoencoder(model, x_train: np.ndarray, x_val: np.ndarray, prop_matrix: np.ndarray,
                       learning_rate: float, patience: int, min_delta: float, max_epochs: int, seed: int):
    """Train `model` (any module with a forward(x, prop_matrix) signature)
    on `x_train`, monitoring loss on `x_val` after every epoch.

    Stops once the validation loss has not improved by at least
    `min_delta` for `patience` consecutive epochs (or after `max_epochs`,
    whichever comes first), and returns the model from the epoch with the
    best validation loss, not necessarily the last one, so an overfit
    stretch at the end of training does not get used.

    Returns (model, train_loss_history, val_loss_history, best_epoch).
    """
    torch.manual_seed(seed)
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    prop_t = torch.tensor(prop_matrix, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    train_history, val_history = [], []
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_since_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        x_hat = model(x_train_t, prop_t)
        train_loss = loss_fn(x_hat, x_train_t)
        train_loss.backward()
        optimizer.step()
        train_history.append(train_loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val_t, prop_t), x_val_t).item()
        val_history.append(val_loss)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            break

    model.load_state_dict(best_state)
    return model, np.array(train_history), np.array(val_history), best_epoch
