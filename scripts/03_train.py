"""Stage 3: train the graph autoencoder and the classical baselines (PCA,
Isolation Forest) on healthy windows only, and verify the Sec 6.3
acceptance criterion that training loss decreases and converges.

Baselines are trained first deliberately (guideline Sec 6.1): a GNN that
can't beat PCA is a finding worth knowing now, not on Day 5.

The graph autoencoder is trained with early stopping on a held-out
healthy validation set (never a fixed epoch count), and both the
training and validation loss are plotted together so overfitting would
be visible directly, rather than only checking the training loss.

Run from the repo root (after 01_generate_synthetic.py and
02_extract_features.py):
    python scripts/03_train.py
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler

from src.fem.geometry import generate_lattice_geometry
from src.graph.build import build_sensor_graph
from src.models.baselines import IsolationForestBaseline, PCABaseline
from src.models.gcn_ae import GCNAutoencoder, per_node_error
from src.models.train_utils import train_autoencoder

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "synthetic"
MODELS_DIR = REPO_ROOT / "models"
FIGURES_DIR = REPO_ROOT / "figures"

NON_FEATURE_COLUMNS = ("window_id", "sensor_idx", "sensor_node_id", "instance_id",
                        "damage_severity", "temperature_c")


def load_config() -> dict:
    with open(REPO_ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def split_healthy_instances(table: pd.DataFrame, test_fraction: float, val_fraction: float,
                             rng: np.random.Generator):
    """Split healthy windows' instances into train / validation / test,
    keeping every window from a given instance in the same split: windows
    from the same instance share a temperature draw, so splitting by
    individual window instead of by instance would leak information
    across the split.

    Validation is used only for early stopping and the overfitting check
    below; it is never used for the detection/localisation numbers
    reported in DAY_3.md, those come from the test split alone.
    """
    healthy_instances = np.sort(table.loc[table["damage_severity"] == 0.0, "instance_id"].unique())
    n_total = len(healthy_instances)
    n_test = max(1, round(n_total * test_fraction))
    n_val = max(1, round(n_total * val_fraction))
    shuffled = rng.permutation(healthy_instances)
    test_instances = set(shuffled[:n_test].tolist())
    val_instances = set(shuffled[n_test:n_test + n_val].tolist())
    train_instances = set(shuffled[n_test + n_val:].tolist())
    return train_instances, val_instances, test_instances


def build_feature_tensor(table: pd.DataFrame, feature_cols: list):
    """Pivot the long (window, sensor) table into a dense
    (n_windows, n_sensors, n_features) array, sensors ordered by
    `sensor_idx` to match the sensor graph's node order.
    """
    table_sorted = table.sort_values(["window_id", "sensor_idx"])
    window_ids = table_sorted["window_id"].unique()
    n_windows = len(window_ids)
    n_sensors = table_sorted["sensor_idx"].nunique()
    x = table_sorted[feature_cols].to_numpy().reshape(n_windows, n_sensors, len(feature_cols))
    return x, window_ids


def check_training_convergence(train_history: np.ndarray, val_history: np.ndarray, best_epoch: int,
                                max_epochs: int) -> None:
    """Verify Sec 6.3: training loss decreases and converges, and check
    for overfitting (training loss still falling while validation loss
    rises or stalls).
    """
    assert train_history[-1] < train_history[0], "Training loss did not decrease"
    stopped_early = len(train_history) < max_epochs
    print(f"[OK] Training loss decreased from {train_history[0]:.4f} to {train_history[-1]:.4f} "
          f"over {len(train_history)} epochs "
          f"({'stopped early once validation loss plateaued' if stopped_early else 'hit the max-epoch safety cap'})")

    best_val = val_history[best_epoch]
    final_val = val_history[-1]
    overfit_gap = (final_val - best_val) / best_val
    status = "[OK]" if overfit_gap < 0.05 else "[FINDING]"
    print(f"{status} Best validation loss {best_val:.4f} at epoch {best_epoch} (model restored to this "
          f"checkpoint, not the last epoch); by epoch {len(train_history) - 1} validation loss had drifted "
          f"{overfit_gap:+.1%} from that best point "
          f"({'no meaningful overfitting' if overfit_gap < 0.05 else 'some drift, which is exactly why the checkpoint was restored instead of using the final weights'})")

    plt.figure(figsize=(6, 4))
    plt.plot(train_history, label="Training loss")
    plt.plot(val_history, label="Validation loss (held-out healthy)")
    plt.axvline(best_epoch, color="gray", linestyle="--", linewidth=0.8, label="Restored checkpoint")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Graph autoencoder training: loss vs validation")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    out_path = FIGURES_DIR / "day3_training_loss.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main() -> None:
    config = load_config()
    model_cfg = config["model"]
    rng = np.random.default_rng(model_cfg["random_seed"])

    table = pd.read_csv(DATA_DIR / "features_raw.csv")
    feature_cols = [c for c in table.columns if c not in NON_FEATURE_COLUMNS]

    train_instances, val_instances, test_instances = split_healthy_instances(
        table, model_cfg["healthy_test_fraction"], model_cfg["healthy_val_fraction"], rng,
    )
    print(f"Healthy instances: {len(train_instances)} train, {len(val_instances)} validation, "
          f"{len(test_instances)} test (damaged instances are never used for training)")

    # Re-fit the scaler on the healthy TRAIN split only. Day 2 fit it on all
    # healthy data as a placeholder before a train/test split existed; now
    # that one does, this matches the guideline's original intent exactly
    # (Sec 5.3: "fit scaler on healthy train split, apply to everything").
    train_mask = table["instance_id"].isin(train_instances)
    scaler = StandardScaler().fit(table.loc[train_mask, feature_cols])
    scaled_table = table.copy()
    scaled_table[feature_cols] = scaler.transform(table[feature_cols])

    x_all, window_ids = build_feature_tensor(scaled_table, feature_cols)
    window_meta = (
        scaled_table[["window_id", "instance_id", "damage_severity"]]
        .drop_duplicates()
        .set_index("window_id")
        .loc[window_ids]
    )

    train_window_mask = window_meta["instance_id"].isin(train_instances).to_numpy()
    val_window_mask = window_meta["instance_id"].isin(val_instances).to_numpy()
    x_train = x_all[train_window_mask]
    x_val = x_all[val_window_mask]
    print(f"Training tensor: {x_train.shape}, validation tensor: {x_val.shape} (windows, sensors, features)")

    npz = np.load(DATA_DIR / "windows.npz")
    sensor_node_ids = npz["sensor_node_ids"]
    geometry = generate_lattice_geometry(config)
    adjacency, a_hat = build_sensor_graph(geometry, sensor_node_ids, config["graph"]["n_distance_levels"])
    print(f"Sensor graph: {len(sensor_node_ids)} nodes, {int(adjacency.sum()) // 2} edges")

    gcn_model = GCNAutoencoder(n_features=len(feature_cols), hidden_dim=model_cfg["hidden_dim"],
                                latent_dim=model_cfg["latent_dim"])
    gcn_model, train_history, val_history, best_epoch = train_autoencoder(
        gcn_model, x_train, x_val, a_hat,
        learning_rate=model_cfg["learning_rate"], patience=model_cfg["early_stopping_patience"],
        min_delta=model_cfg["early_stopping_min_delta"], max_epochs=model_cfg["max_epochs"],
        seed=model_cfg["random_seed"],
    )
    check_training_convergence(train_history, val_history, best_epoch, model_cfg["max_epochs"])

    # Per-sensor calibration for the localisation fix (see DAY_3.md): each
    # sensor's own mean/std reconstruction error on held-out healthy data,
    # used later to judge "is this sensor's error unusual *for this
    # sensor*" instead of comparing raw error across sensors directly.
    gcn_model.eval()
    with torch.no_grad():
        x_val_t = torch.tensor(x_val, dtype=torch.float32)
        a_hat_t = torch.tensor(a_hat, dtype=torch.float32)
        val_node_err = per_node_error(x_val_t, gcn_model(x_val_t, a_hat_t)).numpy()
    sensor_error_mean = val_node_err.mean(axis=0)
    sensor_error_std = val_node_err.std(axis=0)
    print(f"[OK] Per-sensor error calibration computed from {val_node_err.shape[0]} held-out "
          f"healthy validation windows")

    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    pca_baseline = PCABaseline(n_components=model_cfg["pca_n_components"]).fit(x_train_flat)
    if_baseline = IsolationForestBaseline(
        n_estimators=model_cfg["isolation_forest_n_estimators"], random_state=model_cfg["random_seed"],
    ).fit(x_train_flat)
    print(f"[OK] Baselines fit on {x_train_flat.shape[0]} healthy training windows "
          f"({x_train_flat.shape[1]}-dim flattened features)")

    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(gcn_model.state_dict(), MODELS_DIR / "gcn_ae.pt")
    with open(MODELS_DIR / "baselines.pkl", "wb") as f:
        pickle.dump({"pca": pca_baseline, "isolation_forest": if_baseline}, f)
    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    np.savez(MODELS_DIR / "graph.npz", adjacency=adjacency, a_hat=a_hat, sensor_node_ids=sensor_node_ids)
    np.savez(MODELS_DIR / "sensor_error_calibration.npz",
             mean=sensor_error_mean, std=sensor_error_std)

    split_labels = np.where(train_window_mask, "healthy_train",
                             np.where(val_window_mask, "healthy_val", "eval"))
    window_meta.assign(split=split_labels).to_csv(MODELS_DIR / "split.csv")
    print(f"Saved model artefacts to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
