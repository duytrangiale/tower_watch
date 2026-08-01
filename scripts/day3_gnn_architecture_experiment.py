"""Day 3 follow-up experiment (not part of the main pipeline): does
replacing the plain GCN's fixed-weight neighbour averaging with a richer
message-passing layer, mean-aggregated or attention-weighted (a graph
attention network, GAT), improve detection or localisation? Also compares
the opposite extreme: a per-sensor autoencoder with no neighbour mixing
at all (src/models/per_sensor_ae.py), to test whether the graph's
cross-sensor mixing is helping localisation or diluting it, an idea from
the SHM literature (see DAY_3.md, "Richer architectures", for why this
was worth testing, and the result).

Reuses the exact healthy train/validation/test split and scaler saved by
scripts/03_train.py, so the comparison against the plain GCN is
apples-to-apples. Every architecture is trained the same way: early
stopping on the held-out healthy validation set (never a fixed epoch
count), from several random seeds, since any single run is somewhat
noisy on its own. Localisation is reported both raw and per-sensor
z-scored (Sec 6.3 fix, see DAY_3.md).

Run from the repo root (after scripts/01-04):
    python scripts/day3_gnn_architecture_experiment.py
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import yaml

from src.evaluate.metrics import detection_roc_auc, localization_rank, localization_topk_accuracy
from src.fem.geometry import generate_lattice_geometry
from src.graph.build import nearest_sensor_by_hops
from src.models.gcn_ae import GCNAutoencoder, per_node_error
from src.models.message_passing_ae import MessagePassingAutoencoder
from src.models.per_sensor_ae import PerSensorAutoencoder
from src.models.train_utils import train_autoencoder

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "synthetic"
MODELS_DIR = REPO_ROOT / "models"

NON_FEATURE_COLUMNS = ("window_id", "sensor_idx", "sensor_node_id", "instance_id",
                        "damage_severity", "temperature_c")

N_SEEDS = 5


def load_config() -> dict:
    with open(REPO_ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def build_feature_tensor(table: pd.DataFrame, feature_cols: list):
    table_sorted = table.sort_values(["window_id", "sensor_idx"])
    window_ids = table_sorted["window_id"].unique()
    n_windows = len(window_ids)
    n_sensors = table_sorted["sensor_idx"].nunique()
    x = table_sorted[feature_cols].to_numpy().reshape(n_windows, n_sensors, len(feature_cols))
    return x, window_ids


def evaluate_one(model, prop, x_val, x_eval, is_damaged, expected_idx, k):
    model.eval()
    with torch.no_grad():
        val_t = torch.tensor(x_val, dtype=torch.float32)
        eval_t = torch.tensor(x_eval, dtype=torch.float32)
        prop_t = torch.tensor(prop, dtype=torch.float32)
        val_node_err = per_node_error(val_t, model(val_t, prop_t)).numpy()
        node_err = per_node_error(eval_t, model(eval_t, prop_t)).numpy()

    scores = node_err.mean(axis=1)
    auc = detection_roc_auc(is_damaged, scores)

    damaged_err = node_err[is_damaged]
    sensor_mean, sensor_std = val_node_err.mean(axis=0), val_node_err.std(axis=0)
    zscored_err = (damaged_err - sensor_mean) / sensor_std

    topk_raw = localization_topk_accuracy(damaged_err, expected_idx, k)
    rank_raw = np.mean([localization_rank(row, expected_idx) for row in damaged_err])
    topk_z = localization_topk_accuracy(zscored_err, expected_idx, k)
    rank_z = np.mean([localization_rank(row, expected_idx) for row in zscored_err])
    return auc, topk_raw, rank_raw, topk_z, rank_z


def main() -> None:
    config = load_config()
    model_cfg = config["model"]

    table = pd.read_csv(DATA_DIR / "features_raw.csv")
    feature_cols = [c for c in table.columns if c not in NON_FEATURE_COLUMNS]
    split = pd.read_csv(MODELS_DIR / "split.csv")

    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    graph_npz = np.load(MODELS_DIR / "graph.npz")
    adjacency = graph_npz["adjacency"]  # no self-loops, used by the new message-passing layer
    a_hat = graph_npz["a_hat"]          # normalised with self-loops, used by the plain GCN
    sensor_node_ids = graph_npz["sensor_node_ids"]

    scaled_table = table.copy()
    scaled_table[feature_cols] = scaler.transform(table[feature_cols])
    x_all, window_ids = build_feature_tensor(scaled_table, feature_cols)
    window_meta = split.set_index("window_id").loc[window_ids]

    train_mask = (window_meta["split"] == "healthy_train").to_numpy()
    val_mask = (window_meta["split"] == "healthy_val").to_numpy()
    eval_mask = (window_meta["split"] == "eval").to_numpy()
    x_train = x_all[train_mask]
    x_val = x_all[val_mask]
    x_eval = x_all[eval_mask]
    is_damaged = (window_meta.loc[eval_mask, "damage_severity"] > 0).to_numpy()

    npz = np.load(DATA_DIR / "windows.npz")
    geometry = generate_lattice_geometry(config)
    expected_idx = nearest_sensor_by_hops(geometry, sensor_node_ids, npz["damaged_element_nodes"])
    k = model_cfg["localization_top_k"]
    n_sensors = len(sensor_node_ids)
    n_features = len(feature_cols)

    architectures = {
        "GCN (Day 3 baseline)": (
            lambda: GCNAutoencoder(n_features, model_cfg["hidden_dim"], model_cfg["latent_dim"]), a_hat,
        ),
        "Message passing, mean": (
            lambda: MessagePassingAutoencoder(n_features, model_cfg["hidden_dim"], model_cfg["latent_dim"],
                                               use_attention=False), adjacency,
        ),
        "Message passing, attention (GAT)": (
            lambda: MessagePassingAutoencoder(n_features, model_cfg["hidden_dim"], model_cfg["latent_dim"],
                                               use_attention=True), adjacency,
        ),
        "Per-sensor (no graph)": (
            lambda: PerSensorAutoencoder(n_sensors, n_features, model_cfg["hidden_dim"], model_cfg["latent_dim"]),
            a_hat,  # ignored by this architecture, passed only because train_autoencoder expects some prop_matrix
        ),
    }

    print(f"Comparing architectures over {N_SEEDS} random seeds each, each trained with early "
          f"stopping on the held-out healthy validation set "
          f"(same train/val/eval split and scaler as scripts/03_train.py):\n")
    print(f"{'Architecture':<34} {'epochs':>7} {'AUC (mean +/- std)':>21} "
          f"{'top-' + str(k) + ' raw':>10} {'top-' + str(k) + ' z-score':>13} "
          f"{'rank raw':>9} {'rank z':>7}")
    for name, (model_fn, prop) in architectures.items():
        aucs, topk_raws, rank_raws, topk_zs, rank_zs, epochs_used = [], [], [], [], [], []
        for seed in range(N_SEEDS):
            model, train_hist, val_hist, best_epoch = train_autoencoder(
                model_fn(), x_train, x_val, prop,
                learning_rate=model_cfg["learning_rate"], patience=model_cfg["early_stopping_patience"],
                min_delta=model_cfg["early_stopping_min_delta"], max_epochs=model_cfg["max_epochs"], seed=seed,
            )
            auc, topk_raw, rank_raw, topk_z, rank_z = evaluate_one(
                model, prop, x_val, x_eval, is_damaged, expected_idx, k,
            )
            aucs.append(auc)
            topk_raws.append(topk_raw)
            rank_raws.append(rank_raw)
            topk_zs.append(topk_z)
            rank_zs.append(rank_z)
            epochs_used.append(len(train_hist))
        print(f"{name:<34} {np.mean(epochs_used):>7.0f} "
              f"{np.mean(aucs):>9.3f} +/- {np.std(aucs):<7.3f} "
              f"{np.mean(topk_raws):>9.1%} {np.mean(topk_zs):>13.1%} "
              f"{np.mean(rank_raws):>9.1f} {np.mean(rank_zs):>7.1f}")

    chance_rank = (n_sensors - 1) / 2
    print(f"\nChance reference: AUC=0.500, top-{k} localisation={k / n_sensors:.1%}, mean rank={chance_rank:.1f}")


if __name__ == "__main__":
    main()
