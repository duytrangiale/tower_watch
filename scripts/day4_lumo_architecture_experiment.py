"""Day 4 follow-up (not part of the main pipeline): does the per-sensor
autoencoder (src/models/per_sensor_ae.py) show the same pattern on LUMO's
real data that it showed on the synthetic tower (DAY_3.md, "Trying
different architectures"), a less biased raw error and a better
z-scored mean rank than the plain GCN, even though its top-3 accuracy
wasn't clearly better?

Also tests localisation idea #3 from the Eltouny/Gomaa/Liang 2023 SHM
review (Sony & Sadhu 2022; Soman 2020): instead of z-scoring each
sensor's error against its own healthy history, compare it to the OTHER
sensors in the same window, right now. Computed straight from the
already-trained models' error arrays, no extra training needed, see
"relative_node_err" in evaluate() below.

Reuses the exact same LUMO data loading and healthy train/val/test split
as scripts/05_lumo_transfer.py. Trains both the plain GCN and the
per-sensor autoencoder from scratch on LUMO's healthy windows, single
seed each (matching scripts/05_lumo_transfer.py's own single-run
approach, see DAY_4.md's "Limits of this test"). Skips the baselines and
temperature-detrending comparison from that script, this is specifically
about the architecture and scoring questions.

Run from the repo root (needs LUMO's exemplary datasets downloaded and
unzipped, see DAY_4.md):
    python scripts/day4_lumo_architecture_experiment.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler

from src.data.lumo import (
    ACCEL_CHANNELS,
    DAM_POSITION_MLS,
    ML_HEIGHTS_M,
    build_feature_table,
    list_blocks,
    lumo_sensor_graph,
    nearest_levels_for_dam,
)
from src.evaluate.metrics import detection_roc_auc, localization_rank, localization_topk_accuracy
from src.models.gcn_ae import GCNAutoencoder, per_node_error
from src.models.per_sensor_ae import PerSensorAutoencoder
from src.models.train_utils import train_autoencoder

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "figures"

N_LEVELS = len(ML_HEIGHTS_M) - 1  # ML1-ML9 have accelerometers; ML10 does not


def load_config() -> dict:
    with open(REPO_ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def split_healthy_blocks(meta: pd.DataFrame, test_fraction: float, val_fraction: float, rng) -> tuple:
    healthy_blocks = np.sort(meta.loc[meta["is_healthy"], "block_id"].unique())
    n_total = len(healthy_blocks)
    n_test = max(1, round(n_total * test_fraction))
    n_val = max(1, round(n_total * val_fraction))
    shuffled = rng.permutation(healthy_blocks)
    test_blocks = set(shuffled[:n_test].tolist())
    val_blocks = set(shuffled[n_test:n_test + n_val].tolist())
    train_blocks = set(shuffled[n_test + n_val:].tolist())
    return train_blocks, val_blocks, test_blocks


def build_feature_tensor(table: pd.DataFrame, feature_cols: list, n_sensors: int):
    table_sorted = table.sort_values(["window_id", "sensor_idx"])
    window_ids = table_sorted["window_id"].unique()
    x = table_sorted[feature_cols].to_numpy().reshape(len(window_ids), n_sensors, len(feature_cols))
    return x, window_ids


def evaluate(model, prop, x_val, x_eval, is_damaged_eval, dam_position_eval, k=3):
    model.eval()
    with torch.no_grad():
        x_val_t = torch.tensor(x_val, dtype=torch.float32)
        x_eval_t = torch.tensor(x_eval, dtype=torch.float32)
        prop_t = torch.tensor(prop, dtype=torch.float32)
        val_node_err = per_node_error(x_val_t, model(x_val_t, prop_t)).numpy()
        node_err = per_node_error(x_eval_t, model(x_eval_t, prop_t)).numpy()

    scores = node_err.mean(axis=1)
    pooled_auc = detection_roc_auc(is_damaged_eval, scores)

    sensor_error_mean, sensor_error_std = val_node_err.mean(axis=0), val_node_err.std(axis=0)
    zscored_node_err = (node_err - sensor_error_mean) / sensor_error_std
    # Relative index (idea #3): compare each sensor to the other 17 in the
    # SAME window, right now, instead of to its own healthy history. No
    # held-out baseline needed, so unlike the z-score this would also work
    # with no validation split at all.
    relative_node_err = (node_err - node_err.mean(axis=1, keepdims=True)) / node_err.std(axis=1, keepdims=True)

    by_position = {}
    for pos in sorted(set(dam_position_eval[is_damaged_eval])):
        pos_mask = dam_position_eval == pos
        level_err_raw = node_err[pos_mask].reshape(-1, N_LEVELS, 2).mean(axis=2)
        level_err_z = zscored_node_err[pos_mask].reshape(-1, N_LEVELS, 2).mean(axis=2)
        level_err_rel = relative_node_err[pos_mask].reshape(-1, N_LEVELS, 2).mean(axis=2)
        expected_levels = nearest_levels_for_dam(int(pos))

        best_raw_topk = max(localization_topk_accuracy(level_err_raw, e, k=k) for e in expected_levels)
        best_raw_rank = min(np.mean([localization_rank(r, e) for r in level_err_raw]) for e in expected_levels)
        best_z_topk = max(localization_topk_accuracy(level_err_z, e, k=k) for e in expected_levels)
        best_z_rank = min(np.mean([localization_rank(r, e) for r in level_err_z]) for e in expected_levels)
        best_rel_topk = max(localization_topk_accuracy(level_err_rel, e, k=k) for e in expected_levels)
        best_rel_rank = min(np.mean([localization_rank(r, e) for r in level_err_rel]) for e in expected_levels)

        healthy_eval_mask = ~is_damaged_eval
        mask = healthy_eval_mask | pos_mask
        pos_auc = detection_roc_auc(pos_mask[mask].astype(int), scores[mask])

        by_position[int(pos)] = {
            "auc": pos_auc, "raw_topk": best_raw_topk, "raw_rank": best_raw_rank,
            "z_topk": best_z_topk, "z_rank": best_z_rank,
            "rel_topk": best_rel_topk, "rel_rank": best_rel_rank,
        }

    return pooled_auc, by_position


def main() -> None:
    config = load_config()
    model_cfg = config["model"]
    lumo_cfg = config["lumo"]
    seed = model_cfg["random_seed"]
    rng = np.random.default_rng(seed)

    data_dirs = [REPO_ROOT / d for d in lumo_cfg["data_dirs"]]
    missing = [d for d in data_dirs if not d.exists()]
    if missing:
        raise FileNotFoundError(f"LUMO data not found at {missing}. See DAY_4.md for download instructions.")

    blocks = pd.concat([list_blocks(d) for d in data_dirs], ignore_index=True)
    sampling_rate_hz = 1651.6129032258063
    table = build_feature_table(blocks, lumo_cfg["window_length"], sampling_rate_hz, config["features"]["band_edges_hz"])
    non_feature_cols = ("window_id", "sensor_idx", "block_id", "is_healthy", "dam_position", "temperature_c")
    feature_cols = [c for c in table.columns if c not in non_feature_cols]
    print(f"Feature table: {table.shape[0]} rows, {len(feature_cols)} features")

    train_blocks, val_blocks, test_blocks = split_healthy_blocks(
        table, lumo_cfg["healthy_test_fraction"], lumo_cfg["healthy_val_fraction"], rng,
    )
    train_mask = table["block_id"].isin(train_blocks).to_numpy()
    scaler = StandardScaler().fit(table.loc[train_mask, feature_cols])
    scaled_table = table.copy()
    scaled_table[feature_cols] = scaler.transform(table[feature_cols])
    x_scaled, window_ids = build_feature_tensor(scaled_table, feature_cols, len(ACCEL_CHANNELS))

    window_meta = table[["window_id", "block_id", "is_healthy", "dam_position", "temperature_c"]] \
        .drop_duplicates().set_index("window_id").loc[window_ids]
    train_window_mask = window_meta["block_id"].isin(train_blocks).to_numpy()
    val_window_mask = window_meta["block_id"].isin(val_blocks).to_numpy()
    eval_window_mask = (
        window_meta["block_id"].isin(test_blocks) | ~window_meta["is_healthy"].astype(bool)
    ).to_numpy()

    adjacency, a_hat = lumo_sensor_graph()
    n_sensors = len(ACCEL_CHANNELS)
    n_features = len(feature_cols)

    x_train, x_val = x_scaled[train_window_mask], x_scaled[val_window_mask]
    x_eval = x_scaled[eval_window_mask]
    is_damaged_eval = ~window_meta.loc[eval_window_mask, "is_healthy"].astype(bool).to_numpy()
    dam_position_eval = window_meta.loc[eval_window_mask, "dam_position"].to_numpy()
    print(f"Training on {x_train.shape[0]} healthy windows, evaluating on {x_eval.shape[0]} "
          f"({is_damaged_eval.sum()} damaged)\n")

    architectures = {
        "GCN": (lambda: GCNAutoencoder(n_features, model_cfg["hidden_dim"], model_cfg["latent_dim"]), a_hat),
        "Per-sensor (no graph)": (
            lambda: PerSensorAutoencoder(n_sensors, n_features, model_cfg["hidden_dim"], model_cfg["latent_dim"]),
            a_hat,  # ignored by this architecture
        ),
    }

    results = {}
    for name, (model_fn, prop) in architectures.items():
        print(f"Training {name}...")
        model, train_history, val_history, best_epoch = train_autoencoder(
            model_fn(), x_train, x_val, prop,
            learning_rate=model_cfg["learning_rate"], patience=model_cfg["early_stopping_patience"],
            min_delta=model_cfg["early_stopping_min_delta"], max_epochs=model_cfg["max_epochs"], seed=seed,
        )
        stopped_early = len(train_history) < model_cfg["max_epochs"]
        print(f"  {len(train_history)} epochs ({'stopped early' if stopped_early else 'hit max_epochs cap'})")
        pooled_auc, by_position = evaluate(model, prop, x_val, x_eval, is_damaged_eval, dam_position_eval,
                                            k=model_cfg["localization_top_k"])
        results[name] = {"pooled_auc": pooled_auc, "by_position": by_position}
        print(f"  Pooled detection AUC: {pooled_auc:.3f}")
        for pos, m in by_position.items():
            ml_pair = DAM_POSITION_MLS[pos]
            print(f"  DAM{pos} (between ML{ml_pair[0]} and ML{ml_pair[1]}): AUC {m['auc']:.3f}, "
                  f"raw top-3 {m['raw_topk']:.1%}/rank {m['raw_rank']:.1f}, "
                  f"z-score top-3 {m['z_topk']:.1%}/rank {m['z_rank']:.1f}, "
                  f"relative top-3 {m['rel_topk']:.1%}/rank {m['rel_rank']:.1f}")
        print()

    chance_topk = model_cfg["localization_top_k"] / N_LEVELS
    chance_rank = (N_LEVELS - 1) / 2
    print(f"Chance: AUC 0.500, top-{model_cfg['localization_top_k']} {chance_topk:.1%}, mean rank {chance_rank:.1f}\n")

    print("Summary, GCN vs per-sensor, averaged across the 3 DAM positions:")
    for name, r in results.items():
        positions = r["by_position"].values()
        print(f"  {name}: AUC {r['pooled_auc']:.3f}, "
              f"raw rank {np.mean([m['raw_rank'] for m in positions]):.1f}, "
              f"z-score rank {np.mean([m['z_rank'] for m in positions]):.1f}, "
              f"z-score top-3 {np.mean([m['z_topk'] for m in positions]):.1%}, "
              f"relative rank {np.mean([m['rel_rank'] for m in positions]):.1f}, "
              f"relative top-3 {np.mean([m['rel_topk'] for m in positions]):.1%}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey="row")
    positions_list = sorted(next(iter(results.values()))["by_position"].keys())
    x_pos = np.arange(len(positions_list))
    width = 0.35
    panel_specs = [
        (axes[0, 0], "z_rank", "Mean rank, z-score\n(own history baseline, lower is better)", chance_rank),
        (axes[0, 1], "rel_rank", "Mean rank, relative index\n(other sensors, same moment, lower is better)", chance_rank),
        (axes[1, 0], "z_topk", "Top-3, z-score", chance_topk),
        (axes[1, 1], "rel_topk", "Top-3, relative index", chance_topk),
    ]
    for ax, metric, ylabel, chance_line in panel_specs:
        for i, (name, r) in enumerate(results.items()):
            values = [r["by_position"][p][metric] for p in positions_list]
            ax.bar(x_pos + (i - 0.5) * width, values, width, label=name)
        ax.axhline(chance_line, color="black", linestyle="--", linewidth=1, label="Chance")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"DAM{p}" for p in positions_list])
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("LUMO: GCN vs per-sensor autoencoder, z-score vs relative-index scoring, by damage position")
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    out_path = FIGURES_DIR / "day4_lumo_architecture_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
