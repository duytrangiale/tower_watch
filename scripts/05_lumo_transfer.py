"""Stage 4: transfer the same detection pipeline to LUMO, a real 9 m
lattice-mast benchmark (Wernitz et al. 2021, CC-BY 3.0), instead of the
synthetic tower. Uses src/data/lumo.py to reshape LUMO's real .mat
measurement blocks into the exact schema the synthetic pipeline already
emits, then reuses the SAME feature extraction, graph autoencoder,
baselines, training loop, and evaluation metrics as scripts/02-04,
unchanged, per the guideline's requirement (Sec 7.2): "If you find
yourself special-casing LUMO in the feature or model code, the
abstraction is wrong; fix the adapter instead."

A fresh graph autoencoder and both baselines are trained from scratch on
LUMO's own healthy data (never on the synthetic tower), matching Sec 7.3's
experiment list: detection ROC-AUC, localisation ("does peak node error
correspond to the level where braces were removed?"), and temperature
robustness (with vs without a per-sensor temperature detrend).

Needs LUMO's exemplary datasets downloaded and unzipped first (see
DAY_4.md for exact download links, ~600 MB per ZIP); point
configs/default.yaml's lumo.data_dirs at the unzipped folders.

Run from the repo root:
    python scripts/05_lumo_transfer.py
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
from sklearn.linear_model import LinearRegression
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
from src.evaluate.metrics import detection_roc_auc, localization_rank, localization_topk_accuracy, roc_curve_points
from src.models.baselines import IsolationForestBaseline, PCABaseline
from src.models.gcn_ae import GCNAutoencoder, per_node_error
from src.models.train_utils import train_autoencoder

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "figures"
MODELS_DIR = REPO_ROOT / "models"


def load_config() -> dict:
    with open(REPO_ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def split_healthy_blocks(meta: pd.DataFrame, test_fraction: float, val_fraction: float, rng) -> tuple:
    """Split healthy 10-minute BLOCKS (not windows) into train/val/test.
    Windows from the same block are consecutive seconds of one continuous
    recording, not independent draws, so splitting by window instead of
    by block would leak near-duplicate information across the split, the
    same reasoning scripts/03_train.py applies to the synthetic
    instances.
    """
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


def train_full_pipeline(x_train, x_val, adjacency, a_hat, model_cfg, n_features, seed):
    """Train a fresh GCN autoencoder + PCA + Isolation Forest on healthy
    LUMO data, exactly mirroring scripts/03_train.py's approach, just
    reused as a function since scripts/05 trains it twice (raw features,
    then again on temperature-detrended features).
    """
    gcn_model = GCNAutoencoder(n_features=n_features, hidden_dim=model_cfg["hidden_dim"],
                                latent_dim=model_cfg["latent_dim"])
    gcn_model, train_history, val_history, best_epoch = train_autoencoder(
        gcn_model, x_train, x_val, a_hat,
        learning_rate=model_cfg["learning_rate"], patience=model_cfg["early_stopping_patience"],
        min_delta=model_cfg["early_stopping_min_delta"], max_epochs=model_cfg["max_epochs"], seed=seed,
    )
    gcn_model.eval()
    with torch.no_grad():
        x_val_t = torch.tensor(x_val, dtype=torch.float32)
        a_hat_t = torch.tensor(a_hat, dtype=torch.float32)
        val_node_err = per_node_error(x_val_t, gcn_model(x_val_t, a_hat_t)).numpy()
    sensor_error_mean, sensor_error_std = val_node_err.mean(axis=0), val_node_err.std(axis=0)

    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    pca = PCABaseline(n_components=min(model_cfg["pca_n_components"], x_train_flat.shape[0] - 1)).fit(x_train_flat)
    isoforest = IsolationForestBaseline(
        n_estimators=model_cfg["isolation_forest_n_estimators"], random_state=seed,
    ).fit(x_train_flat)

    return {
        "gcn_model": gcn_model, "pca": pca, "isoforest": isoforest,
        "sensor_error_mean": sensor_error_mean, "sensor_error_std": sensor_error_std,
        "train_history": train_history, "val_history": val_history, "best_epoch": best_epoch,
    }


def score_eval_set(pipeline, a_hat, x_eval):
    with torch.no_grad():
        x_eval_t = torch.tensor(x_eval, dtype=torch.float32)
        a_hat_t = torch.tensor(a_hat, dtype=torch.float32)
        node_err = per_node_error(x_eval_t, pipeline["gcn_model"](x_eval_t, a_hat_t)).numpy()
    x_eval_flat = x_eval.reshape(x_eval.shape[0], -1)
    return {
        "gcn": node_err.mean(axis=1), "node_err": node_err,
        "pca": pipeline["pca"].score(x_eval_flat), "isoforest": pipeline["isoforest"].score(x_eval_flat),
    }


def temperature_detrend(x: np.ndarray, temperature_c: np.ndarray, train_mask: np.ndarray):
    """Fit one linear regression per (sensor, feature) of value ~ a*T + b
    on the healthy TRAIN rows only, then subtract the fitted trend
    everywhere. x: (n_windows, n_sensors, n_features).
    """
    n_windows, n_sensors, n_features = x.shape
    detrended = np.empty_like(x)
    t_train = temperature_c[train_mask].reshape(-1, 1)
    for s in range(n_sensors):
        for f in range(n_features):
            reg = LinearRegression().fit(t_train, x[train_mask, s, f])
            predicted = reg.predict(temperature_c.reshape(-1, 1))
            detrended[:, s, f] = x[:, s, f] - predicted
    return detrended


def main() -> None:
    config = load_config()
    model_cfg = config["model"]
    lumo_cfg = config["lumo"]
    seed = model_cfg["random_seed"]
    rng = np.random.default_rng(seed)

    data_dirs = [REPO_ROOT / d for d in lumo_cfg["data_dirs"]]
    missing = [d for d in data_dirs if not d.exists()]
    if missing:
        raise FileNotFoundError(
            f"LUMO data not found at {missing}. Download and unzip the exemplary datasets first "
            f"(see DAY_4.md for links), or point lumo.data_dirs at wherever they were extracted."
        )

    blocks = pd.concat([list_blocks(d) for d in data_dirs], ignore_index=True)
    print(f"Found {len(blocks)} 10-minute blocks: {blocks['is_healthy'].sum()} healthy, "
          f"{(~blocks['is_healthy']).sum()} damaged "
          f"(positions {sorted(blocks.loc[~blocks['is_healthy'], 'dam_position'].unique())})")

    print("Loading, windowing, and extracting features block by block "
          "(streamed one 10-minute file at a time, not held all in memory at once)...")
    sampling_rate_hz = 1651.6129032258063  # from the README / every block's Dat.Fs
    table = build_feature_table(blocks, lumo_cfg["window_length"], sampling_rate_hz, config["features"]["band_edges_hz"])
    non_feature_cols = ("window_id", "sensor_idx", "block_id", "is_healthy", "dam_position", "temperature_c")
    feature_cols = [c for c in table.columns if c not in non_feature_cols]
    assert np.isfinite(table[feature_cols].to_numpy()).all(), "LUMO feature matrix has NaN/inf"
    print(f"[OK] Feature table: {table.shape[0]} rows, {len(feature_cols)} features, no NaN/inf")

    train_blocks, val_blocks, test_blocks = split_healthy_blocks(
        table, lumo_cfg["healthy_test_fraction"], lumo_cfg["healthy_val_fraction"], rng,
    )
    print(f"Healthy blocks: {len(train_blocks)} train, {len(val_blocks)} val, {len(test_blocks)} test "
          f"(damaged blocks are never used for training)")

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
    print(f"LUMO sensor graph: {len(ACCEL_CHANNELS)} nodes (9 levels x 2 axes), "
          f"{int(adjacency.sum()) // 2} edges (height-chain, see src/data/lumo.py)")

    x_train, x_val = x_scaled[train_window_mask], x_scaled[val_window_mask]
    x_eval = x_scaled[eval_window_mask]
    is_damaged_eval = ~window_meta.loc[eval_window_mask, "is_healthy"].astype(bool).to_numpy()
    dam_position_eval = window_meta.loc[eval_window_mask, "dam_position"].to_numpy()

    print(f"\nTraining the graph autoencoder + baselines on {x_train.shape[0]} healthy LUMO windows "
          f"(fresh model, never trained on the synthetic tower)...")
    pipeline = train_full_pipeline(x_train, x_val, adjacency, a_hat, model_cfg, len(feature_cols), seed)
    stopped_early = len(pipeline["train_history"]) < model_cfg["max_epochs"]
    print(f"[OK] Trained {len(pipeline['train_history'])} epochs "
          f"({'stopped early' if stopped_early else 'hit max_epochs cap'}), "
          f"best validation loss at epoch {pipeline['best_epoch']}")

    plt.figure(figsize=(6, 4))
    plt.plot(pipeline["train_history"], label="Training loss")
    plt.plot(pipeline["val_history"], label="Validation loss (held-out healthy LUMO)")
    plt.axvline(pipeline["best_epoch"], color="gray", linestyle="--", linewidth=0.8, label="Restored checkpoint")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("LUMO graph autoencoder training")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURES_DIR / "day4_lumo_training_loss.png", dpi=150)
    print(f"Saved {FIGURES_DIR / 'day4_lumo_training_loss.png'}")

    scores = score_eval_set(pipeline, a_hat, x_eval)
    print("\nDetection ROC-AUC (held-out healthy LUMO vs damaged LUMO, pooled across DAM3/DAM4/DAM6):")
    plt.figure(figsize=(5, 5))
    auc_by_model = {}
    for name, key in [("Graph autoencoder", "gcn"), ("PCA reconstruction", "pca"), ("Isolation Forest", "isoforest")]:
        auc = detection_roc_auc(is_damaged_eval, scores[key])
        auc_by_model[name] = auc
        print(f"  {name}: {auc:.3f}")
        fpr, tpr, _ = roc_curve_points(is_damaged_eval, scores[key])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("LUMO detection ROC")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "day4_lumo_roc_curves.png", dpi=150)
    print(f"Saved {FIGURES_DIR / 'day4_lumo_roc_curves.png'}")
    best = max(auc_by_model, key=auc_by_model.get)
    print(f"Best pooled: {best} at {auc_by_model[best]:.3f}")

    print("\nDetection ROC-AUC by DAM position (held-out healthy vs that position only):")
    healthy_eval_mask = ~is_damaged_eval
    for pos in sorted(set(dam_position_eval[is_damaged_eval])):
        pos_mask = dam_position_eval == pos
        mask = healthy_eval_mask | pos_mask
        row = f"  DAM{int(pos)}: "
        for name, key in [("gcn", "gcn"), ("pca", "pca"), ("isoforest", "isoforest")]:
            auc = detection_roc_auc(pos_mask[mask].astype(int), scores[key][mask])
            row += f"{name}={auc:.3f}  "
        print(row)

    print("\nLocalisation: does the highest per-level error land on/near the level where braces "
          "were removed? (9 levels, each level's error is the mean of its x and y channel errors)")
    n_levels = len(ML_HEIGHTS_M) - 1  # ML1-ML9 have accelerometers; ML10 does not
    node_err_eval = scores["node_err"]
    zscored_node_err = (node_err_eval - pipeline["sensor_error_mean"]) / pipeline["sensor_error_std"]
    for pos in sorted(set(dam_position_eval[is_damaged_eval])):
        pos_mask = dam_position_eval == pos
        level_err_raw = node_err_eval[pos_mask].reshape(-1, n_levels, 2).mean(axis=2)
        level_err_z = zscored_node_err[pos_mask].reshape(-1, n_levels, 2).mean(axis=2)
        expected_levels = nearest_levels_for_dam(int(pos))
        # A real DAM position sits BETWEEN two measurement levels, not at
        # either one exactly, so a hit on either bracketing level counts:
        # best top-3 (higher is better) and best mean rank (lower is
        # better) are taken independently, and may come from different
        # candidate levels.
        best_raw_topk = max(localization_topk_accuracy(level_err_raw, e, k=3) for e in expected_levels)
        best_raw_rank = min(np.mean([localization_rank(r, e) for r in level_err_raw]) for e in expected_levels)
        best_z_topk = max(localization_topk_accuracy(level_err_z, e, k=3) for e in expected_levels)
        best_z_rank = min(np.mean([localization_rank(r, e) for r in level_err_z]) for e in expected_levels)
        ml_pair = DAM_POSITION_MLS[int(pos)]
        print(f"  DAM{int(pos)} (between ML{ml_pair[0]} and ML{ml_pair[1]}, best-of-either-bracketing-level): "
              f"raw top-3 {best_raw_topk:.1%}/mean rank {best_raw_rank:.1f}, "
              f"z-score top-3 {best_z_topk:.1%}/mean rank {best_z_rank:.1f} (chance top-3 {3/n_levels:.1%}, "
              f"chance rank {(n_levels - 1) / 2:.1f})")

    fig, ax = plt.subplots(figsize=(6, 5))
    heights = [ML_HEIGHTS_M[i + 1] for i in range(n_levels)]
    mean_level_err = zscored_node_err[is_damaged_eval].reshape(-1, n_levels, 2).mean(axis=(0, 2))
    ax.plot(mean_level_err, heights, marker="o")
    for pos in sorted(set(dam_position_eval[is_damaged_eval])):
        ml_pair = DAM_POSITION_MLS[int(pos)]
        y0, y1 = ML_HEIGHTS_M[ml_pair[0]], ML_HEIGHTS_M.get(ml_pair[1], 0.15)
        ax.axhspan(min(y0, y1), max(y0, y1), color="red", alpha=0.08)
        ax.annotate(f"DAM{int(pos)}", xy=(mean_level_err.max() * 1.02, (y0 + y1) / 2), fontsize=8, color="tab:red")
    ax.set_xlabel("Mean per-sensor z-score error (damaged windows, all positions pooled)")
    ax.set_ylabel("Height (m)")
    ax.set_title("LUMO localisation: error by level vs damage positions (shaded)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "day4_lumo_localization.png", dpi=150)
    print(f"Saved {FIGURES_DIR / 'day4_lumo_localization.png'}")

    print("\nTemperature robustness: healthy vs damaged block temperatures (are they even comparable?):")
    healthy_temps = window_meta.loc[window_meta["is_healthy"].astype(bool), "temperature_c"]
    damaged_temps = window_meta.loc[~window_meta["is_healthy"].astype(bool), "temperature_c"]
    print(f"  Healthy: {healthy_temps.min():.1f} to {healthy_temps.max():.1f} C (mean {healthy_temps.mean():.1f})")
    print(f"  Damaged: {damaged_temps.min():.1f} to {damaged_temps.max():.1f} C (mean {damaged_temps.mean():.1f})")

    print("\nRetraining on temperature-detrended features (per-sensor, per-feature linear fit vs "
          "temperature on healthy training data, subtracted everywhere) for comparison...")
    temperature_all = window_meta["temperature_c"].to_numpy()
    x_detrended = temperature_detrend(x_scaled, temperature_all, train_window_mask)
    pipeline_dt = train_full_pipeline(
        x_detrended[train_window_mask], x_detrended[val_window_mask], adjacency, a_hat,
        model_cfg, len(feature_cols), seed,
    )
    scores_dt = score_eval_set(pipeline_dt, a_hat, x_detrended[eval_window_mask])
    print("Detection ROC-AUC, with vs without temperature detrending:")
    for name, key in [("Graph autoencoder", "gcn"), ("PCA reconstruction", "pca"), ("Isolation Forest", "isoforest")]:
        auc_raw = detection_roc_auc(is_damaged_eval, scores[key])
        auc_dt = detection_roc_auc(is_damaged_eval, scores_dt[key])
        print(f"  {name}: {auc_raw:.3f} (raw) -> {auc_dt:.3f} (temperature-detrended)")

    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(pipeline["gcn_model"].state_dict(), MODELS_DIR / "lumo_gcn_ae.pt")
    with open(MODELS_DIR / "lumo_baselines.pkl", "wb") as f:
        pickle.dump({"pca": pipeline["pca"], "isoforest": pipeline["isoforest"]}, f)
    with open(MODELS_DIR / "lumo_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"\nSaved model artefacts to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
