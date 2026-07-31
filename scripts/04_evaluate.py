"""Stage 3 (part 2): evaluate the graph autoencoder against the PCA and
Isolation Forest baselines on identical held-out splits, and verify the
Sec 6.3 acceptance criteria: healthy validation error is clearly below
damaged error, ROC-AUC computed against both baselines, and localisation
demonstrated (highest per-sensor error sits near the actual damage).

Run from the repo root (after 03_train.py):
    python scripts/04_evaluate.py
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

from src.evaluate.metrics import (
    detection_roc_auc,
    localization_rank,
    localization_topk_accuracy,
    roc_curve_points,
)
from src.fem.geometry import generate_lattice_geometry
from src.fem.visualize import plot_localization_heatmap
from src.graph.build import nearest_sensor_by_hops
from src.models.gcn_ae import GCNAutoencoder, per_node_error

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "synthetic"
MODELS_DIR = REPO_ROOT / "models"
FIGURES_DIR = REPO_ROOT / "figures"

NON_FEATURE_COLUMNS = ("window_id", "sensor_idx", "sensor_node_id", "instance_id",
                        "damage_severity", "temperature_c")


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


def compute_roc_and_plot(is_damaged, severities, scores_by_model):
    print("\nDetection ROC-AUC (held-out healthy vs damaged, identical split):")
    auc_by_model = {}
    plt.figure(figsize=(5, 5))
    for name, scores in scores_by_model.items():
        auc = detection_roc_auc(is_damaged, scores)
        auc_by_model[name] = auc
        print(f"  {name}: {auc:.3f}")
        fpr, tpr, _ = roc_curve_points(is_damaged, scores)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Detection ROC: graph autoencoder vs baselines")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    out_path = FIGURES_DIR / "day3_roc_curves.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    # Broken down by severity: does detection get easier as damage gets worse?
    # This uses only the healthy windows and one severity at a time, so the
    # ROC-AUC values below are not directly comparable in sample size to the
    # pooled figure above, but the trend (or lack of one) is informative.
    print("\nDetection ROC-AUC by severity (held-out healthy vs that severity only):")
    healthy_mask = ~is_damaged
    for severity in sorted(set(severities[is_damaged])):
        sev_mask = severities == severity
        mask = healthy_mask | sev_mask
        row = "  d={:.1f}: ".format(severity)
        for name, scores in scores_by_model.items():
            auc = detection_roc_auc(sev_mask[mask].astype(int), scores[mask])
            row += f"{name}={auc:.3f}  "
        print(row)

    best = max(auc_by_model, key=auc_by_model.get)
    print(f"\n[OK] ROC-AUC computed for graph autoencoder and both baselines on identical splits "
          f"(best pooled: {best} at {auc_by_model[best]:.3f})")
    return auc_by_model


def check_error_separation(is_damaged, gcn_scores):
    healthy_scores = gcn_scores[~is_damaged]
    damaged_scores = gcn_scores[is_damaged]
    separates = np.median(damaged_scores) > np.median(healthy_scores)
    status = "[OK]" if separates else "[FINDING]"
    print(f"\n{status} GCN error: healthy median {np.median(healthy_scores):.4f}, "
          f"damaged median {np.median(damaged_scores):.4f} "
          f"({'damaged is higher, as expected' if separates else 'damaged is NOT higher -- see DAY_3.md'})")

    plt.figure(figsize=(5, 4))
    plt.hist(healthy_scores, bins=15, alpha=0.6, label="Healthy (held out)", density=True)
    plt.hist(damaged_scores, bins=15, alpha=0.6, label="Damaged (all severities)", density=True)
    plt.xlabel("Global anomaly score (mean per-node MSE)")
    plt.ylabel("Density")
    plt.title("Graph autoencoder: healthy vs damaged error distribution")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = FIGURES_DIR / "day3_error_distribution.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def _report_localization(label, node_err, expected_idx, n_sensors, k):
    top1_acc = localization_topk_accuracy(node_err, expected_idx, k=1)
    topk_acc = localization_topk_accuracy(node_err, expected_idx, k=k)
    mean_rank = np.mean([localization_rank(row, expected_idx) for row in node_err])
    chance_topk = k / n_sensors
    beats_chance = topk_acc > 2 * chance_topk
    status = "[OK]" if beats_chance else "[FINDING]"
    print(f"  {label}: top-1 {top1_acc:.1%}, top-{k} {topk_acc:.1%} (chance {chance_topk:.1%}), "
          f"mean rank {mean_rank:.1f}/{ (n_sensors - 1) / 2:.1f} chance "
          f"-- {status} {'clearly better than chance' if beats_chance else 'NOT clearly better than chance'}")
    return topk_acc, mean_rank


def check_localization(geometry, sensor_node_ids, damaged_element_nodes, expected_idx, damaged_node_err,
                        sensor_error_mean, sensor_error_std, config):
    """Sec 6.3 localisation check, computed two ways: raw per-node error
    (as specified in the guideline), and per-sensor z-scored error (this
    project's fix, see DAY_3.md): each sensor's error judged against its
    *own* healthy baseline rather than compared directly across sensors,
    since some sensors run structurally hotter than others regardless of
    damage (see DAY_3.md, "Why this happens").
    """
    model_cfg = config["model"]
    n_sensors = len(sensor_node_ids)
    k = model_cfg["localization_top_k"]

    print(f"\nLocalisation: expected sensor is node {sensor_node_ids[expected_idx]} "
          f"(nearest to the damaged brace, {n_sensors} sensors total)")

    zscored_node_err = (damaged_node_err - sensor_error_mean) / sensor_error_std
    _report_localization("Raw error       ", damaged_node_err, expected_idx, n_sensors, k)
    _report_localization("Per-sensor z-score", zscored_node_err, expected_idx, n_sensors, k)

    fig, _ = plot_localization_heatmap(
        geometry, sensor_node_ids, damaged_node_err.mean(axis=0), damaged_element_nodes=damaged_element_nodes,
        title="Mean per-sensor reconstruction error across all damaged windows (raw)",
    )
    out_path = FIGURES_DIR / "day3_localization_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")

    fig, _ = plot_localization_heatmap(
        geometry, sensor_node_ids, zscored_node_err.mean(axis=0), damaged_element_nodes=damaged_element_nodes,
        title="Mean per-sensor error across all damaged windows (per-sensor z-score)",
        value_label="z-score",
    )
    out_path = FIGURES_DIR / "day3_localization_heatmap_normalized.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


def check_feature_direct_localization(table, window_meta, sensor_node_ids, geometry,
                                       damaged_element_nodes, expected_idx, config):
    """Alternative localisation signal, bypassing the autoencoder entirely:
    use only the single raw feature Day 2 found actually carries a damage
    signal (band_power_50_100hz, Cohen's d 1.27), z-scored per sensor the
    same way as the fix above. The autoencoder's reconstruction error
    blends this one informative feature in with 14 much less informative
    ones; this checks whether skipping that blending helps. See DAY_3.md.
    """
    model_cfg = config["model"]
    k = model_cfg["localization_top_k"]
    n_sensors = len(sensor_node_ids)
    feature_name = "band_power_50_100hz"

    table_sorted = table.sort_values(["window_id", "sensor_idx"])
    window_ids = table_sorted["window_id"].unique()
    feature_values = table_sorted[feature_name].to_numpy().reshape(len(window_ids), n_sensors)
    meta = window_meta.loc[window_ids]

    calib_mask = meta["split"].isin(["healthy_train", "healthy_val"]).to_numpy()
    sensor_mean = feature_values[calib_mask].mean(axis=0)
    sensor_std = feature_values[calib_mask].std(axis=0)

    is_damaged = (meta["damage_severity"] > 0).to_numpy()
    damaged_mask = is_damaged & (meta["split"] == "eval").to_numpy()
    zscored = np.abs((feature_values[damaged_mask] - sensor_mean) / sensor_std)

    print(f"\nFeature-direct localisation ({feature_name} only, per-sensor z-score, |z|):")
    _report_localization("Feature z-score   ", zscored, expected_idx, n_sensors, k)

    fig, _ = plot_localization_heatmap(
        geometry, sensor_node_ids, zscored.mean(axis=0), damaged_element_nodes=damaged_element_nodes,
        title=f"Mean |z-score| per sensor, {feature_name} only, across damaged windows",
        value_label="|z-score|",
    )
    out_path = FIGURES_DIR / "day3_localization_heatmap_feature.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


def main() -> None:
    config = load_config()
    model_cfg = config["model"]

    table = pd.read_csv(DATA_DIR / "features_raw.csv")
    feature_cols = [c for c in table.columns if c not in NON_FEATURE_COLUMNS]
    split = pd.read_csv(MODELS_DIR / "split.csv")

    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODELS_DIR / "baselines.pkl", "rb") as f:
        baselines = pickle.load(f)
    graph_npz = np.load(MODELS_DIR / "graph.npz")
    a_hat = graph_npz["a_hat"]
    sensor_node_ids = graph_npz["sensor_node_ids"]
    calibration = np.load(MODELS_DIR / "sensor_error_calibration.npz")
    sensor_error_mean, sensor_error_std = calibration["mean"], calibration["std"]

    scaled_table = table.copy()
    scaled_table[feature_cols] = scaler.transform(table[feature_cols])
    x_all, window_ids = build_feature_tensor(scaled_table, feature_cols)

    window_meta = split.set_index("window_id").loc[window_ids]
    eval_mask = (window_meta["split"] == "eval").to_numpy()
    x_eval = x_all[eval_mask]
    is_damaged = (window_meta.loc[eval_mask, "damage_severity"] > 0).to_numpy()
    severities_eval = window_meta.loc[eval_mask, "damage_severity"].to_numpy()
    print(
        f"Evaluation set: {len(is_damaged)} windows "
        f"({(~is_damaged).sum()} held-out healthy, {is_damaged.sum()} damaged across severities "
        f"{sorted(set(severities_eval[severities_eval > 0]))})"
    )

    gcn_model = GCNAutoencoder(n_features=len(feature_cols), hidden_dim=model_cfg["hidden_dim"],
                                latent_dim=model_cfg["latent_dim"])
    gcn_model.load_state_dict(torch.load(MODELS_DIR / "gcn_ae.pt"))
    gcn_model.eval()

    x_eval_t = torch.tensor(x_eval, dtype=torch.float32)
    a_hat_t = torch.tensor(a_hat, dtype=torch.float32)
    with torch.no_grad():
        x_hat = gcn_model(x_eval_t, a_hat_t)
        node_err = per_node_error(x_eval_t, x_hat).numpy()  # (n_eval, n_sensors)
    gcn_scores = node_err.mean(axis=1)  # global score, Sec 6.3

    x_eval_flat = x_eval.reshape(x_eval.shape[0], -1)
    pca_scores = baselines["pca"].score(x_eval_flat)
    if_scores = baselines["isolation_forest"].score(x_eval_flat)

    scores_by_model = {
        "Graph autoencoder": gcn_scores,
        "PCA reconstruction": pca_scores,
        "Isolation Forest": if_scores,
    }
    compute_roc_and_plot(is_damaged, severities_eval, scores_by_model)
    check_error_separation(is_damaged, gcn_scores)

    npz = np.load(DATA_DIR / "windows.npz")
    damaged_element_nodes = npz["damaged_element_nodes"]
    geometry = generate_lattice_geometry(config)
    expected_idx = nearest_sensor_by_hops(geometry, sensor_node_ids, damaged_element_nodes)

    check_localization(geometry, sensor_node_ids, damaged_element_nodes, expected_idx, node_err[is_damaged],
                        sensor_error_mean, sensor_error_std, config)
    check_feature_direct_localization(table, window_meta, sensor_node_ids, geometry,
                                       damaged_element_nodes, expected_idx, config)


if __name__ == "__main__":
    main()
