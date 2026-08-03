"""Stage 5: maintenance-priority ranking stub (Sec 8.0). Scores every
already-evaluated instance from both datasets (Stage 3's synthetic test
split, Stage 4's LUMO test split plus all its damaged blocks) with their
already-trained graph autoencoders, reduces each instance's several
windows down to a current anomaly score, a severity trend, and a
localisation, then combines those into a single ranked table via
src/evaluate/priority.py.

No retraining happens here: this reuses the model artefacts
scripts/03_train.py and scripts/05_lumo_transfer.py already saved under
models/. Run those first.

Run from the repo root:
    python scripts/06_priority_ranking.py
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.lumo import ACCEL_CHANNELS, ML_HEIGHTS_M, build_feature_table, list_blocks, lumo_sensor_graph
from src.evaluate.metrics import detection_roc_auc, false_alarm_rate_at_detection_rate
from src.evaluate.priority import build_priority_table, severity_trend
from src.fem.geometry import generate_lattice_geometry
from src.models.gcn_ae import GCNAutoencoder, per_node_error

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "synthetic"
MODELS_DIR = REPO_ROOT / "models"
FIGURES_DIR = REPO_ROOT / "figures"

SYNTH_NON_FEATURE_COLUMNS = ("window_id", "sensor_idx", "sensor_node_id", "instance_id",
                             "damage_severity", "temperature_c")


def load_config() -> dict:
    with open(REPO_ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def build_feature_tensor(table: pd.DataFrame, feature_cols: list, n_sensors: int):
    table_sorted = table.sort_values(["window_id", "sensor_idx"])
    window_ids = table_sorted["window_id"].unique()
    x = table_sorted[feature_cols].to_numpy().reshape(len(window_ids), n_sensors, len(feature_cols))
    return x, window_ids


def split_healthy_blocks(meta: pd.DataFrame, test_fraction: float, val_fraction: float, rng) -> tuple:
    """Same block-level split scripts/05_lumo_transfer.py used to train
    the saved LUMO model, reproduced here (rather than imported, since a
    numeral-prefixed script cannot be imported) so the eval set matches
    exactly: same seed, same deterministic permutation.
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


def synthetic_instance_records(config: dict):
    """Score every instance in Stage 3's held-out test split (the same
    split scripts/04_evaluate.py used) with the saved graph autoencoder,
    and reduce each instance's windows to one row: a current anomaly
    score (the most recent window), a severity trend (slope across that
    instance's windows), and a localisation (the sensor with the highest
    mean per-sensor z-score, mapped to its height).

    Returns (records, is_damaged_windows, scores_windows); the latter two
    are the pooled window-level arrays used for the AUC / false-alarm
    report, kept separate from the per-instance table above.
    """
    model_cfg = config["model"]
    table = pd.read_csv(DATA_DIR / "features_raw.csv")
    feature_cols = [c for c in table.columns if c not in SYNTH_NON_FEATURE_COLUMNS]
    split = pd.read_csv(MODELS_DIR / "split.csv")

    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    graph_npz = np.load(MODELS_DIR / "graph.npz")
    a_hat = graph_npz["a_hat"]
    sensor_node_ids = graph_npz["sensor_node_ids"]
    calibration = np.load(MODELS_DIR / "sensor_error_calibration.npz")
    sensor_error_mean, sensor_error_std = calibration["mean"], calibration["std"]

    scaled_table = table.copy()
    scaled_table[feature_cols] = scaler.transform(table[feature_cols])
    x_all, window_ids = build_feature_tensor(scaled_table, feature_cols, len(sensor_node_ids))

    window_meta = split.set_index("window_id").loc[window_ids]
    eval_mask = (window_meta["split"] == "eval").to_numpy()
    x_eval = x_all[eval_mask]

    gcn_model = GCNAutoencoder(n_features=len(feature_cols), hidden_dim=model_cfg["hidden_dim"],
                                latent_dim=model_cfg["latent_dim"])
    gcn_model.load_state_dict(torch.load(MODELS_DIR / "gcn_ae.pt"))
    gcn_model.eval()

    with torch.no_grad():
        x_eval_t = torch.tensor(x_eval, dtype=torch.float32)
        a_hat_t = torch.tensor(a_hat, dtype=torch.float32)
        node_err = per_node_error(x_eval_t, gcn_model(x_eval_t, a_hat_t)).numpy()  # (n_eval, n_sensors)
    global_score = node_err.mean(axis=1)
    zscored = (node_err - sensor_error_mean) / sensor_error_std
    is_damaged_windows = (window_meta.loc[eval_mask, "damage_severity"] > 0).to_numpy()

    geometry = generate_lattice_geometry(config)
    heights = geometry.nodes[sensor_node_ids, 2]

    meta_eval = window_meta.loc[eval_mask].copy()
    meta_eval["window_id"] = meta_eval.index
    meta_eval = meta_eval.reset_index(drop=True)
    meta_eval["_row"] = np.arange(len(meta_eval))

    records = []
    for instance_id, group in meta_eval.sort_values("window_id").groupby("instance_id"):
        rows = group["_row"].to_numpy()
        scores_seq = global_score[rows]
        mean_z = zscored[rows].mean(axis=0)
        localised_idx = int(np.argmax(mean_z))
        records.append({
            "source": "synthetic",
            "instance_id": f"synthetic-{instance_id}",
            "is_damaged": bool(group["damage_severity"].iloc[0] > 0),
            "ground_truth_note": (
                "healthy" if group["damage_severity"].iloc[0] == 0
                else f"damage severity {group['damage_severity'].iloc[0]:.1f}"
            ),
            "current_anomaly_score": float(scores_seq[-1]),
            "severity_trend": severity_trend(scores_seq),
            "localised_height_m": float(heights[localised_idx]),
            "localised_label": f"sensor node {int(sensor_node_ids[localised_idx])} (~{heights[localised_idx]:.1f} m up)",
        })
    return pd.DataFrame(records), is_damaged_windows, global_score


def lumo_block_records(config: dict):
    """Same idea as synthetic_instance_records, for LUMO: score every
    block in Stage 4's held-out healthy test split plus every damaged
    block, with the saved LUMO graph autoencoder, and reduce each
    block's windows to one row.
    """
    model_cfg = config["model"]
    lumo_cfg = config["lumo"]
    rng = np.random.default_rng(model_cfg["random_seed"])

    data_dirs = [REPO_ROOT / d for d in lumo_cfg["data_dirs"]]
    missing = [d for d in data_dirs if not d.exists()]
    if missing:
        raise FileNotFoundError(f"LUMO data not found at {missing}. See DAY_4.md for download instructions.")

    blocks = pd.concat([list_blocks(d) for d in data_dirs], ignore_index=True)
    sampling_rate_hz = 1651.6129032258063
    table = build_feature_table(blocks, lumo_cfg["window_length"], sampling_rate_hz, config["features"]["band_edges_hz"])
    non_feature_cols = ("window_id", "sensor_idx", "block_id", "is_healthy", "dam_position", "temperature_c")
    feature_cols = [c for c in table.columns if c not in non_feature_cols]

    train_blocks, val_blocks, test_blocks = split_healthy_blocks(
        table, lumo_cfg["healthy_test_fraction"], lumo_cfg["healthy_val_fraction"], rng,
    )

    with open(MODELS_DIR / "lumo_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    scaled_table = table.copy()
    scaled_table[feature_cols] = scaler.transform(table[feature_cols])
    x_scaled, window_ids = build_feature_tensor(scaled_table, feature_cols, len(ACCEL_CHANNELS))

    window_meta = table[["window_id", "block_id", "is_healthy", "dam_position", "temperature_c"]] \
        .drop_duplicates().set_index("window_id").loc[window_ids]
    val_window_mask = window_meta["block_id"].isin(val_blocks).to_numpy()
    eval_window_mask = (
        window_meta["block_id"].isin(test_blocks) | ~window_meta["is_healthy"].astype(bool)
    ).to_numpy()

    _, a_hat = lumo_sensor_graph()
    n_features = len(feature_cols)

    gcn_model = GCNAutoencoder(n_features=n_features, hidden_dim=model_cfg["hidden_dim"],
                                latent_dim=model_cfg["latent_dim"])
    gcn_model.load_state_dict(torch.load(MODELS_DIR / "lumo_gcn_ae.pt"))
    gcn_model.eval()

    with torch.no_grad():
        a_hat_t = torch.tensor(a_hat, dtype=torch.float32)
        x_val_t = torch.tensor(x_scaled[val_window_mask], dtype=torch.float32)
        val_node_err = per_node_error(x_val_t, gcn_model(x_val_t, a_hat_t)).numpy()
        x_eval_t = torch.tensor(x_scaled[eval_window_mask], dtype=torch.float32)
        node_err = per_node_error(x_eval_t, gcn_model(x_eval_t, a_hat_t)).numpy()

    sensor_error_mean, sensor_error_std = val_node_err.mean(axis=0), val_node_err.std(axis=0)
    global_score = node_err.mean(axis=1)
    zscored = (node_err - sensor_error_mean) / sensor_error_std
    is_damaged_windows = ~window_meta.loc[eval_window_mask, "is_healthy"].astype(bool).to_numpy()

    meta_eval = window_meta.loc[eval_window_mask].copy()
    meta_eval["window_id"] = meta_eval.index
    meta_eval = meta_eval.reset_index(drop=True)
    meta_eval["_row"] = np.arange(len(meta_eval))

    n_levels = len(ML_HEIGHTS_M) - 1  # ML1-ML9 have accelerometers; ML10 does not
    records = []
    for block_id, group in meta_eval.sort_values("window_id").groupby("block_id"):
        rows = group["_row"].to_numpy()
        scores_seq = global_score[rows]
        mean_z = zscored[rows].mean(axis=0)
        level_z = mean_z.reshape(n_levels, 2).mean(axis=1)
        localised_level = int(np.argmax(level_z)) + 1  # 1-indexed ML number
        is_healthy = bool(group["is_healthy"].iloc[0])
        dam_position = group["dam_position"].iloc[0]
        records.append({
            "source": "LUMO",
            "instance_id": f"lumo-block-{block_id}",
            "is_damaged": not is_healthy,
            "ground_truth_note": "healthy" if is_healthy else f"DAM{int(dam_position)}",
            "current_anomaly_score": float(scores_seq[-1]),
            "severity_trend": severity_trend(scores_seq),
            "localised_height_m": float(ML_HEIGHTS_M[localised_level]),
            "localised_label": f"ML{localised_level} (~{ML_HEIGHTS_M[localised_level]:.2f} m up)",
        })
    return pd.DataFrame(records), is_damaged_windows, global_score


def main() -> None:
    config = load_config()
    priority_cfg = config["priority"]

    print("Scoring already-evaluated synthetic instances (Stage 3 test split)...")
    synth_records, synth_is_damaged, synth_scores = synthetic_instance_records(config)
    print(f"  {len(synth_records)} instances ({int(synth_records['is_damaged'].sum())} damaged)")

    print("Scoring already-evaluated LUMO blocks (Stage 4 test split, plus every damaged block)...")
    lumo_records, lumo_is_damaged, lumo_scores = lumo_block_records(config)
    print(f"  {len(lumo_records)} blocks ({int(lumo_records['is_damaged'].sum())} damaged)")

    target_tpr = priority_cfg["detection_target_tpr"]
    print(f"\nOperating point (Sec 8.3): false alarm rate at {target_tpr:.0%} detection rate, "
          f"window-level, using the saved models with no retraining:")
    for name, is_damaged, scores in [
        ("Synthetic", synth_is_damaged, synth_scores), ("LUMO", lumo_is_damaged, lumo_scores),
    ]:
        auc = detection_roc_auc(is_damaged, scores)
        far = false_alarm_rate_at_detection_rate(is_damaged, scores, target_tpr)
        print(f"  {name}: AUC {auc:.3f}, false alarm rate at {target_tpr:.0%} detection = {far:.1%}")

    records = pd.concat([synth_records, lumo_records], ignore_index=True)
    ranked = build_priority_table(
        records, priority_cfg["base_inspection_cost_aud"], priority_cfg["climb_cost_per_metre_aud"],
        priority_cfg["risk_scale_aud"],
    )

    FIGURES_DIR.mkdir(exist_ok=True)
    out_path = FIGURES_DIR / "priority_ranking.csv"
    ranked.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(ranked)} rows, ranked most urgent first)")

    print("\nTop 10 by priority score:")
    top10 = ranked.head(10)[[
        "source", "instance_id", "ground_truth_note", "current_anomaly_score", "severity_trend",
        "priority_score", "localised_label", "inspection_cost_aud", "risk_of_delay_aud", "worth_inspecting_now",
    ]]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(top10.to_string(index=False))

    overall_damaged_fraction = records["is_damaged"].mean()
    top10_damaged_fraction = top10["ground_truth_note"].ne("healthy").mean()
    print(f"\n{top10_damaged_fraction:.0%} of the top 10 ranked instances are actually damaged "
          f"(the dataset overall is {overall_damaged_fraction:.0%} damaged).")


if __name__ == "__main__":
    main()
