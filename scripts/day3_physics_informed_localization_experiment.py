"""Day 3 follow-up experiment (not part of the main pipeline): does
combining a physics-derived per-sensor sensitivity (modal strain energy,
src/fem/modal_energy.py) with the existing data-driven per-sensor
z-score improve damage localisation?

This is the classical "model-based SHM" idea, damage localisation via
modal strain energy (Pandey, Biswas & Samman 1991; Stubbs & Kim 1996),
combined here with this project's already-trained graph autoencoder in
the simplest way that keeps both signals on a comparable, symmetric
scale: z-score each one separately, the autoencoder's error against its
own healthy-history baseline (already established, see DAY_3.md), the
physics sensitivity across the 18 sensors (it has no "healthy history"
of its own, it is one fixed number per sensor, not a per-window
measurement), then add the two.

No retraining: reuses the exact model/scaler/split artefacts
scripts/03_train.py already saved, and computes the physics side fresh
from the same healthy finite-element model scripts/01_generate_synthetic.py
already validated (Sec 4.2's acceptance checks).

Run from the repo root (after 03_train.py):
    python scripts/day3_physics_informed_localization_experiment.py
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import yaml

from src.evaluate.metrics import localization_rank, localization_topk_accuracy
from src.fem.geometry import generate_lattice_geometry
from src.fem.modal_energy import sensor_modal_sensitivity
from src.fem.truss import build_truss_model, solve_modal
from src.graph.build import nearest_sensor_by_hops
from src.models.gcn_ae import GCNAutoencoder, per_node_error

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "synthetic"
MODELS_DIR = REPO_ROOT / "models"

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


def main() -> None:
    config = load_config()
    model_cfg = config["model"]
    k = model_cfg["localization_top_k"]

    print("Computing physics-derived per-sensor sensitivity from the healthy finite-element model...")
    geometry = generate_lattice_geometry(config)
    truss_model = build_truss_model(geometry, config["material"])
    _, mode_shapes = solve_modal(truss_model, config["modal"]["n_modes"])

    npz = np.load(DATA_DIR / "windows.npz")
    sensor_node_ids = npz["sensor_node_ids"]
    damaged_element_nodes = npz["damaged_element_nodes"]
    n_sensors = len(sensor_node_ids)

    sensitivity = sensor_modal_sensitivity(truss_model, mode_shapes, sensor_node_ids)
    physics_z = (sensitivity - sensitivity.mean()) / sensitivity.std()
    print("Per-sensor physics sensitivity (z-scored across the 18 sensors):")
    for node_id, val in zip(sensor_node_ids, physics_z):
        print(f"  node {node_id}: {val:+.2f}")

    print("\nLoading the already-trained graph autoencoder and calibration...")
    table = pd.read_csv(DATA_DIR / "features_raw.csv")
    feature_cols = [c for c in table.columns if c not in NON_FEATURE_COLUMNS]
    split = pd.read_csv(MODELS_DIR / "split.csv")

    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    graph_npz = np.load(MODELS_DIR / "graph.npz")
    a_hat = graph_npz["a_hat"]
    calibration = np.load(MODELS_DIR / "sensor_error_calibration.npz")
    sensor_error_mean, sensor_error_std = calibration["mean"], calibration["std"]

    scaled_table = table.copy()
    scaled_table[feature_cols] = scaler.transform(table[feature_cols])
    x_all, window_ids = build_feature_tensor(scaled_table, feature_cols)

    window_meta = split.set_index("window_id").loc[window_ids]
    eval_mask = (window_meta["split"] == "eval").to_numpy()
    x_eval = x_all[eval_mask]
    is_damaged = (window_meta.loc[eval_mask, "damage_severity"] > 0).to_numpy()

    gcn_model = GCNAutoencoder(n_features=len(feature_cols), hidden_dim=model_cfg["hidden_dim"],
                                latent_dim=model_cfg["latent_dim"])
    gcn_model.load_state_dict(torch.load(MODELS_DIR / "gcn_ae.pt"))
    gcn_model.eval()

    with torch.no_grad():
        x_eval_t = torch.tensor(x_eval, dtype=torch.float32)
        a_hat_t = torch.tensor(a_hat, dtype=torch.float32)
        node_err = per_node_error(x_eval_t, gcn_model(x_eval_t, a_hat_t)).numpy()

    zscored = (node_err - sensor_error_mean) / sensor_error_std
    combined = zscored + physics_z[None, :]

    expected_idx = nearest_sensor_by_hops(geometry, sensor_node_ids, damaged_element_nodes)
    print(f"\nLocalisation: expected sensor is node {sensor_node_ids[expected_idx]} "
          f"(nearest to the damaged brace, {n_sensors} sensors total)")

    for label, err in [
        ("Autoencoder z-score alone (baseline)      ", zscored[is_damaged]),
        ("Autoencoder z-score + physics, combined   ", combined[is_damaged]),
    ]:
        top1 = localization_topk_accuracy(err, expected_idx, k=1)
        topk = localization_topk_accuracy(err, expected_idx, k=k)
        mean_rank = np.mean([localization_rank(row, expected_idx) for row in err])
        chance_topk = k / n_sensors
        beats_chance = topk > 2 * chance_topk
        status = "[OK]" if beats_chance else "[FINDING]"
        print(f"  {label}: top-1 {top1:.1%}, top-{k} {topk:.1%} (chance {chance_topk:.1%}), "
              f"mean rank {mean_rank:.1f}/{(n_sensors - 1) / 2:.1f} chance "
              f"-- {status} {'clearly better than chance' if beats_chance else 'NOT clearly better than chance'}")


if __name__ == "__main__":
    main()
