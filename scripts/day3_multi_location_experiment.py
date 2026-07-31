"""Day 3 follow-up experiment (not part of the main pipeline): does damage
localisation generalise to a location it has never seen, or does every
result so far only look good because it has always been the same single
brace?

Every number in DAY_3.md up to this point, including the supervised
classifier's 49.0% top-3 accuracy, was trained and tested on the same one
damaged element (call it location A). This experiment simulates three more
damage locations, spread across the tower's height and legs
(src/fem/geometry.py generates diagonal braces panel by panel, 6 per
panel: 2 directions x 3 faces, so picking elements from different
panel/face combinations spreads them out physically, see `pick_locations`
below), trains the supervised classifier on three locations (A, B, C) and
tests it on the fourth (D), held out entirely, never touched during
training. It also re-checks the two location-agnostic methods (the
autoencoder's per-sensor z-score, and the single-feature z-score) on B, C,
and D: those never needed damage labels to begin with, only healthy data,
so they should transfer to a new location "for free" if the underlying
fix is sound.

The three new locations use fewer simulated instances per class (40, vs
the main pipeline's 150) to keep runtime reasonable: this is a
supplementary generalisation check, not a replacement for the main
dataset. No new healthy data is simulated, healthy vibration behaviour
does not depend on where the (as yet nonexistent) damage would be, so the
existing healthy split from scripts/03_train.py is reused for calibration.

Run from the repo root (after scripts/01-04). Takes roughly 10-12
minutes, almost all of it simulating the 3 new locations:
    python scripts/day3_multi_location_experiment.py
"""

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

from src.evaluate.metrics import localization_rank, localization_topk_accuracy
from src.features.spectral import spectral_features
from src.features.windows import time_domain_features
from src.fem.damage import apply_damage, default_damage_element
from src.fem.geometry import generate_lattice_geometry
from src.fem.truss import build_truss_model
from src.fem.visualize import plot_localization_heatmap
from src.graph.build import nearest_sensor_by_hops
from src.models.gcn_ae import GCNAutoencoder, per_node_error
from src.simulate.environment import material_at_temperature
from src.simulate.response import simulate_sensor_windows

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "synthetic"
MODELS_DIR = REPO_ROOT / "models"
FIGURES_DIR = REPO_ROOT / "figures"

NON_FEATURE_COLUMNS = ("window_id", "sensor_idx", "sensor_node_id", "instance_id",
                        "damage_severity", "temperature_c")

N_INSTANCES_PER_CLASS = 40  # reduced from the main pipeline's 150, see module docstring
SIM_SEED = 101               # separate from model_cfg.random_seed: independently reproducible


def load_config() -> dict:
    with open(REPO_ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def pick_locations(geometry) -> dict:
    """Location A is the element used throughout every other part of this
    project. B and C are two more, used for training alongside A. D is
    held out entirely, used only to test generalisation. Chosen as
    fractions of the total panel count (rather than hardcoded indices) so
    this still makes sense if the geometry config changes; with the
    current 21-panel tower this lands at roughly 1.5 m (B), 4.5 m (A, the
    original), 6.2 m (C), and 7.9 m (D) up the 9 m tower, on three
    different faces.
    """
    diagonal_idx = np.where(geometry.element_type == "diagonal")[0]
    n_panels = len(diagonal_idx) // 6  # 2 directions x 3 faces per panel

    def element_at(panel_fraction, face):
        panel = round(n_panels * panel_fraction)
        return int(diagonal_idx[panel * 6 + face * 2])

    return {
        "A": default_damage_element(geometry.element_type),
        "B": element_at(0.15, face=0),
        "C": element_at(0.68, face=2),
        "D": element_at(0.88, face=1),
    }


def simulate_location(element_idx, geometry, sensor_node_ids, config, rng) -> pd.DataFrame:
    """Simulate every damage severity for one element and extract features
    directly (no healthy class: healthy data is shared across locations
    and already exists from 01_generate_synthetic.py).
    """
    sim_cfg = config["simulate"]
    env_cfg = config["environment"]
    severities = config["damage"]["severities"]
    instance_rngs = rng.spawn(len(severities) * N_INSTANCES_PER_CLASS)

    all_windows, rows = [], []
    instance_id = 0
    for severity in severities:
        for _ in range(N_INSTANCES_PER_CLASS):
            instance_rng = instance_rngs[instance_id]
            temperature_c = instance_rng.uniform(env_cfg["temperature_min_c"], env_cfg["temperature_max_c"])
            material_T = material_at_temperature(config["material"], temperature_c, env_cfg)
            model_T = build_truss_model(geometry, material_T)
            model_T, _ = apply_damage(model_T, [(element_idx, severity)])

            windows = simulate_sensor_windows(model_T, sensor_node_ids, config, instance_rng)
            all_windows.append(windows)
            for _ in range(windows.shape[0]):
                rows.append({"window_id": len(rows), "instance_id": instance_id,
                             "damage_severity": severity, "temperature_c": temperature_c})
            instance_id += 1

    windows_array = np.concatenate(all_windows, axis=0)
    meta = pd.DataFrame(rows)

    time_feats = time_domain_features(windows_array)
    freq_feats = spectral_features(windows_array, sim_cfg["sampling_rate_hz"], config["features"]["band_edges_hz"])
    all_feats = {**time_feats, **freq_feats}

    n_windows, n_sensors = windows_array.shape[0], windows_array.shape[1]
    table = pd.DataFrame({
        "window_id": np.repeat(meta["window_id"].to_numpy(), n_sensors),
        "sensor_idx": np.tile(np.arange(n_sensors), n_windows),
        "sensor_node_id": np.tile(sensor_node_ids, n_windows),
        "instance_id": np.repeat(meta["instance_id"].to_numpy(), n_sensors),
        "damage_severity": np.repeat(meta["damage_severity"].to_numpy(), n_sensors),
        "temperature_c": np.repeat(meta["temperature_c"].to_numpy(), n_sensors),
    })
    for name, arr in all_feats.items():
        table[name] = arr.ravel()
    return table


def pivot_features(table: pd.DataFrame, feature_cols: list, n_sensors: int):
    """(window, sensor) long table -> (n_windows, n_sensors, n_features), window_ids."""
    table_sorted = table.sort_values(["window_id", "sensor_idx"])
    window_ids = table_sorted["window_id"].unique()
    values = table_sorted[feature_cols].to_numpy().reshape(len(window_ids), n_sensors, len(feature_cols))
    return values, window_ids


def _report(label, node_err, expected_idx, n_sensors, k):
    top1 = localization_topk_accuracy(node_err, expected_idx, k=1)
    topk = localization_topk_accuracy(node_err, expected_idx, k=k)
    mean_rank = np.mean([localization_rank(row, expected_idx) for row in node_err])
    print(f"    {label}: top-1 {top1:.1%}, top-{k} {topk:.1%}, mean rank {mean_rank:.1f}")
    return top1, topk, mean_rank


def main() -> None:
    config = load_config()
    model_cfg = config["model"]
    k = model_cfg["localization_top_k"]

    main_table = pd.read_csv(DATA_DIR / "features_raw.csv")
    split = pd.read_csv(MODELS_DIR / "split.csv")
    feature_cols = [c for c in main_table.columns if c not in NON_FEATURE_COLUMNS]
    # peak_frequency_hz excluded for the classifier and feature-direct checks: a
    # quantisation artefact caught during the original supervised experiment
    # (see DAY_3.md), inflates z-scores instead of carrying a real signal.
    feature_cols_clf = [c for c in feature_cols if c != "peak_frequency_hz"]
    band_idx = feature_cols_clf.index("band_power_50_100hz")

    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    graph_npz = np.load(MODELS_DIR / "graph.npz")
    a_hat = graph_npz["a_hat"]
    sensor_node_ids = graph_npz["sensor_node_ids"]
    n_sensors = len(sensor_node_ids)
    calibration = np.load(MODELS_DIR / "sensor_error_calibration.npz")
    sensor_error_mean, sensor_error_std = calibration["mean"], calibration["std"]

    gcn_model = GCNAutoencoder(n_features=len(feature_cols), hidden_dim=model_cfg["hidden_dim"],
                                latent_dim=model_cfg["latent_dim"])
    gcn_model.load_state_dict(torch.load(MODELS_DIR / "gcn_ae.pt"))
    gcn_model.eval()

    geometry = generate_lattice_geometry(config)
    locations = pick_locations(geometry)
    expected_idx = {
        name: nearest_sensor_by_hops(geometry, sensor_node_ids, geometry.elements[idx])
        for name, idx in locations.items()
    }
    print("Damage locations (element index, height, nearest sensor):")
    for name, idx in locations.items():
        height = geometry.nodes[geometry.elements[idx], 2].mean()
        role = "used throughout Stages 1-3, and for training here" if name == "A" \
            else "held out, testing only" if name == "D" else "new, for training here"
        print(f"  {name}: element {idx}, height {height:.1f} m, "
              f"nearest sensor node {sensor_node_ids[expected_idx[name]]} ({role})")

    # Per-sensor calibration, from the SAME healthy data used everywhere else in
    # this project: healthy behaviour does not depend on where the damage is
    # (there isn't any yet), so one calibration applies to every location.
    healthy_train_val = set(split.loc[split["split"].isin(["healthy_train", "healthy_val"]), "instance_id"])
    raw_a, window_ids_a = pivot_features(main_table, feature_cols_clf, n_sensors)
    meta_a = split.set_index("window_id").loc[window_ids_a]
    calib_mask = meta_a["instance_id"].isin(healthy_train_val).to_numpy()
    sensor_mean = raw_a[calib_mask].mean(axis=0)
    sensor_std = raw_a[calib_mask].std(axis=0)

    is_damaged_a = (meta_a["damage_severity"] > 0).to_numpy()
    z_healthy = (raw_a[calib_mask] - sensor_mean) / sensor_std
    z_a_damaged = (raw_a[is_damaged_a] - sensor_mean) / sensor_std

    print(f"\nSimulating locations B, C, D ({N_INSTANCES_PER_CLASS} instances/class, "
          f"{len(config['damage']['severities'])} severities each)...")
    rng = np.random.default_rng(SIM_SEED)
    location_rngs = dict(zip(["B", "C", "D"], rng.spawn(3)))

    tables = {}
    t_start = time.time()
    for name in ["B", "C", "D"]:
        tables[name] = simulate_location(locations[name], geometry, sensor_node_ids, config, location_rngs[name])
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tables[name].to_csv(DATA_DIR / f"multi_location_{name}_features.csv", index=False)
        print(f"  {name} done ({time.time() - t_start:.0f}s elapsed)")

    # Location-agnostic methods: no retraining, just applied to the new
    # locations, using the healthy-only calibration from above (feature
    # z-score) and from scripts/03_train.py (autoencoder error z-score).
    print("\nLocation-agnostic methods on the 3 new locations (never used for any training):")
    z_by_location = {}
    for name in ["B", "C", "D"]:
        raw, window_ids = pivot_features(tables[name], feature_cols_clf, n_sensors)
        z = (raw - sensor_mean) / sensor_std
        z_by_location[name] = z

        scaled = tables[name].copy()
        scaled[feature_cols] = scaler.transform(tables[name][feature_cols])
        x_scaled, _ = pivot_features(scaled, feature_cols, n_sensors)
        with torch.no_grad():
            x_t = torch.tensor(x_scaled, dtype=torch.float32)
            a_hat_t = torch.tensor(a_hat, dtype=torch.float32)
            node_err = per_node_error(x_t, gcn_model(x_t, a_hat_t)).numpy()
        node_err_z = (node_err - sensor_error_mean) / sensor_error_std

        print(f"  Location {name} (height {geometry.nodes[geometry.elements[locations[name]], 2].mean():.1f} m):")
        _report("Autoencoder, raw error       ", node_err, expected_idx[name], n_sensors, k)
        _report("Autoencoder, per-sensor z-score", node_err_z, expected_idx[name], n_sensors, k)
        _report("Single feature, per-sensor z-score", np.abs(z[:, :, band_idx]), expected_idx[name], n_sensors, k)

    # Supervised classifier: train on A, B, C (pooled, sensor identity withheld,
    # per-sensor z-scored features, exactly as in
    # day3_supervised_localization_experiment.py), test on D, held out entirely.
    print("\nSupervised classifier, trained on locations A + B + C, tested on D (held out entirely):")

    def labelled_damaged(z, expected):
        """Every window here IS damaged at `expected`'s location: label 1
        for that sensor, 0 for every other sensor in the same window."""
        n_windows, n_s, n_f = z.shape
        labels = np.zeros((n_windows, n_s), dtype=int)
        labels[:, expected] = 1
        return z.reshape(-1, n_f), labels.reshape(-1)

    def labelled_healthy(z):
        """No damage at all: label 0 for every sensor."""
        n_windows, n_s, n_f = z.shape
        return z.reshape(-1, n_f), np.zeros(n_windows * n_s, dtype=int)

    x_healthy, y_healthy = labelled_healthy(z_healthy)
    x_a, y_a = labelled_damaged(z_a_damaged, expected_idx["A"])
    x_b, y_b = labelled_damaged(z_by_location["B"], expected_idx["B"])
    x_c, y_c = labelled_damaged(z_by_location["C"], expected_idx["C"])
    x_d, y_d = labelled_damaged(z_by_location["D"], expected_idx["D"])

    x_train = np.concatenate([x_healthy, x_a, x_b, x_c], axis=0)
    y_train = np.concatenate([y_healthy, y_a, y_b, y_c], axis=0)
    print(f"  Training rows: {x_train.shape[0]} ({y_train.sum()} positive, {y_train.mean():.2%}), "
          f"from healthy windows + damaged windows at locations A, B, C")
    print(f"  Test rows: {x_d.shape[0]} ({y_d.sum()} positive, {y_d.mean():.2%}), "
          f"location D only, never seen during training")

    x_train_res, y_train_res = SMOTE(random_state=model_cfg["random_seed"]).fit_resample(x_train, y_train)
    clf = RandomForestClassifier(n_estimators=200, random_state=model_cfg["random_seed"], n_jobs=-1)
    clf.fit(x_train_res, y_train_res)

    n_test_windows = x_d.shape[0] // n_sensors
    proba_d = clf.predict_proba(x_d)[:, 1].reshape(n_test_windows, n_sensors)
    top1, top3, mean_rank = _report("Held-out location D", proba_d, expected_idx["D"], n_sensors, k)
    chance_top3 = k / n_sensors
    print(f"  Chance: top-1 {1 / n_sensors:.1%}, top-{k} {chance_top3:.1%}, mean rank {(n_sensors - 1) / 2:.1f}")
    print(f"  For comparison, the same classifier trained AND tested on location A alone scored "
          f"top-3 49.0% (see DAY_3.md); this is the same method's accuracy on a location it never trained on.")

    fig, _ = plot_localization_heatmap(
        geometry, sensor_node_ids, proba_d.mean(axis=0), damaged_element_nodes=geometry.elements[locations["D"]],
        title="Mean predicted P(nearest sensor), supervised RF trained on A+B+C, tested on held-out D",
        value_label="P(nearest sensor)",
    )
    out_path = FIGURES_DIR / "day3_localization_heatmap_multi_location.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
