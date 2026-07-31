"""Day 3 follow-up experiment (not part of the main pipeline): a more
thorough version of day3_multi_location_experiment.py's generalisation
test. That experiment trained the supervised classifier on 3 locations
and tested on a 4th, held out entirely, and found it did *worse* than
chance, evidence the classifier had memorised 3 specific answers rather
than learned a transferable pattern. The open question that leaves: was
that just because 3 locations isn't enough to learn from, or does the
underlying signal simply not generalise at all, no matter how much is
given? This tests the first possibility directly: 12 damage locations,
spread across the tower's height and legs, evaluated with
leave-one-location-out cross-validation (train on the other 11, test on
the held-out one, repeated for every one of the 12), instead of a single
train-on-3/test-on-1 split, so no single unlucky choice of held-out
location can make the result look better or worse than it really is.

Simulates 11 new damage locations (location A, used throughout the
project, is reused from the existing dataset) at 40 instances/class each,
the same reduced count used in day3_multi_location_experiment.py. Takes
roughly 30-35 minutes, almost all of it simulation; the 12 classifier
folds themselves take a couple of minutes on top of that.

Run from the repo root (after scripts/01-04):
    python scripts/day3_leave_one_location_out_experiment.py
"""

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
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

N_LOCATIONS = 12
N_INSTANCES_PER_CLASS = 40   # matches day3_multi_location_experiment.py
SIM_SEED = 202                # separate from that script's seed (101) and model_cfg.random_seed


def load_config() -> dict:
    with open(REPO_ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def pick_locations(geometry, n_locations: int = N_LOCATIONS) -> dict:
    """Location A is the element used throughout every other part of the
    project. The rest are spread evenly across the tower's height,
    cycling through the three faces, so 12 locations cover roughly every
    1.5-2 panels from just above the base to near the top.
    """
    diagonal_idx = np.where(geometry.element_type == "diagonal")[0]
    n_panels = len(diagonal_idx) // 6  # 2 directions x 3 faces per panel
    default_idx = default_damage_element(geometry.element_type)

    locations = {"A": default_idx}
    fractions = np.linspace(0.05, 0.95, n_locations - 1)
    for i, frac in enumerate(fractions):
        panel = round(frac * (n_panels - 1))
        face = i % 3
        idx = int(diagonal_idx[panel * 6 + face * 2])
        if idx == default_idx:  # avoid an exact collision with A
            idx = int(diagonal_idx[min(panel + 1, n_panels - 1) * 6 + face * 2])
        locations[chr(ord("B") + i)] = idx
    return locations


def simulate_location(element_idx, geometry, sensor_node_ids, config, rng) -> pd.DataFrame:
    """Every damage severity for one element; healthy data is shared
    across locations and not re-simulated (see day3_multi_location_experiment.py)."""
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


def _metrics(node_err, expected_idx, k):
    top1 = localization_topk_accuracy(node_err, expected_idx, k=1)
    topk = localization_topk_accuracy(node_err, expected_idx, k=k)
    mean_rank = np.mean([localization_rank(row, expected_idx) for row in node_err])
    return top1, topk, mean_rank


def labelled_damaged(z, expected):
    """Every window here IS damaged at `expected`'s location: label 1 for
    that sensor, 0 for every other sensor in the same window."""
    n_windows, n_s, n_f = z.shape
    labels = np.zeros((n_windows, n_s), dtype=int)
    labels[:, expected] = 1
    return z.reshape(-1, n_f), labels.reshape(-1)


def labelled_healthy(z):
    """No damage at all: label 0 for every sensor."""
    n_windows, n_s, n_f = z.shape
    return z.reshape(-1, n_f), np.zeros(n_windows * n_s, dtype=int)


def main() -> None:
    config = load_config()
    model_cfg = config["model"]
    k = model_cfg["localization_top_k"]

    main_table = pd.read_csv(DATA_DIR / "features_raw.csv")
    split = pd.read_csv(MODELS_DIR / "split.csv")
    feature_cols = [c for c in main_table.columns if c not in NON_FEATURE_COLUMNS]
    # peak_frequency_hz excluded: a quantisation artefact caught during the
    # original supervised experiment (see DAY_3.md), inflates z-scores
    # instead of carrying a real signal.
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
    print(f"{len(locations)} damage locations (element index, height, nearest sensor):")
    for name, idx in locations.items():
        height = geometry.nodes[geometry.elements[idx], 2].mean()
        note = " (used throughout Stages 1-3)" if name == "A" else ""
        print(f"  {name}: element {idx}, height {height:.1f} m, nearest sensor node "
              f"{sensor_node_ids[expected_idx[name]]}{note}")

    # Per-sensor calibration, from the same healthy data used everywhere else
    # (healthy behaviour does not depend on where the damage is).
    healthy_train_val = set(split.loc[split["split"].isin(["healthy_train", "healthy_val"]), "instance_id"])
    raw_a, window_ids_a = pivot_features(main_table, feature_cols_clf, n_sensors)
    meta_a = split.set_index("window_id").loc[window_ids_a]
    calib_mask = meta_a["instance_id"].isin(healthy_train_val).to_numpy()
    sensor_mean = raw_a[calib_mask].mean(axis=0)
    sensor_std = raw_a[calib_mask].std(axis=0)

    is_damaged_a = (meta_a["damage_severity"] > 0).to_numpy()
    z_healthy = (raw_a[calib_mask] - sensor_mean) / sensor_std
    z_by_location = {"A": (raw_a[is_damaged_a] - sensor_mean) / sensor_std}

    new_names = [n for n in locations if n != "A"]
    print(f"\nSimulating {len(new_names)} new locations ({N_INSTANCES_PER_CLASS} instances/class, "
          f"{len(config['damage']['severities'])} severities each)...")
    rng = np.random.default_rng(SIM_SEED)
    location_rngs = dict(zip(new_names, rng.spawn(len(new_names))))

    tables = {}
    t_start = time.time()
    for name in new_names:
        tables[name] = simulate_location(locations[name], geometry, sensor_node_ids, config, location_rngs[name])
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tables[name].to_csv(DATA_DIR / f"loo_location_{name}_features.csv", index=False)
        raw, _ = pivot_features(tables[name], feature_cols_clf, n_sensors)
        z_by_location[name] = (raw - sensor_mean) / sensor_std
        print(f"  {name} done ({time.time() - t_start:.0f}s elapsed)")

    # Location-agnostic methods on the new locations: no retraining needed,
    # both are calibrated on healthy data only.
    print("\nLocation-agnostic methods on the 11 new locations (never used for any training):")
    ae_top3s, feat_top3s = [], []
    for name in new_names:
        scaled = tables[name].copy()
        scaled[feature_cols] = scaler.transform(tables[name][feature_cols])
        x_scaled, _ = pivot_features(scaled, feature_cols, n_sensors)
        with torch.no_grad():
            x_t = torch.tensor(x_scaled, dtype=torch.float32)
            a_hat_t = torch.tensor(a_hat, dtype=torch.float32)
            node_err = per_node_error(x_t, gcn_model(x_t, a_hat_t)).numpy()
        node_err_z = (node_err - sensor_error_mean) / sensor_error_std

        _, ae_top3, _ = _metrics(node_err_z, expected_idx[name], k)
        _, feat_top3, _ = _metrics(np.abs(z_by_location[name][:, :, band_idx]), expected_idx[name], k)
        ae_top3s.append(ae_top3)
        feat_top3s.append(feat_top3)
        print(f"  {name}: autoencoder z-score top-3 {ae_top3:.1%}, feature z-score top-3 {feat_top3:.1%}")
    chance_top3 = k / n_sensors
    print(f"  Average over {len(new_names)} new locations: autoencoder z-score {np.mean(ae_top3s):.1%}, "
          f"feature z-score {np.mean(feat_top3s):.1%} (chance {chance_top3:.1%})")

    # Leave-one-location-out: train the classifier on 11 locations, test on
    # the 12th, repeated for every location.
    print(f"\nLeave-one-location-out: training the supervised classifier on 11 of "
          f"{len(locations)} locations, testing on the held-out 12th, repeated for every location:")
    healthy_x, healthy_y = labelled_healthy(z_healthy)
    damaged_x, damaged_y = {}, {}
    for name in locations:
        damaged_x[name], damaged_y[name] = labelled_damaged(z_by_location[name], expected_idx[name])

    all_names = list(locations.keys())
    results = []
    t_fit_start = time.time()
    for held_out in all_names:
        train_names = [n for n in all_names if n != held_out]
        x_train = np.concatenate([healthy_x] + [damaged_x[n] for n in train_names], axis=0)
        y_train = np.concatenate([healthy_y] + [damaged_y[n] for n in train_names], axis=0)

        x_train_res, y_train_res = SMOTE(random_state=model_cfg["random_seed"]).fit_resample(x_train, y_train)
        clf = RandomForestClassifier(n_estimators=200, random_state=model_cfg["random_seed"], n_jobs=-1)
        clf.fit(x_train_res, y_train_res)

        n_test_windows = damaged_x[held_out].shape[0] // n_sensors
        proba = clf.predict_proba(damaged_x[held_out])[:, 1].reshape(n_test_windows, n_sensors)
        top1, top3, mean_rank = _metrics(proba, expected_idx[held_out], k)
        results.append({"location": held_out, "top1": top1, "top3": top3, "mean_rank": mean_rank})
        print(f"  held out {held_out}: top-1 {top1:.1%}, top-3 {top3:.1%}, mean rank {mean_rank:.1f}")
    print(f"  ({time.time() - t_fit_start:.0f}s for all {len(all_names)} folds)")

    results_df = pd.DataFrame(results)
    chance_rank = (n_sensors - 1) / 2
    print(f"\nLeave-one-location-out average over {len(all_names)} folds: "
          f"top-1 {results_df['top1'].mean():.1%} +/- {results_df['top1'].std():.1%}, "
          f"top-3 {results_df['top3'].mean():.1%} +/- {results_df['top3'].std():.1%}, "
          f"mean rank {results_df['mean_rank'].mean():.1f} +/- {results_df['mean_rank'].std():.1f}")
    print(f"Chance: top-1 {1 / n_sensors:.1%}, top-3 {chance_top3:.1%}, mean rank {chance_rank:.1f}")
    print("For comparison (from the earlier, smaller experiments): trained and tested on location A "
          "alone scored top-3 49.0%; trained on 3 locations and tested on 1 held-out location scored "
          "top-3 5.3%, worse than chance.")

    results_df.to_csv(DATA_DIR / "loo_results.csv", index=False)

    heights = np.array([geometry.nodes[geometry.elements[locations[n]], 2].mean()
                         for n in results_df["location"]])
    order = np.argsort(heights)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(results_df)), results_df["top3"].to_numpy()[order], color="tab:blue")
    ax.axhline(chance_top3, color="black", linestyle="--", linewidth=1, label=f"Chance ({chance_top3:.1%})")
    ax.axhline(results_df["top3"].mean(), color="tab:red", linestyle="-", linewidth=1,
               label=f"Mean across folds ({results_df['top3'].mean():.1%})")
    ax.set_xticks(range(len(results_df)))
    ax.set_xticklabels([f"{results_df['location'].to_numpy()[order][i]}\n{heights[order[i]]:.1f}m"
                         for i in range(len(order))], fontsize=8)
    ax.set_ylabel("Top-3 accuracy")
    ax.set_xlabel("Held-out location (letter, height up the tower)")
    ax.set_title("Leave-one-location-out: supervised classifier top-3 accuracy per held-out location")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    out_path = FIGURES_DIR / "day3_leave_one_out_topk.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
