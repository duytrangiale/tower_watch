"""Day 3 follow-up experiment (not part of the main pipeline): can a
supervised classifier, given the known damage location during training,
localise damage better than the unsupervised autoencoder's reconstruction
error did? Directly inspired by Samudra, Barbosh & Sadhu (2023, Sensors
23, 3365), who found an unsupervised clustering approach (a
self-organising map) could not cleanly separate their SHM anomaly
classes and switched to a supervised random forest instead.

This is a deliberate, flagged departure from the guideline's "healthy
windows only" training convention (Sec 6.3), done here specifically to
check whether the labels this synthetic study happens to have would
help, not to replace the main detector. It would not directly transfer
to a real deployment with no confirmed damage history to train on.

Sensor identity is deliberately withheld from the classifier's input
features: every damaged window in this project has the *same* damaged
element, so a classifier that could see which physical sensor a row came
from could trivially learn "always predict sensor 27" and score
perfectly without learning anything from the actual sensor readings.
Withholding identity and z-scoring each sensor's own features (so a
given feature value means the same thing regardless of which physical
sensor it came from) forces the classifier to generalise from feature
values, not memorise a position.

Also applies SMOTE (from the same paper) to the severe class imbalance
in the training data (roughly 1 positive row per 18).

Run from the repo root (after scripts/01-04):
    python scripts/day3_supervised_localization_experiment.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

from src.evaluate.metrics import localization_rank, localization_topk_accuracy
from src.fem.geometry import generate_lattice_geometry
from src.fem.visualize import plot_localization_heatmap
from src.graph.build import nearest_sensor_by_hops

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "synthetic"
MODELS_DIR = REPO_ROOT / "models"
FIGURES_DIR = REPO_ROOT / "figures"

NON_FEATURE_COLUMNS = ("window_id", "sensor_idx", "sensor_node_id", "instance_id",
                        "damage_severity", "temperature_c")


def load_config() -> dict:
    with open(REPO_ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def split_damaged_instances(split: pd.DataFrame, test_fraction: float, rng: np.random.Generator):
    """A new train/test split, over the damaged instances specifically
    (separate from the healthy train/val/test split from 03_train.py,
    which never touches damaged data)."""
    damaged_instances = np.sort(split.loc[split["damage_severity"] > 0, "instance_id"].unique())
    shuffled = rng.permutation(damaged_instances)
    n_test = round(len(damaged_instances) * test_fraction)
    return set(shuffled[n_test:].tolist()), set(shuffled[:n_test].tolist())


def raw_feature_array(table: pd.DataFrame, feature_cols: list, n_sensors: int):
    """(window, sensor) long table -> (n_windows, n_sensors, n_features), window_ids."""
    table_sorted = table.sort_values(["window_id", "sensor_idx"])
    window_ids = table_sorted["window_id"].unique()
    raw = table_sorted[feature_cols].to_numpy().reshape(len(window_ids), n_sensors, len(feature_cols))
    return raw, window_ids


def flatten_for_classifier(z_features: np.ndarray, window_mask: np.ndarray, is_damaged: np.ndarray,
                            expected_idx: int):
    """(n_windows, n_sensors, n_features) -> (n_windows*n_sensors, n_features)
    rows, with a binary label per row: 1 only for the expected sensor's
    row in a damaged window, 0 everywhere else (all sensors in healthy
    windows, and every non-expected sensor in damaged windows). Sensor
    identity itself is never included as a feature.
    """
    selected = z_features[window_mask]
    n_windows, n_sensors, n_features = selected.shape
    labels = np.zeros((n_windows, n_sensors), dtype=int)
    labels[is_damaged[window_mask], expected_idx] = 1
    return selected.reshape(-1, n_features), labels.reshape(-1)


def main() -> None:
    config = load_config()
    model_cfg = config["model"]
    rng = np.random.default_rng(model_cfg["random_seed"])

    table = pd.read_csv(DATA_DIR / "features_raw.csv")
    feature_cols = [c for c in table.columns if c not in NON_FEATURE_COLUMNS]
    # peak_frequency_hz is excluded deliberately: it only takes 2-4 discrete
    # FFT-bin values per sensor with very low healthy-state variance, so
    # per-sensor z-scoring divides by a near-zero standard deviation and
    # inflates any deviation. Checked directly (see DAY_3.md): including it
    # pushes top-3 to 99.8%, excluding it gives 49.0%. The lower number is
    # reported here as the more trustworthy, less fragile result.
    feature_cols = [c for c in feature_cols if c != "peak_frequency_hz"]
    split = pd.read_csv(MODELS_DIR / "split.csv")

    graph_npz = np.load(MODELS_DIR / "graph.npz")
    sensor_node_ids = graph_npz["sensor_node_ids"]
    n_sensors = len(sensor_node_ids)

    npz = np.load(DATA_DIR / "windows.npz")
    damaged_element_nodes = npz["damaged_element_nodes"]
    geometry = generate_lattice_geometry(config)
    expected_idx = nearest_sensor_by_hops(geometry, sensor_node_ids, damaged_element_nodes)

    damaged_train_instances, damaged_test_instances = split_damaged_instances(
        split, model_cfg["healthy_test_fraction"], rng,
    )
    print(f"Damaged instances: {len(damaged_train_instances)} train, {len(damaged_test_instances)} test "
          f"(a new split, over damaged instances specifically, separate from the healthy split)")

    raw, window_ids = raw_feature_array(table, feature_cols, n_sensors)
    meta = split.set_index("window_id").loc[window_ids]
    is_damaged = (meta["damage_severity"] > 0).to_numpy()

    healthy_train_val = set(split.loc[split["split"].isin(["healthy_train", "healthy_val"]), "instance_id"])
    healthy_test = set(split.loc[(split["split"] == "eval") & (split["damage_severity"] == 0.0), "instance_id"])

    calib_mask = meta["instance_id"].isin(healthy_train_val).to_numpy()
    sensor_mean = raw[calib_mask].mean(axis=0)
    sensor_std = raw[calib_mask].std(axis=0)
    z_features = (raw - sensor_mean) / sensor_std

    train_window_mask = (
        meta["instance_id"].isin(healthy_train_val) | meta["instance_id"].isin(damaged_train_instances)
    ).to_numpy()
    test_window_mask = (
        meta["instance_id"].isin(healthy_test) | meta["instance_id"].isin(damaged_test_instances)
    ).to_numpy()

    x_train, y_train = flatten_for_classifier(z_features, train_window_mask, is_damaged, expected_idx)
    x_test, y_test = flatten_for_classifier(z_features, test_window_mask, is_damaged, expected_idx)
    print(f"Training rows: {x_train.shape[0]} ({y_train.sum()} positive, {y_train.mean():.1%}); "
          f"test rows: {x_test.shape[0]} ({y_test.sum()} positive, {y_test.mean():.1%})")

    x_train_res, y_train_res = SMOTE(random_state=model_cfg["random_seed"]).fit_resample(x_train, y_train)
    print(f"After SMOTE: {x_train_res.shape[0]} rows ({y_train_res.mean():.1%} positive)")

    clf = RandomForestClassifier(n_estimators=200, random_state=model_cfg["random_seed"], n_jobs=-1)
    clf.fit(x_train_res, y_train_res)

    print("\nFeature importance (Random Forest's built-in ranking, a simpler stand-in for "
          "the paper's MRMR):")
    importance = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    for name, score in importance.head(5).items():
        print(f"  {name}: {score:.3f}")

    n_test_windows = test_window_mask.sum()
    proba = clf.predict_proba(x_test)[:, 1].reshape(n_test_windows, n_sensors)
    is_damaged_test = is_damaged[test_window_mask]
    damaged_proba = proba[is_damaged_test]

    k = model_cfg["localization_top_k"]
    top1 = localization_topk_accuracy(damaged_proba, expected_idx, k=1)
    topk = localization_topk_accuracy(damaged_proba, expected_idx, k=k)
    mean_rank = np.mean([localization_rank(row, expected_idx) for row in damaged_proba])
    chance_topk = k / n_sensors
    beats_chance = topk > 2 * chance_topk
    status = "[OK]" if beats_chance else "[FINDING]"
    print(f"\nSupervised Random Forest localisation ({damaged_proba.shape[0]} held-out damaged "
          f"test windows, from damaged instances never seen during training):")
    print(f"  top-1 {top1:.1%}, top-{k} {topk:.1%} (chance {chance_topk:.1%}), "
          f"mean rank {mean_rank:.1f}/{(n_sensors - 1) / 2:.1f} chance")
    print(f"{status} {'Clearly better than chance' if beats_chance else 'NOT clearly better than chance'}")

    fig, _ = plot_localization_heatmap(
        geometry, sensor_node_ids, damaged_proba.mean(axis=0), damaged_element_nodes=damaged_element_nodes,
        title="Mean predicted P(nearest sensor), supervised RF, held-out damaged test instances",
        value_label="P(nearest sensor)",
    )
    out_path = FIGURES_DIR / "day3_localization_heatmap_supervised.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")

    print("\nReminder: trained and tested on the SAME single damage location used throughout this "
          "project (a mid-height diagonal brace). This shows the classifier can find THIS location "
          "better than the unsupervised approaches could, on instances of it never trained on; it "
          "does not show the approach generalises to other damage locations. See DAY_3.md.")


if __name__ == "__main__":
    main()
