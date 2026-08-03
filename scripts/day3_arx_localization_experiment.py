"""Day 3 follow-up experiment (not part of the main pipeline): does an
autoregressive-with-exogenous-input (ARX) model localise damage better
than the graph autoencoder, following a specific finding from the SHM
literature (Shahidi et al. 2015, cited in Eltouny, Gomaa & Liang 2023,
Section 3.2): comparing single-variate regression, collinear regression,
AR, and ARX models on a scaled steel frame test bed, ARX gave the best
localisation performance of the four, even though all four could detect
damage.

Unlike every other model in this project, ARX operates directly on the
raw windowed acceleration signal (data/synthetic/windows.npz), not on
the extracted statistical/spectral features scripts/02_extract_features.py
produces: AR/ARX is fundamentally a raw-signal time-series technique.

ARX needs a second, "exogenous" input channel. This project's data is
output-only (ambient wind or traffic loading only, no measured
excitation/force channel), a standard adaptation for output-only ARX is
to use a reference sensor's own signal as the exogenous input instead.
The lowest simulated sensor level (closest to the fixed base) is used
here: physically the least likely to be disturbed by damage itself (see
DAY_4.md's DAM6 discussion), which makes it a reasonable stand-in for
"the input the structure is reacting to". Every other sensor's ARX model
uses its own past values plus this reference sensor's past values; the
reference sensor itself gets a plain AR model (no exogenous input for
itself).

One shared (p, q) order is chosen once via Akaike's information
criterion (AIC), named in the review as a standard way to pick
time-series model order, then reused for all 17 non-reference sensors (a
per-sensor order search would be 17x the cost, for structurally
near-identical channels sampled at the same rate).

Reuses the exact same healthy train/validation/eval split
scripts/03_train.py saved, so results are directly comparable to
DAY_3.md's "All localisation methods side by side" table.

Run from the repo root (after 03_train.py):
    python scripts/day3_arx_localization_experiment.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from src.evaluate.metrics import detection_roc_auc, localization_rank, localization_topk_accuracy
from src.fem.geometry import generate_lattice_geometry
from src.graph.build import nearest_sensor_by_hops
from src.models.arx import arx_residuals, fit_arx, select_arx_order

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "synthetic"
MODELS_DIR = REPO_ROOT / "models"

REFERENCE_SENSOR_IDX = 0  # lowest simulated sensor level, closest to the fixed base
# The excitation is low-pass filtered at 60 Hz (configs/default.yaml,
# simulate.excitation_cutoff_hz); at 1651.61 Hz sampling, one full cycle
# at that frequency is ~27.5 samples, so orders much below that would
# structurally be unable to capture the fastest oscillatory content.
# The grid below was widened after an initial [2,4,6,8,10] search picked
# the top of its range in both p and q, a sign it had not yet found an
# interior optimum.
P_GRID = [10, 15, 20, 30, 40, 50]
Q_GRID = [5, 10, 15, 20, 25, 30]
ORDER_SEARCH_N_WINDOWS = 120  # subset of healthy training windows, for a fast AIC search


def load_config() -> dict:
    with open(REPO_ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config()
    model_cfg = config["model"]
    k = model_cfg["localization_top_k"]

    npz = np.load(DATA_DIR / "windows.npz")
    windows = npz["windows"]  # (n_windows, n_sensors, window_length)
    sensor_node_ids = npz["sensor_node_ids"]
    damaged_element_nodes = npz["damaged_element_nodes"]
    n_sensors = windows.shape[1]

    split = pd.read_csv(MODELS_DIR / "split.csv").set_index("window_id")
    train_ids = split.index[split["split"] == "healthy_train"].to_numpy()
    val_ids = split.index[split["split"] == "healthy_val"].to_numpy()
    eval_ids = split.index[split["split"] == "eval"].to_numpy()

    print(f"Reference sensor: index {REFERENCE_SENSOR_IDX} (node {sensor_node_ids[REFERENCE_SENSOR_IDX]}, "
          f"the lowest simulated sensor level)")
    print(f"Training windows: {len(train_ids)}, validation: {len(val_ids)}, eval: {len(eval_ids)}")

    ref_train = windows[train_ids, REFERENCE_SENSOR_IDX, :]
    search_subset = ref_train[:ORDER_SEARCH_N_WINDOWS]
    other_idx = [i for i in range(n_sensors) if i != REFERENCE_SENSOR_IDX]
    example_sensor_train = windows[train_ids, other_idx[0], :][:ORDER_SEARCH_N_WINDOWS]

    p, q, aic = select_arx_order(example_sensor_train, search_subset, P_GRID, Q_GRID)
    print(f"\nAIC-selected order: AR order p={p}, exogenous order q={q} (AIC={aic:.1f}, "
          f"searched on sensor {other_idx[0]} vs the reference, {ORDER_SEARCH_N_WINDOWS} windows)")

    print(f"\nFitting {n_sensors} per-sensor models on {len(train_ids)} healthy training windows...")
    coeffs_by_sensor = {REFERENCE_SENSOR_IDX: fit_arx(ref_train, None, p, 0)[0]}
    for i in other_idx:
        y_train = windows[train_ids, i, :]
        coeffs_by_sensor[i], _ = fit_arx(y_train, ref_train, p, q)
    print("[OK] All per-sensor ARX/AR models fit")

    def per_sensor_error(window_ids: np.ndarray) -> np.ndarray:
        """Mean squared one-step-ahead residual per (window, sensor)."""
        ref = windows[window_ids, REFERENCE_SENSOR_IDX, :]
        errors = np.empty((len(window_ids), n_sensors))
        errors[:, REFERENCE_SENSOR_IDX] = np.mean(
            arx_residuals(ref, None, coeffs_by_sensor[REFERENCE_SENSOR_IDX], p, 0) ** 2, axis=1,
        )
        for i in other_idx:
            y = windows[window_ids, i, :]
            errors[:, i] = np.mean(arx_residuals(y, ref, coeffs_by_sensor[i], p, q) ** 2, axis=1)
        return errors

    print("\nComputing residuals on held-out healthy validation windows (for per-sensor calibration)...")
    val_err = per_sensor_error(val_ids)
    print("Computing residuals on the evaluation set (held-out healthy + all damaged)...")
    eval_err = per_sensor_error(eval_ids)
    sensor_error_mean, sensor_error_std = val_err.mean(axis=0), val_err.std(axis=0)

    is_damaged = (split.loc[eval_ids, "damage_severity"] > 0).to_numpy()
    global_score = eval_err.mean(axis=1)
    auc = detection_roc_auc(is_damaged, global_score)
    print(f"\nDetection ROC-AUC (ARX residual, pooled across all severities): {auc:.3f}")

    geometry = generate_lattice_geometry(config)
    expected_idx = nearest_sensor_by_hops(geometry, sensor_node_ids, damaged_element_nodes)
    print(f"\nLocalisation: expected sensor is node {sensor_node_ids[expected_idx]} "
          f"(nearest to the damaged brace, {n_sensors} sensors total)")

    damaged_err = eval_err[is_damaged]
    zscored = (damaged_err - sensor_error_mean) / sensor_error_std

    for label, err in [("Raw error       ", damaged_err), ("Per-sensor z-score", zscored)]:
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
