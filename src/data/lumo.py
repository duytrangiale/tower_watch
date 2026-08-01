"""LUMO real-data adapter (Sec 7.2). Reformats LUMO's raw .mat measurement
blocks into the same (n_windows, n_sensors, window_length) schema the
synthetic pipeline already emits (src/simulate/response.py), so the
existing feature extraction, graph autoencoder, and evaluation code runs
unchanged on real data. See DAY_4.md for how this reconciles LUMO's real
sensor layout with the synthetic model's assumptions.

Source: Wernitz, Hofmeister, Jonscher, Grießmann, Rolfes (2021), "LUMO -
Leibniz University Test Structure for Monitoring", LUIS,
https://doi.org/10.25835/0027803, CC-BY 3.0.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

# From the README (Table 1): 9 accelerometer measurement levels (ML1 at
# the top, 8.95 m, down to ML9 at 1.00 m), each with ONE biaxial sensor
# (x and y direction) -- not one sensor per leg at each level, unlike the
# synthetic model's assumption (see DAY_4.md, "What LUMO's real layout
# turned out to be"). ML10 (the base, 0.15 m) carries strain gauges and
# the temperature sensor only, no accelerometer, so it is not one of the
# 18 channels used as graph nodes here.
ML_HEIGHTS_M = {
    1: 8.95, 2: 8.00, 3: 7.00, 4: 5.95, 5: 5.00,
    6: 4.00, 7: 2.95, 8: 2.00, 9: 1.00, 10: 0.15,
}
N_LEVELS = 9

# Ordered top (ML1) to bottom (ML9); index i = level (i // 2) + 1, axis
# "x" if i is even else "y". This ordering is the sensor-node order used
# everywhere below (graph, windows, features).
ACCEL_CHANNELS = [f"accel{level:02d}{axis}" for level in range(1, N_LEVELS + 1) for axis in ("x", "y")]
TEMPERATURE_CHANNEL = "temp01"

# Best-effort mapping from LUMO's 6 damage positions (DAM1-DAM6) to the
# pair of measurement levels each sits between, read from the README's
# Figure 1/2. The healthy FE model (lumo_fem_healthy.inp) confirms exact
# sensor node heights (matches Table 1 exactly) but has no explicit
# DAM-labelled node or element sets, so the DAM-to-level pairing below is
# a figure reading, not an independently verified label -- flagged in
# DAY_4.md.
DAM_POSITION_MLS = {
    1: (1, 2), 2: (3, 4), 3: (5, 6), 4: (6, 7), 5: (8, 9), 6: (9, 10),
}


def list_blocks(root_dir) -> pd.DataFrame:
    """Walk one unzipped LUMO exemplary-dataset directory (e.g.
    exemplary_datasets_dam3_111/), which contains subfolders like
    05_Healthy/ and 06_DAM3_111/, each holding several 10-minute .mat
    files. The folder name is the ground-truth label for every file
    inside it for these pre-sorted exemplary subsets -- LUMO's general
    metafile/state-label scheme (README Sec 5.3) is for the full,
    unsorted monthly archive, not needed here.
    """
    root_dir = Path(root_dir)
    rows = []
    for state_dir in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        is_healthy = "healthy" in state_dir.name.lower()
        dam_match = re.search(r"DAM(\d)", state_dir.name, re.IGNORECASE)
        dam_position = int(dam_match.group(1)) if dam_match else None
        for mat_path in sorted(state_dir.glob("*.mat")):
            rows.append({
                "path": str(mat_path), "state_dir": state_dir.name,
                "is_healthy": is_healthy, "dam_position": dam_position,
            })
    return pd.DataFrame(rows)


def load_block(path) -> dict:
    """Load one 10-minute SHMTS_<timestamp>.mat measurement block."""
    mat = scipy.io.loadmat(path, simplify_cells=True)
    return mat["Dat"]


def block_to_windows(block: dict, window_length: int) -> np.ndarray:
    """One 10-minute block's 18 acceleration channels -> non-overlapping
    windows, shape (n_windows, 18, window_length). Real continuous sensor
    data needs no burn-in trim (that was only for letting the synthetic
    simulation's modal startup transient decay).
    """
    names = list(block["ChannelNames"])
    col_idx = [names.index(c) for c in ACCEL_CHANNELS]
    accel = block["Data"][:, col_idx].T  # (18, n_samples), float32 as stored
    n_windows = accel.shape[1] // window_length
    trimmed = accel[:, :n_windows * window_length]
    return np.stack(np.split(trimmed, n_windows, axis=1), axis=0)  # (n_windows, 18, window_length)


def block_mean_temperature(block: dict) -> float:
    names = list(block["ChannelNames"])
    return float(block["Data"][:, names.index(TEMPERATURE_CHANNEL)].mean())


def build_dataset(block_paths: pd.DataFrame, window_length: int):
    """block_paths: the concatenation of list_blocks(...) over every
    downloaded exemplary directory. Returns (windows_array, meta):
    windows_array is (n_windows, 18, window_length) float32; meta is one
    row per window with block_id (the source 10-minute file, the
    splitting unit -- windows from the same block are consecutive seconds
    of the same continuous recording, not independent draws, so they must
    never be split across train/val/test the way individual windows
    would be), is_healthy, dam_position, temperature_c, source file.

    Not used by scripts/05_lumo_transfer.py directly (see
    build_feature_table below, which streams block-by-block instead of
    holding every block's raw signal in memory at once); kept for
    ad-hoc inspection of the raw windows, e.g. in a notebook.
    """
    all_windows = []
    rows = []
    window_id = 0
    for block_id, row in enumerate(block_paths.itertuples()):
        block = load_block(row.path)
        windows = block_to_windows(block, window_length)
        temperature_c = block_mean_temperature(block)
        all_windows.append(windows)
        for _ in range(windows.shape[0]):
            rows.append({
                "window_id": window_id, "block_id": block_id,
                "is_healthy": row.is_healthy, "dam_position": row.dam_position,
                "temperature_c": temperature_c, "source_path": row.path,
            })
            window_id += 1
    windows_array = np.concatenate(all_windows, axis=0)
    meta = pd.DataFrame(rows)
    return windows_array, meta


def build_feature_table(block_paths: pd.DataFrame, window_length: int, sampling_rate_hz: float,
                         band_edges_hz: list) -> pd.DataFrame:
    """Stream through each 10-minute block, window it, and compute
    features immediately, rather than holding every block's raw
    (n_windows, 18, window_length) signal in memory at once: the raw
    signal for the full exemplary dataset is several GB, the resulting
    feature table is a few tens of MB. Reuses the exact same feature
    functions the synthetic pipeline uses (src/features/), the entire
    point of the adapter.
    """
    from src.features.spectral import spectral_features
    from src.features.windows import time_domain_features

    tables = []
    window_id = 0
    for block_id, row in enumerate(block_paths.itertuples()):
        block = load_block(row.path)
        windows = block_to_windows(block, window_length)
        temperature_c = block_mean_temperature(block)

        time_feats = time_domain_features(windows)
        freq_feats = spectral_features(windows, sampling_rate_hz, band_edges_hz)
        all_feats = {**time_feats, **freq_feats}

        n_windows, n_sensors = windows.shape[0], windows.shape[1]
        block_table = pd.DataFrame({
            "window_id": np.repeat(np.arange(window_id, window_id + n_windows), n_sensors),
            "sensor_idx": np.tile(np.arange(n_sensors), n_windows),
            "block_id": block_id,
            "is_healthy": row.is_healthy,
            "dam_position": row.dam_position,
            "temperature_c": temperature_c,
        })
        for name, arr in all_feats.items():
            block_table[name] = arr.ravel()
        tables.append(block_table)
        window_id += n_windows

    return pd.concat(tables, ignore_index=True)


def lumo_sensor_graph():
    """Sensor graph for LUMO's 18 acceleration channels. Unlike the
    synthetic tower -- one sensor per leg per level, connected through a
    full 3-leg braced mesh -- LUMO's real accelerometers sit at a single
    point per level (see DAY_4.md), so a full hop-distance mesh graph
    (src/graph/build.py) does not apply. The natural graph here is a
    chain by height: each channel connects to its same-level partner (the
    other axis, physically co-located) and to the corresponding channel
    one level above and below.

    Returns (adjacency, a_hat) in the same format as
    src/graph/build.py's build_sensor_graph, so it plugs into the same
    GCN/message-passing code unchanged.
    """
    from src.graph.build import normalized_adjacency

    n = len(ACCEL_CHANNELS)
    adjacency = np.zeros((n, n))
    for level in range(N_LEVELS):
        x_idx, y_idx = 2 * level, 2 * level + 1
        adjacency[x_idx, y_idx] = adjacency[y_idx, x_idx] = 1.0
        if level > 0:
            prev_x, prev_y = 2 * (level - 1), 2 * (level - 1) + 1
            for a in (x_idx, y_idx):
                for b in (prev_x, prev_y):
                    adjacency[a, b] = adjacency[b, a] = 1.0
    return adjacency, normalized_adjacency(adjacency)


def nearest_levels_for_dam(dam_position: int) -> tuple:
    """The 0-indexed level pair (into the 9 measurement levels ML1-ML9,
    NOT the 18-channel sensor list) that a given DAM position sits
    between. ML10 (dam position 6's lower neighbour) has no accelerometer,
    so for DAM6 only the single accelerometer level (ML9) is meaningful.
    """
    ml_a, ml_b = DAM_POSITION_MLS[dam_position]
    levels = [ml - 1 for ml in (ml_a, ml_b) if ml <= N_LEVELS]
    return tuple(levels)
