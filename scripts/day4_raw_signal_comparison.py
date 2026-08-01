"""Ad-hoc plot (not part of the main pipeline): what does the raw vibration
signal actually look like, before any windowing or feature extraction?
Plots 3 synthetic sensors at the same height (one per leg) next to 1 real
LUMO sensor at a similar height, for a direct "what does the data look
like" comparison. See DAY_4.md for why the synthetic and LUMO layouts
differ (3 sensors per level vs 1 biaxial sensor per level).

Both signals are now recorded (real) or simulated (synthetic) at the same
rate, LUMO's real 1651.61 Hz, since configs/default.yaml's
simulate.sampling_rate_hz was updated to match it (the original 500 Hz
was only ever a placeholder, see that file's comment). Both are also
converted to the same unit, m/s^2 (LUMO's raw files are calibrated in g,
1 g = 9.80665 m/s^2), so amplitude is now directly comparable, not just
shape.

That comparison turns out to matter: the synthetic model's vibration
amplitude and LUMO's real ambient vibration amplitude are on very
different scales (see the printed ratio when this script runs), because
the synthetic model's excitation force was only ever tuned to land
natural frequencies in a plausible range, not to match real-world
amplitude. The top panel below shows both on the same axis to make that
gap visible directly; the bottom panel zooms in on LUMO alone so its own
shape stays readable.

Run from the repo root (needs data/synthetic/windows.npz from
01_generate_synthetic.py, and one unzipped LUMO healthy block):
    python scripts/day4_raw_signal_comparison.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.lumo import list_blocks, load_block

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "synthetic"
LUMO_DIR = REPO_ROOT / "data" / "lumo" / "exemplary_datasets"
FIGURES_DIR = REPO_ROOT / "figures"

G_TO_MS2 = 9.80665

SYNTHETIC_LEVEL_SENSOR_IDX = [6, 7, 8]  # one level, all 3 legs; height ~3.86 m (see script docstring)
LUMO_LEVEL = 6  # ML6, height 4.00 m, the closest real level to the synthetic one above


def main() -> None:
    npz = np.load(DATA_DIR / "windows.npz")
    windows = npz["windows"]
    sampling_rate_hz = float(npz["sampling_rate_hz"])
    labels = pd.read_csv(DATA_DIR / "labels.csv")

    healthy_window_id = int(labels.loc[labels["damage_severity"] == 0.0, "window_id"].iloc[0])
    synthetic_signal = windows[healthy_window_id][SYNTHETIC_LEVEL_SENSOR_IDX]  # (3, window_length), m/s^2
    synthetic_t = np.arange(synthetic_signal.shape[1]) / sampling_rate_hz
    sensor_height_m = 3.857  # from src/simulate/response.py's select_sensor_nodes, level index 2

    blocks = list_blocks(LUMO_DIR)
    lumo_path = blocks.loc[blocks["is_healthy"]].iloc[0]["path"]
    block = load_block(lumo_path)
    names = list(block["ChannelNames"])
    lumo_fs = float(block["Fs"])
    n_samples = synthetic_signal.shape[1]  # same sample count as the synthetic window, now that rates match
    lumo_x = block["Data"][:n_samples, names.index(f"accel{LUMO_LEVEL:02d}x")] * G_TO_MS2
    lumo_y = block["Data"][:n_samples, names.index(f"accel{LUMO_LEVEL:02d}y")] * G_TO_MS2
    lumo_t = np.arange(n_samples) / lumo_fs

    print(f"Synthetic sampling rate: {sampling_rate_hz:.2f} Hz, LUMO: {lumo_fs:.2f} Hz")
    print(f"Synthetic std: {synthetic_signal.std():.4f} m/s^2, LUMO std (x,y): "
          f"{lumo_x.std():.4f}, {lumo_y.std():.4f} m/s^2")
    print(f"Ratio (synthetic / LUMO x): {synthetic_signal.std() / lumo_x.std():.0f}x")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    for leg, color in zip(range(3), ["tab:blue", "tab:orange", "tab:green"]):
        ax1.plot(synthetic_t, synthetic_signal[leg], color=color, linewidth=0.8, label=f"synthetic leg {leg}")
    ax1.plot(lumo_t, lumo_x, color="tab:purple", linewidth=0.8, label="LUMO x direction")
    ax1.plot(lumo_t, lumo_y, color="tab:brown", linewidth=0.8, label="LUMO y direction")
    ax1.set_ylabel("acceleration (m/s$^2$)")
    ax1.set_xlabel("time (s)")
    ax1.set_title("Same axis, same units: the synthetic tower vibrates far more than LUMO's real one")
    ax1.legend(loc="upper right", fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2.plot(lumo_t, lumo_x, color="tab:purple", linewidth=0.8, label="x direction")
    ax2.plot(lumo_t, lumo_y, color="tab:brown", linewidth=0.8, label="y direction")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("acceleration (m/s$^2$)")
    ax2.set_title(f"LUMO alone, zoomed in: 1 real sensor (both directions), ML{LUMO_LEVEL}, height 4.00 m")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Raw sensor signal at matching rate ({sampling_rate_hz:.0f} Hz) and units (m/s$^2$):\n"
                 f"synthetic (height {sensor_height_m:.2f} m) vs LUMO real data (height 4.00 m)", fontsize=11)
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    out_path = FIGURES_DIR / "day4_raw_signal_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
