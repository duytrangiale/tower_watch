"""Day 4 follow-up (not part of the main pipeline): visualise the
synthetic model's assumed sensor layout side by side with LUMO's actual
real layout, so the mismatch described in DAY_4.md ("Before downloading
anything: what the README said") is visible directly, not just described
in prose.

Run from the repo root (after 01_generate_synthetic.py, for the config):
    python scripts/day4_sensor_layout_comparison.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.data.lumo import ML_HEIGHTS_M
from src.fem.geometry import generate_lattice_geometry
from src.simulate.response import select_sensor_nodes

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "figures"

LEG_X = {0: 0.0, 1: 1.2, 2: 0.6}  # spread the 3 legs out for a readable 2D layout, matches src/fem/visualize.py


def load_config() -> dict:
    with open(REPO_ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def plot_synthetic_panel(ax, geometry, sensor_node_ids) -> None:
    def local_xy(node_id):
        return LEG_X[node_id % 3], geometry.nodes[node_id, 2]

    # Just the 3 legs as reference lines, not the full braced mesh (already
    # shown in DAY_1.md): the point here is the sensor pattern, and the full
    # X-bracing is dense enough to bury it visually.
    max_height = geometry.nodes[:, 2].max()
    for leg in range(3):
        ax.plot([LEG_X[leg], LEG_X[leg]], [0, max_height], color="lightgray", linewidth=1.2, zorder=1)

    sensor_xy = np.array([local_xy(n) for n in sensor_node_ids])
    ax.scatter(sensor_xy[:, 0], sensor_xy[:, 1], marker="^", s=160, color="tab:green",
               edgecolors="black", linewidths=0.8, zorder=3, label="Sensor (x-axis only)")
    for x, z in sensor_xy:
        ax.annotate("x", (x, z), textcoords="offset points", xytext=(9, -3), fontsize=7, color="tab:green")

    ax.set_xticks(list(LEG_X.values()))
    ax.set_xticklabels([f"leg {k}" for k in LEG_X])
    ax.set_xlabel("leg (spread out for readability)")
    ax.set_title("Synthetic model (assumed)\n3 legs x 6 levels x 1 axis = 18 sensors")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)


def plot_lumo_panel(ax) -> None:
    x = LEG_X[2]  # centred, one physical point per level, not spread across legs
    ax.plot([x, x], [0, 9.1], color="lightgray", linewidth=1.2, zorder=1)

    for ml in range(1, 10):
        z = ML_HEIGHTS_M[ml]
        ax.scatter([x], [z], marker="o", s=200, color="tab:purple",
                   edgecolors="black", linewidths=0.8, zorder=3,
                   label="Sensor (biaxial x+y)" if ml == 1 else None)
        ax.annotate(f"ML{ml}", (x, z), textcoords="offset points", xytext=(10, -3), fontsize=8)
        ax.annotate("x,y", (x, z), textcoords="offset points", xytext=(-28, -3), fontsize=7, color="tab:purple")

    z_base = ML_HEIGHTS_M[10]
    ax.scatter([x], [z_base], marker="s", s=90, color="gray", zorder=3,
               label="ML10: strain + temp only, no accelerometer")
    ax.annotate("ML10", (x, z_base), textcoords="offset points", xytext=(10, -3), fontsize=8, color="gray")

    ax.set_xticks([x])
    ax.set_xticklabels(["one point per level"])
    ax.set_xlim(-0.3, 1.5)
    ax.set_title("LUMO (real)\n1 point x 9 levels x 2 axes = 18 channels")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)


def main() -> None:
    config = load_config()
    geometry = generate_lattice_geometry(config)
    n_rings = geometry.nodes.shape[0] // 3
    sensor_node_ids = select_sensor_nodes(n_rings, config["simulate"]["n_sensor_levels"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 8), sharey=True)
    plot_synthetic_panel(ax1, geometry, sensor_node_ids)
    plot_lumo_panel(ax2)
    ax1.set_ylabel("height (m)")
    ax1.set_ylim(-0.3, 9.5)
    fig.suptitle("Same sensor count, different physical layout: synthetic assumption vs LUMO's real sensors")
    fig.tight_layout()

    FIGURES_DIR.mkdir(exist_ok=True)
    out_path = FIGURES_DIR / "day4_sensor_layout_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
