"""Render the tower with its 18 simulated sensor locations marked, so the
layout (mirroring LUMO's 18 sensors over 6 measurement levels) can be
inspected visually rather than only as node indices.

Run from the repo root:
    python scripts/visualize_sensors.py

By default this also opens an interactive window for the 3D view (click
and drag to rotate, scroll to zoom). Pass --no-show to only save the PNGs.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import yaml

from src.fem.geometry import generate_lattice_geometry
from src.fem.visualize import plot_face_elevation, plot_geometry_3d
from src.simulate.response import select_sensor_nodes

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "figures"


def load_config() -> dict:
    with open(REPO_ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true",
                         help="save the PNGs without opening an interactive window")
    args = parser.parse_args()

    config = load_config()
    geometry = generate_lattice_geometry(config)
    n_rings = geometry.nodes.shape[0] // 3
    sensor_node_ids = select_sensor_nodes(n_rings, config["simulate"]["n_sensor_levels"])
    print(f"{len(sensor_node_ids)} sensors at node ids: {sensor_node_ids.tolist()}")
    FIGURES_DIR.mkdir(exist_ok=True)

    fig_3d, _ = plot_geometry_3d(
        geometry, sensor_node_ids=sensor_node_ids,
        title="Sensor placement (3D): 18 sensors over 6 levels, mirroring LUMO",
    )
    out_3d = FIGURES_DIR / "sensor_placement_3d.png"
    fig_3d.savefig(out_3d, dpi=150, bbox_inches="tight")
    print(f"Saved {out_3d}")

    fig_elev, _ = plot_face_elevation(
        geometry, leg_a=0, leg_b=1, sensor_node_ids=sensor_node_ids,
        title="Sensor placement, elevation view (face: leg 0 to leg 1)",
    )
    out_elev = FIGURES_DIR / "sensor_placement_elevation.png"
    fig_elev.savefig(out_elev, dpi=150, bbox_inches="tight")
    print(f"Saved {out_elev}")

    if not args.no_show:
        print("Opening interactive window: click and drag to rotate, close it to continue.")
        plt.show()


if __name__ == "__main__":
    main()
