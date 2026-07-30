"""Render the synthetic tower's geometry so its shape and bracing pattern
can be inspected visually, rather than only read as numbers.

Run from the repo root:
    python scripts/visualize_geometry.py

By default this also opens interactive windows for the two 3D views (click
and drag to rotate, scroll to zoom). Pass --no-show to only save the PNGs,
e.g. when regenerating figures for the README without wanting to click
through windows.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import yaml

from src.fem.geometry import generate_lattice_geometry
from src.fem.visualize import plot_face_elevation, plot_geometry_3d

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "figures"


def load_config() -> dict:
    with open(REPO_ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true",
                         help="save the PNGs without opening interactive windows")
    args = parser.parse_args()

    config = load_config()
    geometry = generate_lattice_geometry(config)
    FIGURES_DIR.mkdir(exist_ok=True)

    fig_3d, _ = plot_geometry_3d(geometry)
    out_3d = FIGURES_DIR / "tower_geometry_3d.png"
    fig_3d.savefig(out_3d, dpi=150, bbox_inches="tight")
    print(f"Saved {out_3d}")

    # The full tower is 9 m tall but only ~1.2 m across, so at true scale
    # the whole-tower view is very slender. Zoom into the bottom segment
    # (one full 3 m segment, 7 bracing panels) so the triangular
    # cross-section and X-bracing are legible up close.
    segment_height = config["geometry"]["segment_height_m"]
    fig_detail, _ = plot_geometry_3d(
        geometry, z_max=segment_height,
        title="Lattice mast geometry (3D), bottom segment detail",
    )
    out_detail = FIGURES_DIR / "tower_geometry_3d_detail.png"
    fig_detail.savefig(out_detail, dpi=150, bbox_inches="tight")
    print(f"Saved {out_detail}")

    fig_elev, _ = plot_face_elevation(geometry, leg_a=0, leg_b=1)
    out_elev = FIGURES_DIR / "tower_geometry_elevation.png"
    fig_elev.savefig(out_elev, dpi=150, bbox_inches="tight")
    print(f"Saved {out_elev}")

    if not args.no_show:
        print("Opening interactive windows: click and drag to rotate, close them to continue.")
        plt.show()


if __name__ == "__main__":
    main()
