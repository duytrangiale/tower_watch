"""Day 1: build the synthetic lattice mast, run modal analysis, and verify
the Sec 4.2 / 4.3 acceptance criteria (plausible frequencies, damage lowers
them monotonically). Response simulation (Day 2) is added in a later pass.

Run from the repo root:
    python scripts/01_generate_synthetic.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.fem.damage import apply_damage
from src.fem.geometry import generate_lattice_geometry
from src.fem.truss import (
    apply_fixity,
    assemble_global_mass,
    assemble_global_stiffness,
    build_truss_model,
    fixed_dofs_from_base_nodes,
    solve_modal,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "figures"


def load_config() -> dict:
    with open(REPO_ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def check_healthy_model(model, config) -> np.ndarray:
    """Verify Sec 4.2 acceptance criteria on the undamaged structure."""
    K = assemble_global_stiffness(model)
    M = assemble_global_mass(model)
    fixed_dofs = fixed_dofs_from_base_nodes(model.base_node_ids)
    K_ff, M_ff, free_dofs = apply_fixity(K, M, fixed_dofs)

    assert np.allclose(K, K.T), "K is not symmetric"
    eigvals_K = np.linalg.eigvalsh(K_ff)
    assert eigvals_K.min() > 0, "K_ff is not positive-definite after constraint removal"

    n_modes = config["modal"]["n_modes"]
    frequencies_hz, mode_shapes = solve_modal(model, n_modes)
    assert np.all(np.isfinite(frequencies_hz)), "Non-finite frequency: unstable assembly"
    assert np.all(frequencies_hz > 0), "Rigid-body mode present (zero or negative frequency)"

    phi_free = mode_shapes[free_dofs, :]
    orthonormality_error = np.max(np.abs(phi_free.T @ M_ff @ phi_free - np.eye(n_modes)))
    assert orthonormality_error < 1e-6, (
        f"Mode shapes not M-orthonormal (max error {orthonormality_error:.2e})"
    )

    print(f"[OK] K symmetric, K_ff positive-definite (min eigenvalue {eigvals_K.min():.3e})")
    print(f"[OK] {n_modes} modes computed, all frequencies real and positive")
    print(f"[OK] Mode shapes M-orthonormal (max error {orthonormality_error:.2e})")
    print(f"First 5 natural frequencies (Hz): {np.round(frequencies_hz[:5], 3)}")

    return frequencies_hz


def check_damage_monotonicity(model, config) -> None:
    """Verify Sec 4.3: damage shifts frequency down, more so with larger severity."""
    severities = config["damage"]["severities"]
    n_modes = config["modal"]["n_modes"]

    # Target a mid-height diagonal rather than an arbitrary brace: near the
    # free top, curvature (and hence strain energy) in the first bending
    # mode is small, so damage there barely shifts the frequency.
    diagonal_idx = np.where(model.element_type == "diagonal")[0]
    target_element = int(diagonal_idx[len(diagonal_idx) // 2])
    node_i, node_j = model.elements[target_element]
    print(
        f"\nDamaging element {target_element} "
        f"({model.element_type[target_element]}, nodes {node_i}-{node_j}) "
        f"at severities {severities}"
    )

    f_healthy, _ = solve_modal(model, n_modes)
    f1_healthy = f_healthy[0]

    delta_f = []
    for d in severities:
        damaged_model, _ = apply_damage(model, [(target_element, d)])
        f_damaged, _ = solve_modal(damaged_model, n_modes)
        delta_f.append(f1_healthy - f_damaged[0])
        print(f"  d={d:.1f}: f1 = {f_damaged[0]:.4f} Hz (shift {delta_f[-1]:.4f} Hz)")

    delta_f = np.array(delta_f)
    assert np.all(np.diff(delta_f) >= -1e-9), "Frequency shift is not monotonic with severity"
    print("[OK] Frequency shift increases monotonically with damage severity")

    plt.figure(figsize=(5, 4))
    plt.plot(severities, delta_f, marker="o")
    plt.xlabel("Damage severity d")
    plt.ylabel("Δf1 (Hz)")
    plt.title(
        f"First-mode frequency shift vs damage severity\n"
        f"(element {target_element}, {model.element_type[target_element]})"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    out_path = FIGURES_DIR / "day1_damage_frequency_shift.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main() -> None:
    config = load_config()
    geometry = generate_lattice_geometry(config)
    model = build_truss_model(geometry, config["material"])

    n_free_dof = model.n_dof - len(model.base_node_ids) * 3
    print(
        f"Geometry: {geometry.nodes.shape[0]} nodes, {geometry.elements.shape[0]} elements "
        f"({model.n_dof} DOF, {n_free_dof} free)"
    )

    check_healthy_model(model, config)
    check_damage_monotonicity(model, config)


if __name__ == "__main__":
    main()
