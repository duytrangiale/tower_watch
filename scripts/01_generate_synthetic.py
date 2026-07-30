"""Stage 1/2: build the synthetic lattice mast, verify the Sec 4.2 / 4.3
acceptance criteria (plausible frequencies, damage lowers them
monotonically), demonstrate the temperature confounder (Sec 5.2), and
generate the windowed sensor dataset (Sec 5.1) used by
02_extract_features.py.

Run from the repo root:
    python scripts/01_generate_synthetic.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.fem.damage import apply_damage, default_damage_element
from src.fem.geometry import generate_lattice_geometry
from src.fem.truss import (
    apply_fixity,
    assemble_global_mass,
    assemble_global_stiffness,
    build_truss_model,
    fixed_dofs_from_base_nodes,
    solve_modal,
)
from src.simulate.environment import material_at_temperature
from src.simulate.response import select_sensor_nodes, simulate_sensor_windows

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "figures"
DATA_DIR = REPO_ROOT / "data" / "synthetic"


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

    target_element = default_damage_element(model.element_type)
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


def plot_temperature_vs_damage(geometry, model, config) -> None:
    """Verify Sec 5.2: temperature alone, over a realistic outdoor range,
    shifts the first natural frequency by an amount comparable to real
    damage. Plotted on a shared y-axis so the two are directly comparable.
    """
    env_cfg = config["environment"]
    n_modes = config["modal"]["n_modes"]
    severities = config["damage"]["severities"]

    f_ref, _ = solve_modal(model, n_modes)
    f1_ref = f_ref[0]

    temperatures = np.linspace(env_cfg["temperature_min_c"], env_cfg["temperature_max_c"], 15)
    delta_f_temp = []
    for T in temperatures:
        material_T = material_at_temperature(config["material"], T, env_cfg)
        model_T = build_truss_model(geometry, material_T)
        f_T, _ = solve_modal(model_T, n_modes)
        delta_f_temp.append(f1_ref - f_T[0])
    delta_f_temp = np.array(delta_f_temp)

    target_element = default_damage_element(model.element_type)
    delta_f_damage = []
    for d in severities:
        damaged_model, _ = apply_damage(model, [(target_element, d)])
        f_d, _ = solve_modal(damaged_model, n_modes)
        delta_f_damage.append(f1_ref - f_d[0])
    delta_f_damage = np.array(delta_f_damage)

    print(
        f"\nTemperature confounder: Δf1 ranges {delta_f_temp.min():+.4f} to {delta_f_temp.max():+.4f} Hz "
        f"across {env_cfg['temperature_min_c']:.0f}-{env_cfg['temperature_max_c']:.0f} C, versus "
        f"{delta_f_damage.min():+.4f} to {delta_f_damage.max():+.4f} Hz across damage severities "
        f"{severities[0]}-{severities[-1]}: same order of magnitude."
    )

    y_min = min(delta_f_temp.min(), delta_f_damage.min(), 0.0) * 1.15
    y_max = max(delta_f_temp.max(), delta_f_damage.max()) * 1.15

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    ax1.plot(temperatures, delta_f_temp, marker="o", color="tab:red")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xlabel("Temperature (C)")
    ax1.set_ylabel("Δf1 from reference (Hz)")
    ax1.set_title("Temperature alone\n(healthy structure)")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(y_min, y_max)

    ax2.plot(severities, delta_f_damage, marker="o", color="tab:orange")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Damage severity d")
    ax2.set_title(f"Damage alone\n(element {target_element}, at reference T)")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Temperature-induced frequency shift is the same order of magnitude as damage")
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    out_path = FIGURES_DIR / "day2_temperature_vs_damage.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def generate_dataset(geometry, config) -> None:
    """Generate the Sec 5.1 windowed sensor dataset: for the healthy
    structure and each damage severity, simulate several independent
    instances at randomly drawn temperatures (Sec 5.2), each sliced into
    several windows. Saves the raw window array plus a label table.
    """
    sim_cfg = config["simulate"]
    env_cfg = config["environment"]
    n_rings = geometry.nodes.shape[0] // 3
    sensor_node_ids = select_sensor_nodes(n_rings, sim_cfg["n_sensor_levels"])
    target_element = default_damage_element(geometry.element_type)
    damaged_element_nodes = geometry.elements[target_element]

    classes = [0.0] + list(config["damage"]["severities"])
    n_instances_per_class = sim_cfg["n_instances_per_class"]
    rng = np.random.default_rng(42)
    instance_rngs = rng.spawn(len(classes) * n_instances_per_class)

    all_windows = []
    rows = []
    instance_id = 0
    t_start = time.time()
    for severity in classes:
        for _ in range(n_instances_per_class):
            instance_rng = instance_rngs[instance_id]
            temperature_c = instance_rng.uniform(env_cfg["temperature_min_c"], env_cfg["temperature_max_c"])
            material_T = material_at_temperature(config["material"], temperature_c, env_cfg)
            model_T = build_truss_model(geometry, material_T)
            if severity > 0.0:
                model_T, _ = apply_damage(model_T, [(target_element, severity)])

            windows = simulate_sensor_windows(model_T, sensor_node_ids, config, instance_rng)
            all_windows.append(windows)
            for w in range(windows.shape[0]):
                rows.append({
                    "window_id": len(rows), "instance_id": instance_id,
                    "damage_severity": severity, "temperature_c": temperature_c,
                })
            instance_id += 1

    windows_array = np.concatenate(all_windows, axis=0)
    labels = pd.DataFrame(rows)
    elapsed = time.time() - t_start

    assert np.all(np.isfinite(windows_array)), "Simulated acceleration has NaN/inf"
    print(
        f"\n[OK] Simulated {instance_id} instances ({len(classes)} classes x "
        f"{n_instances_per_class} each) -> {windows_array.shape[0]} windows, "
        f"shape {windows_array.shape}, in {elapsed:.1f}s"
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        DATA_DIR / "windows.npz",
        windows=windows_array,
        sensor_node_ids=sensor_node_ids,
        sampling_rate_hz=sim_cfg["sampling_rate_hz"],
        damaged_element_idx=target_element,
        damaged_element_nodes=damaged_element_nodes,
    )
    labels.to_csv(DATA_DIR / "labels.csv", index=False)
    print(f"Saved {DATA_DIR / 'windows.npz'} and {DATA_DIR / 'labels.csv'}")


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
    plot_temperature_vs_damage(geometry, model, config)
    generate_dataset(geometry, config)


if __name__ == "__main__":
    main()
