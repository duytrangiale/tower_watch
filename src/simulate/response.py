"""Modal-superposition response simulation: given a TrussModel, simulate
the acceleration a set of sensors would record under ambient (wind-like)
excitation. See TowerWatch_guideline.md Sec 5.1.

Each retained mode behaves as an independent damped single-degree-of-freedom
(SDOF) oscillator (this falls out of the mode shapes being M-orthonormal),
driven by the excitation force projected onto that mode. Each mode's
response is integrated exactly, given the sampled input, via a continuous
state-space model (scipy.signal.lsim), then the modal responses are
recombined back into physical nodal accelerations.
"""

import numpy as np
from scipy.signal import StateSpace, butter, filtfilt, lsim

from src.fem.truss import solve_modal


def select_sensor_nodes(n_rings: int, n_levels: int = 6) -> np.ndarray:
    """Pick `n_levels` evenly spaced height rings (excluding the fixed base
    ring) and return the node ids of all 3 legs at each, mirroring LUMO's
    real layout of 18 uniaxial sensors over 6 measurement levels.
    """
    ring_indices = np.unique(np.linspace(1, n_rings - 1, n_levels).astype(int))
    return np.concatenate([ring_idx * 3 + np.arange(3) for ring_idx in ring_indices])


def lateral_free_dofs(model) -> np.ndarray:
    """DOF indices for horizontal (x, y) motion at every free (non-base)
    node; these are the DOFs excited by simulated wind buffeting.
    """
    free_node_ids = np.setdiff1d(np.arange(model.nodes.shape[0]), model.base_node_ids)
    return np.concatenate([3 * free_node_ids + 0, 3 * free_node_ids + 1])


def _band_limited_white_noise(rng: np.random.Generator, n_dof: int, n_samples: int,
                               sampling_rate_hz: float, std_n: float, cutoff_hz: float) -> np.ndarray:
    """Independent Gaussian white noise per DOF, low-pass filtered to
    `cutoff_hz`, standing in for band-limited wind buffeting.
    """
    noise = rng.normal(0.0, std_n, size=(n_dof, n_samples))
    nyquist = sampling_rate_hz / 2.0
    b, a = butter(4, cutoff_hz / nyquist, btype="low")
    return filtfilt(b, a, noise, axis=1)


def _modal_acceleration_response(modal_force: np.ndarray, omega_hz: np.ndarray,
                                  damping_ratio: float, dt: float) -> np.ndarray:
    """Integrate each mode's SDOF equation of motion to get modal
    acceleration q_ddot_i(t), given modal force f_i(t) = phi_i^T @ F(t).

    `modal_force` has shape (n_modes, n_samples). Mode shapes are
    M-orthonormal, so each mode's generalized mass is 1 and the equation of
    motion is q_ddot + 2*zeta*omega*q_dot + omega^2*q = f(t). Returns modal
    acceleration of the same shape as `modal_force`.
    """
    n_modes, n_samples = modal_force.shape
    t = np.arange(n_samples) * dt
    modal_accel = np.empty_like(modal_force)
    for i in range(n_modes):
        omega = 2 * np.pi * omega_hz[i]
        # State x = [q, q_dot]; output y = q_ddot = -omega^2*q - 2*zeta*omega*q_dot + f
        a_mat = [[0.0, 1.0], [-omega**2, -2 * damping_ratio * omega]]
        b_mat = [[0.0], [1.0]]
        c_mat = [[-omega**2, -2 * damping_ratio * omega]]
        d_mat = [[1.0]]
        system = StateSpace(a_mat, b_mat, c_mat, d_mat)
        _, y_out, _ = lsim(system, U=modal_force[i], T=t)
        modal_accel[i] = y_out
    return modal_accel


def simulate_sensor_windows(model, sensor_node_ids: np.ndarray, config: dict,
                             rng: np.random.Generator) -> np.ndarray:
    """Simulate sensor acceleration and slice it into non-overlapping
    windows.

    Returns an array of shape (n_windows, n_sensors, window_length): the
    horizontal (x-direction) acceleration at each sensor node, with
    measurement noise added, matching TowerWatch_guideline.md Sec 5.1's
    output schema and LUMO's real uniaxial accelerometers (Sec 7.2). A
    single linear axis is used deliberately, not the (ax, ay) magnitude:
    that magnitude is a nonlinear function of two oscillatory signals and
    spreads energy into frequencies neither ax nor ay actually contains,
    which would corrupt the spectral features computed in Sec 5.3.
    """
    sim_cfg = config["simulate"]
    n_modes = config["modal"]["n_modes"]
    sampling_rate_hz = sim_cfg["sampling_rate_hz"]
    window_length = sim_cfg["window_length"]
    n_windows = sim_cfg["windows_per_instance"]
    dt = 1.0 / sampling_rate_hz

    burn_in_samples = int(round(sim_cfg["burn_in_seconds"] * sampling_rate_hz))
    n_samples = burn_in_samples + n_windows * window_length

    frequencies_hz, mode_shapes = solve_modal(model, n_modes)

    excited_dofs = lateral_free_dofs(model)
    force = np.zeros((model.n_dof, n_samples))
    force[excited_dofs, :] = _band_limited_white_noise(
        rng, len(excited_dofs), n_samples, sampling_rate_hz,
        sim_cfg["excitation_std_n"], sim_cfg["excitation_cutoff_hz"],
    )

    modal_force = mode_shapes.T @ force  # (n_modes, n_samples)
    modal_accel = _modal_acceleration_response(modal_force, frequencies_hz, sim_cfg["damping_ratio"], dt)
    nodal_accel = mode_shapes @ modal_accel  # (n_dof, n_samples)

    sensor_signal = nodal_accel[3 * sensor_node_ids + 0, :]  # (n_sensors, n_samples), x-axis only
    sensor_signal = sensor_signal[:, burn_in_samples:]  # drop the startup transient

    rms = np.sqrt(np.mean(sensor_signal**2, axis=1, keepdims=True))
    noise = rng.normal(0.0, sim_cfg["measurement_noise_rms_fraction"] * rms, size=sensor_signal.shape)
    sensor_signal = sensor_signal + noise

    return np.stack(np.split(sensor_signal, n_windows, axis=1), axis=0)  # (n_windows, n_sensors, window_length)
