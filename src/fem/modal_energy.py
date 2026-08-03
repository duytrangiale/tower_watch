"""Physics-informed sensor sensitivity from modal strain energy, an idea
from the classical structural-dynamics damage-localisation literature
(Pandey, Biswas & Samman 1991, "Damage detection from changes in
curvature mode shapes", Journal of Sound and Vibration; Stubbs & Kim
1996). This is part of what TowerWatch_guideline.md's source review
(Eltouny, Gomaa & Liang 2023) calls "model-based SHM" in its
introduction, the physics-driven sibling of the data-driven,
machine-learning methods that make up the rest of this project.

The idea: a structural element's contribution to a given mode shape's
total strain energy measures how much that element's own stiffness
matters to that mode. An element carrying a lot of strain energy is one
whose local stiffness change (damage) would be expected to disturb the
structure's vibration the most; an element carrying almost none would
barely be felt anywhere, healthy or damaged. This is a fixed property of
the structure and its mode shapes, computed once from the known, healthy
finite-element model, never from observed vibration data.

Used as a physics-derived weight, combined with the existing data-driven
per-sensor z-score, in
scripts/day3_physics_informed_localization_experiment.py.
"""

import numpy as np

from src.fem.truss import element_length_and_direction


def element_strain_energy(model, mode_shapes: np.ndarray) -> np.ndarray:
    """Each truss element's contribution to each mode's strain energy.

    For a 2-node axial bar with stiffness EA/L along unit direction c,
    the strain energy stored under a displacement field u is
    0.5 * (EA/L) * (c . (u_i - u_j))^2: the same k_local = (EA/L) *
    outer(c, c) already used to assemble the global stiffness matrix
    (src/fem/truss.py's assemble_global_stiffness), evaluated as a scalar
    energy instead of assembled into a matrix.

    `mode_shapes`: (n_dof, n_modes), as returned by solve_modal.
    Returns (n_elements, n_modes).
    """
    length, direction = element_length_and_direction(model.nodes, model.elements)
    i_nodes, j_nodes = model.elements[:, 0], model.elements[:, 1]

    dof_i = 3 * i_nodes[:, None] + np.arange(3)[None, :]  # (n_elements, 3)
    dof_j = 3 * j_nodes[:, None] + np.arange(3)[None, :]
    u_i = mode_shapes[dof_i, :]  # (n_elements, 3, n_modes)
    u_j = mode_shapes[dof_j, :]

    axial_disp = np.einsum("ed,edm->em", direction, u_i - u_j)  # (n_elements, n_modes)
    stiffness = model.ea / length  # (n_elements,)
    return 0.5 * stiffness[:, None] * axial_disp ** 2


def sensor_modal_sensitivity(model, mode_shapes: np.ndarray, sensor_node_ids: np.ndarray) -> np.ndarray:
    """Per-sensor physics sensitivity: total strain energy, summed over
    every retained mode, of every element touching that sensor's node.
    A purely physics-derived quantity, the same for every window and
    every instance, computed once from the healthy model.
    """
    energy = element_strain_energy(model, mode_shapes)  # (n_elements, n_modes)
    total_energy_per_element = energy.sum(axis=1)  # (n_elements,)

    sensitivity = np.zeros(len(sensor_node_ids))
    for s, node_id in enumerate(sensor_node_ids):
        incident = np.any(model.elements == node_id, axis=1)
        sensitivity[s] = total_energy_per_element[incident].sum()
    return sensitivity
