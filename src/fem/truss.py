"""3D space-truss finite element assembly and modal analysis.

Standard 2-node bar element, 3 translational DOF per node (truss members
carry axial force only, so no rotational DOF is needed). See
TowerWatch_guideline.md Sec 4.2.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh


@dataclass
class TrussModel:
    nodes: np.ndarray
    elements: np.ndarray
    element_type: np.ndarray
    ea: np.ndarray               # axial stiffness E*A per element [N]
    mass_per_length: np.ndarray  # [kg/m] per element
    base_node_ids: np.ndarray
    n_dof: int


def build_truss_model(geometry, material_config: dict) -> TrussModel:
    """Attach material properties to a LatticeGeometry to form a TrussModel.

    Legs and braces (diagonal/horizontal) get separate cross-sectional
    areas from `material_config`; Young's modulus and density are uniform.
    """
    E = material_config["youngs_modulus_pa"]
    rho = material_config["density_kg_m3"]
    leg_area = material_config["leg_area_m2"]
    brace_area = material_config["brace_area_m2"]

    area = np.where(geometry.element_type == "leg", leg_area, brace_area)
    ea = E * area
    mass_per_length = rho * area

    n_dof = geometry.nodes.shape[0] * 3
    return TrussModel(
        nodes=geometry.nodes,
        elements=geometry.elements,
        element_type=geometry.element_type,
        ea=ea,
        mass_per_length=mass_per_length,
        base_node_ids=geometry.base_node_ids,
        n_dof=n_dof,
    )


def element_length_and_direction(nodes: np.ndarray, elements: np.ndarray):
    """Return per-element length (M,) and unit direction-cosine vectors (M, 3)."""
    p0 = nodes[elements[:, 0]]
    p1 = nodes[elements[:, 1]]
    delta = p1 - p0
    length = np.linalg.norm(delta, axis=1)
    direction = delta / length[:, None]
    return length, direction


def assemble_global_stiffness(model: TrussModel) -> np.ndarray:
    """Assemble the global stiffness matrix K (n_dof x n_dof) by DOF mapping."""
    length, direction = element_length_and_direction(model.nodes, model.elements)
    K = np.zeros((model.n_dof, model.n_dof))
    for e, (i, j) in enumerate(model.elements):
        c = direction[e]
        k_local = (model.ea[e] / length[e]) * np.outer(c, c)  # 3x3
        dofs_i = 3 * i + np.arange(3)
        dofs_j = 3 * j + np.arange(3)
        K[np.ix_(dofs_i, dofs_i)] += k_local
        K[np.ix_(dofs_j, dofs_j)] += k_local
        K[np.ix_(dofs_i, dofs_j)] -= k_local
        K[np.ix_(dofs_j, dofs_i)] -= k_local
    return K


def assemble_global_mass(model: TrussModel) -> np.ndarray:
    """Assemble the lumped global mass matrix M (n_dof x n_dof), diagonal.

    Half of each element's mass is placed on each end node, split equally
    across that node's 3 translational DOFs.
    """
    length, _ = element_length_and_direction(model.nodes, model.elements)
    diag = np.zeros(model.n_dof)
    for e, (i, j) in enumerate(model.elements):
        half_mass = 0.5 * model.mass_per_length[e] * length[e]
        diag[3 * i: 3 * i + 3] += half_mass
        diag[3 * j: 3 * j + 3] += half_mass
    return np.diag(diag)


def fixed_dofs_from_base_nodes(base_node_ids: np.ndarray) -> np.ndarray:
    """DOF indices to constrain: all 3 translational DOF of each base node."""
    return np.concatenate([3 * n + np.arange(3) for n in base_node_ids])


def apply_fixity(K: np.ndarray, M: np.ndarray, fixed_dofs: np.ndarray):
    """Remove fixed DOFs, returning the free-free submatrices and free DOF indices."""
    free_dofs = np.setdiff1d(np.arange(K.shape[0]), fixed_dofs)
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    M_ff = M[np.ix_(free_dofs, free_dofs)]
    return K_ff, M_ff, free_dofs


def modal_analysis(K_ff: np.ndarray, M_ff: np.ndarray, n_modes: int,
                    n_dof_full: int, free_dofs: np.ndarray):
    """Solve the generalized eigenproblem K phi = lambda M phi.

    Returns natural frequencies in Hz (ascending) and mode shapes expanded
    back to the full (unconstrained) DOF space, zero at fixed DOFs.
    scipy.linalg.eigh with `M_ff` as the b-matrix returns eigenvectors that
    are already M-orthonormal (phi.T @ M @ phi = I).
    """
    eigenvalues, eigenvectors = eigh(K_ff, M_ff)
    frequencies_hz = np.sqrt(eigenvalues) / (2 * np.pi)

    n_modes = min(n_modes, len(frequencies_hz))
    frequencies_hz = frequencies_hz[:n_modes]
    mode_shapes = np.zeros((n_dof_full, n_modes))
    mode_shapes[free_dofs, :] = eigenvectors[:, :n_modes]
    return frequencies_hz, mode_shapes


def solve_modal(model: TrussModel, n_modes: int):
    """Convenience wrapper: assemble K, M, apply base fixity, solve modal analysis.

    Returns (frequencies_hz, mode_shapes), with mode_shapes in the full
    (unconstrained) DOF space.
    """
    K = assemble_global_stiffness(model)
    M = assemble_global_mass(model)
    fixed_dofs = fixed_dofs_from_base_nodes(model.base_node_ids)
    K_ff, M_ff, free_dofs = apply_fixity(K, M, fixed_dofs)
    return modal_analysis(K_ff, M_ff, n_modes, model.n_dof, free_dofs)
