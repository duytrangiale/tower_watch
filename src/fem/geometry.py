"""Lattice mast geometry generator.

Builds a 3-legged (triangular, isosceles cross-section) lattice tower as a
list of nodes and 2-node truss elements, mirroring LUMO's stated geometry:
9 m height, three 3 m segments, 7 bracing levels per segment. See
TowerWatch_guideline.md Sec 4.1.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class LatticeGeometry:
    nodes: np.ndarray          # (N, 3) node coordinates [m]
    elements: np.ndarray       # (M, 2) node index pairs
    element_type: np.ndarray   # (M,) array of 'leg' | 'diagonal' | 'horizontal'
    base_node_ids: np.ndarray  # node indices fixed at the foundation


def generate_lattice_geometry(config: dict) -> LatticeGeometry:
    """Generate a 3-legged lattice mast from the geometry block of a config dict.

    The footprint is an isosceles triangle: two "back" legs separated by
    `base_width_m`, and a third "apex" leg set forward by `depth_m`. The
    mast height is divided into `n_segments` segments, each further divided
    into `bracing_levels_per_segment` equal panels. Every panel boundary
    gets a horizontal brace ring, and every panel gets one diagonal brace
    per face, alternating direction between panels to form a zigzag
    (Warren-truss) bracing pattern.
    """
    geom_cfg = config["geometry"]
    n_segments = geom_cfg["n_segments"]
    segment_height = geom_cfg["segment_height_m"]
    levels_per_segment = geom_cfg["bracing_levels_per_segment"]
    base_width = geom_cfg["base_width_m"]
    depth = geom_cfg["depth_m"]

    assert np.isclose(n_segments * segment_height, geom_cfg["height_m"]), (
        "height_m must equal n_segments * segment_height_m"
    )

    panel_height = segment_height / levels_per_segment
    n_panels = n_segments * levels_per_segment
    n_rings = n_panels + 1

    # Leg 0 and leg 1 form the base of the isosceles triangle; leg 2 is the
    # apex, set back by `depth`.
    leg_xy = np.array([
        [0.0, 0.0],
        [base_width, 0.0],
        [base_width / 2.0, depth],
    ])

    def node_id(ring_idx, leg_idx):
        return ring_idx * 3 + leg_idx

    ring_heights = np.arange(n_rings) * panel_height
    nodes = np.zeros((n_rings * 3, 3))
    for ring_idx, z in enumerate(ring_heights):
        for leg_idx in range(3):
            nodes[node_id(ring_idx, leg_idx), 0:2] = leg_xy[leg_idx]
            nodes[node_id(ring_idx, leg_idx), 2] = z

    elements = []
    element_type = []

    # Legs: consecutive rings, same leg.
    for leg_idx in range(3):
        for ring_idx in range(n_rings - 1):
            elements.append((node_id(ring_idx, leg_idx), node_id(ring_idx + 1, leg_idx)))
            element_type.append("leg")

    # Horizontals: the 3 members of the triangle at every ring, including
    # the base ring and the top cap.
    faces = [(0, 1), (1, 2), (2, 0)]
    for ring_idx in range(n_rings):
        for (a, b) in faces:
            elements.append((node_id(ring_idx, a), node_id(ring_idx, b)))
            element_type.append("horizontal")

    # Diagonals: X-bracing, i.e. both diagonals per face per panel. A single
    # zigzag diagonal is only just enough to brace each panel; removing it
    # entirely (severity 1.0, matching LUMO's fully-removed-brace state)
    # then leaves that panel with near-zero shear stiffness, which is not
    # representative of a real mast built to survive brace removal for
    # testing. X-bracing keeps one diagonal in place when the other is
    # damaged, so the structure stays well-conditioned at full severity.
    for panel_idx in range(n_panels):
        bottom, top = panel_idx, panel_idx + 1
        for (a, b) in faces:
            elements.append((node_id(bottom, a), node_id(top, b)))
            element_type.append("diagonal")
            elements.append((node_id(bottom, b), node_id(top, a)))
            element_type.append("diagonal")

    base_node_ids = np.array([node_id(0, leg_idx) for leg_idx in range(3)])

    return LatticeGeometry(
        nodes=nodes,
        elements=np.array(elements, dtype=int),
        element_type=np.array(element_type, dtype=object),
        base_node_ids=base_node_ids,
    )
