"""3D and elevation visualization of the lattice mast geometry.

Draws nodes and members (legs, horizontals, diagonals) so the tower's
shape and bracing pattern can be inspected visually. See
TowerWatch_guideline.md Sec 8.2 (figure 1: tower geometry).
"""

import matplotlib.pyplot as plt
import numpy as np

ELEMENT_STYLE = {
    "leg": dict(color="black", linewidth=2.5, label="Leg"),
    "horizontal": dict(color="tab:blue", linewidth=1.2, label="Horizontal"),
    "diagonal": dict(color="tab:orange", linewidth=1.0, label="Diagonal (X-brace)"),
}


def plot_geometry_3d(geometry, ax=None, elev=22, azim=-35, title="Lattice mast geometry (3D)",
                      z_max=None):
    """Draw the tower in 3D: every leg, horizontal, and diagonal member, plus
    nodes, with fixed base nodes marked separately from free nodes.

    `z_max`, if given, only draws members and nodes at or below that height
    (metres). The full tower is 9 m tall but only about 1.2 m across, so at
    true scale the whole-tower view is very slender; passing e.g. `z_max=3`
    zooms in on the bottom segment so the triangular cross-section and
    X-bracing are legible up close.
    """
    if ax is None:
        fig = plt.figure(figsize=(7, 9))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    nodes = geometry.nodes
    visible = np.ones(nodes.shape[0], dtype=bool) if z_max is None else nodes[:, 2] <= z_max + 1e-9

    drawn_labels = set()
    for (i, j), etype in zip(geometry.elements, geometry.element_type):
        if not (visible[i] and visible[j]):
            continue
        style = ELEMENT_STYLE[etype]
        label = style["label"] if style["label"] not in drawn_labels else None
        drawn_labels.add(style["label"])
        p0, p1 = nodes[i], nodes[j]
        ax.plot(
            [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
            color=style["color"], linewidth=style["linewidth"], label=label,
        )

    base_ids = geometry.base_node_ids[visible[geometry.base_node_ids]]
    free_ids = np.setdiff1d(np.arange(nodes.shape[0]), geometry.base_node_ids)
    free_ids = free_ids[visible[free_ids]]
    ax.scatter(*nodes[free_ids].T, color="tab:blue", s=12, depthshade=False, label="Node")
    ax.scatter(*nodes[base_ids].T, color="red", s=45, marker="s", depthshade=False,
               label="Fixed base node")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m), height")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    _set_equal_3d_aspect(ax, nodes[visible])
    ax.legend(loc="upper left", fontsize=8)
    return fig, ax


def _set_equal_3d_aspect(ax, nodes):
    """Force equal x/y/z scaling so the tall, narrow tower isn't squashed
    or stretched by matplotlib's default per-axis scaling.
    """
    ranges = nodes.max(axis=0) - nodes.min(axis=0)
    centers = (nodes.max(axis=0) + nodes.min(axis=0)) / 2
    max_range = ranges.max() / 2
    ax.set_xlim(centers[0] - max_range, centers[0] + max_range)
    ax.set_ylim(centers[1] - max_range, centers[1] + max_range)
    ax.set_zlim(centers[2] - max_range, centers[2] + max_range)
    ax.set_box_aspect([1, 1, 1])


def plot_face_elevation(geometry, leg_a=0, leg_b=1, ax=None, title=None):
    """Flat 2D elevation of a single face: the two legs `leg_a`/`leg_b` and
    every horizontal and diagonal connecting them, drawn like a standard
    structural elevation drawing (no depth, so nothing from the other two
    faces overlaps it). The bracing pattern is identical on all three
    faces, so any one face is representative of the whole tower.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 9))
    else:
        fig = ax.get_figure()

    nodes = geometry.nodes
    face_width = np.linalg.norm(nodes[leg_b, :2] - nodes[leg_a, :2])

    def local_x(node_id):
        return 0.0 if node_id % 3 == leg_a else face_width

    drawn_labels = set()
    for (i, j), etype in zip(geometry.elements, geometry.element_type):
        leg_i, leg_j = i % 3, j % 3
        if leg_i not in (leg_a, leg_b) or leg_j not in (leg_a, leg_b):
            continue
        style = ELEMENT_STYLE[etype]
        label = style["label"] if style["label"] not in drawn_labels else None
        drawn_labels.add(style["label"])
        ax.plot(
            [local_x(i), local_x(j)], [nodes[i, 2], nodes[j, 2]],
            color=style["color"], linewidth=style["linewidth"], label=label,
        )

    ax.set_xlabel(f"distance across face (m): leg {leg_a} to leg {leg_b}")
    ax.set_ylabel("height z (m)")
    ax.set_title(title or f"Elevation view: face between leg {leg_a} and leg {leg_b}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    return fig, ax
