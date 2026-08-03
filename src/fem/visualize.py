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
                      z_max=None, sensor_node_ids=None):
    """Draw the tower in 3D: every leg, horizontal, and diagonal member, plus
    nodes, with fixed base nodes marked separately from free nodes.

    `z_max`, if given, only draws members and nodes at or below that height
    (metres). The full tower is 9 m tall but only about 1.2 m across, so at
    true scale the whole-tower view is very slender; passing e.g. `z_max=3`
    zooms in on the bottom segment so the triangular cross-section and
    X-bracing are legible up close.

    `sensor_node_ids`, if given, marks those nodes separately (a green
    triangle) instead of as a plain node, e.g. the sensor layout from
    `src.simulate.response.select_sensor_nodes`.
    """
    if ax is None:
        fig = plt.figure(figsize=(7, 9))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    nodes = geometry.nodes
    visible = np.ones(nodes.shape[0], dtype=bool) if z_max is None else nodes[:, 2] <= z_max + 1e-9
    sensor_node_ids = np.asarray(sensor_node_ids) if sensor_node_ids is not None else np.array([], dtype=int)

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
    plain_ids = np.setdiff1d(np.arange(nodes.shape[0]), geometry.base_node_ids)
    plain_ids = np.setdiff1d(plain_ids, sensor_node_ids)
    plain_ids = plain_ids[visible[plain_ids]]
    sensor_ids = sensor_node_ids[visible[sensor_node_ids]] if sensor_node_ids.size else sensor_node_ids

    ax.scatter(*nodes[plain_ids].T, color="tab:blue", s=12, depthshade=False, label="Node")
    ax.scatter(*nodes[base_ids].T, color="red", s=45, marker="s", depthshade=False,
               label="Fixed base node")
    if sensor_ids.size:
        ax.scatter(*nodes[sensor_ids].T, color="tab:green", s=60, marker="^", depthshade=False,
                   label="Sensor")

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


def plot_face_elevation(geometry, leg_a=0, leg_b=1, ax=None, title=None, sensor_node_ids=None):
    """Flat 2D elevation of a single face: the two legs `leg_a`/`leg_b` and
    every horizontal and diagonal connecting them, drawn like a standard
    structural elevation drawing (no depth, so nothing from the other two
    faces overlaps it). The bracing pattern is identical on all three
    faces, so any one face is representative of the whole tower.

    `sensor_node_ids`, if given, marks any of this face's nodes that are
    sensors (a green triangle), e.g. the sensor layout from
    `src.simulate.response.select_sensor_nodes`.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 9))
    else:
        fig = ax.get_figure()

    nodes = geometry.nodes
    face_width = np.linalg.norm(nodes[leg_b, :2] - nodes[leg_a, :2])
    sensor_node_ids = set(sensor_node_ids.tolist()) if sensor_node_ids is not None else set()

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

    face_sensor_ids = [n for n in sensor_node_ids if n % 3 in (leg_a, leg_b)]
    if face_sensor_ids:
        xs = [local_x(n) for n in face_sensor_ids]
        zs = [nodes[n, 2] for n in face_sensor_ids]
        ax.scatter(xs, zs, color="tab:green", s=90, marker="^", zorder=5, label="Sensor")

    ax.set_xlabel(f"distance across face (m): leg {leg_a} to leg {leg_b}")
    ax.set_ylabel("height z (m)")
    ax.set_title(title or f"Elevation view: face between leg {leg_a} and leg {leg_b}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    return fig, ax


def plot_sensor_graph(geometry, sensor_node_ids, adjacency, ax=None,
                       title="Sensor graph used by the GCN"):
    """2D elevation-style plot of the sensor-to-sensor graph the GCN
    actually operates on (src/graph/build.py's `build_sensor_graph`),
    distinct from the tower's own structural members: two sensors can be
    graph-connected without being joined by a single physical member, if
    they are close in hop distance through the mesh. The tower's members
    are drawn faintly in the background for context, exactly as in
    `plot_localization_heatmap`, so the two graphs (structural, sensor)
    can be compared directly. See TowerWatch_guideline.md Sec 8.2 (figure
    1: tower geometry with sensor nodes and graph edges overlaid).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 9))
    else:
        fig = ax.get_figure()

    nodes = geometry.nodes
    leg_x = {0: 0.0, 1: 1.2, 2: 0.6}  # spread the 3 legs out for a readable 2D layout

    def local_xy(node_id):
        return leg_x[node_id % 3], nodes[node_id, 2]

    for (i, j), etype in zip(geometry.elements, geometry.element_type):
        (x0, z0), (x1, z1) = local_xy(i), local_xy(j)
        ax.plot([x0, x1], [z0, z1], color="lightgray", linewidth=0.8, zorder=1)

    n_sensors = len(sensor_node_ids)
    drawn_label = False
    for i in range(n_sensors):
        for j in range(i + 1, n_sensors):
            if adjacency[i, j] == 0:
                continue
            (x0, z0) = local_xy(sensor_node_ids[i])
            (x1, z1) = local_xy(sensor_node_ids[j])
            ax.plot([x0, x1], [z0, z1], color="tab:red", linewidth=1.8, alpha=0.8, zorder=2,
                     label="Sensor graph edge" if not drawn_label else None)
            drawn_label = True

    sensor_xy = np.array([local_xy(n) for n in sensor_node_ids])
    ax.scatter(sensor_xy[:, 0], sensor_xy[:, 1], color="tab:green", marker="^", s=140,
               edgecolors="black", linewidths=0.8, zorder=3, label="Sensor")

    ax.set_xlabel("leg (spread out for readability, not true x position)")
    ax.set_ylabel("height z (m)")
    ax.set_title(title)
    ax.set_xticks(list(leg_x.values()))
    ax.set_xticklabels([f"leg {k}" for k in leg_x])
    ax.legend(loc="upper right", fontsize=8)
    return fig, ax


def plot_localization_heatmap(geometry, sensor_node_ids, values, damaged_element_nodes=None,
                               ax=None, title="Per-sensor reconstruction error", value_label="error"):
    """2D elevation-style heatmap: the tower's members drawn faintly in the
    background, sensor nodes coloured by `values` (e.g. reconstruction
    error), and the actual damaged brace marked, so predicted and true
    damage location can be compared visually. See
    TowerWatch_guideline.md Sec 8.2 (figure 5: localisation heat map).

    Projects every node onto (leg index position, height) rather than a
    single face, so all 18 sensors are visible at once, spread out
    left-to-right by leg.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 9))
    else:
        fig = ax.get_figure()

    nodes = geometry.nodes
    leg_x = {0: 0.0, 1: 1.2, 2: 0.6}  # spread the 3 legs out for a readable 2D layout

    def local_xy(node_id):
        return leg_x[node_id % 3], nodes[node_id, 2]

    for (i, j), etype in zip(geometry.elements, geometry.element_type):
        (x0, z0), (x1, z1) = local_xy(i), local_xy(j)
        ax.plot([x0, x1], [z0, z1], color="lightgray", linewidth=0.8, zorder=1)

    sensor_xy = np.array([local_xy(n) for n in sensor_node_ids])
    scatter = ax.scatter(sensor_xy[:, 0], sensor_xy[:, 1], c=values, cmap="viridis", s=220,
                          edgecolors="black", linewidths=0.8, zorder=3)
    fig.colorbar(scatter, ax=ax, label=value_label, shrink=0.7)

    if damaged_element_nodes is not None:
        damage_xy = np.mean([local_xy(n) for n in damaged_element_nodes], axis=0)
        ax.scatter(*damage_xy, marker="x", color="red", s=200, linewidths=3, zorder=4,
                   label="Actual damage location")
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("leg (spread out for readability, not true x position)")
    ax.set_ylabel("height z (m)")
    ax.set_title(title)
    ax.set_xticks(list(leg_x.values()))
    ax.set_xticklabels([f"leg {k}" for k in leg_x])
    return fig, ax
