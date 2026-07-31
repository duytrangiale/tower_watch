"""Sensor graph construction from truss connectivity. See
TowerWatch_guideline.md Sec 6.2.
"""

from collections import deque

import numpy as np


def _adjacency_list(elements: np.ndarray, n_nodes: int) -> list:
    adjacency = [[] for _ in range(n_nodes)]
    for i, j in elements:
        adjacency[i].append(int(j))
        adjacency[j].append(int(i))
    return adjacency


def _hop_distances_from(elements: np.ndarray, n_nodes: int,
                         source_ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    """Shortest-path hop distance, via the full truss mesh (not just direct
    single-member links, sensor nodes are rarely directly joined), from
    each of `source_ids` to each of `target_ids`. Breadth-first search
    from each source. Shape (len(source_ids), len(target_ids)).
    """
    adjacency = _adjacency_list(elements, n_nodes)
    dist = np.full((len(source_ids), len(target_ids)), np.inf)
    for a, start in enumerate(source_ids):
        visited = {int(start): 0}
        queue = deque([int(start)])
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                if v not in visited:
                    visited[v] = visited[u] + 1
                    queue.append(v)
        for b, target in enumerate(target_ids):
            if int(target) in visited:
                dist[a, b] = visited[int(target)]
    return dist


def _hop_distance_matrix(elements: np.ndarray, n_nodes: int, node_ids: np.ndarray) -> np.ndarray:
    """Shortest-path hop distance between every pair of nodes in `node_ids`."""
    return _hop_distances_from(elements, n_nodes, node_ids, node_ids)


def nearest_sensor_by_hops(geometry, sensor_node_ids: np.ndarray, target_node_ids: np.ndarray) -> int:
    """Index (into `sensor_node_ids`) of the sensor with the smallest total
    hop distance, through the full truss mesh, to `target_node_ids` (e.g.
    the two end nodes of a damaged brace) -- the sensor structurally
    closest to that location, used for the Sec 6.3 localisation check.
    """
    n_nodes = geometry.nodes.shape[0]
    dist = _hop_distances_from(geometry.elements, n_nodes, target_node_ids, sensor_node_ids)
    return int(np.argmin(dist.sum(axis=0)))


def build_sensor_graph(geometry, sensor_node_ids: np.ndarray, n_distance_levels: int = 2):
    """Build the sensor-level structural graph.

    Nodes are the sensor locations. Two sensors are connected if their
    hop distance through the full truss mesh (not the flattened
    per-window feature order) is one of the `n_distance_levels` smallest
    distances seen from either sensor, e.g. with n_distance_levels=2 that
    is "same measurement level" (distance 1, joined by a horizontal) and
    "next level up/down on the same leg" (the next-smallest distance).
    This is derived from the mesh rather than hardcoded, so it adapts if
    the sensor layout or geometry changes.

    Returns (adjacency, a_hat): `adjacency` is the (n_sensors, n_sensors)
    binary adjacency matrix (no self loops), `a_hat` is the symmetric
    normalised adjacency `D^-1/2 (A+I) D^-1/2` used by the GCN layers.
    """
    n_nodes = geometry.nodes.shape[0]
    dist = _hop_distance_matrix(geometry.elements, n_nodes, sensor_node_ids)
    n_sensors = len(sensor_node_ids)

    adjacency = np.zeros((n_sensors, n_sensors))
    for i in range(n_sensors):
        others = dist[i].copy()
        others[i] = np.inf
        finite = others[np.isfinite(others)]
        keep_distances = np.unique(finite)[:n_distance_levels]
        connect = np.isin(others, keep_distances)
        adjacency[i, connect] = 1.0
    adjacency = np.maximum(adjacency, adjacency.T)  # symmetrize (ties can break this otherwise)

    return adjacency, normalized_adjacency(adjacency)


def normalized_adjacency(adjacency: np.ndarray) -> np.ndarray:
    """The GCN propagation matrix Â = D^(-1/2) (A + I) D^(-1/2)."""
    a_tilde = adjacency + np.eye(adjacency.shape[0])
    degree = a_tilde.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    return d_inv_sqrt @ a_tilde @ d_inv_sqrt
