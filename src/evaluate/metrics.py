"""Detection and localisation metrics. See TowerWatch_guideline.md Sec 6.3.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def detection_roc_auc(is_damaged: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC for detection: does a higher anomaly score correspond to
    actually-damaged windows? `is_damaged` is a boolean array, `scores` is
    the corresponding anomaly score (higher = more anomalous).
    """
    return roc_auc_score(is_damaged, scores)


def roc_curve_points(is_damaged: np.ndarray, scores: np.ndarray):
    """(false_positive_rate, true_positive_rate, thresholds) for plotting."""
    return roc_curve(is_damaged, scores)


def localization_rank(per_node_error: np.ndarray, expected_node_idx: int) -> int:
    """Where the expected (nearest-to-damage) sensor ranks among all nodes
    sorted by descending reconstruction error, for one window. Rank 0
    means it has the single highest error (perfect localisation); rank
    n_nodes - 1 means it had the lowest error (worst possible).
    """
    order = np.argsort(-per_node_error)
    return int(np.where(order == expected_node_idx)[0][0])


def localization_topk_accuracy(per_node_errors: np.ndarray, expected_node_idx: int, k: int) -> float:
    """Fraction of windows (rows of `per_node_errors`, shape (n_windows,
    n_nodes)) where the expected node's error is among the top `k` highest.
    """
    ranks = np.array([localization_rank(row, expected_node_idx) for row in per_node_errors])
    return float(np.mean(ranks < k))
