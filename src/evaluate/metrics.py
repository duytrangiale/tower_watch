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


def false_alarm_rate_at_detection_rate(is_damaged: np.ndarray, scores: np.ndarray,
                                        target_tpr: float) -> float:
    """False positive rate at the lowest threshold that still catches at
    least `target_tpr` of the damaged windows (Sec 8.3: the operating
    point matters more than peak AUC, since a false positive means an
    unnecessary truck roll and climb crew, not a free action). Returns
    1.0 if no threshold reaches `target_tpr` (roc_curve's fpr/tpr are
    already sorted ascending by threshold, ending at (1.0, 1.0), so this
    can only happen if target_tpr > 1.0).
    """
    fpr, tpr, _ = roc_curve_points(is_damaged, scores)
    reaches_target = np.where(tpr >= target_tpr)[0]
    if len(reaches_target) == 0:
        return 1.0
    return float(fpr[reaches_target[0]])


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
