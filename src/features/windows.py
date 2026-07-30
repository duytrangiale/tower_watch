"""Per-window, per-sensor time-domain statistical features. See
TowerWatch_guideline.md Sec 5.3.
"""

import numpy as np
from scipy.stats import kurtosis


def time_domain_features(windows: np.ndarray) -> dict:
    """Compute RMS, variance, kurtosis, crest factor, and peak-to-peak for
    every sensor in every window.

    `windows` has shape (n_windows, n_sensors, window_length). Returns a
    dict of (n_windows, n_sensors) arrays, one per feature.
    """
    rms = np.sqrt(np.mean(windows**2, axis=-1))
    peak = np.max(np.abs(windows), axis=-1)
    return {
        "rms": rms,
        "variance": np.var(windows, axis=-1),
        "kurtosis": kurtosis(windows, axis=-1, fisher=False),
        "crest_factor": peak / rms,
        "peak_to_peak": np.max(windows, axis=-1) - np.min(windows, axis=-1),
    }
