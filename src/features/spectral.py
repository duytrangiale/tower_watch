"""Per-window, per-sensor frequency-domain features via Welch's method for
power spectral density (PSD) estimation. See TowerWatch_guideline.md
Sec 5.3.
"""

import numpy as np
from scipy.signal import welch


def spectral_features(windows: np.ndarray, sampling_rate_hz: float, band_edges_hz: list) -> dict:
    """Compute Welch-PSD band powers, spectral centroid, and dominant peak
    frequency/amplitude for every sensor in every window.

    `windows` has shape (n_windows, n_sensors, window_length).
    `band_edges_hz` is a fixed grid, e.g. [0, 5, 10, 20, 40, 80]; one band
    power feature is produced per consecutive pair of edges. Returns a
    dict of (n_windows, n_sensors) arrays.
    """
    window_length = windows.shape[-1]
    nperseg = min(256, window_length)
    freqs, psd = welch(windows, fs=sampling_rate_hz, nperseg=nperseg, axis=-1)
    # freqs: (n_freq,), psd: (n_windows, n_sensors, n_freq)

    features = {}
    for lo, hi in zip(band_edges_hz[:-1], band_edges_hz[1:]):
        mask = (freqs >= lo) & (freqs < hi)
        features[f"band_power_{lo:g}_{hi:g}hz"] = np.trapezoid(psd[..., mask], freqs[mask], axis=-1)

    total_power = np.trapezoid(psd, freqs, axis=-1)
    features["spectral_centroid_hz"] = np.trapezoid(psd * freqs, freqs, axis=-1) / total_power

    peak_idx = np.argmax(psd, axis=-1)
    features["peak_frequency_hz"] = freqs[peak_idx]
    features["peak_amplitude"] = np.max(psd, axis=-1)

    return features
