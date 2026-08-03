"""Autoregressive with exogenous input (ARX) modelling, an idea from the
SHM literature (Shahidi et al. 2015, cited in Eltouny, Gomaa & Liang
2023, Section 3.2: compared against single-variate regression, collinear
regression, and plain AR models on a scaled steel frame, ARX gave the
best damage-localisation performance of the four).

An ARX model predicts a sensor's current raw reading from its own past
readings (the autoregressive, "AR", part) plus a second, "exogenous"
channel's past readings (the "X" part):

    y(t) = sum_i a_i * y(t-i) + sum_j b_j * x(t-j) + c + e(t)

This project has no measured excitation/force channel (both the
synthetic tower and LUMO are output-only: ambient wind or traffic
loading, never recorded directly), so the exogenous input here is a
second sensor's own signal, a standard adaptation of ARX for output-only
vibration data. See scripts/day3_arx_localization_experiment.py for the
specific reference-sensor choice and reasoning.

Fit by ordinary least squares: for a fixed lag order this is a plain
linear regression, no iterative optimisation needed.
"""

import numpy as np


def build_arx_design(y_windows: np.ndarray, x_windows: np.ndarray | None, p: int, q: int):
    """Build a pooled OLS design matrix and target vector from every
    window's own lagged samples. Lags never cross a window boundary,
    each window is an independent short recording, not one continuous
    series. `y_windows`/`x_windows`: (n_windows, window_length). Pass
    `x_windows=None` for a pure AR model (q is then ignored).

    Columns: y(t-1)..y(t-p), then [x(t)..x(t-q+1) if x given], then a
    constant term. Returns (design, target), each with
    n_windows * (window_length - start) rows.
    """
    n_windows, window_length = y_windows.shape
    start = max(p, (q - 1) if x_windows is not None else 0)
    n_t = window_length - start
    if n_t <= 0:
        raise ValueError(f"window_length {window_length} too short for order p={p}, q={q}")

    target = y_windows[:, start:].reshape(-1)

    cols = [y_windows[:, start - i: window_length - i].reshape(-1) for i in range(1, p + 1)]
    if x_windows is not None:
        cols += [x_windows[:, start - j: window_length - j].reshape(-1) for j in range(q)]
    cols.append(np.ones_like(target))

    design = np.stack(cols, axis=1)
    return design, target


def fit_arx(y_windows: np.ndarray, x_windows: np.ndarray | None, p: int, q: int):
    """Fit one ARX(p, q) model (or AR(p) if x_windows is None) by OLS on
    every row `build_arx_design` produces. Returns (coeffs, aic).
    """
    design, target = build_arx_design(y_windows, x_windows, p, q)
    coeffs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    residuals = target - design @ coeffs
    n = len(target)
    k = design.shape[1]
    rss = float(np.sum(residuals ** 2))
    aic = n * np.log(rss / n) + 2 * k
    return coeffs, aic


def select_arx_order(y_windows: np.ndarray, x_windows: np.ndarray | None,
                      p_grid: list, q_grid: list) -> tuple:
    """Akaike's information criterion (AIC) order selection, named
    explicitly in Eltouny, Gomaa & Liang 2023 (Section 3.2) as a common
    way researchers choose time-series model order. Tries every (p, q)
    in the grid (q_grid ignored if x_windows is None) and returns the
    (p, q, aic) of the best one.
    """
    best = None
    for p in p_grid:
        q_options = q_grid if x_windows is not None else [0]
        for q in q_options:
            _, aic = fit_arx(y_windows, x_windows, p, max(q, 1) if x_windows is not None else 0)
            if best is None or aic < best[2]:
                best = (p, q, aic)
    return best


def arx_residuals(y_windows: np.ndarray, x_windows: np.ndarray | None, coeffs: np.ndarray,
                   p: int, q: int) -> np.ndarray:
    """One-step-ahead prediction residuals for every window, using the
    window's own true past samples at each step (not a multi-step
    forecast). Returns (n_windows, window_length - start); the first
    `start` samples of each window have no prediction (not enough lags
    yet) and are dropped, matching `build_arx_design`.
    """
    n_windows, window_length = y_windows.shape
    start = max(p, (q - 1) if x_windows is not None else 0)
    design, target = build_arx_design(y_windows, x_windows, p, q)
    residuals = target - design @ coeffs
    return residuals.reshape(n_windows, window_length - start)
