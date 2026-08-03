"""Stage 5: maintenance-priority ranking stub (TowerWatch_guideline.md
Sec 8.0). Combines an anomaly score, a severity trend, and a naive cost
line into a single ranked "what needs attention first" table.
Deliberately small: a stand-in demonstrating the shape of a Milestone 2
problem (turning a detection signal into a maintenance schedule), not a
solved instance of it. No real cost data, no scheduling optimisation,
see README.md's "Operational framing" section for the explicit scope
note.
"""

import numpy as np
import pandas as pd


def severity_trend(window_scores: np.ndarray) -> float:
    """Slope of anomaly score across a sequence of windows, ordered
    oldest to newest, from a simple linear fit against window position.
    Positive means the score is rising, i.e. getting worse, right now. A
    single window (no history within this instance to compare against)
    has no trend to measure, so returns 0.0.
    """
    n = len(window_scores)
    if n < 2:
        return 0.0
    slope, _ = np.polyfit(np.arange(n), window_scores, 1)
    return float(slope)


def priority_score(current_anomaly_score: np.ndarray, severity_trend: np.ndarray) -> np.ndarray:
    """Sec 8.0's suggested combination: trend x current level. An
    instance with a high but stable anomaly score scores lower than one
    climbing quickly, on the reasoning that a fast-worsening signal is
    the more urgent thing to schedule an inspection for. Deliberately
    naive, not calibrated against any real degradation model.
    """
    return severity_trend * current_anomaly_score


def inspection_cost_aud(height_m: np.ndarray, base_cost_aud: float, cost_per_metre_aud: float) -> np.ndarray:
    """Placeholder cost of sending a crew to inspect the flagged
    location: a flat call-out cost plus an amount that grows with how
    high up the tower the crew has to climb (Sec 8.3: a truck roll and
    climb crew are costly and, given tower climbing's safety record, not
    risk-free). Arbitrary units, not real quoted prices.
    """
    return base_cost_aud + cost_per_metre_aud * height_m


def risk_of_delay_aud(priority: np.ndarray, risk_scale_aud: float) -> np.ndarray:
    """Placeholder cost of NOT inspecting now: the priority score
    converted to a dollar figure by one arbitrary scale constant, just
    enough to state the inspect-now-vs-wait tradeoff explicitly (Sec
    8.0). Not a calibrated estimate of real risk.
    """
    return priority * risk_scale_aud


def build_priority_table(records: pd.DataFrame, base_cost_aud: float, cost_per_metre_aud: float,
                          risk_scale_aud: float) -> pd.DataFrame:
    """records: one row per already-evaluated tower/damage instance, with
    columns "current_anomaly_score", "severity_trend", and
    "localised_height_m" (height of the sensor with the highest per-sensor
    error, used for the climb-cost placeholder). Returns records with
    "priority_score", "inspection_cost_aud", "risk_of_delay_aud", and
    "worth_inspecting_now" added, sorted by priority_score descending
    (most urgent first).
    """
    table = records.copy()
    table["priority_score"] = priority_score(
        table["current_anomaly_score"].to_numpy(), table["severity_trend"].to_numpy(),
    )
    table["inspection_cost_aud"] = inspection_cost_aud(
        table["localised_height_m"].to_numpy(), base_cost_aud, cost_per_metre_aud,
    )
    table["risk_of_delay_aud"] = risk_of_delay_aud(table["priority_score"].to_numpy(), risk_scale_aud)
    table["worth_inspecting_now"] = table["risk_of_delay_aud"] > table["inspection_cost_aud"]
    return table.sort_values("priority_score", ascending=False).reset_index(drop=True)
