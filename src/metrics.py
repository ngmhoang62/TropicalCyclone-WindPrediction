"""Regression, event metrics, storm-cluster uncertainty and offline drift diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(y, prediction, persistence, threshold=33.0) -> dict:
    y, prediction, persistence = map(
        lambda x: np.asarray(x, dtype=float), (y, prediction, persistence)
    )
    baseline_mae = mean_absolute_error(y, persistence)
    extreme = y >= threshold
    return {
        "samples": len(y),
        "mae_mps": float(mean_absolute_error(y, prediction)),
        "rmse_mps": float(np.sqrt(mean_squared_error(y, prediction))),
        "r2": float(r2_score(y, prediction)) if len(y) > 1 else None,
        "mae_skill_vs_persistence": float(1 - mean_absolute_error(y, prediction) / baseline_mae)
        if baseline_mae > 0
        else None,
        "change_direction_accuracy": float(
            (np.sign(prediction - persistence) == np.sign(y - persistence)).mean()
        ),
        "extreme_samples": int(extreme.sum()),
        "extreme_mae_mps": float(mean_absolute_error(y[extreme], prediction[extreme]))
        if extreme.any()
        else None,
        "mean_bias_mps": float((prediction - y).mean()),
    }


def conformal_radius(y, prediction, alpha=0.1) -> float:
    residuals = np.abs(np.asarray(y) - np.asarray(prediction))
    if not len(residuals):
        raise ValueError("Calibration set is empty")
    rank = min(len(residuals), int(np.ceil((len(residuals) + 1) * (1 - alpha))))
    return float(np.partition(residuals, rank - 1)[rank - 1])


def cluster_mae_interval(frame, repetitions=500, seed=42):
    """Resample entire storms, preserving within-storm serial dependence."""
    errors = (
        pd.DataFrame(
            {"storm_id": frame["storm_id"], "error": np.abs(frame["prediction"] - frame["target"])}
        )
        .groupby("storm_id")["error"]
        .agg(["sum", "count"])
    )
    rng = np.random.default_rng(seed)
    sums, counts = errors["sum"].to_numpy(), errors["count"].to_numpy()
    values = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(errors), size=len(errors))
        values.append(sums[indices].sum() / counts[indices].sum())
    return [float(v) for v in np.quantile(values, [0.025, 0.975])]


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> list[dict]:
    """PSI is a descriptive retrospective check; never tune on the final holdout."""
    report = []
    for column in reference.columns:
        left = reference[column].dropna().to_numpy()
        right = current[column].dropna().to_numpy()
        if len(left) < 20 or len(right) < 20:
            continue
        edges = np.unique(np.quantile(left, np.linspace(0, 1, 11)))
        if len(edges) < 3:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        a = np.histogram(left, edges)[0].astype(float) + 0.5
        b = np.histogram(right, edges)[0].astype(float) + 0.5
        a, b = a / a.sum(), b / b.sum()
        report.append(
            {
                "feature": column,
                "psi": float(((b - a) * np.log(b / a)).sum()),
                "reference_missing_rate": float(reference[column].isna().mean()),
                "current_missing_rate": float(current[column].isna().mean()),
            }
        )
    return sorted(report, key=lambda x: -x["psi"])
