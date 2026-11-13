"""Focused, post-experiment analysis; no model selection on the inspected holdout."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import TARGET
from .features import attach_targets, feature_frame
from .metrics import conformal_radius, evaluate
from .models import build_model, model_columns, predict
from .splits import assert_disjoint, development_calibration_test


def reproduce_selected(observations, config, selection):
    """Refit frozen development-selected parameters; do not search or retune."""
    started = time.perf_counter()
    features, columns, sequence_columns = feature_frame(observations, config["history_hours"])
    metrics, predictions, partitions = [], [], []
    for row in selection.loc[selection["is_champion"]].itertuples():
        horizon = int(row.horizon_hours)
        aligned = attach_targets(observations, features, horizon)
        development, calibration, test = development_calibration_test(
            aligned, config["development_end"], config["calibration_end"]
        )
        assert_disjoint(development, calibration)
        assert_disjoint(calibration, test)
        for name, part in [("development", development), ("calibration", calibration), ("test", test)]:
            partitions.append({"horizon_hours": horizon, "partition": name,
                               "samples": len(part), "storms": part.storm_id.nunique()})
        used = model_columns(row.model, columns, sequence_columns)
        model = build_model(row.model, json.loads(row.parameters), config, horizon, sequence_columns)
        fit_start = time.perf_counter()
        with threadpool_limits(limits=config.get("n_jobs", 2)):
            model.fit(development[used], development.target)
            cal_prediction = predict(model, calibration[used])
            forecast = predict(model, test[used])
        radius = conformal_radius(calibration.target, cal_prediction,
                                  config["prediction_interval_alpha"])
        output = test[["storm_id", "timestamp", "target_time", "target"]].copy()
        output["current_wind"] = test[TARGET].to_numpy()
        output["prediction"] = forecast
        output["lower"] = np.maximum(0, forecast - radius)
        output["upper"] = forecast + radius
        output["horizon_hours"] = horizon
        output["model"] = row.model
        metrics.append({
            "horizon_hours": horizon, "model": row.model,
            **evaluate(test.target, forecast, test[TARGET],
                       config["extreme_wind_threshold_mps"]),
            "interval_coverage": float(
                ((output.target >= output.lower) & (output.target <= output.upper)).mean()
            ),
            "fit_predict_seconds": time.perf_counter() - fit_start
        })
        predictions.append(output)
    return (
        pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True),
        pd.DataFrame(partitions), time.perf_counter() - started
    )


def error_slices(predictions):
    """Descriptive strata use observed outcomes for diagnosis, never as new inputs."""
    frame = predictions.copy()
    frame["wind_band"] = pd.cut(
        frame.target, [0, 17, 33, 50, np.inf], right=False,
        labels=["[0,17)", "[17,33)", "[33,50)", "[50,inf)"]
    )
    frame["absolute_change"] = (frame.target - frame.current_wind).abs()
    frame["change_band"] = pd.cut(
        frame.absolute_change, [0, 5, 10, np.inf], right=False,
        labels=["[0,5)", "[5,10)", "[10,inf)"]
    )
    frame["year"] = frame.timestamp.dt.year.astype(str)
    rows = []
    for group in ["wind_band", "change_band", "year"]:
        for (horizon, label), part in frame.groupby(["horizon_hours", group], observed=True):
            model_mae = (part.prediction - part.target).abs().mean()
            baseline_mae = (part.current_wind - part.target).abs().mean()
            rows.append({
                "horizon_hours": int(horizon), "group": group, "slice": str(label),
                "samples": len(part), "storms": part.storm_id.nunique(),
                "model_mae_mps": model_mae, "persistence_mae_mps": baseline_mae,
                "skill_vs_persistence": 1 - model_mae / baseline_mae if baseline_mae else np.nan,
                "bias_mps": (part.prediction - part.target).mean(),
                "interval_coverage": (
                    (part.target >= part.lower) & (part.target <= part.upper)
                ).mean()
            })
    return pd.DataFrame(rows)


def plot_diagnostics(predictions):
    import matplotlib.pyplot as plt
    horizons = sorted(predictions.horizon_hours.unique())
    fig, axes = plt.subplots(2, len(horizons), figsize=(13, 7), squeeze=False)
    for column, horizon in enumerate(horizons):
        part = predictions.loc[predictions.horizon_hours == horizon]
        axes[0, column].scatter(part.target, part.prediction, alpha=0.35, s=12)
        limit = max(part.target.max(), part.prediction.max())
        axes[0, column].plot([0, limit], [0, limit], "--", color="black", lw=1)
        axes[0, column].set(title=f"{horizon}h", xlabel="Observed wind (m/s)",
                            ylabel="Forecast wind (m/s)")
        axes[1, column].hist(part.prediction - part.target, bins=25)
        axes[1, column].axvline(0, color="black", linestyle="--", lw=1)
        axes[1, column].set(xlabel="Forecast minus observed (m/s)", ylabel="Count")
    fig.suptitle("Fixed-model replay: observed wind and residuals")
    fig.tight_layout()
    return fig


def plot_slices(slices):
    import matplotlib.pyplot as plt
    change = slices.loc[slices.group == "change_band"]
    horizons = sorted(change.horizon_hours.unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(13, 4), squeeze=False)
    for i, horizon in enumerate(horizons):
        part = change.loc[change.horizon_hours == horizon].set_index("slice")
        part[["persistence_mae_mps", "model_mae_mps"]].plot.bar(ax=axes[0, i], rot=0)
        axes[0, i].set(title=f"{horizon}h", ylabel="MAE (m/s)",
                       xlabel="Observed absolute wind change (m/s)")
        axes[0, i].legend(["Persistence", "Selected model"], fontsize=8)
    fig.suptitle("Post-hoc error analysis: change magnitude is NOT an input feature")
    fig.tight_layout()
    return fig
