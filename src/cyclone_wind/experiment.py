"""Auditable research execution; tuning, calibration and holdout have distinct roles."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import TARGET, load_observations
from .features import attach_targets, feature_frame
from .metrics import cluster_mae_interval, conformal_radius, drift_report, evaluate
from .models import build_model, candidate_parameters, model_columns, predict
from .reporting import plots, write_summary
from .splits import assert_disjoint, development_calibration_test, window_split


def save_json(path: Path, value):
    def convert(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, (pd.Timestamp, Path)):
            return str(obj)
        raise TypeError(type(obj).__name__)

    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=convert, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not config["validation_windows"]:
        raise ValueError("At least one development validation window is required")
    for start, end in config["validation_windows"]:
        if not start < end <= config["development_end"]:
            raise ValueError("Validation windows must end before development_end")
    if not config["development_end"] < config["calibration_end"]:
        raise ValueError("Development must precede calibration and final test")
    if not 0 < config["prediction_interval_alpha"] < 1:
        raise ValueError("Invalid prediction interval alpha")
    if not config["horizons_hours"] or min(config["horizons_hours"]) <= 0:
        raise ValueError("Positive horizons required")
    if 6 not in config["history_hours"]:
        raise ValueError("6h history required for the trend baseline")
    if "persistence" not in config["models"]:
        raise ValueError("Persistence baseline required")
    return config


def cv_score(frame, columns, sequence_columns, name, parameters, config, horizon):
    scores, fold_rows = [], []
    used_columns = model_columns(name, columns, sequence_columns)
    for start, end in config["validation_windows"]:
        train, validation = window_split(frame, start, end)
        assert_disjoint(train, validation)
        model = build_model(name, parameters, config, horizon, sequence_columns)
        started = time.perf_counter()
        with threadpool_limits(limits=config.get("n_jobs", 2)):
            model.fit(train[used_columns], train["target"])
            prediction = predict(model, validation[used_columns])
        metrics = evaluate(validation["target"], prediction, validation[TARGET])
        scores.append(metrics["mae_mps"])
        fold_rows.append(
            {
                "validation_start": start,
                "validation_end": end,
                "training_rows": len(train),
                "validation_rows": len(validation),
                "training_storms": train["storm_id"].nunique(),
                "validation_storms": validation["storm_id"].nunique(),
                "train_label_max": str(train["target_time"].max()),
                "validation_history_min": str(validation["history_start"].min()),
                "fit_predict_seconds": time.perf_counter() - started,
                **metrics,
            }
        )
    return float(np.mean(scores)), fold_rows


def run_experiment(data_path, config_path, output, evaluate_holdout=False):
    config = load_config(config_path)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a fresh run directory; refusing overwrite: {output}")
    output.mkdir(parents=True)
    observations, audit = load_observations(data_path)
    frame, columns, seq_columns = feature_frame(observations, config["history_hours"])
    code_digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        code_digest.update(path.name.encode())
        code_digest.update(path.read_bytes())
    metadata = {
        "run_id": output.name,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "config": config,
        "code_sha256": code_digest.hexdigest(),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "data_audit": audit,
        "features": columns,
        "sequence_features": seq_columns,
        "evaluate_holdout": evaluate_holdout,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "scipy", "joblib"]
        },
        "partitions": {},
    }
    for optional in ["torch", "fastapi"]:
        try:
            metadata["versions"][optional] = importlib.metadata.version(optional)
        except importlib.metadata.PackageNotFoundError:
            pass
    save_json(output / "metadata.json", metadata)
    # Written before reading holdout labels for evaluation; this protocol is immutable.
    save_json(
        output / "LOCKED_PROTOCOL.json",
        {
            "config": config,
            "data_sha256": audit["input_sha256"],
            "code_sha256": metadata["code_sha256"],
            "holdout_enabled": evaluate_holdout,
            "selection_metric": "unweighted mean MAE over calendar validation windows",
        },
    )
    trials, selections, holdout_rows, prediction_rows, ablations = [], [], [], [], []
    champions = {}
    try:
        for horizon in config["horizons_hours"]:
            print(f"\nHorizon {horizon}h: preparing aligned samples", flush=True)
            aligned = attach_targets(observations, frame, horizon)
            dev, calibration, test = development_calibration_test(
                aligned, config["development_end"], config["calibration_end"]
            )
            assert_disjoint(dev, calibration)
            assert_disjoint(calibration, test)
            metadata["partitions"][str(horizon)] = {
                name: {
                    "rows": len(part),
                    "storms": part["storm_id"].nunique(),
                    "origin_min": str(part["timestamp"].min()),
                    "origin_max": str(part["timestamp"].max()),
                }
                for name, part in [
                    ("development", dev),
                    ("calibration", calibration),
                    ("test", test),
                ]
            }
            candidates = {}
            for name in config["models"]:
                best = None
                for trial_id, params in enumerate(candidate_parameters(name, config)):
                    score, folds = cv_score(
                        dev, columns, seq_columns, name, params, config, horizon
                    )
                    trials.extend(
                        {
                            "horizon_hours": horizon,
                            "model": name,
                            "trial": trial_id,
                            "parameters": json.dumps(params, sort_keys=True),
                            **fold,
                        }
                        for fold in folds
                    )
                    if best is None or score < best[0]:
                        best = (score, params)
                    print(f"  {name} trial {trial_id}: CV MAE={score:.4f}", flush=True)
                candidates[name] = best
            champion = min(candidates, key=lambda name: candidates[name][0])
            champions[str(horizon)] = champion
            for name, (score, params) in candidates.items():
                selections.append(
                    {
                        "horizon_hours": horizon,
                        "model": name,
                        "cv_mae_mps": score,
                        "parameters": json.dumps(params, sort_keys=True),
                        "is_champion": name == champion,
                    }
                )
            # Ablations belong to development only; never optimize them on final test.
            if champion not in {"persistence", "trend", "torch_lstm"}:
                for label, subset in [
                    (
                        "without_era5",
                        [
                            c
                            for c in columns
                            if "ERA5" not in c
                            and not c.startswith(("wind_magnitude", "wind_shear"))
                        ],
                    ),
                    (
                        "without_past_lags",
                        [
                            c
                            for c in columns
                            if not c.startswith("seq_")
                            and c not in {"wind_history_mean", "wind_history_std", "wind_change_6h"}
                        ],
                    ),
                ]:
                    score, _ = cv_score(
                        dev, subset, seq_columns, champion, candidates[champion][1], config, horizon
                    )
                    ablations.append(
                        {
                            "horizon_hours": horizon,
                            "model": champion,
                            "ablation": label,
                            "cv_mae_mps": score,
                            "delta_vs_full_mps": score - candidates[champion][0],
                        }
                    )
            frozen = {
                name: {"cv_mae_mps": score, "parameters": params}
                for name, (score, params) in candidates.items()
            }
            save_json(
                output / f"frozen_selection_{horizon}h.json",
                {"champion": champion, "models": frozen},
            )
            # Fit only on development. Calibration is reserved for interval calibration.
            for name, (_, params) in candidates.items():
                used_columns = model_columns(name, columns, seq_columns)
                model = build_model(name, params, config, horizon, seq_columns)
                with threadpool_limits(limits=config.get("n_jobs", 2)):
                    model.fit(dev[used_columns], dev["target"])
                    calibration_prediction = predict(model, calibration[used_columns])
                radius = conformal_radius(
                    calibration["target"],
                    calibration_prediction,
                    config["prediction_interval_alpha"],
                )
                if name == champion:
                    bundle = {
                        "model": model,
                        "features": used_columns,
                        "history_hours": config["history_hours"],
                        "horizon_hours": horizon,
                        "interval_radius_mps": radius,
                        "alpha": config["prediction_interval_alpha"],
                        "model_name": name,
                        "data_sha256": audit["input_sha256"],
                        "code_sha256": metadata["code_sha256"],
                    }
                    joblib.dump(bundle, output / f"champion_{horizon}h.joblib")
                if not evaluate_holdout:
                    continue
                started = time.perf_counter()
                with threadpool_limits(limits=config.get("n_jobs", 2)):
                    forecast = predict(model, test[used_columns])
                elapsed = time.perf_counter() - started
                metrics = evaluate(
                    test["target"], forecast, test[TARGET], config["extreme_wind_threshold_mps"]
                )
                lower, upper = np.maximum(0, forecast - radius), forecast + radius
                pred = test[["storm_id", "timestamp", "target_time", "target"]].copy()
                pred["prediction"] = forecast
                pred["persistence"] = test[TARGET]
                pred["lower"] = lower
                pred["upper"] = upper
                pred["horizon_hours"] = horizon
                pred["model"] = name
                pred["is_champion"] = name == champion
                prediction_rows.append(pred)
                holdout_rows.append(
                    {
                        "horizon_hours": horizon,
                        "model": name,
                        "is_champion": name == champion,
                        **metrics,
                        "batch_prediction_ms": elapsed * 1000,
                        "ms_per_sample_amortized": elapsed * 1000 / len(test),
                        "interval_radius_mps": radius,
                        "interval_coverage": float(
                            ((test["target"] >= lower) & (test["target"] <= upper)).mean()
                        ),
                        "mean_interval_width_mps": float((upper - lower).mean()),
                        "mae_storm_bootstrap_95ci": json.dumps(
                            cluster_mae_interval(
                                pred, config["bootstrap_repetitions"], config["seed"]
                            )
                        ),
                    }
                )
                if name == champion:
                    save_json(
                        output / f"offline_drift_{horizon}h.json",
                        drift_report(dev[used_columns], test[used_columns]),
                    )
            save_json(output / "metadata.json", metadata)
            pd.DataFrame(trials).to_csv(output / "cv_trials.csv", index=False)
            pd.DataFrame(selections).to_csv(output / "selection.csv", index=False)
            pd.DataFrame(holdout_rows).to_csv(output / "holdout_metrics.csv", index=False)
            pd.DataFrame(ablations).to_csv(output / "development_ablations.csv", index=False)
        selections_df, metrics_df = pd.DataFrame(selections), pd.DataFrame(holdout_rows)
        if prediction_rows:
            predictions = pd.concat(prediction_rows, ignore_index=True)
            predictions.to_csv(output / "holdout_predictions.csv", index=False)
            plots(predictions, output / "figures")
        metadata["status"] = "COMPLETE"
        metadata["champions"] = champions
        metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(output / "metadata.json", metadata)
        write_summary(output, metadata, selections_df, metrics_df)
        print(f"Complete: {output}", flush=True)
    except Exception as exc:
        metadata["status"] = "FAILED"
        metadata["failure"] = f"{type(exc).__name__}: {exc}"
        save_json(output / "metadata.json", metadata)
        raise
    return metadata
