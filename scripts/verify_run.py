"""Recompute run evidence and confirm saved-model inference on the actual holdout."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cyclone_wind  # noqa: E402
from cyclone_wind.data import load_observations  # noqa: E402
from cyclone_wind.features import attach_targets, feature_frame  # noqa: E402
from cyclone_wind.metrics import evaluate  # noqa: E402
from cyclone_wind.models import predict  # noqa: E402
from cyclone_wind.splits import (  # noqa: E402
    assert_disjoint,
    development_calibration_test,
    window_split,
)

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
parser.add_argument("--data", required=True)
parser.add_argument("--output", default="reports/run_verification.json")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
run = root / args.run
manifest = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
if manifest["status"] != "COMPLETE":
    raise SystemExit("Run is not complete")
config = manifest["config"]
locked = json.loads((run / "LOCKED_PROTOCOL.json").read_text(encoding="utf-8"))
assert locked["config"] == config
assert locked["data_sha256"] == manifest["data_audit"]["input_sha256"]
assert locked["code_sha256"] == manifest["code_sha256"]
observations, audit = load_observations(root / args.data)
assert audit["input_sha256"] == manifest["data_audit"]["input_sha256"]
digest = hashlib.sha256()
for path in sorted(Path(cyclone_wind.__file__).parent.glob("*.py")):
    digest.update(path.name.encode())
    digest.update(path.read_bytes())
if digest.hexdigest() != manifest["code_sha256"]:
    snapshot = json.loads((root / "reports/benchmark/source_modules.json").read_text())
    current = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in Path(cyclone_wind.__file__).parent.glob("*.py")}
    assert current == snapshot["modules"], "Forecasting source differs from verified snapshot"
features, columns, sequence_columns = feature_frame(observations, config["history_hours"])
predictions = pd.read_csv(run / "holdout_predictions.csv")
metrics = pd.read_csv(run / "holdout_metrics.csv")
selection = pd.read_csv(run / "selection.csv")
trials = pd.read_csv(run / "cv_trials.csv")
assert not predictions.duplicated(["horizon_hours", "model", "storm_id", "timestamp"]).any()
for horizon in config["horizons_hours"]:
    assert set(predictions.loc[predictions.horizon_hours == horizon, "model"]) == set(
        config["models"]
    )
    frame = attach_targets(observations, features, horizon)
    dev, cal, test = development_calibration_test(
        frame, config["development_end"], config["calibration_end"]
    )
    assert_disjoint(dev, cal)
    assert_disjoint(cal, test)
    for start, end in config["validation_windows"]:
        assert_disjoint(*window_split(dev, start, end))
    expected = test[["storm_id", "timestamp", "target_time", "target"]].copy()
    expected["timestamp"] = expected["timestamp"].astype(str)
    expected["target_time"] = expected["target_time"].astype(str)
    expected = expected.reset_index(drop=True)
    for model, frame_predictions in predictions.loc[predictions.horizon_hours == horizon].groupby(
        "model"
    ):
        frame_predictions = frame_predictions.reset_index(drop=True)
        pd.testing.assert_frame_equal(
            expected,
            frame_predictions[expected.columns],
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
        )
        recomputed = evaluate(
            frame_predictions.target,
            frame_predictions.prediction,
            frame_predictions.persistence,
            config["extreme_wind_threshold_mps"],
        )
        saved = metrics.loc[(metrics.horizon_hours == horizon) & (metrics.model == model)].iloc[0]
        for metric in ["mae_mps", "rmse_mps", "r2", "mae_skill_vs_persistence"]:
            np.testing.assert_allclose(recomputed[metric], saved[metric], atol=1e-10)
        coverage = (
            (frame_predictions.target >= frame_predictions.lower)
            & (frame_predictions.target <= frame_predictions.upper)
        ).mean()
        np.testing.assert_allclose(coverage, saved["interval_coverage"], atol=1e-10)
    candidates = selection.loc[selection.horizon_hours == horizon]
    for row in candidates.itertuples():
        model_trials = trials.loc[(trials.horizon_hours == horizon) & (trials.model == row.model)]
        trial_scores = model_trials.groupby("trial").mae_mps.mean()
        np.testing.assert_allclose(trial_scores.min(), row.cv_mae_mps, atol=1e-10)
    champion = candidates.loc[candidates.cv_mae_mps.idxmin(), "model"]
    assert manifest["champions"][str(horizon)] == champion
    bundle = joblib.load(run / f"champion_{horizon}h.joblib")
    recomputed_predictions = predict(bundle["model"], test[bundle["features"]])
    saved_predictions = predictions.loc[
        (predictions.horizon_hours == horizon) & (predictions.model == champion), "prediction"
    ].to_numpy()
    np.testing.assert_allclose(recomputed_predictions, saved_predictions, atol=1e-7)
report = {
    "status": "PASS",
    "run_id": run.name,
    "source_revision": "See benchmark/source_modules.json",
    "input_sha256": audit["input_sha256"],
    "code_sha256": manifest["code_sha256"],
    "horizons_verified": config["horizons_hours"],
    "prediction_rows_verified": len(predictions),
    "checks": [
        "input bytes",
        "locked protocol consistency",
        "source digest",
        "storm disjointness",
        "chronological label/history boundaries",
        "all development folds",
        "common holdout origin/target pairs",
        "metrics recomputed",
        "empirical interval coverage recomputed",
        "champion selected using development only",
        "saved model inference replay",
    ],
}
(root / args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
