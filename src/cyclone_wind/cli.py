"""Command line entry points, with explicit filesystem paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_observations
from .experiment import load_config, run_experiment, save_json
from .features import feature_frame
from .models import predict


def main():
    parser = argparse.ArgumentParser(description="Cyclone wind forecasting research")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Inspect original CSV without training")
    audit.add_argument("--data", required=True)
    audit.add_argument("--output", required=True)
    run = commands.add_parser("train", help="Run frozen calendar-window research")
    run.add_argument("--data", required=True)
    run.add_argument("--config", default="configs/research.json")
    run.add_argument("--output", required=True)
    run.add_argument("--evaluate-holdout", action="store_true")
    inference = commands.add_parser("predict", help="Forecast from a trusted local model bundle")
    inference.add_argument("--data", required=True, help="Original-schema observation history CSV")
    inference.add_argument(
        "--model", required=True, help="Trusted joblib artifact from this project"
    )
    inference.add_argument("--output", required=True)
    commands.add_parser("config", help="Validate and print the default research configuration")
    args = parser.parse_args()
    if args.command == "audit":
        _, report = load_observations(args.data)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(path, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.command == "train":
        run_experiment(args.data, args.config, args.output, args.evaluate_holdout)
    elif args.command == "predict":
        import joblib
        import numpy as np
        import pandas as pd

        bundle = joblib.load(args.model)
        observations, _ = load_observations(args.data)
        features, _, _ = feature_frame(observations, bundle["history_hours"])
        if features.empty:
            raise ValueError("Insufficient exact-clock observation history")
        # Return latest valid origin per storm; future target labels are not needed.
        latest = features.sort_values("timestamp").groupby("storm_id").tail(1)
        forecast = predict(bundle["model"], latest[bundle["features"]])
        output = latest[["storm_id", "timestamp"]].copy()
        output["target_time"] = output["timestamp"] + pd.Timedelta(hours=bundle["horizon_hours"])
        output["forecast_wind_mps"] = forecast
        output["lower_mps"] = np.maximum(0, forecast - bundle["interval_radius_mps"])
        output["upper_mps"] = forecast + bundle["interval_radius_mps"]
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(path, index=False)
        print(f"Wrote {len(output)} forecasts to {path}")
    else:
        print(json.dumps(load_config("configs/research.json"), indent=2))
