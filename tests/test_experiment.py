import json

import pandas as pd
import pytest

from cyclone_wind.experiment import run_experiment


def test_small_research_run_and_saved_inference(tmp_path):
    rows = []
    for year in range(2000, 2005):
        for storm in range(2):
            dates = pd.date_range(f"{year}-06-{storm + 1:02d}", periods=12, freq="6h")
            for i, timestamp in enumerate(dates):
                rows.append(
                    {
                        "storm_id": f"{year}-WP-{storm}",
                        "pt_dateTimeISO": timestamp.strftime("%d-%m-%y %H:%M"),
                        "pt_windSpeedMPS": 10 + (i % 5) * 2,
                        "pt_lat": 10,
                        "pt_lon": 120,
                    }
                )
    data = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(data, index=False)
    configuration = {
        "seed": 42,
        "horizons_hours": [6],
        "history_hours": [0, 6, 12, 18],
        "development_end": "2003-01-01",
        "calibration_end": "2004-01-01",
        "validation_windows": [["2001-01-01", "2003-01-01"]],
        "models": ["persistence", "trend"],
        "search_trials": 1,
        "n_jobs": 1,
        "bootstrap_repetitions": 10,
        "prediction_interval_alpha": 0.1,
        "extreme_wind_threshold_mps": 33.0,
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(configuration))
    output = tmp_path / "run"
    metadata = run_experiment(data, config, output, evaluate_holdout=True)
    assert metadata["status"] == "COMPLETE"
    assert (output / "champion_6h.joblib").exists()
    predictions = pd.read_csv(output / "holdout_predictions.csv")
    assert predictions.groupby("model").size().nunique() == 1
    assert metadata["champions"]["6"] in configuration["models"]
    with pytest.raises(FileExistsError):
        run_experiment(data, config, output)
