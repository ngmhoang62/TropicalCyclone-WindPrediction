import json

import numpy as np
import pandas as pd
import pytest

from cyclone_wind.data import TARGET
from cyclone_wind.metrics import conformal_radius, evaluate
from cyclone_wind.models import build_model, predict
from cyclone_wind.splits import assert_disjoint, development_calibration_test, window_split


def frame():
    timestamps = pd.to_datetime(
        ["2000-06-01", "2001-12-31 18:00", "2002-01-01", "2002-06-01", "2003-01-01", "2004-01-01"],
        format="mixed",
        utc=True,
    )
    return pd.DataFrame(
        {
            "storm_id": ["old", "crossing", "crossing", "validation", "calibration", "test"],
            "timestamp": timestamps,
            "history_start": timestamps - pd.Timedelta(hours=18),
            "target_time": timestamps + pd.Timedelta(hours=24),
            "target": np.arange(6, dtype=float),
        }
    )


def test_temporal_fold_purges_overlapping_labels_and_storms():
    train, validation = window_split(frame(), "2002-01-01", "2003-01-01")
    assert list(train.storm_id) == ["old"]
    assert list(validation.storm_id) == ["validation"]
    assert_disjoint(train, validation)


def test_three_partitions_are_chronological_and_storm_disjoint():
    source = frame()
    dev, cal, test = development_calibration_test(source, "2002-01-01", "2003-06-01")
    assert_disjoint(dev, cal)
    assert_disjoint(cal, test)


def test_scaler_is_fitted_on_training_only():
    config = {"seed": 42}
    model = build_model("elasticnet", {"alpha": 0.1, "l1_ratio": 0.5}, config, 6, [])
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    model.fit(train, [0.0, 1.0, 2.0, 3.0])
    before = model.named_steps["standardscaler"].mean_.copy()
    predict(model, pd.DataFrame({"x": [1e9]}))
    np.testing.assert_array_equal(before, [1.5])
    np.testing.assert_array_equal(model.named_steps["standardscaler"].mean_, before)


def test_persistence_and_skill_are_consistent():
    X = pd.DataFrame({TARGET: [10.0, 20.0]})
    baseline = build_model("persistence", {}, {"seed": 42}, 6, []).fit(X, [11, 22])
    result = evaluate([11, 22], predict(baseline, X), X[TARGET])
    assert result["mae_mps"] == 1.5
    assert result["mae_skill_vs_persistence"] == 0
    json.dumps(result, allow_nan=False)


def test_conformal_calibration_does_not_take_holdout_labels():
    assert conformal_radius([1, 2, 3, 4], [1, 1, 1, 1], alpha=0.5) == 2.0


@pytest.mark.parametrize("kind", ["torch_mlp", "torch_lstm"])
def test_torch_model_artifact_roundtrip(tmp_path, kind):
    pytest.importorskip("torch")
    import joblib

    config = {"seed": 42, "history_hours": [0, 6, 12, 18], "torch_epochs": 2}
    model = build_model(kind, {"hidden_size": 8}, config, 6, [])
    X = pd.DataFrame(np.arange(160).reshape(20, 8) / 100)
    y = np.arange(20) / 10
    model.fit(X, y)
    original = predict(model, X)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(predict(joblib.load(path), X), original)
