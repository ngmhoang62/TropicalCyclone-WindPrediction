import numpy as np
import pandas as pd

from src.analysis import error_slices


def test_diagnostic_strata_cover_all_rows_without_modifying_predictions():
    frame = pd.DataFrame({
        "storm_id": ["a"] * 4, "timestamp": pd.to_datetime(["2023-01-01"] * 4, utc=True),
        "horizon_hours": [6] * 4, "target": [10.0, 20.0, 40.0, 60.0],
        "current_wind": [10.0, 15.0, 30.0, 40.0],
        "prediction": [11.0, 18.0, 35.0, 50.0],
        "lower": [0.0] * 4, "upper": [100.0] * 4,
    })
    original = frame.copy(deep=True)
    result = error_slices(frame)
    for _, group in result.groupby("group"):
        assert group.samples.sum() == len(frame)
    pd.testing.assert_frame_equal(frame, original)


def test_zero_persistence_error_does_not_produce_infinite_skill():
    frame = pd.DataFrame({
        "storm_id": ["a"], "timestamp": pd.to_datetime(["2023-01-01"], utc=True),
        "horizon_hours": [6], "target": [20.0], "current_wind": [20.0],
        "prediction": [21.0], "lower": [0.0], "upper": [50.0],
    })
    result = error_slices(frame)
    assert result.skill_vs_persistence.isna().all()
    assert not np.isinf(result.model_mae_mps).any()
