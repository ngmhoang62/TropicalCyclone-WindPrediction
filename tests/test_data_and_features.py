import numpy as np
import pandas as pd
import pytest

from cyclone_wind.data import TARGET, load_observations
from cyclone_wind.features import attach_targets, feature_frame


def observations(n=12):
    return pd.DataFrame(
        {
            "storm_id": ["storm-a"] * n,
            "timestamp": pd.date_range("2000-01-01", periods=n, freq="6h", tz="UTC"),
            TARGET: np.arange(n, dtype=float) + 10,
            "pt_lat": [10.0] * n,
            "pt_lon": [120.0] * n,
            "pt_pressureMB": np.arange(n) + 990.0,
        }
    )


def test_future_perturbation_cannot_change_past_features():
    source = observations()
    original, columns, _ = feature_frame(source, [0, 6, 12, 18])
    perturbed = source.copy()
    perturbed.loc[8:, TARGET] = 120.0
    changed, _, _ = feature_frame(perturbed, [0, 6, 12, 18])
    cutoff = source.loc[8, "timestamp"]
    pd.testing.assert_frame_equal(
        original.loc[original.timestamp < cutoff, columns],
        changed.loc[changed.timestamp < cutoff, columns],
    )


@pytest.mark.parametrize("horizon", [6, 12, 24])
def test_labels_use_exact_future_clock_time(horizon):
    source = observations()
    features, _, _ = feature_frame(source, [0, 6, 12, 18])
    aligned = attach_targets(source, features, horizon)
    assert np.all(aligned.target - aligned[TARGET] == horizon / 6)
    assert np.all(aligned.target_time - aligned.timestamp == pd.Timedelta(hours=horizon))


def test_missing_clock_step_is_not_filled_with_nearest_observation():
    source = observations().drop(index=4).reset_index(drop=True)
    features, _, _ = feature_frame(source, [0, 6, 12, 18])
    assert source.loc[4, "timestamp"] not in set(features.timestamp)


def test_other_storm_cannot_supply_history():
    source = observations(8)
    source.loc[4:, "storm_id"] = "storm-b"
    features, _, _ = feature_frame(source, [0, 6, 12, 18])
    assert len(features) == 2
    assert len(features[features.storm_id == "storm-b"]) == 1


def test_ambiguous_duplicate_is_removed_without_averaging_label(tmp_path):
    rows = pd.DataFrame(
        {
            "storm_id": ["1980-WP-01"] * 4,
            "pt_dateTimeISO": ["18-03-80 0:00", "18-03-80 0:00", "18-03-80 6:00", "18-03-80 6:00"],
            TARGET: [20, 20, 30, 40],
            "pt_lat": [10] * 4,
            "pt_lon": [120] * 4,
            "pr_windMaxMPS": [999] * 4,
        }
    )
    path = tmp_path / "raw.csv"
    rows.to_csv(path, index=False)
    cleaned, audit = load_observations(path)
    assert len(cleaned) == 1
    assert audit["removed_equivalent_observations"] == 1
    assert audit["conflicting_keys"] == 1
    assert "pr_windMaxMPS" not in cleaned


def test_bad_timestamp_fails_closed(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame(
        {
            "storm_id": ["a"],
            "pt_dateTimeISO": ["unparseable"],
            TARGET: [20],
            "pt_lat": [10],
            "pt_lon": [120],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Unparseable"):
        load_observations(path)


def test_metadata_and_lifetime_extrema_are_not_features():
    source = observations()
    source["pr_windMaxMPS"] = 999
    source["pr_endISO"] = "tomorrow"
    _, columns, _ = feature_frame(source, [0, 6, 12, 18])
    assert "pr_windMaxMPS" not in columns
    assert "pr_endISO" not in columns
