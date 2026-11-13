"""Expanding calendar windows with purged label boundaries and storm disjointness."""

from __future__ import annotations

import numpy as np
import pandas as pd


def utc(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def before(frame: pd.DataFrame, boundary: str | pd.Timestamp) -> pd.DataFrame:
    boundary = utc(boundary) if isinstance(boundary, str) else boundary
    return frame.loc[frame["target_time"] < boundary].copy()


def window_split(frame: pd.DataFrame, start: str, end: str):
    lo, hi = utc(start), utc(end)
    # Entire input window and target must lie within validation's calendar window.
    validation = frame.loc[(frame["history_start"] >= lo) & (frame["target_time"] < hi)].copy()
    train = frame.loc[frame["target_time"] < lo].copy()
    # Storms crossing calendar boundaries are never shared across partitions.
    train = train.loc[~train["storm_id"].isin(validation["storm_id"])].copy()
    return train, validation


def development_calibration_test(frame: pd.DataFrame, dev_end: str, cal_end: str):
    test = frame.loc[frame["history_start"] >= utc(cal_end)].copy()
    calibration = frame.loc[
        (frame["history_start"] >= utc(dev_end))
        & (frame["target_time"] < utc(cal_end))
        & ~frame["storm_id"].isin(test["storm_id"])
    ].copy()
    development = frame.loc[
        (frame["target_time"] < utc(dev_end))
        & ~frame["storm_id"].isin(set(calibration["storm_id"]) | set(test["storm_id"]))
    ].copy()
    return development, calibration, test


def assert_disjoint(train: pd.DataFrame, validation: pd.DataFrame):
    if train.empty or validation.empty:
        raise ValueError("Empty partition: choose windows supported by the data")
    if set(train["storm_id"]) & set(validation["storm_id"]):
        raise AssertionError("Storm leakage between partitions")
    if train["target_time"].max() >= validation["history_start"].min():
        raise AssertionError("Future label or overlapping history at partition boundary")
    if not np.isfinite(train["target"]).all():
        raise ValueError("Non-finite training labels")
