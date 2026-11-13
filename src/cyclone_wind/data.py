"""Read the original observations; discard ambiguous duplicates without averaging labels."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "pt_windSpeedMPS"
KEY = ["storm_id", "timestamp"]
OBSERVATIONS = [
    TARGET,
    "pt_lat",
    "pt_lon",
    "pt_pressureMB",
    "pt_moveSpeedMPS",
    "pt_moveDirectionDEG",
]
WEATHER = [
    f"ERA5_{variable}_{level}"
    for level in [850, 500, 200]
    for variable in ["u", "v", "z", "r", "w", "vo"]
] + ["ERA5_SST", "ERA5_MSLP", "ERA5_TCWV", "ERA5_CAPE"]
ALLOWED = OBSERVATIONS + WEATHER


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_observations(path: str | Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path)
    return clean_observations(raw, sha256(path))


def clean_observations(raw: pd.DataFrame, input_sha256=None) -> tuple[pd.DataFrame, dict]:
    required = {"storm_id", "pt_dateTimeISO", TARGET, "pt_lat", "pt_lon"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    # The original export is DD-MM-YY HH:MM. Never infer ambiguous month/day order.
    timestamp = pd.to_datetime(
        raw["pt_dateTimeISO"], format="%d-%m-%y %H:%M", utc=True, errors="coerce"
    )
    if timestamp.isna().any():
        raise ValueError(f"Unparseable timestamps: {int(timestamp.isna().sum())}")
    frame = raw[["storm_id"] + [c for c in ALLOWED if c in raw]].copy()
    frame["timestamp"] = timestamp
    if frame["storm_id"].isna().any():
        raise ValueError("Missing storm identifiers")
    for column in ALLOWED:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
    frame = frame.sort_values(KEY, kind="stable")
    n_input = len(frame)
    # Drop duplicates in the predictive observations; ignore bookkeeping differences,
    # such as pt_index, but never silently choose between contradictory wind labels.
    frame = frame.drop_duplicates()
    n_exact = n_input - len(frame)
    conflicting = frame.duplicated(KEY, keep=False)
    conflicts = frame.loc[conflicting, KEY].drop_duplicates()
    n_conflicting_rows = int(conflicting.sum())
    frame = frame.loc[~conflicting].copy()
    invalid = (
        frame[TARGET].isna()
        | frame[TARGET].lt(0)
        | frame[TARGET].gt(150)
        | ~frame["pt_lat"].between(-90, 90)
        | ~frame["pt_lon"].between(-180, 180)
    )
    n_invalid = int(invalid.sum())
    frame = frame.loc[~invalid].reset_index(drop=True)
    gaps = frame.groupby("storm_id")["timestamp"].diff().dt.total_seconds().div(3600)
    audit = {
        "input_sha256": input_sha256,
        "input_rows": n_input,
        "original_columns": len(raw.columns),
        "removed_equivalent_observations": n_exact,
        "conflicting_keys": len(conflicts),
        "conflicting_rows_removed": n_conflicting_rows,
        "conflict_examples": conflicts.astype(str).head(20).to_dict("records"),
        "invalid_rows_removed": n_invalid,
        "clean_rows": len(frame),
        "storms": int(frame["storm_id"].nunique()),
        "time_min": str(frame["timestamp"].min()),
        "time_max": str(frame["timestamp"].max()),
        "gap_hours_counts": {str(k): int(v) for k, v in gaps.value_counts().items()},
        "missing_by_column": frame.isna().sum().astype(int).to_dict(),
        "excluded_columns": sorted(
            set(raw.columns) - {"storm_id", "pt_dateTimeISO"} - set(ALLOWED)
        ),
        "availability_caveat": "ERA5 reanalysis is retrospective; as-of-time availability is unverified.",
    }
    return frame, audit
