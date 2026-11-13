"""Exact-clock past features and aligned future labels. No future interpolation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import ALLOWED, KEY, TARGET

SEQUENCE_VARIABLES = [
    TARGET,
    "pt_pressureMB",
    "pt_lat",
    "pt_lon",
    "ERA5_MSLP",
    "ERA5_u_850",
    "ERA5_v_850",
]


def lookup(frame: pd.DataFrame, column: str, offsets_hours: int) -> np.ndarray:
    """Read same-storm observations at timestamp + offset, never nearest/future fill."""
    source = frame.set_index(KEY)[column]
    index = pd.MultiIndex.from_arrays(
        [frame["storm_id"], frame["timestamp"] + pd.Timedelta(hours=offsets_hours)], names=KEY
    )
    return source.reindex(index).to_numpy()


def feature_frame(
    observations: pd.DataFrame, history_hours: list[int]
) -> tuple[pd.DataFrame, list[str], list[str]]:
    if not history_hours or history_hours[0] != 0 or sorted(set(history_hours)) != history_hours:
        raise ValueError("history_hours must start at zero and increase without duplicates")
    frame = observations.sort_values(KEY).reset_index(drop=True).copy()
    result = frame[KEY].copy()
    columns = []
    for name in ALLOWED:
        if name in frame:
            result[name] = frame[name]
            columns.append(name)
    # A complete wind history is shared by every model, so model comparison uses
    # the same origin timestamps and target horizon, including DL and persistence.
    sequence_columns = []
    for lag in history_hours:
        for name in SEQUENCE_VARIABLES:
            if name in frame:
                key = f"seq_{name}_lag{lag}"
                result[key] = lookup(frame, name, -lag)
                columns.append(key)
                sequence_columns.append(key)
    wind_history = [f"seq_{TARGET}_lag{lag}" for lag in history_hours]
    result["wind_history_mean"] = result[wind_history].mean(axis=1)
    result["wind_history_std"] = result[wind_history].std(axis=1)
    # Express rates per actual clock hour, rather than assuming every row is 6h apart.
    if 6 in history_hours:
        result["wind_change_6h"] = result[TARGET] - result[f"seq_{TARGET}_lag6"]
        columns.append("wind_change_6h")
    columns += ["wind_history_mean", "wind_history_std"]
    for level in [850, 500, 200]:
        u, v = f"ERA5_u_{level}", f"ERA5_v_{level}"
        if u in result and v in result:
            key = f"wind_magnitude_{level}"
            result[key] = np.hypot(result[u], result[v])
            columns.append(key)
    if all(c in result for c in ["ERA5_u_850", "ERA5_v_850", "ERA5_u_200", "ERA5_v_200"]):
        result["wind_shear_850_200"] = np.hypot(
            result["ERA5_u_850"] - result["ERA5_u_200"], result["ERA5_v_850"] - result["ERA5_v_200"]
        )
        columns.append("wind_shear_850_200")
    for period, values in [
        ("hour", frame["timestamp"].dt.hour / 24),
        ("month", (frame["timestamp"].dt.month - 1) / 12),
        ("longitude", (frame["pt_lon"] + 180) / 360),
    ]:
        for kind, function in [("sin", np.sin), ("cos", np.cos)]:
            key = f"{period}_{kind}"
            result[key] = function(2 * np.pi * values)
            columns.append(key)
    keep = result[wind_history].notna().all(axis=1)
    result["history_start"] = frame["timestamp"] - pd.Timedelta(hours=max(history_hours))
    result = result.loc[keep].reset_index(drop=True)
    return result, columns, sequence_columns


def attach_targets(
    observations: pd.DataFrame, frame: pd.DataFrame, horizon_hours: int
) -> pd.DataFrame:
    if horizon_hours <= 0:
        raise ValueError("Forecast horizon must be positive")
    index = pd.MultiIndex.from_arrays(
        [frame["storm_id"], frame["timestamp"] + pd.Timedelta(hours=horizon_hours)], names=KEY
    )
    result = frame.copy()
    result["target"] = observations.set_index(KEY)[TARGET].reindex(index).to_numpy()
    result["target_time"] = frame["timestamp"] + pd.Timedelta(hours=horizon_hours)
    return result.dropna(subset=["target"]).reset_index(drop=True)
