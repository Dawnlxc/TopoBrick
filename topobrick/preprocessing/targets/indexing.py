"""Enumerate valid window-index rows for a target sensor."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def enumerate_sensor_windows(
    timestamps: np.ndarray,
    values: np.ndarray,
    observed_mask: np.ndarray,
    outage_mask: np.ndarray,
    lookback: int,
    horizon: int,
    splits: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    strides: dict[str, int],
    target_node_id: str,
    sensor_id: str,
    building_id: str,
    dataset_name: str,
    max_windows: int | None,
    max_horizon_invalid_frac: float,
) -> list[dict[str, Any]]:
    """Enumerate valid windows for one sensor and forecast horizon."""
    if len(timestamps) < lookback + horizon:
        return []
    has_nan = np.isnan(values)
    invalid_target = (~observed_mask) | outage_mask

    cumulative_nan = np.concatenate([[0], np.cumsum(has_nan.astype(np.int32))])
    cumulative_invalid = np.concatenate(
        [[0], np.cumsum(invalid_target.astype(np.int32))]
    )

    def any_nan(lo, hi_excl):
        return (cumulative_nan[hi_excl] - cumulative_nan[lo]) > 0

    def invalid_count(lo, hi_excl):
        return cumulative_invalid[hi_excl] - cumulative_invalid[lo]

    rows: list[dict[str, Any]] = []
    for split, (split_start, split_end) in splits.items():
        stride = strides[split]
        first_target_idx = np.searchsorted(
            timestamps, np.datetime64(split_start), side="left"
        )
        last_target_idx = (
            np.searchsorted(timestamps, np.datetime64(split_end), side="left") - 1
        )
        first_anchor = max(lookback - 1, first_target_idx - 1)
        last_anchor = min(len(timestamps) - horizon - 1, last_target_idx - horizon)
        if last_anchor < first_anchor:
            continue
        anchors = np.arange(first_anchor, last_anchor + 1, stride, dtype=np.int64)
        if anchors.size == 0:
            continue
        lo = anchors - (lookback - 1)
        hi_excl = anchors + 1 + horizon
        bad_nan = any_nan(lo, hi_excl)
        horizon_invalid_fraction = invalid_count(
            anchors + 1, anchors + 1 + horizon
        ).astype(np.float32) / float(horizon)
        invalid_horizon = horizon_invalid_fraction > max_horizon_invalid_frac
        keep = ~(bad_nan | invalid_horizon)
        anchors = anchors[keep]
        horizon_invalid_fraction = horizon_invalid_fraction[keep]
        if anchors.size == 0:
            continue
        if max_windows is not None and max_windows > 0 and anchors.size > max_windows:
            selected = np.linspace(0, anchors.size - 1, max_windows).astype(np.int64)
            anchors = anchors[selected]
            horizon_invalid_fraction = horizon_invalid_fraction[selected]

        anchor_timestamps = timestamps[anchors]
        input_starts = timestamps[anchors - (lookback - 1)]
        input_ends = timestamps[anchors]
        target_starts = timestamps[anchors + 1]
        target_ends = timestamps[anchors + horizon]
        for i in range(anchors.size):
            rows.append(
                {
                    "dataset": dataset_name,
                    "building_id": building_id,
                    "target_node_id": target_node_id,
                    "sensor_id": sensor_id,
                    "t_anchor": anchor_timestamps[i],
                    "input_start": input_starts[i],
                    "input_end": input_ends[i],
                    "target_start": target_starts[i],
                    "target_end": target_ends[i],
                    "split": split,
                    "horizon": horizon,
                    "subgraph_cache_key": target_node_id,
                    "horizon_invalid_frac": float(horizon_invalid_fraction[i]),
                }
            )
    return rows
