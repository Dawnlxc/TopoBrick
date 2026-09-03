"""L4: window-level target validation."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

from topobrick.preprocessing.targets.indexing import enumerate_sensor_windows
from topobrick.preprocessing.targets.selection import load_forecast_targets
from topobrick.utils.paths import resolve_processed_root

PROCESSED_ROOT = resolve_processed_root()
DATASET_FREQUENCY_MINUTES = {
    "LBNL59": 15,
    "BTS_Site_B": 15,
    "BTS_Site_C": 15,
}
ABSOLUTE_FLAT_STD = 1e-4
MAX_CONSTANT_RUN_FRACTION = 0.75
MIN_SENSOR_RELATIVE_STD = 0.005
MIN_SENSOR_ABSOLUTE_STD = 0.05


def add_months(timestamp: pd.Timestamp, months: int) -> pd.Timestamp:
    return (timestamp.to_period("M") + months).to_timestamp()


def build_validated_windows(
    dataset: str,
    start_date: str,
    end_date: str,
    train_months: int,
    validation_months: int,
    test_months: int,
    lookback: int,
    horizon: int,
    stride: int = 16,
    max_horizon_invalid_fraction: float = 0.05,
    processed_root: str = PROCESSED_ROOT,
) -> None:
    if not 0 <= max_horizon_invalid_fraction <= 1:
        raise ValueError("max_horizon_invalid_fraction must be between 0 and 1")

    processed_directory = os.path.join(processed_root, dataset)
    print(f"\n=== {dataset} ===")
    print(
        f"  date range: {start_date} → {end_date}  "
        f"({train_months}m train / {validation_months}m val / {test_months}m test)"
    )

    targets = load_forecast_targets(processed_directory)
    node_to_sensor = dict(zip(targets["target_node_id"], targets["sensor_id"]))
    print(f"  {len(node_to_sensor)} L3 forecast targets")

    t_start = pd.Timestamp(start_date)
    train_end = add_months(t_start, train_months)
    val_end = add_months(t_start, train_months + validation_months)
    test_end = add_months(t_start, train_months + validation_months + test_months)
    if test_end > pd.Timestamp(end_date):
        test_end = pd.Timestamp(end_date)
    splits = {
        "train": (t_start, train_end),
        "val": (train_end, val_end),
        "test": (val_end, test_end),
    }
    print(f"  train: {splits['train'][0]} → {splits['train'][1]}")
    print(f"  val:   {splits['val'][0]} → {splits['val'][1]}")
    print(f"  test:  {splits['test'][0]} → {splits['test'][1]}")
    strides = {"train": stride, "val": stride, "test": stride}
    frequency_minutes = DATASET_FREQUENCY_MINUTES[dataset]
    print(
        f"  window stride: {stride} steps "
        f"({stride * frequency_minutes / 60:.1f}h @ {frequency_minutes}-min)"
    )

    lookback_pad = pd.Timedelta(minutes=frequency_minutes * (lookback + 5))
    t_lo = t_start - lookback_pad
    t_hi = test_end + pd.Timedelta(days=1)

    # Push target and time predicates into PyArrow before materialization.
    series_dataset = pads.dataset(
        os.path.join(processed_directory, "series.parquet"), format="parquet"
    )
    series_filter = (
        pads.field("sensor_id").isin(sorted(node_to_sensor.values()))
        & (pads.field("timestamp") >= t_lo)
        & (pads.field("timestamp") <= t_hi)
    )
    series = series_dataset.to_table(
        columns=[
            "sensor_id",
            "timestamp",
            "value",
            "observed_mask",
            "outage_mask",
            "building_id",
        ],
        filter=series_filter,
    ).to_pandas()
    series["timestamp"] = pd.to_datetime(series["timestamp"])
    print(f"  filtered series: {len(series):,} rows")

    sensor_to_node = {sensor: node for node, sensor in node_to_sensor.items()}
    all_windows: list[dict[str, Any]] = []
    broken_window_count = 0
    invalid_window_count = 0
    kept_window_count = 0
    low_variability_sensor_count = 0

    for sensor_id, sensor_series in series.groupby("sensor_id"):
        sensor_series = sensor_series.sort_values("timestamp")
        timestamps = sensor_series["timestamp"].values
        values = sensor_series["value"].values.astype(np.float32)
        observed_mask = sensor_series["observed_mask"].values.astype(bool)
        outage_mask = sensor_series["outage_mask"].values.astype(bool)
        building_id = sensor_series["building_id"].iloc[0]
        target_node_id = sensor_to_node.get(sensor_id)
        if target_node_id is None:
            continue
        if observed_mask.sum() > 100:
            sensor_std = float(np.std(values[observed_mask]))
            sensor_mean_abs = abs(float(np.mean(values[observed_mask])))
        else:
            sensor_std = 0.0
            sensor_mean_abs = 0.0
        if sensor_std < ABSOLUTE_FLAT_STD:
            continue
        if sensor_std < MIN_SENSOR_ABSOLUTE_STD:
            low_variability_sensor_count += 1
            continue
        if (
            sensor_mean_abs > 1.0
            and (sensor_std / sensor_mean_abs) < MIN_SENSOR_RELATIVE_STD
        ):
            low_variability_sensor_count += 1
            continue
        sensor_windows = enumerate_sensor_windows(
            timestamps=timestamps,
            values=values,
            observed_mask=observed_mask,
            outage_mask=outage_mask,
            lookback=lookback,
            horizon=horizon,
            splits=splits,
            strides=strides,
            target_node_id=target_node_id,
            sensor_id=sensor_id,
            building_id=building_id,
            dataset_name=dataset,
            max_windows=None,
            max_horizon_invalid_frac=max_horizon_invalid_fraction,
        )
        for window in sensor_windows:
            target_start_index = int(
                np.searchsorted(
                    timestamps,
                    np.datetime64(window["target_start"]),
                    side="left",
                )
            )
            target_end_index = target_start_index + horizon
            if target_end_index > len(values):
                continue
            target_values = values[target_start_index:target_end_index]
            target_observed = observed_mask[target_start_index:target_end_index]
            target_outage = outage_mask[target_start_index:target_end_index]
            trusted_target = target_observed & ~target_outage
            minimum_trusted = (1.0 - max_horizon_invalid_fraction) * horizon
            if int(trusted_target.sum()) < minimum_trusted:
                invalid_window_count += 1
                continue
            clean_target = target_values[trusted_target]
            if len(clean_target) < 2:
                invalid_window_count += 1
                continue
            if float(np.std(clean_target)) < ABSOLUTE_FLAT_STD:
                broken_window_count += 1
                continue
            constant_differences = np.abs(np.diff(clean_target)) < 1e-6
            if constant_differences.any():
                run_length = 0
                longest_run = 0
                for is_constant in constant_differences:
                    if is_constant:
                        run_length += 1
                    else:
                        longest_run = max(longest_run, run_length)
                        run_length = 0
                longest_run = max(longest_run, run_length) + 1
            else:
                longest_run = 1
            if longest_run > MAX_CONSTANT_RUN_FRACTION * horizon:
                broken_window_count += 1
                continue
            kept_window_count += 1
            all_windows.append(window)
    window_index = pd.DataFrame(all_windows)
    invalid_percent = 100 * max_horizon_invalid_fraction
    print(
        f"  filter: kept {kept_window_count}, "
        f"dropped {broken_window_count} broken/stuck windows, "
        f"{invalid_window_count} windows with >{invalid_percent:g}% invalid targets, "
        f"and {low_variability_sensor_count} low-variability sensors"
    )
    print(f"  total windows generated: {len(window_index):,}")
    if len(window_index) == 0:
        print("  (no windows; skip)")
        return

    for split in ("train", "val", "test"):
        split_windows = window_index[window_index["split"] == split].reset_index(
            drop=True
        )
        out_path = os.path.join(
            processed_directory,
            f"windows_L{lookback}_H{horizon}_{split}_clean.parquet",
        )
        split_windows.to_parquet(out_path, index=False)
        print(f"    {split}: {len(split_windows):,} windows → {out_path}")
