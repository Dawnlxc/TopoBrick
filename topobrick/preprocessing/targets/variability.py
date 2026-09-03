"""Remove L4 targets with degenerate training-split variation."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

from topobrick.utils.paths import resolve_processed_root
from topobrick.utils.progress import log

PROCESSED_ROOT = resolve_processed_root()


def compute_per_sensor_train_std(
    processed_dir: str, horizon: int, windows_filename: str | None = None
) -> dict[str, float]:
    """Per-sensor std of the target value over TRAIN-split windows only."""
    log("  computing per-sensor train standard deviations")
    windows_file = windows_filename or f"windows_H{horizon}_train.parquet"
    windows = pd.read_parquet(os.path.join(processed_dir, windows_file))
    sensor_ids = windows["sensor_id"].dropna().astype(str).unique().tolist()
    if not sensor_ids or windows.empty:
        log("    no training windows found; using fallback scales at evaluation")
        return {}

    time_start = pd.Timestamp(windows["input_start"].min())
    time_end = pd.Timestamp(windows["target_end"].max())
    series = pads.dataset(
        os.path.join(processed_dir, "series.parquet"), format="parquet"
    )
    row_filter = (
        pads.field("sensor_id").isin(sensor_ids)
        & (pads.field("timestamp") >= time_start)
        & (pads.field("timestamp") <= time_end)
        & pads.field("observed_mask")
    )
    values = series.to_table(
        columns=["sensor_id", "value"], filter=row_filter
    ).to_pandas()

    train_std_by_sensor: dict[str, float] = {}
    for sensor_id, group in values.groupby("sensor_id", sort=False):
        observed = group["value"].astype(np.float32).values
        train_std_by_sensor[str(sensor_id)] = (
            float(observed.std()) + 1e-6 if observed.size >= 2 else 1.0
        )
    if train_std_by_sensor:
        std_values = train_std_by_sensor.values()
        log(
            f"    train standard deviations for {len(train_std_by_sensor)} sensors "
            f"(range {min(std_values):.4f} - {max(std_values):.2f})"
        )
    else:
        log("    no training observations found; using fallback scales at evaluation")
    return train_std_by_sensor


def drop_low_variability_targets(
    dataset: str,
    processed_root: str = PROCESSED_ROOT,
    horizons=(24, 48, 72, 96),
    lookback: int = 96,
    min_train_std: float = 0.1,
    dry_run: bool = False,
) -> set:
    proc = os.path.join(processed_root, dataset)
    print(f"=== {dataset}  L4 degenerate-sensor drop (std_train < {min_train_std}) ===")

    bad_sids: set = set()
    for horizon in horizons:
        wf = f"windows_L{lookback}_H{horizon}_train_clean.parquet"
        sid2std = compute_per_sensor_train_std(proc, horizon, windows_filename=wf)
        bad = [sid for sid, s in sid2std.items() if s < min_train_std]
        print(f"  H={horizon}: {len(bad)}/{len(sid2std)} sensors below threshold")
        bad_sids.update(bad)
    print(f"\nsensors dropped (union over horizons): {len(bad_sids)}")

    if dry_run:
        print("[dry_run] no writes.")
        return bad_sids

    for horizon in horizons:
        for split in ("train", "val", "test"):
            p = os.path.join(
                proc, f"windows_L{lookback}_H{horizon}_{split}_clean.parquet"
            )
            df = pd.read_parquet(p)
            n0 = len(df)
            df = df[~df["sensor_id"].astype(str).isin(bad_sids)].reset_index(drop=True)
            df.to_parquet(p, index=False)
            pct = (n0 - len(df)) / n0 * 100 if n0 else 0.0
            print(
                f"  {split:<5} H={horizon:>3}: {n0:,} -> {len(df):,}  "
                f"({n0 - len(df):,} dropped, {pct:.1f}%)"
            )
    return bad_sids
