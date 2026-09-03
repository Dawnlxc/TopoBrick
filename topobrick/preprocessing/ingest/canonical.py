"""Resample to the 15-minute grid, and say what each slot means."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from topobrick.utils.progress import log


def _interp_with_masks(
    s: pd.Series,
    max_gap: int,
    edge_max: int,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    observed = s.notna().to_numpy()
    interp = s.interpolate(
        method="linear", limit=max_gap, limit_area="inside", limit_direction="both"
    )
    # `fillna(method=...)` was removed in pandas 3; ffill/bfill are the same op.
    interp = interp.ffill(limit=edge_max)
    interp = interp.bfill(limit=edge_max)
    imputed = (~observed) & interp.notna().to_numpy()
    return interp, pd.Series(observed, index=s.index), pd.Series(imputed, index=s.index)


def _runs_at_least(condition: np.ndarray, min_len: int) -> np.ndarray:
    """Return mask True at positions inside a run of `condition` of length >= min_len."""
    n = condition.size
    out = np.zeros(n, dtype=bool)
    if n == 0 or min_len <= 0:
        return out
    cond = condition.astype(bool)
    # find run boundaries
    pad = np.concatenate([[False], cond, [False]])
    diff = np.diff(pad.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]  # exclusive
    for s, e in zip(starts, ends):
        if (e - s) >= min_len:
            out[s:e] = True
    return out


def _is_setpoint_or_command_class(brick_class: str) -> bool:
    """Setpoints / commands / parameters are SUPPOSED to be flat for long periods (user changes them rarely)."""
    if not brick_class:
        return False
    bc = brick_class.lower()
    return any(
        t in bc
        for t in (
            "setpoint",
            "command",
            "parameter",
            "limit",
            "deadband",
            "schedule",
        )
    )


def _detect_spikes(values: np.ndarray, k: float = 10.0) -> np.ndarray:
    """Mark sentinel/sensor-fail values via robust dispersion estimate."""
    n = len(values)
    out = np.zeros(n, dtype=bool)
    finite = np.isfinite(values)
    if finite.sum() < 10:
        return out
    v = values[finite]
    med = float(np.nanmedian(v))
    mad = float(np.nanmedian(np.abs(v - med)))
    q25, q75 = np.nanpercentile(v, [25, 75])
    iqr_sigma = float((q75 - q25) / 1.349)
    scale = abs(med) * 0.10 + 1.0
    thr = k * max(mad, iqr_sigma, scale)
    out |= (np.abs(values - med) > thr) & finite
    return out


def _detect_outages(
    values: np.ndarray,
    stuck_flat_steps: int,
    zero_threshold: float,
    drop_to_zero_steps: int,
    brick_class: str | None = None,
    spike_k: float = 10.0,
) -> np.ndarray:
    """Vectorised outage detection."""
    n = len(values)
    out = np.zeros(n, dtype=bool)
    if n < max(stuck_flat_steps, drop_to_zero_steps):
        return out

    # Apply spike detection to all classes, including control signals.
    out |= _detect_spikes(values, k=spike_k)

    # Setpoints/commands: skip the stuck-flat and drop-to-zero detectors
    # below (those are wrong signals for setpoint-typed series).
    if _is_setpoint_or_command_class(brick_class or ""):
        return out

    finite = np.isfinite(values)
    abs_med = float(np.nanmedian(np.abs(values[finite]))) if finite.any() else 0.0

    # 1. stuck-flat (non-zero plateau)
    if stuck_flat_steps > 1 and n >= 2:
        same = np.zeros(n, dtype=bool)
        diff_ok = (np.abs(np.diff(values)) < 1e-6) & finite[1:] & finite[:-1]
        same[1:] = diff_ok
        same[:-1] |= diff_ok
        non_zero = np.abs(values) > 1e-6
        flat_cond = same & non_zero & finite
        out |= _runs_at_least(flat_cond, stuck_flat_steps)

    # 2. drop-to-zero on non-zero sensor
    if abs_med > zero_threshold and drop_to_zero_steps > 1:
        zero_run = (np.abs(values) < zero_threshold) & finite
        out |= _runs_at_least(zero_run, drop_to_zero_steps)

    return out


def recompute_outage_with_classes(
    series_df: pd.DataFrame,
    sid2class: Dict[str, str],
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Recompute `outage_mask` now that each sensor's Brick class is known."""
    if series_df.empty or not sid2class:
        return series_df
    stuck_steps = int(cfg["preprocessing"].get("stuck_flat_steps", 24))
    zero_thresh = float(cfg["preprocessing"].get("drop_zero_threshold", 1e-3))
    drop_zero_steps = int(cfg["preprocessing"].get("drop_to_zero_steps", 12))

    out = series_df.sort_values(["sensor_id", "timestamp"]).copy()
    mask = out["outage_mask"].to_numpy(dtype=bool, copy=True)
    pos = 0
    n_sensors = n_changed = 0
    for sid, grp in out.groupby("sensor_id", sort=False):
        k = len(grp)
        bc = sid2class.get(str(sid))
        if bc:
            n_sensors += 1
            fresh = _detect_outages(
                grp["value"].to_numpy(dtype=np.float64),
                stuck_flat_steps=stuck_steps,
                zero_threshold=zero_thresh,
                drop_to_zero_steps=drop_zero_steps,
                brick_class=bc,
            )
            if not np.array_equal(fresh, mask[pos : pos + k]):
                n_changed += 1
            mask[pos : pos + k] = fresh
        pos += k
    out["outage_mask"] = mask
    log(
        f"recomputed outage_mask with Brick classes: {n_sensors} sensors classed, "
        f"{n_changed} changed; outage {100 * series_df['outage_mask'].mean():.2f}% "
        f"-> {100 * out['outage_mask'].mean():.2f}%"
    )
    return out


def resample_and_clean(
    raw_df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Per-sensor resample + clean to a uniform grid."""
    drop_ratio = cfg["preprocessing"]["drop_missing_ratio"]
    max_gap = cfg["preprocessing"]["max_interpolation_gap"]
    edge_max = cfg["preprocessing"]["edge_fill_max"]
    L = cfg["L"]
    Hmax = max(cfg["horizons"])
    min_points = L + Hmax + cfg["preprocessing"].get("min_required_points_extra", 0)
    stuck_steps = int(cfg["preprocessing"].get("stuck_flat_steps", 24))  # 6 h @ 15 min
    zero_thresh = float(cfg["preprocessing"].get("drop_zero_threshold", 1e-3))
    drop_zero_steps = int(
        cfg["preprocessing"].get("drop_to_zero_steps", 12)
    )  # 3 h @ 15 min

    out_chunks: List[pd.DataFrame] = []
    dropped_summary: List[Dict[str, Any]] = []
    missing_ratios: List[float] = []
    n_total = 0
    n_kept = 0

    # Streaming mode: iterate shard files
    if "_shard_path" in raw_df.columns:
        shards = raw_df["_shard_path"].tolist()
        log(f"resample_and_clean (streaming): {len(shards)} shards")

        def _shard_groups():
            for sp in shards:
                shard = pd.read_parquet(sp)
                shard["timestamp"] = pd.to_datetime(shard["timestamp"], errors="coerce")
                shard = shard.dropna(subset=["timestamp"])
                for key, grp in shard.groupby(
                    ["dataset", "building_id", "sensor_id"], sort=False
                ):
                    yield key, grp
                # free shard memory before reading the next
                del shard

        groups_iter = _shard_groups()
    else:
        raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], errors="coerce")
        raw_df = raw_df.dropna(subset=["timestamp"])
        groups_iter = raw_df.groupby(
            ["dataset", "building_id", "sensor_id"], sort=False
        )

    for (dataset_name, building_id, sensor_id), grp in groups_iter:
        n_total += 1
        # Sort + dedup by averaging duplicate timestamps. Loader pre-resamples to
        # `freq` already, so we only enforce the canonical grid here.
        grp = grp.sort_values("timestamp")
        grp = grp.drop_duplicates(subset=["timestamp"], keep="last")
        s = pd.Series(
            pd.to_numeric(grp["value"], errors="coerce").values,
            index=pd.DatetimeIndex(grp["timestamp"].values),
        )
        if s.index.tz is not None:
            s.index = s.index.tz_convert(None)
        # cheap re-bin only if grid is irregular
        if not s.index.is_monotonic_increasing or s.index.has_duplicates:
            s = s.groupby(level=0).mean().sort_index()
        if len(s) == 0:
            dropped_summary.append(
                {"sensor_id": sensor_id, "reason": "empty_after_resample"}
            )
            continue
        miss_ratio = float(s.isna().mean())
        missing_ratios.append(miss_ratio)
        if miss_ratio > drop_ratio:
            dropped_summary.append(
                {
                    "sensor_id": sensor_id,
                    "building_id": building_id,
                    "reason": "missing_ratio_too_high",
                    "ratio": miss_ratio,
                }
            )
            continue
        filled, observed_mask, imputed_mask = _interp_with_masks(s, max_gap, edge_max)
        valid_count = int(filled.notna().sum())
        if valid_count < min_points:
            dropped_summary.append(
                {
                    "sensor_id": sensor_id,
                    "building_id": building_id,
                    "reason": "too_few_valid_points",
                    "n": valid_count,
                    "min_required": min_points,
                }
            )
            continue
        n_kept += 1
        unit = grp["unit"].dropna().iloc[0] if grp["unit"].notna().any() else None
        raw_name = grp["raw_name"].iloc[0]
        outage_mask = _detect_outages(
            filled.values.astype(np.float64),
            stuck_flat_steps=stuck_steps,
            zero_threshold=zero_thresh,
            drop_to_zero_steps=drop_zero_steps,
        )
        out_chunks.append(
            pd.DataFrame(
                {
                    "dataset": dataset_name,
                    "building_id": building_id,
                    "sensor_id": sensor_id,
                    "kg_node_id": pd.NA,
                    "timestamp": filled.index.values,
                    "value": filled.values.astype(np.float32),
                    "unit": unit,
                    "raw_name": raw_name,
                    "brick_class": pd.NA,
                    "is_numeric": True,
                    "observed_mask": observed_mask.values,
                    "imputed_mask": imputed_mask.values,
                    "outage_mask": outage_mask,
                }
            )
        )

    if out_chunks:
        series = pd.concat(out_chunks, ignore_index=True)
    else:
        series = pd.DataFrame(
            columns=[
                "dataset",
                "building_id",
                "sensor_id",
                "kg_node_id",
                "timestamp",
                "value",
                "unit",
                "raw_name",
                "brick_class",
                "is_numeric",
                "observed_mask",
                "imputed_mask",
            ]
        )
    stats = {
        "num_raw_timeseries": n_total,
        "num_kept_timeseries": n_kept,
        "num_dropped_timeseries": n_total - n_kept,
        "missing_ratio_summary": {
            "mean": float(np.mean(missing_ratios)) if missing_ratios else None,
            "median": float(np.median(missing_ratios)) if missing_ratios else None,
            "p90": float(np.quantile(missing_ratios, 0.9)) if missing_ratios else None,
        },
        "dropped_sensor_summary": dropped_summary[:200],  # cap report size
        "dropped_sensor_count": len(dropped_summary),
    }
    log(f"resample_and_clean: kept {n_kept}/{n_total} sensors")
    return series, stats
