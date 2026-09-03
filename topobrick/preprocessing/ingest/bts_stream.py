"""All-in-one streaming pipeline for BTS sites."""

from __future__ import annotations

import glob
import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from topobrick.preprocessing.ingest.canonical import _detect_outages, _interp_with_masks
from topobrick.utils.paths import ensure_directory, resolve_path
from topobrick.utils.progress import log

_CANONICAL_SCHEMA = pa.schema(
    [
        pa.field("dataset", pa.string()),
        pa.field("building_id", pa.string()),
        pa.field("sensor_id", pa.string()),
        pa.field("kg_node_id", pa.string()),
        pa.field("timestamp", pa.timestamp("ns")),
        pa.field("value", pa.float32()),
        pa.field("unit", pa.string()),
        pa.field("raw_name", pa.string()),
        pa.field("brick_class", pa.string()),
        pa.field("is_numeric", pa.bool_()),
        pa.field("observed_mask", pa.bool_()),
        pa.field("imputed_mask", pa.bool_()),
        pa.field("outage_mask", pa.bool_()),
    ]
)


def _bts_pickle_to_series(
    pickle_path: str,
    building_id: str,
    freq: str,
) -> Tuple[str, str, pd.Series]:
    """Return (sensor_id, raw_name, resampled_series). May raise."""
    with open(pickle_path, "rb") as f:
        obj = pickle.load(f)
    if not (isinstance(obj, list) and len(obj) == 3):
        raise ValueError("unexpected pickle structure")
    stream_name, times, values = obj
    times = np.asarray(times)
    values = np.asarray(values, dtype=np.float32)
    base = os.path.basename(str(stream_name))
    base = base[: -len(".pickle")] if base.endswith(".pickle") else base
    pref = f"{building_id}_"
    uuid = base[len(pref) :] if base.startswith(pref) else base
    if len(times) == 0:
        return uuid, uuid, pd.Series(dtype=np.float32)
    s = pd.Series(values, index=pd.DatetimeIndex(pd.to_datetime(times)))
    s = s.groupby(level=0).mean()
    s = s.resample(freq).mean()
    return uuid, uuid, s


def stream_bts_to_canonical(
    cfg: Dict[str, Any],
    out_dir: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Stream BTS pickles -> series.parquet, return (sensor_meta, stats)."""
    buildings = cfg["bts"]["buildings"]
    only = cfg["bts"].get("only_buildings")
    if only:
        buildings = [b for b in buildings if b["building_id"] in only]
        log(
            f"BTS streaming: only_buildings filter active -> {[b['building_id'] for b in buildings]}"
        )

    freq = cfg["freq"]
    drop_ratio = cfg["preprocessing"]["drop_missing_ratio"]
    max_gap = cfg["preprocessing"]["max_interpolation_gap"]
    edge_max = cfg["preprocessing"]["edge_fill_max"]
    L = cfg["L"]
    Hmax = max(cfg["horizons"])
    min_points = L + Hmax + cfg["preprocessing"].get("min_required_points_extra", 0)
    stuck_steps = int(cfg["preprocessing"].get("stuck_flat_steps", 24))
    zero_thresh = float(cfg["preprocessing"].get("drop_zero_threshold", 1e-3))
    drop_zero_steps = int(cfg["preprocessing"].get("drop_to_zero_steps", 12))

    ensure_directory(out_dir)
    series_path = os.path.join(out_dir, "series.parquet")
    if os.path.exists(series_path):
        os.remove(series_path)
    writer = pq.ParquetWriter(series_path, _CANONICAL_SCHEMA, compression="snappy")

    sensor_meta_rows: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    missing_ratios: List[float] = []
    n_total = 0
    n_kept = 0

    for b in buildings:
        bid = b["building_id"]
        pdir = resolve_path(cfg["paths"]["raw_root"], b["pickle_dir"])
        if not os.path.isdir(pdir):
            log(f"BTS-stream: skip {bid}, dir missing: {pdir}")
            continue
        # Preload metadata.csv for BTS — gives us sensor_id (StreamID) ->
        # Brick class so outage detection can skip Setpoint/Command classes.
        sid2bc: Dict[str, str] = {}
        meta_path = b.get("metadata_csv")
        if meta_path:
            mp = resolve_path(cfg["paths"]["raw_root"], meta_path)
            if os.path.exists(mp):
                meta = pd.read_csv(mp)
                if "StreamID" in meta.columns and "Brick" in meta.columns:
                    sid2bc = dict(
                        zip(meta["StreamID"].astype(str), meta["Brick"].astype(str))
                    )
                    log(f"  loaded {len(sid2bc)} sid->brick mappings from {mp}")
        files = sorted(glob.glob(os.path.join(pdir, "*.pickle")))
        log(f"BTS-stream {bid}: {len(files)} pickles in {pdir}")
        for i, fp in enumerate(files):
            try:
                sensor_id, raw_name, s = _bts_pickle_to_series(fp, bid, freq)
            except Exception as e:
                if i < 5:
                    log(f"  parse fail {os.path.basename(fp)}: {e}")
                continue
            n_total += 1
            if len(s) == 0:
                dropped.append({"sensor_id": sensor_id, "reason": "empty"})
                continue
            if s.index.tz is not None:
                s.index = s.index.tz_convert(None)
            miss = float(s.isna().mean())
            missing_ratios.append(miss)
            if miss > drop_ratio:
                dropped.append(
                    {
                        "sensor_id": sensor_id,
                        "building_id": bid,
                        "reason": "missing_ratio_too_high",
                        "ratio": miss,
                    }
                )
                continue
            filled, observed_mask, imputed_mask = _interp_with_masks(
                s, max_gap, edge_max
            )
            valid_count = int(filled.notna().sum())
            if valid_count < min_points:
                dropped.append(
                    {
                        "sensor_id": sensor_id,
                        "building_id": bid,
                        "reason": "too_few_valid_points",
                        "n": valid_count,
                        "min_required": min_points,
                    }
                )
                continue
            sensor_bc = sid2bc.get(sensor_id, "")
            outage_mask = _detect_outages(
                filled.values.astype(np.float64),
                stuck_flat_steps=stuck_steps,
                zero_threshold=zero_thresh,
                drop_to_zero_steps=drop_zero_steps,
                brick_class=sensor_bc,
            )
            n_kept += 1
            n_rows = len(filled)
            tbl = pa.table(
                {
                    "dataset": pa.array(["BTS"] * n_rows),
                    "building_id": pa.array([bid] * n_rows),
                    "sensor_id": pa.array([sensor_id] * n_rows),
                    "kg_node_id": pa.nulls(n_rows, type=pa.string()),
                    "timestamp": pa.array(filled.index.values, type=pa.timestamp("ns")),
                    "value": pa.array(
                        filled.values.astype(np.float32), type=pa.float32()
                    ),
                    "unit": pa.nulls(n_rows, type=pa.string()),
                    "raw_name": pa.array([raw_name] * n_rows),
                    "brick_class": pa.nulls(n_rows, type=pa.string()),
                    "is_numeric": pa.array([True] * n_rows, type=pa.bool_()),
                    "observed_mask": pa.array(observed_mask.values, type=pa.bool_()),
                    "imputed_mask": pa.array(imputed_mask.values, type=pa.bool_()),
                    "outage_mask": pa.array(outage_mask, type=pa.bool_()),
                },
                schema=_CANONICAL_SCHEMA,
            )
            writer.write_table(tbl)
            sensor_meta_rows.append(
                {
                    "dataset": "BTS",
                    "building_id": bid,
                    "sensor_id": sensor_id,
                    "raw_name": raw_name,
                    "n_rows": n_rows,
                    "n_observed": int(observed_mask.sum()),
                    "n_imputed": int(imputed_mask.sum()),
                    "n_outage": int(outage_mask.sum()),
                }
            )
            del s, filled, observed_mask, imputed_mask, outage_mask, tbl
            if (i + 1) % 500 == 0:
                log(f"  {bid}: processed {i + 1}/{len(files)} (kept {n_kept})")

    writer.close()
    log(f"BTS-stream: kept {n_kept}/{n_total} sensors -> {series_path}")

    sensor_meta = pd.DataFrame(sensor_meta_rows)
    stats = {
        "num_raw_timeseries": n_total,
        "num_kept_timeseries": n_kept,
        "num_dropped_timeseries": n_total - n_kept,
        "missing_ratio_summary": {
            "mean": float(np.mean(missing_ratios)) if missing_ratios else None,
            "median": float(np.median(missing_ratios)) if missing_ratios else None,
            "p90": float(np.quantile(missing_ratios, 0.9)) if missing_ratios else None,
        },
        "dropped_sensor_summary": dropped[:200],
        "dropped_sensor_count": len(dropped),
        "series_path": series_path,
    }
    return sensor_meta, stats
