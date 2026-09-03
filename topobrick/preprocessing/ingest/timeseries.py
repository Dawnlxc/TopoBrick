"""Raw time-series loaders for LBNL59 and BTS."""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from topobrick.utils.progress import log

# LBNL59 column units guessed from column name (used only as metadata, not for math).
_LBNL_UNIT_HINTS = [
    ("temp", "degC"),
    ("co2", "ppm"),
    ("flow", "cfm"),
    ("press", "Pa"),
    ("damper", "%"),
    ("valve", "%"),
    ("spd", "%"),
    ("kw", "kW"),
    ("power", "kW"),
    ("mbtu", "MBtu/hr"),
    ("occ", "count"),
]


def _guess_unit(name: str) -> str | None:
    n = name.lower()
    for k, u in _LBNL_UNIT_HINTS:
        if k in n:
            return u
    return None


def load_lbnl59_timeseries(raw_dir: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Load LBNL59 Building 59 raw per-CSV files (Dryad clean dataset)."""
    csv_dir = os.path.join(raw_dir, "Building_59", "Bldg59_clean data")
    if not os.path.isdir(csv_dir):
        raise FileNotFoundError(csv_dir)
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    log(f"LBNL59: {len(csv_files)} raw CSV files in {csv_dir}")

    freq = cfg["freq"]
    frames: List[pd.DataFrame] = []
    n_sensors = 0
    for fp in csv_files:
        try:
            df = pd.read_csv(fp, low_memory=False)
        except Exception as e:
            log(f"  skip {os.path.basename(fp)}: {e}")
            continue
        if "date" not in df.columns:
            log(f"  skip {os.path.basename(fp)}: no 'date' column")
            continue
        df.columns = [c.strip() for c in df.columns]
        df = df.loc[:, [c for c in df.columns if c]]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        sensor_cols = [c for c in df.columns if c != "date"]
        for col in sensor_cols:
            v = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
            s = pd.Series(v.values, index=pd.DatetimeIndex(df["date"].values))
            # collapse duplicate raw timestamps
            s = s.groupby(level=0).mean()
            s = s.resample(freq).mean()
            sub = pd.DataFrame(
                {
                    "dataset": "LBNL59",
                    "building_id": "Bldg59",
                    "sensor_id": col,
                    "timestamp": s.index.values,
                    "value": s.values.astype(np.float32),
                    "unit": _guess_unit(col),
                    "raw_name": col,
                    "source_file": os.path.basename(fp),
                }
            )
            frames.append(sub)
            n_sensors += 1
        log(f"  parsed {os.path.basename(fp)}: {len(sensor_cols)} sensors")
    if not frames:
        raise RuntimeError(f"No LBNL59 series parsed from {csv_dir}")
    out = pd.concat(frames, ignore_index=True)
    log(f"LBNL59: {n_sensors} sensors, {len(out):,} resampled rows")
    return out
