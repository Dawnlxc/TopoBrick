"""Per-sensor unit inference (statistical, not hard-coded by class)."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from topobrick.utils.progress import log

# Infer units from class-specific plausible value ranges.

_TEMP_UNIT_PROFILES = {
    # zone air / supply air / room air — comfort range
    "zone_air_temperature_sensor": [
        ("degC", (10, 35), None),
        ("degF", (55, 100), None),
    ],
    "air_temperature_sensor": [("degC", (-10, 40), None), ("degF", (-10, 110), None)],
    "supply_air_temperature": [("degC", (5, 45), None), ("degF", (40, 110), None)],
    "return_air_temperature": [("degC", (10, 40), None), ("degF", (55, 100), None)],
    "mixed_air_temperature": [("degC", (0, 40), None), ("degF", (32, 105), None)],
    "discharge_air_temperature": [("degC", (5, 45), None), ("degF", (40, 110), None)],
    "outdoor_air_temperature_sensor": [
        ("degC", (-30, 35), (-50, 50)),
        ("degF", (-20, 100), (-40, 130)),
    ],
    "chilled_water_temperature": [("degC", (1, 25), None), ("degF", (33, 80), None)],
    "hot_water_temperature": [("degC", (25, 95), None), ("degF", (80, 210), None)],
    "water_temperature": [("degC", (0, 100), None), ("degF", (32, 220), None)],
    "cooling_temperature_setpoint": [
        ("degC", (15, 30), None),
        ("degF", (60, 85), None),
    ],
    "heating_temperature_setpoint": [
        ("degC", (15, 25), None),
        ("degF", (55, 80), None),
    ],
    "temperature_setpoint": [("degC", (5, 35), None), ("degF", (40, 95), None)],
    "room_air_temperature_setpoint": [
        ("degC", (15, 30), None),
        ("degF", (60, 85), None),
    ],
}

# Other (non-temperature) classes have well-defined units, so a single accept.
_FIXED_UNIT_BY_CLASS = {
    "co2_sensor": "ppm",
    "relative_humidity_sensor": "%",
    "outdoor_air_humidity_sensor": "%",
    "humidity_sensor": "%",
    "valve_position_sensor": "%",
    "damper_position_sensor": "%",
    "position_sensor": "%",
    "voltage_sensor": "V",
    "current_sensor": "A",
    "frequency_sensor": "Hz",
    "electrical_power_sensor": "kW (or W)",
    "active_power_sensor": "kW (or W)",
    "reactive_power_sensor": "kVAR (or VAR)",
    "demand_sensor": "kW",
    "electrical_energy_sensor": "kWh",
}


def infer_unit_for_sensor(
    bc: str,
    median: float,
    p5: float,
    p95: float,
    n_obs: int,
) -> Tuple[Optional[str], str]:
    """Return (inferred_unit, confidence_reason)."""
    if not bc or n_obs < 50:
        return None, "no_class_or_too_few_samples"
    bcl = bc.lower()

    # Fixed-unit classes (humidity / position / electrical etc).
    for k, u in _FIXED_UNIT_BY_CLASS.items():
        if k in bcl:
            return u, "fixed_by_class"

    # Temperature classes: pick best-matching (degC vs degF) by median.
    profile = None
    for k, p in _TEMP_UNIT_PROFILES.items():
        if k in bcl:
            profile = p
            break
    if profile is None and "temperature" in bcl:
        profile = [("degC", (-50, 100), None), ("degF", (-50, 220), None)]
    if profile is None:
        return None, f"no_profile_for_class:{bc}"

    matches = []
    for unit, med_range, p5_p95_range in profile:
        if not (med_range[0] <= median <= med_range[1]):
            continue
        if p5_p95_range is not None and not (
            p5_p95_range[0] <= p5 and p95 <= p5_p95_range[1]
        ):
            continue
        # tighter ranges = higher confidence
        width = med_range[1] - med_range[0]
        matches.append((unit, width))
    if not matches:
        return None, f"median={median:.2f} outside all candidates"
    matches.sort(key=lambda x: x[1])
    return matches[0][0], "temperature_inferred_by_median"


def infer_units(processed_directory: str) -> pd.DataFrame:
    """Infer sensor units from streamed value summaries."""
    series_path = os.path.join(processed_directory, "series.parquet")
    pmap = pd.read_parquet(os.path.join(processed_directory, "point_map.parquet"))
    sid2bc = dict(
        zip(pmap["sensor_id"].astype(str), pmap["brick_class"].fillna("").astype(str))
    )

    pf = pq.ParquetFile(series_path)
    cols = ["sensor_id", "value", "observed_mask"]
    schema = pf.schema_arrow
    if any(f.name == "outage_mask" for f in schema):
        cols.append("outage_mask")

    # First pass: collect per-sensor value samples (cap memory by sampling).
    per_sensor: Dict[str, List[float]] = {}
    cap_per_sensor = 50_000
    fsz = os.path.getsize(series_path)
    log(
        f"audit: scanning {fsz / (1024**3):.2f} GB across {pf.num_row_groups} row groups"
    )
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=cols).to_pandas()
        # Drop outage + non-observed positions (we want UNTAMPERED values for
        # unit inference).
        if "outage_mask" in df.columns:
            df = df[(df["observed_mask"]) & (~df["outage_mask"])]
        else:
            df = df[df["observed_mask"]]
        for sid, g in df.groupby("sensor_id", sort=False):
            buf = per_sensor.setdefault(sid, [])
            if len(buf) >= cap_per_sensor:
                continue
            need = cap_per_sensor - len(buf)
            buf.extend(g["value"].iloc[:need].tolist())
        if (rg + 1) % 5 == 0:
            log(
                f"  row-group {rg + 1}/{pf.num_row_groups}, sensors so far={len(per_sensor)}"
            )

    rows = []
    for sid, vals in per_sensor.items():
        a = np.asarray(vals, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size < 10:
            continue
        med = float(np.median(a))
        p5 = float(np.percentile(a, 5))
        p95 = float(np.percentile(a, 95))
        bc = sid2bc.get(sid, "")
        unit, reason = infer_unit_for_sensor(bc, med, p5, p95, n_obs=a.size)
        rows.append(
            {
                "sensor_id": sid,
                "brick_class": bc,
                "n_samples": int(a.size),
                "median": med,
                "p5": p5,
                "p95": p95,
                "inferred_unit": unit,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)
