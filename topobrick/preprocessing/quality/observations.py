"""L1: observation-level physical range filtering."""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from topobrick.utils.paths import resolve_processed_root

PROCESSED_ROOT = resolve_processed_root()

# Sentinel codes: values a BMS emits to mean "no reading".
SENTINEL_VALUES = {
    # BMS / OPC error codes
    -666.0,
    -333.0,
    -444.0,
    -438.0,
    -432.0,
    -436.0,
    -439.0,
    -437.0,
    # generic missing / placeholder
    -999.0,
    -9999.0,
    9999.0,
    99999.0,
    # observed scale errors in BTS
    15000.0,
    62100.0,
    5505.0,
    11270.0,
    1188.0,
    2023.0,
    # uint32 overflow patterns observed in BTS water-flow streams
    2147485.75,
    4294967.5,
    4294967.0,
    # uint16 0xFFFF sentinel (BTS Status/Limit/Demand/Energy report 65535 on fault)
    65535.0,
    65534.0,
}

# Catch overflow values for classes without tighter bounds.
GENERIC_MAX_ABS = 1e9

# Treat exact zero as missing only for selected physical sensors.
ZERO_IS_SENTINEL_QUANTITIES = ("Temperature", "Humidity", "Dewpoint", "Enthalpy")

# Apply first-match physical bounds in canonical units.
PHYSICAL_BOUNDS: List[Tuple[str, Optional[Tuple[float, float]], str]] = [
    # --- Specific Temperature subclasses (must come before generic Temperature_Sensor) ---
    ("Hot_Water_Temperature", (5, 100), "degC"),
    ("Chilled_Water_Temperature", (-5, 30), "degC"),
    ("Outside_Air_Wet_Bulb_Temperature", (-50, 50), "degC"),
    ("Outdoor_Air_Wet_Bulb_Temperature", (-50, 50), "degC"),
    ("Outside_Air_Temperature", (-50, 60), "degC"),
    ("Outdoor_Air_Temperature", (-50, 60), "degC"),
    ("Mixed_Air_Temperature", (-10, 60), "degC"),
    ("Supply_Air_Temperature", (-10, 60), "degC"),
    ("Return_Air_Temperature", (-10, 60), "degC"),
    ("Discharge_Air_Temperature", (-10, 80), "degC"),
    ("Exhaust_Air_Temperature", (-10, 80), "degC"),
    # Retain low but valid zone-air temperatures.
    ("Zone_Air_Temperature", (5, 40), "degC"),
    ("Air_Temperature", (-10, 60), "degC"),
    ("Water_Temperature", (-5, 110), "degC"),
    # --- Dewpoint / Differential Temperature ---
    ("Dewpoint", (-60, 60), "degC"),
    ("Differential_Temperature", (-30, 80), "degC"),
    # --- Setpoint family (mirrors sensor bounds) ---
    ("Cooling_Temperature_Setpoint", (5, 40), "degC"),
    ("Heating_Temperature_Setpoint", (5, 40), "degC"),
    ("Room_Air_Temperature_Setpoint", (5, 40), "degC"),
    ("Discharge_Air_Temperature_Setpoint", (-10, 80), "degC"),
    ("Temperature_Setpoint", (-50, 150), "degC"),  # generic fallback
    ("Outside_Air_Lockout_Temperature_Setpoint", (-30, 30), "degC"),
    ("Low_Outside_Air_Temperature_Enable_Setpoint", (-30, 30), "degC"),
    # --- Generic Temperature fallback (catches Temperature_Sensor, Temperature) ---
    ("Temperature", (-50, 150), "degC"),
    # --- Humidity (0-105 % tolerance) ---
    ("Humidity", (-5, 105), "%"),
    # --- Enthalpy kJ/kg ---
    ("Enthalpy", (-50, 200), "kJ/kg"),
    # Allow signed duct pressure but reject reversed filter differentials.
    ("Filter_Differential_Pressure", (0, 2000), "Pa-like"),
    ("Static_Pressure", (-200, 5000), "Pa-like"),
    ("Differential_Pressure", (-200, 2000), "Pa-like"),
    ("Pressure_Setpoint", (-200, 5000), "Pa-like"),
    ("Pressure", (-200, 5000), "Pa-like"),
    # --- Flow / Air flow (non-negative) ---
    ("Discharge_Air_Flow", (-10, 100000), "CFM/L·s⁻¹"),
    ("Supply_Air_Flow", (0, 100000), "CFM/L·s⁻¹"),
    ("Fume_Hood_Air_Flow", (-10, 10000), "CFM"),
    ("Air_Flow", (0, 100000), "CFM/L·s⁻¹"),
    ("Flow_Rate", (0, 100000), ""),
    ("Flow_Sensor", (0, 100000), ""),
    # --- Position / Damper / Valve (%) ---
    ("Damper_Position", (-5, 105), "%"),
    ("Valve_Position", (-5, 105), "%"),
    ("Position", (-5, 105), "%"),
    # --- CO2 ppm ---
    ("CO2", (300, 5000), "ppm"),
    # --- Weather ---
    ("Solar_Radiance", (0, 1500), "W/m²"),
    ("Solar_Radiation", (0, 1500), "W/m²"),
    ("Wind_Speed", (0, 60), "m/s"),
    ("Wind_Direction", (0, 360), "deg"),
    ("Rain", (0, 500), "mm/hr"),
    # --- Electrical ---
    ("Electrical_Power", (-1000, 100000), "kW"),
    ("Electrical_Energy", None, "cumulative kWh — bounds skipped"),
    ("Gas_Meter", None, "cumulative"),
    ("Voltage", (-50, 500), "V"),
    ("Current", (-1, 5000), "A"),
    ("Frequency", (40, 65), "Hz"),
    ("Thermal_Power", (-1000, 100000), "kW"),
    # --- Categorical / status / command — skip bounds ---
    ("Status", None, "discrete"),
    ("Command", None, "discrete"),
    ("Mode", None, "discrete"),
    ("Alarm", None, "discrete"),
    ("Enable", None, "discrete"),
    ("Stages", None, "discrete"),
    ("Parameter", None, "discrete-like"),
    ("Setpoint_Limit", None, "limit-config"),
    ("Counter", None, "cumulative"),
]


def _rewrite_series(
    series_path: str,
    canon: np.ndarray,
    all_bad: np.ndarray,
    batch_size: int = 4_000_000,
) -> None:
    """Stream the series through, replacing `value`, `observed_mask`, `outage_mask`."""
    pf = pq.ParquetFile(series_path)
    schema = pf.schema_arrow
    tmp = series_path + ".tmp"
    off = 0
    with pq.ParquetWriter(tmp, schema) as w:
        for batch in pf.iter_batches(batch_size=batch_size):
            n = batch.num_rows
            sl = slice(off, off + n)
            bad = all_bad[sl]
            cols = []
            for name in schema.names:
                col = batch.column(batch.schema.get_field_index(name))
                if name == "value":
                    col = pa.array(canon[sl], type=schema.field(name).type)
                elif name == "observed_mask":
                    obs = col.to_numpy(zero_copy_only=False).astype(bool) & ~bad
                    col = pa.array(obs, type=schema.field(name).type)
                elif name == "outage_mask":
                    out = col.to_numpy(zero_copy_only=False).astype(bool) | bad
                    col = pa.array(out, type=schema.field(name).type)
                cols.append(col)
            w.write_table(pa.Table.from_arrays(cols, schema=schema))
            off += n
    if off != len(canon):
        os.remove(tmp)
        raise RuntimeError(
            f"row-count mismatch on rewrite: wrote {off}, expected {len(canon)}"
        )
    os.replace(tmp, series_path)


def _bounds_for_class(brick_class: str) -> Tuple[Optional[Tuple[float, float]], str]:
    """Substring match, first hit wins. Returns (bounds_or_None, unit_label)."""
    if not brick_class:
        return (None, "no-class")
    for sub, bnds, unit in PHYSICAL_BOUNDS:
        if sub in brick_class:
            return (bnds, unit)
    return (None, "no-match")


def _to_canonical(values: np.ndarray, unit: Optional[str], is_temp: bool) -> np.ndarray:
    """Convert °F/K → °C for temp classes; identity otherwise."""
    if not is_temp or unit is None:
        return values
    u = str(unit).strip()
    if u == "degF":
        return (values - 32.0) * (5.0 / 9.0)
    if u == "K":
        return values - 273.15
    return values


def detect_drop_to_zero(
    values: np.ndarray, observed: np.ndarray, run_len: int = 4
) -> np.ndarray:
    """Mark consecutive runs of ≥run_len zeros where the surrounding context is non-zero."""
    bad = np.zeros(len(values), dtype=bool)
    if len(values) < run_len + 2:
        return bad
    is_zero = (values == 0.0) & observed
    diff = np.diff(is_zero.astype(int), prepend=0, append=0)
    run_starts = np.where(diff == 1)[0]
    run_ends = np.where(diff == -1)[0]  # exclusive
    for s, e in zip(run_starts, run_ends):
        if e - s < run_len:
            continue
        # A sensor that reads zero for its whole record is not dropping to zero,
        # it is off or unmapped; require a non-zero reading on one side.
        before_nonzero = s > 0 and observed[s - 1] and abs(values[s - 1]) > 1e-6
        after_nonzero = e < len(values) and observed[e] and abs(values[e]) > 1e-6
        if before_nonzero or after_nonzero:
            bad[s:e] = True
    return bad


def detect_frozen_run(
    values: np.ndarray, observed: np.ndarray, max_run: int = 48
) -> np.ndarray:
    """Mark observations within a run of identical values longer than max_run."""
    bad = np.zeros(len(values), dtype=bool)
    if len(values) <= max_run + 1:
        return bad
    same_as_prev = np.zeros(len(values), dtype=bool)
    same_as_prev[1:] = (values[1:] == values[:-1]) & observed[1:] & observed[:-1]
    run = np.zeros(len(values), dtype=int)
    for i in range(1, len(values)):
        run[i] = run[i - 1] + 1 if same_as_prev[i] else 0
    for i in np.where(run > max_run)[0]:
        bad[i - max_run : i + 1] = True  # the whole run, not just its tail
    return bad


def detect_mad_outlier(
    values: np.ndarray, observed: np.ndarray, window: int = 96, mad_k: float = 5.0
) -> np.ndarray:
    """Rolling-median ± mad_k × MAD. Robust to sentinel-contaminated data."""
    bad = np.zeros(len(values), dtype=bool)
    if len(values) < window // 2:
        return bad
    s = pd.Series(values).where(pd.Series(observed))
    rmed = s.rolling(window, center=True, min_periods=window // 4).median()
    rmad = (
        (s - rmed).abs().rolling(window, center=True, min_periods=window // 4).median()
    )
    rmad = rmad.replace(
        0, np.nan
    )  # a flat window has MAD 0; leave it to detect_frozen_run
    z = (s - rmed).abs() / rmad
    bad = (z.fillna(0) > mad_k).values
    return bad & observed


def filter_one_sensor(
    sub: pd.DataFrame,
    brick_class: str,
    unit: Optional[str],
    drop_to_zero_run: int,
    frozen_run_steps: int,
    mad_window: int,
    mad_k: float,
    skip_outlier_classes: List[str],
) -> Tuple[np.ndarray, Dict[str, int], np.ndarray]:
    """Returns (rows to mask, per-check stats, values in canonical units)."""
    n = len(sub)
    v_raw = sub["value"].to_numpy(dtype=np.float64)
    is_temp = bool(brick_class) and (
        ("Temperature" in brick_class) or ("Dewpoint" in brick_class)
    )
    # A conversion happened iff the sensor is temperature-like AND its unit is one
    # we convert from. Comparing arrays would be wrong: NaN != NaN.
    converted = is_temp and str(unit).strip() in ("degF", "K")
    v = _to_canonical(v_raw, unit, is_temp)
    obs = sub["observed_mask"].to_numpy(dtype=bool)
    stats = {"n_total": n, "n_already_masked": int((~obs).sum())}

    sentinel = np.isin(v, list(SENTINEL_VALUES))
    stats["A_sentinel"] = int((sentinel & obs).sum())

    zero_sentinel = np.zeros(n, dtype=bool)
    if (
        brick_class
        and brick_class.endswith("_Sensor")
        and any(q in brick_class for q in ZERO_IS_SENTINEL_QUANTITIES)
    ):
        zero_sentinel = v == 0.0
    stats["A3_zero_sentinel"] = int((zero_sentinel & obs).sum())

    overflow = np.isfinite(v) & (np.abs(v) > GENERIC_MAX_ABS)
    stats["A2_overflow"] = int((overflow & obs).sum())

    nonfinite = ~np.isfinite(v)
    stats["C_nonfinite"] = int((nonfinite & obs).sum())

    # The bounds table is written in canonical units and `v` is already canonical,
    # so no conversion happens here.
    bnds, _bunit = _bounds_for_class(brick_class)
    bounds_violation = np.zeros(n, dtype=bool)
    if bnds is not None:
        bounds_violation = ((v < bnds[0]) | (v > bnds[1])) & np.isfinite(v)
    stats["B_bounds"] = int((bounds_violation & obs).sum())

    # Skip dynamics checks for piecewise-constant controls and unclassified points.
    skip_outlier = (
        any(s in brick_class for s in skip_outlier_classes) if brick_class else True
    )

    drop_zero = np.zeros(n, dtype=bool)
    if not skip_outlier:
        drop_zero = detect_drop_to_zero(v, obs, run_len=drop_to_zero_run)
    stats["D_drop_to_zero"] = int(drop_zero.sum())

    frozen = np.zeros(n, dtype=bool)
    if not skip_outlier:
        frozen = detect_frozen_run(v, obs, max_run=frozen_run_steps)
    stats["E_frozen"] = int(frozen.sum())

    mad_out = np.zeros(n, dtype=bool)
    if not skip_outlier:
        mad_out = detect_mad_outlier(v, obs, window=mad_window, mad_k=mad_k)
    stats["F_mad"] = int(mad_out.sum())

    bad = (
        sentinel
        | overflow
        | nonfinite
        | bounds_violation
        | drop_zero
        | frozen
        | mad_out
        | zero_sentinel
    ) & obs
    stats["total_newly_masked"] = int(bad.sum())
    stats["G_unit_converted"] = int(converted)
    return bad, stats, v


def run_filter(
    dataset: str,
    processed_root: str = PROCESSED_ROOT,
    dry_run: bool = False,
    drop_to_zero_run: int = 4,
    frozen_run_steps: int = 48,
    mad_window: int = 96,
    mad_k: float = 5.0,
    skip_outlier_classes=(
        "Status",
        "Command",
        "Mode",
        "Alarm",
        "Enable",
        "Stages",
        "Parameter",
        "Setpoint",
        "Limit",
        "Fan_Speed",
        "Position",
        "Damper",
        "Valve",
    ),
) -> Dict:
    proc = os.path.join(processed_root, dataset)
    series_path = os.path.join(proc, "series.parquet")
    nodes_path = os.path.join(proc, "kg_nodes.parquet")
    if not os.path.exists(series_path):
        raise FileNotFoundError(series_path)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    bak = series_path + f".pre_qc_layer1_{ts}.bak"
    if not dry_run and not os.path.exists(bak):
        shutil.copyfile(series_path, bak)
        print(f"[backup] {bak}")

    print("[load] point_map + kg_nodes for sensor_id → (brick_class, unit) map")
    # point_map.parquet is the authoritative sensor_id ↔ kg_node_id ↔ brick_class
    # mapping (kg_nodes.ts_refs is empty in some datasets).
    pmap = pd.read_parquet(
        os.path.join(proc, "point_map.parquet"),
        columns=["sensor_id", "kg_node_id", "brick_class"],
    )
    nodes = pd.read_parquet(
        nodes_path, columns=["node_id", "brick_class", "resolved_unit"]
    )
    nid2unit = {
        str(r["node_id"]): (
            r["resolved_unit"] if pd.notna(r["resolved_unit"]) else None
        )
        for _, r in nodes.iterrows()
    }
    sid2bc, sid2unit = {}, {}
    for _, r in pmap.iterrows():
        sid = str(r["sensor_id"])
        nid = str(r["kg_node_id"])
        sid2bc[sid] = str(r["brick_class"]) if pd.notna(r["brick_class"]) else ""
        u = nid2unit.get(nid)
        if u:
            sid2unit[sid] = u
    print(f"  mapped {len(sid2bc)} sensor_ids")

    # Dictionary-encode string columns to bound memory use.
    print("[load] series.parquet (sensor_id, unit, value, observed_mask)")
    tbl = pq.read_table(
        series_path,
        columns=["sensor_id", "unit", "value", "observed_mask"],
        read_dictionary=["sensor_id", "unit"],
    )
    n_rows = tbl.num_rows
    s = tbl.to_pandas()  # dictionary columns arrive as pandas `category`
    del tbl
    print(f"  rows={n_rows:,}  observed=True={int(s['observed_mask'].sum()):,}")

    # Process contiguous sensor blocks directly to avoid groupby memory overhead.
    codes = s["sensor_id"].cat.codes.to_numpy()
    cats = s["sensor_id"].cat.categories
    seen = np.zeros(len(cats), dtype=bool)
    seen[codes] = True  # O(n), and no 616 M-element sort
    n_sensors = int(seen.sum())
    starts = np.concatenate(([0], np.flatnonzero(np.diff(codes)) + 1))
    contiguous = len(starts) == n_sensors
    if contiguous:
        bounds = list(zip(starts, np.concatenate((starts[1:], [n_rows]))))
        order = None
    else:
        order = np.argsort(codes, kind="stable")
        b = np.concatenate(([0], np.flatnonzero(np.diff(codes[order])) + 1))
        bounds = list(zip(b, np.concatenate((b[1:], [n_rows]))))
    print(
        f"[filter] processing {len(bounds):,} sensors "
        f"({'contiguous' if contiguous else 'reordered'}) ..."
    )

    all_bad = np.zeros(n_rows, dtype=bool)
    # canonical-unit values, seeded with the raw column and overwritten per sensor
    canon = s["value"].to_numpy().copy()
    n_converted = 0
    per_class_stats: Dict[str, Dict[str, int]] = {}
    n_sensors_processed = 0
    for a, b_ in bounds:
        rows = np.arange(a, b_) if order is None else order[a:b_]
        if not len(rows):
            continue
        grp = s.take(rows)
        sid = str(cats[codes[rows[0]]])
        bc = sid2bc.get(sid, "") or ""
        unit = sid2unit.get(sid)
        if unit is None:  # fall back to the series' own label
            u = grp["unit"].iloc[0]
            unit = None if pd.isna(u) else str(u)
        bad, st, v_canon = filter_one_sensor(
            grp,
            bc,
            unit,
            drop_to_zero_run,
            frozen_run_steps,
            mad_window,
            mad_k,
            skip_outlier_classes=list(skip_outlier_classes),
        )
        all_bad[rows] = bad
        if st.get("G_unit_converted"):
            canon[rows] = v_canon.astype(canon.dtype)
            n_converted += 1
        cs = per_class_stats.setdefault(
            bc, {"sensors": 0, **{k: 0 for k in st if k != "n_total"}}
        )
        cs["sensors"] += 1
        for k in st:
            if k == "n_total":
                continue
            cs[k] = cs.get(k, 0) + st[k]
        n_sensors_processed += 1
        if n_sensors_processed % 500 == 0:
            print(f"  {n_sensors_processed} sensors processed")

    masked_count = int(all_bad.sum())
    masked_fraction = 100 * masked_count / n_rows
    print(
        f"\n[filter] DONE.  total newly masked: {masked_count:,} / {n_rows:,} "
        f"({masked_fraction:.3f}%)"
    )
    print(f"[filter] canonicalized to degC: {n_converted} sensors")

    print("\n[summary] per Brick class (top 30 by # newly masked)")
    rows = []
    for bc, st in per_class_stats.items():
        rows.append({"class": bc, **st})
    df_s = pd.DataFrame(rows).sort_values("total_newly_masked", ascending=False)
    print(df_s.head(30).to_string(index=False))

    if dry_run:
        print("\n[dry_run] no writes.")
        return {
            "n_newly_masked": int(all_bad.sum()),
            "per_class": per_class_stats,
            "n_unit_converted": n_converted,
        }

    # Rewrite stage-owned columns in streaming batches.
    del s
    print(f"\n[write] series.parquet ({n_rows:,} rows, streaming)")
    _rewrite_series(series_path, canon, all_bad)
    print("  saved.")

    audit_path = os.path.join(proc, "qc_layer1_audit.csv")
    df_s.to_csv(audit_path, index=False)
    print(f"  audit log → {audit_path}")
    return {"n_newly_masked": int(all_bad.sum()), "per_class": per_class_stats}
