"""The unit a sensor's vendor *declares*, as opposed to the one its values imply."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from rdflib import Graph

from topobrick.utils.paths import resolve_path
from topobrick.utils.progress import log

# Free-text unit strings from a corpus's own metadata -> the canonical tokens
# `units.infer` also emits, so the two can be cross-checked.
_UNIT_NORMALISE = {
    # temperature
    "°C": "degC",
    "degC": "degC",
    "DegreeCelsius": "degC",
    "C": "degC",
    "°F": "degF",
    "degF": "degF",
    "DegreeFahrenheit": "degF",
    "F": "degF",
    "K": "K",
    "Kelvin": "K",
    # ratios / scales
    "%": "%",
    "percent": "%",
    "Percent": "%",
    "ppm": "ppm",
    "PartsPerMillion": "ppm",
    # pressure
    "psi": "psi",
    "Pascal": "Pa",
    "Pa": "Pa",
    "kPa": "kPa",
    "InchesH2O": "inH2O",
    # flow
    "cfm": "cfm",
    "CFM": "cfm",
    "m3/s": "m3/s",
    "L/s": "L/s",
    # power
    "kW": "kW",
    "KiloW": "kW",
    "Kilowatt": "kW",
    "W": "W",
    "Watt": "W",
    "kVA": "kVA",
    "KiloV-A": "kVA",
    "kVAR": "kVAR",
    "KiloV-A-Reactive": "kVAR",
    # energy
    "kWh": "kWh",
    "Kilowatt-Hour": "kWh",
    "Wh": "Wh",
    "MBTU/h": "MBTU/h",
    "mbtuph": "MBTU/h",
    # electrical
    "V": "V",
    "Volt": "V",
    "kV": "kV",
    "A": "A",
    "Amp": "A",
    "Ampere": "A",
    "Hz": "Hz",
    "Hertz": "Hz",
    # solar
    "W/m2": "W/m^2",
    "W/m^2": "W/m^2",
    # placeholder
    "/": None,  # the xlsx uses "/" for "no unit"; treat as unknown
}


def _norm_unit(u: Any) -> Optional[str]:
    if u is None or (isinstance(u, float) and np.isnan(u)):
        return None
    s = str(u).strip()
    if s in _UNIT_NORMALISE:
        return _UNIT_NORMALISE[s]
    # try lowercase
    for k, v in _UNIT_NORMALISE.items():
        if k.lower() == s.lower():
            return v
    return s  # pass through as-is


def extract_lbnl59_source_units(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Returns DataFrame with columns: sensor_id, source_unit, source."""
    xlsx_path = resolve_path(
        cfg["paths"]["raw_root"],
        "LBNL59",
        "data_description_table_3year_clean_data.xlsx",
    )
    if not os.path.exists(xlsx_path):
        log(f"LBNL59 xlsx not found: {xlsx_path}")
        return pd.DataFrame(columns=["sensor_id", "source_unit", "source"])
    df = pd.read_excel(xlsx_path, sheet_name="Missing rate_3 years")
    df = df.dropna(subset=["Column name"])
    df["unit_norm"] = df["Unit"].apply(_norm_unit)

    # Build: for each row, expand wildcard "*" against the actual sensor_id list
    # by enumerating CSV columns under the row's File name.
    csv_dir = resolve_path(
        cfg["paths"]["raw_root"], "LBNL59", "Building_59", "Bldg59_clean data"
    )

    rows: List[Dict[str, Any]] = []
    file_cache: Dict[str, List[str]] = {}
    for _, r in df.iterrows():
        col_name = str(r["Column name"]).strip()
        unit = r["unit_norm"]
        # Try to find the file: fillna handles propagation
        fname = r["File name"]
        if pd.isna(fname):
            # fallback: use the previous non-NaN
            fname = None
        if isinstance(fname, str):
            fname = fname.strip()
            if fname not in file_cache:
                fp = os.path.join(csv_dir, fname)
                if os.path.exists(fp):
                    head = pd.read_csv(fp, nrows=0)
                    file_cache[fname] = [c.strip() for c in head.columns if c]
                else:
                    file_cache[fname] = []
            cols = file_cache[fname]
        else:
            cols = []

        if "*" in col_name:
            pat = re.compile(re.escape(col_name).replace(r"\*", ".+") + "$")
            matches = [c for c in cols if pat.match(c)]
        else:
            # exact match (with possible trailing whitespace etc)
            matches = [c for c in cols if c.strip() == col_name.strip()]
            if not matches:
                # also try the original (may not be in this file)
                matches = [col_name]

        for sid in matches:
            rows.append(
                {
                    "sensor_id": sid,
                    "source_unit": unit,
                    "source": "lbnl59_xlsx",
                    "file_name": fname,
                    "raw_unit_str": str(r["Unit"]) if pd.notna(r["Unit"]) else None,
                    "description": str(r["Description"])
                    if pd.notna(r["Description"])
                    else None,
                }
            )
    out = pd.DataFrame(rows).drop_duplicates(subset=["sensor_id"], keep="first")
    log(f"LBNL59: extracted source unit for {len(out)} sensors from xlsx")
    return out


def extract_bts_source_units(cfg: Dict[str, Any]) -> pd.DataFrame:
    only = cfg["bts"].get("only_buildings")
    buildings = cfg["bts"]["buildings"]
    if only:
        buildings = [b for b in buildings if b["building_id"] in only]
    rows: List[Dict[str, Any]] = []
    for b in buildings:
        ttl_path = resolve_path(cfg["paths"]["raw_root"], b["ttl_path"])
        if not os.path.exists(ttl_path):
            continue
        g = Graph()
        g.parse(ttl_path, format="turtle")
        # blank-node value lookup
        bn2unit: Dict[Any, str] = {}
        for s, p, o in g:
            sp = str(p).split("/")[-1].split("#")[-1]
            if sp == "value":
                bn2unit[s] = str(o)
        # Point hasUnit [bn]
        for s, p, o in g:
            sp = str(p).split("/")[-1].split("#")[-1]
            if sp != "hasUnit":
                continue
            unit_str = bn2unit.get(o, str(o))
            # The KG point uri is the full URI; senaps stream_id is the matching field
            stream_id = None
            for p2, o2 in g.predicate_objects(s):
                p2s = str(p2).split("/")[-1].split("#")[-1]
                if p2s == "stream_id":
                    stream_id = str(o2)
                    break
            if stream_id is None:
                continue
            rows.append(
                {
                    "sensor_id": stream_id,
                    "source_unit": _norm_unit(unit_str),
                    "source": "bts_kg_hasUnit",
                    "raw_unit_str": unit_str,
                    "building_id": b["building_id"],
                }
            )
    out = pd.DataFrame(rows).drop_duplicates(subset=["sensor_id"], keep="first")
    log(f"BTS: extracted source unit for {len(out)} sensors from KG hasUnit")
    return out


def compare_source_and_inferred_units(
    processed_directory: str, source_units: pd.DataFrame
) -> pd.DataFrame:
    inferred_csv = os.path.join(processed_directory, "inferred_units.csv")
    if not os.path.exists(inferred_csv):
        log(f"{inferred_csv} missing — run infer_units first")
        return pd.DataFrame()
    inf = pd.read_csv(inferred_csv)
    source_columns = ["sensor_id", "source_unit", "source", "raw_unit_str"]
    merged = inf.merge(
        source_units.reindex(columns=source_columns),
        on="sensor_id",
        how="left",
    )

    # conflict if both present and different
    def _conflict(r):
        sx, ix = r["source_unit"], r["inferred_unit"]
        if pd.isna(sx) or pd.isna(ix):
            return False
        return str(sx) != str(ix)

    merged["conflict"] = merged.apply(_conflict, axis=1)
    return merged


def extract_and_audit_source_units(
    config: Dict[str, Any], dataset: str, processed_directory: str
) -> pd.DataFrame:
    if dataset == "LBNL59":
        source_units = extract_lbnl59_source_units(config)
    elif dataset == "BTS":
        source_units = extract_bts_source_units(config)
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    source_units.to_csv(
        os.path.join(processed_directory, "source_units.csv"), index=False
    )
    log(f"wrote source_units.csv ({len(source_units)} rows)")

    audit = compare_source_and_inferred_units(processed_directory, source_units)
    if len(audit):
        audit.to_csv(os.path.join(processed_directory, "units_audit.csv"), index=False)
    return audit


def print_unit_audit(audit: pd.DataFrame, processed_directory: str) -> None:
    print()
    print(f"=== Source vs inferred unit audit: {processed_directory} ===")
    if audit.empty:
        print("  no sensors available for comparison")
        return

    n = len(audit)
    n_src = audit["source_unit"].notna().sum()
    n_inf = audit["inferred_unit"].notna().sum()
    n_both = ((audit["source_unit"].notna()) & (audit["inferred_unit"].notna())).sum()
    n_conflict = audit["conflict"].sum()
    print(f"  total sensors: {n}")
    print(f"  has source unit:    {n_src} ({n_src / n:.1%})")
    print(f"  has inferred unit:  {n_inf} ({n_inf / n:.1%})")
    print(f"  both present:       {n_both}")
    print(f"  CONFLICTS:          {n_conflict}")
    print()

    if n_conflict > 0:
        print("=== CONFLICTING sensors (source label vs empirical median) ===")
        conflicts = audit[audit["conflict"]]
        for _, row in conflicts.head(20).iterrows():
            print(
                f"  {str(row['sensor_id'])[:30]:30s}  "
                f"bc={str(row['brick_class'])[:32]:32s}  "
                f"source={row['source_unit']:6s} inferred={row['inferred_unit']:6s}  "
                f"median={row['median']:.2f}"
            )
        if len(conflicts) > 20:
            print(f"  ... and {len(conflicts) - 20} more")

    if n_src > 0:
        print()
        print("=== Source unit distribution ===")
        print(audit["source_unit"].value_counts().to_string())
