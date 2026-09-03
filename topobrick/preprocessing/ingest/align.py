"""Align time-series sensor IDs to KG point nodes."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Tuple

import pandas as pd

from topobrick.utils.progress import log


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def _node_id_suffix(uri: str) -> str:
    s = str(uri)
    if "#" in s:
        return s.split("#", 1)[-1]
    return s.rsplit("/", 1)[-1]


def align_timeseries_to_kg(
    series_df: pd.DataFrame,
    kg_nodes_df: pd.DataFrame,
    kg_edges_df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (point_map_df, updated_nodes_df, updated_series_df)."""
    if series_df.empty or kg_nodes_df.empty:
        log("align_timeseries_to_kg: empty input")
        empty_pm = pd.DataFrame(
            columns=[
                "dataset",
                "building_id",
                "sensor_id",
                "kg_node_id",
                "match_method",
                "confidence",
                "raw_name",
                "brick_class",
                "has_timeseries",
                "is_forecast_target",
                "is_context_ts",
            ]
        )
        return empty_pm, kg_nodes_df, series_df

    # Manual mapping (sensor_id -> kg_node_id) optional
    manual_map: Dict[Tuple[str, str], str] = {}
    mp = cfg["preprocessing"].get("manual_mapping_path")
    if mp and os.path.exists(mp):
        m = pd.read_csv(mp)
        for _, r in m.iterrows():
            manual_map[(str(r["building_id"]), str(r["sensor_id"]))] = str(
                r["kg_node_id"]
            )
        log(f"align: loaded {len(manual_map)} manual mapping rows")

    # Build lookup tables per (building_id) to disambiguate.
    nodes_by_bldg: Dict[str, pd.DataFrame] = {
        b: g for b, g in kg_nodes_df.groupby("building_id", sort=False)
    }

    # Pre-compute per-building maps
    building_lookups: Dict[str, Dict[str, Dict[str, str]]] = {}
    for bid, ndf in nodes_by_bldg.items():
        ref_map: Dict[str, str] = {}
        suffix_map: Dict[str, str] = {}
        norm_map: Dict[str, str] = {}
        label_map: Dict[str, str] = {}
        for _, n in ndf.iterrows():
            nid = n["node_id"]
            for ref in n.get("ts_refs") or []:
                ref_map.setdefault(str(ref).strip(), nid)
            suf = _node_id_suffix(nid)
            suffix_map.setdefault(suf, nid)
            suffix_map.setdefault(nid, nid)
            nm = _norm(suf)
            norm_map.setdefault(nm, nid)
            if n.get("label"):
                label_map.setdefault(_norm(n["label"]), nid)
        building_lookups[bid] = {
            "ref": ref_map,
            "suffix": suffix_map,
            "norm": norm_map,
            "label": label_map,
        }

    # Group sensors
    sensor_meta = (
        series_df.groupby(["dataset", "building_id", "sensor_id"], sort=False)
        .agg({"raw_name": "first"})
        .reset_index()
    )

    # Aggregate union of building IDs in KG; if no building match, fall back to global maps
    global_ref: Dict[str, str] = {}
    global_suffix: Dict[str, str] = {}
    global_norm: Dict[str, str] = {}
    for bid, lookup in building_lookups.items():
        for k, v in lookup["ref"].items():
            global_ref.setdefault(k, v)
        for k, v in lookup["suffix"].items():
            global_suffix.setdefault(k, v)
        for k, v in lookup["norm"].items():
            global_norm.setdefault(k, v)

    rows: List[Dict[str, Any]] = []
    matched_node_ids: Dict[
        str, str
    ] = {}  # node_id -> sensor_id (for has_timeseries flag)

    for _, r in sensor_meta.iterrows():
        bid = str(r["building_id"])
        sid = str(r["sensor_id"])
        raw = str(r.get("raw_name") or sid)
        method = "unmatched"
        conf = 0.0
        nid = None

        lookup = building_lookups.get(
            bid, {"ref": {}, "suffix": {}, "norm": {}, "label": {}}
        )
        # 1) manual
        if (bid, sid) in manual_map:
            nid = manual_map[(bid, sid)]
            method, conf = "manual", 1.0
        # 2) ts ref
        if nid is None and sid in lookup["ref"]:
            nid = lookup["ref"][sid]
            method, conf = "timeseries_ref", 1.0
        elif nid is None and sid in global_ref:
            nid = global_ref[sid]
            method, conf = "timeseries_ref", 0.9
        # 3) exact suffix
        if nid is None and sid in lookup["suffix"]:
            nid = lookup["suffix"][sid]
            method, conf = "exact_id", 0.95
        elif nid is None and sid in global_suffix:
            nid = global_suffix[sid]
            method, conf = "exact_id", 0.85
        # 4) normalized name
        if nid is None:
            n = _norm(raw)
            if n in lookup["norm"]:
                nid = lookup["norm"][n]
                method, conf = "name", 0.7
            elif n in lookup["label"]:
                nid = lookup["label"][n]
                method, conf = "label", 0.7
            elif n in global_norm:
                nid = global_norm[n]
                method, conf = "name", 0.6

        rows.append(
            {
                "dataset": r["dataset"],
                "building_id": bid,
                "sensor_id": sid,
                "kg_node_id": nid,
                "match_method": method,
                "confidence": conf,
                "raw_name": raw,
                "brick_class": None,
                "has_timeseries": nid is not None,
                "is_forecast_target": False,
                "is_context_ts": False,
            }
        )
        if nid is not None:
            matched_node_ids[nid] = sid

    point_map = pd.DataFrame(rows)

    # Fill brick_class from KG into point_map
    nid2bc = dict(zip(kg_nodes_df["node_id"], kg_nodes_df["brick_class"]))
    point_map["brick_class"] = point_map["kg_node_id"].map(nid2bc)

    # Update kg_nodes.has_timeseries
    kg_nodes_df = kg_nodes_df.copy()
    kg_nodes_df["has_timeseries"] = kg_nodes_df["node_id"].isin(matched_node_ids.keys())

    # Update series_df.kg_node_id and brick_class
    sid2nid = dict(zip(point_map["sensor_id"], point_map["kg_node_id"]))
    sid2bc = dict(zip(point_map["sensor_id"], point_map["brick_class"]))
    series_df = series_df.copy()
    series_df["kg_node_id"] = series_df["sensor_id"].map(sid2nid)
    series_df["brick_class"] = series_df["sensor_id"].map(sid2bc)

    n_match = int(point_map["has_timeseries"].sum())
    log(f"align: matched {n_match}/{len(point_map)} sensors to KG nodes")
    log(f"align: by method: {point_map['match_method'].value_counts().to_dict()}")
    return point_map, kg_nodes_df, series_df
