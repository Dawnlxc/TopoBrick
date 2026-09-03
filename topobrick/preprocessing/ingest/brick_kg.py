"""Brick KG loader: parse TTL into nodes_df / edges_df."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Tuple

import pandas as pd
from rdflib import RDF, RDFS, Graph, Literal

from topobrick.utils.paths import resolve_path
from topobrick.utils.progress import log

# Retain only canonical Brick structural relations and their inverses.
_REL_NORMALIZE: Dict[str, str] = {
    "haspoint": "hasPoint",
    "ispointof": "isPointOf",
    "haslocation": "hasLocation",
    "islocationof": "isLocationOf",
    "feeds": "feeds",
    "isfedby": "isFedBy",
    "haspart": "hasPart",
    "ispartof": "isPartOf",
}
_INVERSE_OF: Dict[str, str] = {
    "hasPoint": "isPointOf",
    "isPointOf": "hasPoint",
    "hasLocation": "isLocationOf",
    "isLocationOf": "hasLocation",
    "feeds": "isFedBy",
    "isFedBy": "feeds",
    "hasPart": "isPartOf",
    "isPartOf": "hasPart",
}

# Matched as a substring of the URI tail.
_LOCATION_HINTS = (
    "Room",
    "Zone",
    "Floor",
    "Building",
    "Space",
    "HVAC_Zone",
    "Lighting_Zone",
)
_POINT_HINTS = ("Sensor", "Meter", "Setpoint", "Status", "Command", "Alarm", "Point")
_SYSTEM_HINTS = ("System",)
# Equipment is any subclass of brick:Equipment, or a class matching one of these.
_EQUIPMENT_HINTS = (
    "AHU",
    "VAV",
    "RVAV",
    "Chiller",
    "Pump",
    "Fan",
    "Coil",
    "Boiler",
    "Tower",
    "RTU",
    "Rooftop_Unit",
    "Damper",
    "Valve",
    "Heat_Exchanger",
    "VFD",
    "Compressor",
    "Equipment",
    "Unit",
    "Filter",
    "Plant",
)


def _short(uri: str) -> str:
    s = str(uri)
    if "#" in s:
        return s.split("#", 1)[-1]
    return s.rsplit("/", 1)[-1]


# Strip instance UUID prefixes from extension class names.
_UUID_RE = re.compile(
    r"[0-9a-f]{8}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{12}_?",
    re.I,
)


def _scrub_brick_class(bc: str) -> str:
    """Strip UUID-prefixed extension class names down to a generic suffix."""
    if not bc:
        return bc
    if _UUID_RE.search(bc):
        cleaned = _UUID_RE.sub("", bc).strip("_") or "Unknown"
        # Mark as extension to avoid clashing with the unprefixed Brick class.
        return f"{cleaned}_Ext"
    return bc


def _norm_rel(uri: str) -> str | None:
    short = _short(uri).lower()
    return _REL_NORMALIZE.get(short)


def _classify_node_type(brick_class: str) -> str:
    bc = brick_class or ""
    bcl = bc.lower()
    if any(h.lower() in bcl for h in _POINT_HINTS):
        return "Point"
    if any(h.lower() in bcl for h in _LOCATION_HINTS):
        return "Location"
    if any(h.lower() in bcl for h in _EQUIPMENT_HINTS):
        return "Equipment"
    if any(h.lower() in bcl for h in _SYSTEM_HINTS):
        return "System"
    return "Unknown"


def _is_brick_class_uri(uri: str) -> bool:
    s = str(uri)
    return "brickschema.org" in s or "/brick" in s.lower()


def _parse_one_kg(
    ttl_path: str,
    dataset_name: str,
    building_id: str,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    g = Graph()
    g.parse(ttl_path, format="turtle")
    log(f"KG {building_id}: {len(g)} triples loaded from {ttl_path}")

    nodes: Dict[str, Dict[str, Any]] = {}

    # Pass 1: every subject with rdf:type pointing at a Brick instance class
    for s, _, o in g.triples((None, RDF.type, None)):
        if not _is_brick_class_uri(o):
            continue
        # owl class subjects themselves are not instances; skip if subject IS a class URI.
        if _is_brick_class_uri(s):
            continue
        nid = str(s)
        raw_bc = _short(o)
        bc = _scrub_brick_class(raw_bc)
        nt = _classify_node_type(bc)
        if nid not in nodes:
            nodes[nid] = {
                "dataset": dataset_name,
                "building_id": building_id,
                "node_id": nid,
                "node_uri": nid,
                "node_type": nt,
                "brick_class": bc,
                "label": None,
                "tags": [],
                "ts_refs": [],
                "has_timeseries": False,
                "is_forecast_target": False,
                "is_context_ts": False,
                "is_online_graph_node": False,
                "descriptor_text": None,
            }
        else:
            # multiple types; prefer non-Unknown classification, also prefer a
            # non-extension class name over the UUID-derived one.
            cur = nodes[nid]
            cur_is_ext = str(cur.get("brick_class", "")).endswith("_Ext")
            new_is_ext = bc.endswith("_Ext")
            if cur["node_type"] == "Unknown" and nt != "Unknown":
                cur["node_type"] = nt
                cur["brick_class"] = bc
            elif nt == "Point" and cur["node_type"] != "Point":
                cur["node_type"] = nt
                cur["brick_class"] = bc
            elif cur_is_ext and not new_is_ext:
                cur["brick_class"] = bc

    # Pass 2: rdfs:label, senaps:stream_id, ref:* references, etc.
    for s, p, o in g:
        nid = str(s)
        if nid not in nodes:
            continue
        node = nodes[nid]
        if p == RDFS.label and isinstance(o, Literal):
            node["label"] = str(o)
        sp = str(p).lower()
        if (
            sp.endswith("stream_id")
            or sp.endswith("hastimeseriesid")
            or sp.endswith("hasexternalreference")
        ):
            ref_val = str(o).strip()
            # if reference is a UUID-like literal, store directly
            node["ts_refs"].append(ref_val)
        if "tag" in sp and isinstance(o, (Literal,)):
            node["tags"].append(str(o))

    # Pass 3: edges (whitelist only)
    edge_rows: List[Dict[str, Any]] = []
    for s, p, o in g:
        rel = _norm_rel(p)
        if rel is None:
            continue
        # both endpoints must be retained instance nodes
        if str(s) not in nodes or str(o) not in nodes:
            continue
        edge_rows.append(
            {
                "dataset": dataset_name,
                "building_id": building_id,
                "src": str(s),
                "rel": rel,
                "dst": str(o),
                "src_type": nodes[str(s)]["node_type"],
                "dst_type": nodes[str(o)]["node_type"],
                "directed": True,
                "is_inverse": False,
                "base_rel": rel,
                "edge_weight": 1.0,
            }
        )

    seen = {(r["src"], r["rel"], r["dst"]) for r in edge_rows}
    if cfg["kg"].get("add_inverse", True):
        inv_rows: List[Dict[str, Any]] = []
        for r in edge_rows:
            inv_name = _INVERSE_OF.get(r["rel"])
            if inv_name is None:
                continue
            key = (r["dst"], inv_name, r["src"])
            if key in seen:
                continue
            inv_rows.append(
                {
                    "dataset": dataset_name,
                    "building_id": building_id,
                    "src": r["dst"],
                    "rel": inv_name,
                    "dst": r["src"],
                    "src_type": r["dst_type"],
                    "dst_type": r["src_type"],
                    "directed": True,
                    "is_inverse": True,
                    "base_rel": r["rel"],
                    "edge_weight": 1.0,
                }
            )
            seen.add(key)
        edge_rows.extend(inv_rows)

    nodes_df = pd.DataFrame(list(nodes.values()))
    if len(nodes_df) == 0:
        nodes_df = pd.DataFrame(
            columns=[
                "dataset",
                "building_id",
                "node_id",
                "node_uri",
                "node_type",
                "brick_class",
                "label",
                "tags",
                "ts_refs",
                "has_timeseries",
                "is_forecast_target",
                "is_context_ts",
                "is_online_graph_node",
                "descriptor_text",
            ]
        )
    edges_df = pd.DataFrame(edge_rows)
    log(
        f"KG {building_id}: {len(nodes_df)} instance nodes, {len(edges_df)} edges (incl. inverses)"
    )
    return nodes_df, edges_df


def load_lbnl59_kg(
    raw_dir: str, cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ttl_rel = cfg["lbnl59"]["ttl_path"]
    ttl_abs = resolve_path(cfg["paths"]["raw_root"], ttl_rel)
    return _parse_one_kg(ttl_abs, "LBNL59", cfg["lbnl59"]["building_id"], cfg)


def load_bts_kg(raw_dir: str, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes_chunks: List[pd.DataFrame] = []
    edges_chunks: List[pd.DataFrame] = []
    only = cfg["bts"].get("only_buildings")
    buildings = cfg["bts"]["buildings"]
    if only:
        buildings = [b for b in buildings if b["building_id"] in only]
    for b in buildings:
        ttl_abs = resolve_path(cfg["paths"]["raw_root"], b["ttl_path"])
        if not os.path.exists(ttl_abs):
            log(f"BTS KG {b['building_id']}: missing TTL {ttl_abs}, skip")
            continue
        n, e = _parse_one_kg(ttl_abs, "BTS", b["building_id"], cfg)
        nodes_chunks.append(n)
        edges_chunks.append(e)
    if not nodes_chunks:
        raise RuntimeError("no BTS KG loaded")
    nodes_df = pd.concat(nodes_chunks, ignore_index=True)
    edges_df = pd.concat(edges_chunks, ignore_index=True)
    return nodes_df, edges_df
