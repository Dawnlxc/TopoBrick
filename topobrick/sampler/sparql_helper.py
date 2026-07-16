"""KG access layer for the agent sampler.

Loads from `data/processed/<DS>/kg_{nodes,edges}.parquet` (the same KG view
that v2.4 sampler consumes), so agent vs v2.4 are apples-to-apples.

Exposes three primitives the planner agent uses as tools:

  1. `neighbors(node, rels, direction, limit)` → typed neighbour expansion
  2. `metadata(node)` → {brick_class, node_type, has_ts, ...}
  3. `query_class(brick_class, has_ts=None, max_n=None)` → list of nodes of
     a given class (for global lookups like "find an Outdoor_Air_Temp Sensor")
"""
from __future__ import annotations
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd


CANONICAL_RELS = {"hasPoint", "hasPart", "feeds", "isLocationOf"}
INVERSE_TO_CANONICAL = {
    "isPointOf":   "hasPoint",
    "isPartOf":    "hasPart",
    "isFedBy":     "feeds",
    "hasLocation": "isLocationOf",
}
ALL_RELS = CANONICAL_RELS | set(INVERSE_TO_CANONICAL.keys())


class SparqlHelper:
    """Per-dataset KG view backed by the processed parquet files."""

    def __init__(self, processed_dir: str, cache_dir: Optional[str] = None,
                 ttl_path: Optional[str] = None):
        # ttl_path is accepted for API compat but not used; processed parquet wins.
        self.processed_dir = processed_dir
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        nodes = pd.read_parquet(os.path.join(processed_dir, "kg_nodes.parquet"))
        edges = pd.read_parquet(os.path.join(processed_dir, "kg_edges.parquet"))

        # node metadata
        self._node_meta: Dict[str, Dict[str, Any]] = {}
        self._class_to_nodes: Dict[str, List[str]] = defaultdict(list)
        for _, r in nodes.iterrows():
            nid = str(r["node_id"])
            bc = str(r.get("brick_class") or "")
            self._node_meta[nid] = {
                "brick_class": bc,
                "node_type": str(r.get("node_type") or ""),
                "has_ts": bool(r.get("has_timeseries", False)),
                "is_forecast_target": bool(r.get("is_forecast_target", False)),
                "is_context_ts": bool(r.get("is_context_ts", False)),
            }
            if bc:
                self._class_to_nodes[bc].append(nid)

        # short-id → full lookup
        self._short2full: Dict[str, str] = {}
        for full in self._node_meta.keys():
            short = full.split("#", 1)[-1] if "#" in full else full.rsplit("/", 1)[-1]
            self._short2full.setdefault(short, full)

        # adjacency: per-direction, per-rel.  Use canonical rel names from
        # rel_canonical (set by relation_canonicalizer.py); the parquet stores
        # canonical-only post `collapse_inverses_to_canonical`.
        rel_col = "rel_canonical" if "rel_canonical" in edges.columns else "rel"
        # outgoing: src --rel--> dst (canonical direction)
        self._out: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        # incoming: dst -- (rel reversed) -- src
        self._in:  Dict[str, List[Dict[str, str]]] = defaultdict(list)
        allowed_col = "rel_allowed_for_local_sampling"
        # `global_*` edges (building-wide points attached to a structural
        # root) are folded into their base rel for _out/_in, but also kept
        # in a separate index so callers can recover the building-global set.
        self._global_out: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for s, r, d in zip(edges["src"], edges[rel_col].astype(str), edges["dst"]):
            is_global = r.startswith("global_")
            base = r[len("global_"):] if is_global else r
            self._out[s].append({"rel": base, "neighbour": d})
            self._in[d].append({"rel": base, "neighbour": s})
            if is_global:
                self._global_out[s].append({"rel": r, "neighbour": d})

    # ------------------------------------------------------------------
    def resolve(self, node_or_short: str) -> Optional[str]:
        if node_or_short in self._node_meta:
            return node_or_short
        return self._short2full.get(node_or_short)

    def metadata(self, node: str) -> Dict[str, Any]:
        full = self.resolve(node) or node
        return self._node_meta.get(full, {
            "brick_class": "?", "node_type": "?",
            "has_ts": False, "is_forecast_target": False, "is_context_ts": False,
        })

    def query_class(self, brick_class: str, has_ts: Optional[bool] = None,
                     max_n: int = 30) -> List[Dict[str, Any]]:
        """Find nodes of a given brick_class (handy for `Outdoor_Air_Temp` lookups
        that aren't in the target's local neighbourhood)."""
        out = []
        for nid in self._class_to_nodes.get(brick_class, []):
            m = self.metadata(nid)
            if has_ts is not None and m["has_ts"] != has_ts:
                continue
            out.append({"neighbour": nid, **m})
            if len(out) >= max_n:
                break
        return out

    def neighbors(self, node: str, rels: Iterable[str],
                   direction: str = "both", limit: int = 30) -> List[Dict[str, Any]]:
        """Get typed neighbours of a node — direction is interpreted in the
        AGENT's semantic frame, not the canonical storage frame.

        For a canonical rel R (hasPoint, hasPart, feeds, isLocationOf):
          * `direction=outgoing`  →  walk R forward     → scan `_out`
          * `direction=incoming`  →  walk R backward    → scan `_in`

        For an inverse rel R^-1 (isPointOf, isPartOf, isFedBy, hasLocation):
          * `direction=outgoing`  →  walk R^-1 forward  → scan `_in`  (canonical R reversed)
          * `direction=incoming`  →  walk R^-1 backward → scan `_out` (canonical R forward)

        Concretely, `neighbors(zone, [isFedBy], 'outgoing')` returns the things
        that feed the zone (upstream VAV/RTU), even though the underlying
        edge is stored as `vav --feeds--> zone`.
        """
        full = self.resolve(node)
        if full is None:
            return []
        rels_set = {r for r in rels}
        out: List[Dict[str, Any]] = []
        seen: Set[tuple] = set()

        def _emit(label_rel: str, dir_label: str, nbr: str) -> bool:
            key = (label_rel, nbr)
            if key in seen:
                return False
            seen.add(key)
            m = self.metadata(nbr)
            out.append({"rel": label_rel, "direction": dir_label,
                         "neighbour": nbr, **m})
            return len(out) >= limit

        for r in rels_set:
            if r in CANONICAL_RELS:
                # canonical rel: outgoing → _out, incoming → _in
                if direction in ("outgoing", "both"):
                    for e in self._out.get(full, []):
                        if e["rel"] == r and _emit(r, "out", e["neighbour"]):
                            return out
                if direction in ("incoming", "both"):
                    for e in self._in.get(full, []):
                        if e["rel"] == r and _emit(r, "in", e["neighbour"]):
                            return out
            elif r in INVERSE_TO_CANONICAL:
                canon = INVERSE_TO_CANONICAL[r]
                # inverse rel: ignore the direction parameter and ALWAYS
                # return the natural-reading neighbours (e.g. isPointOf returns
                # the equipment that owns this point, isFedBy returns the
                # upstream feeder). The rel name already encodes direction.
                for e in self._in.get(full, []):
                    if e["rel"] == canon and _emit(r, "in", e["neighbour"]):
                        return out
        return out

    def global_neighbors(self, node: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Neighbours reachable via a `global_*` edge — building-wide signals
        (weather, plant-loop temps, building aggregates) the KG ingestion
        attaches to a structural root. `neighbors()` folds these into plain
        hasPoint; this method is the only way to recover the global-only set.
        Each result dict carries node metadata, like `neighbors()`.
        """
        full = self.resolve(node)
        if full is None:
            return []
        out: List[Dict[str, Any]] = []
        for e in self._global_out.get(full, []):
            m = self.metadata(e["neighbour"])
            out.append({"rel": e["rel"], "direction": "out",
                         "neighbour": e["neighbour"], **m})
            if len(out) >= limit:
                break
        return out
