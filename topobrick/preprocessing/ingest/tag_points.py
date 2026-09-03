"""Tag forecast targets, context-TS nodes, and online-graph nodes."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, Set

import pandas as pd

from topobrick.utils.progress import log


def _matches_any(brick_class: str, patterns) -> bool:
    if not brick_class:
        return False
    bc = brick_class.lower()
    for p in patterns:
        if p.lower() in bc:
            return True
    return False


def filter_points_and_nodes(
    kg_nodes_df: pd.DataFrame,
    kg_edges_df: pd.DataFrame,
    point_map_df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Set is_forecast_target / is_context_ts / is_online_graph_node on kg_nodes_df."""
    nodes = kg_nodes_df.copy()
    incl = cfg["points"]["forecast_target_include_substrings"]
    excl = cfg["points"]["forecast_target_exclude_substrings"]
    include_control_as_ctx = cfg["preprocessing"].get(
        "include_control_points_as_context", True
    )
    allow_control_target = cfg["preprocessing"].get(
        "allow_control_points_as_target", False
    )

    def _is_target(row) -> bool:
        if not row["has_timeseries"]:
            return False
        if row["node_type"] != "Point":
            return False
        bc = row["brick_class"] or ""
        # exclude pattern dominates unless allow_control_target
        if (not allow_control_target) and _matches_any(bc, excl):
            return False
        if _matches_any(bc, incl):
            return True
        # by default, only sensors and meters become targets; everything else excluded
        return False

    nodes["is_forecast_target"] = nodes.apply(_is_target, axis=1)

    def _is_ctx_ts(row) -> bool:
        if not row["has_timeseries"]:
            return False
        if row["node_type"] != "Point":
            return False
        bc = row["brick_class"] or ""
        # boolean / status / alarm / mode classes excluded as context_ts
        non_numeric_hints = ("Status", "Mode", "Alarm")
        if any(h in bc for h in non_numeric_hints):
            return False
        # setpoints/commands allowed only if include_control_points_as_context
        control_hints = ("Setpoint", "Command")
        if any(h in bc for h in control_hints):
            return include_control_as_ctx
        return True

    nodes["is_context_ts"] = nodes.apply(_is_ctx_ts, axis=1)

    # Targets and context_ts always included.
    online: Set[str] = set(nodes.loc[nodes["is_forecast_target"], "node_id"])
    online.update(nodes.loc[nodes["is_context_ts"], "node_id"])

    # Add equipment / location / system nodes that are within max_radius of any retained point
    max_radius = cfg["kg"].get("online_node_max_radius", 4)
    if not kg_edges_df.empty:
        adj = defaultdict(list)
        for _, e in kg_edges_df.iterrows():
            adj[e["src"]].append(e["dst"])
        # BFS frontier from current online set
        visited = dict.fromkeys(online, 0)
        q = deque((n, 0) for n in online)
        while q:
            n, h = q.popleft()
            if h >= max_radius:
                continue
            for nbr in adj.get(n, []):
                if nbr not in visited:
                    visited[nbr] = h + 1
                    q.append((nbr, h + 1))
        # Add the visited nodes whose type is Equipment / Location / System / Unknown(non-Point)
        type_map = dict(zip(nodes["node_id"], nodes["node_type"]))
        for nid in visited:
            t = type_map.get(nid)
            if t in ("Equipment", "Location", "System"):
                online.add(nid)

    nodes["is_online_graph_node"] = nodes["node_id"].isin(online)

    # Mirror flags into point_map
    nid2target = dict(zip(nodes["node_id"], nodes["is_forecast_target"]))
    nid2ctx = dict(zip(nodes["node_id"], nodes["is_context_ts"]))
    point_map_df["is_forecast_target"] = (
        point_map_df["kg_node_id"].map(nid2target).fillna(False).astype(bool)
    )
    point_map_df["is_context_ts"] = (
        point_map_df["kg_node_id"].map(nid2ctx).fillna(False).astype(bool)
    )

    n_target = int(nodes["is_forecast_target"].sum())
    n_ctx = int(nodes["is_context_ts"].sum())
    n_online = int(nodes["is_online_graph_node"].sum())
    log(
        f"filter_points: targets={n_target}, ctx_ts={n_ctx}, online={n_online}, no_ts_online={int(((~nodes['has_timeseries']) & nodes['is_online_graph_node']).sum())}"
    )
    return nodes
