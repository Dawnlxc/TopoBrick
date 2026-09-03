from __future__ import annotations

from typing import Any

from topobrick.sampler.graph.points import PointIndex

EXPANSION_SCOPES = {"self", "subtree", "building", "cluster"}


def materialize_actions(
    actions: list[dict[str, Any]],
    anchor_table: dict[str, str],
    target: str,
    points: PointIndex,
) -> tuple[list[str], list[dict[str, Any]]]:
    selected = []
    seen = {target}
    records = []
    for action in actions:
        anchor = anchor_table.get(str(action.get("anchor", "")))
        scope = str(action.get("expand", "self"))
        point_class = action.get("point_class") or None
        resolved = []
        if anchor is not None and scope in EXPANSION_SCOPES:
            if scope == "cluster" or anchor == "__cluster__":
                resolved = points.cluster_peers(target, point_class)
            elif scope == "building" and point_class:
                resolved = [
                    point
                    for host in points.hosts_of_class(point_class)
                    for point in points.points_on(host, point_class)
                ]
            else:
                resolved = points.points_under(
                    anchor,
                    point_class,
                    include_subtree=scope == "subtree",
                )

        kept = []
        for point in resolved:
            if point not in seen:
                seen.add(point)
                kept.append(point)
        selected.extend(kept)
        records.append({**action, "n_resolved": len(kept)})
    return selected, records
