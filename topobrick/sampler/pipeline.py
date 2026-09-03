from __future__ import annotations

from typing import Any

from topobrick.sampler.agents.selection import select_exogenous_actions
from topobrick.sampler.agents.verification import verify_exogenous_points
from topobrick.sampler.context.materialize import materialize_actions
from topobrick.sampler.context.topology import TopologyContext, build_topology_context
from topobrick.sampler.graph.points import PointIndex
from topobrick.sampler.graph.skeleton import BuildingSkeleton

DEFAULT_MAX_POINTS = 60


def _assemble_subgraph(
    target: str,
    points: PointIndex,
    context: TopologyContext,
    selected_points: list[str],
    agent_output: dict[str, Any],
    action_records: list[dict[str, Any]],
    selection_error: Exception | None,
) -> dict[str, Any]:
    structure = context.structure
    ancestry = [] if structure["orphan"] else list(structure["ancestry_nodes"])
    ancestry_edges = (
        [] if structure["orphan"] else list(structure["ancestry_edges"])
    )
    nodes = set(ancestry) | {target} | set(selected_points)
    edges = [
        list(edge)
        for edge in ancestry_edges
        if edge[0] in nodes and edge[2] in nodes
    ]
    for point in selected_points:
        for anchor in points.anchors(point):
            if anchor in nodes:
                edges.append([anchor, "hasPoint", point])
                break
    return {
        "target_uri": target,
        "target_class": points.class_of(target),
        "nodes": sorted(nodes),
        "edges": edges,
        "n_leaves": len(selected_points),
        "leaves": selected_points,
        "spine_size": len(ancestry),
        "orphan": bool(structure["orphan"]),
        "roots_reached": structure["roots_reached"],
        "llm_used": selection_error is None,
        "fallback_used": selection_error is not None,
        "target_interpretation": agent_output.get("target_interpretation", ""),
        "target_structural_summary": agent_output.get(
            "target_structural_summary", ""
        ),
        "pulls": action_records,
        "llm_summary": agent_output.get("summary", ""),
    }


def sample_target(
    target: str,
    skeleton: BuildingSkeleton,
    points: PointIndex,
    client,
    model: str,
    *,
    max_points: int = DEFAULT_MAX_POINTS,
    use_verifier: bool = False,
) -> dict[str, Any]:
    context = build_topology_context(target, skeleton, points)
    agent_output, selection_error = select_exogenous_actions(
        client, model, context.text
    )
    actions = agent_output.get("pulls", []) if selection_error is None else []
    selected_points, action_records = materialize_actions(
        actions,
        context.anchors,
        target,
        points,
    )

    verification = None
    if use_verifier and selection_error is None:
        additions, dropped_classes, verification = verify_exogenous_points(
            client,
            model,
            context,
            selected_points,
            target,
            points,
        )
        selected_points = [
            point
            for point in selected_points
            if points.class_of(point) not in dropped_classes
        ]
        selected_set = set(selected_points)
        for point in additions:
            allowed = points.class_of(point) not in dropped_classes
            if allowed and point not in selected_set:
                selected_set.add(point)
                selected_points.append(point)

    points_before_cap = len(selected_points)
    selected_points = selected_points[:max_points]
    record = _assemble_subgraph(
        target,
        points,
        context,
        selected_points,
        agent_output,
        action_records,
        selection_error,
    )
    record["n_pre_cap"] = points_before_cap
    if verification is not None:
        record["verifier"] = verification
    if selection_error is not None:
        record["llm_error"] = str(selection_error)
    return record
