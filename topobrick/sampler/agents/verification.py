from __future__ import annotations

from collections import Counter
from typing import Any

from topobrick.sampler.agents.client import request_with_retry
from topobrick.sampler.agents.prompts import VERIFIER_SYSTEM_PROMPT
from topobrick.sampler.context.materialize import materialize_actions
from topobrick.sampler.context.topology import TopologyContext
from topobrick.sampler.graph.points import PointIndex


def verify_exogenous_points(
    client,
    model: str,
    context: TopologyContext,
    selected_points: list[str],
    target: str,
    points: PointIndex,
) -> tuple[list[str], set[str], dict[str, Any]]:
    class_counts = Counter(points.class_of(point) or "?" for point in selected_points)
    selection_summary = ", ".join(
        f"{point_class}×{count}" for point_class, count in class_counts.most_common()
    )
    message = (
        context.text
        + "\n\nCOVARIATES THE ENGINEER CHOSE (audit these):\n  "
        + (selection_summary or "(none)")
    )
    output, error = request_with_retry(
        client,
        model,
        VERIFIER_SYSTEM_PROMPT,
        message,
    )
    additions, _ = materialize_actions(
        output.get("add", []),
        context.anchors,
        target,
        points,
    )
    dropped_classes = set(output.get("drop", []) or [])
    record = {
        "verdict": output.get("verdict", ""),
        "added": len(additions),
        "dropped_classes": sorted(dropped_classes),
        "note": output.get("note", ""),
    }
    if error is not None:
        record["error"] = str(error)
    return additions, dropped_classes, record
