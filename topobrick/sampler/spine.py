"""derive_spine — the target's "you are here" structural context.

Step 3 of the per-target pipeline. A target leaf attaches to the Brick skeleton
at one or more resolved attach nodes (from resolver.LeafResolver); from those
nodes we take the full ancestry up-cone in the skeleton, which includes BOTH the
feeds-upstream chain and the containment chain to the building root. The result
is the target's position, consumed downstream by context.render_context.

Returns a dict the prompt layer renders as the target's position; no greedy
single-path pruning (the up-cone fans IN toward roots, so it stays small).
"""
from __future__ import annotations
from typing import Dict, List

from topobrick.sampler.skeleton import Skeleton
from topobrick.sampler.resolver import LeafResolver


def derive_spine(leaf: str, skeleton: Skeleton, resolver: LeafResolver) -> Dict:
    attach = [a for a in resolver.attach_nodes(leaf) if skeleton.is_node(a)]
    if not attach:
        return {
            "leaf": leaf, "attach": [], "ancestry_nodes": [],
            "ancestry_edges": [], "roots_reached": [], "primary_path": [],
            "orphan": True,
        }

    nodes = set()
    edges = []
    seen_e = set()
    for a in attach:
        n, e = skeleton.up_cone(a)
        nodes |= n
        for ed in e:
            k = tuple(ed)
            if k not in seen_e:
                seen_e.add(k)
                edges.append(ed)

    roots_reached = sorted(nodes & skeleton.roots)
    # primary path from the first (highest-priority) attach node, for display
    primary = skeleton.primary_path(attach[0])

    return {
        "leaf": leaf,
        "attach": attach,
        "ancestry_nodes": sorted(nodes),
        "ancestry_edges": edges,
        "roots_reached": roots_reached,
        "primary_path": primary,
        "orphan": False,
    }
