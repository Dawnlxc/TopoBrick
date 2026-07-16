from __future__ import annotations
from typing import Dict, List

from topobrick.sampler.graph.skeleton import Skeleton
from topobrick.sampler.graph.resolver import LeafResolver


def derive_spine(leaf: str, skeleton: Skeleton, resolver: LeafResolver) -> Dict:
    attach = [a for a in resolver.attach_nodes(leaf) if skeleton.is_node(a)]
    if not attach:
        return {
            "leaf": leaf, 
            "attach": [], 
            "ancestry_nodes": [],
            "ancestry_edges": [], 
            "roots_reached": [], 
            "primary_path": [],
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
