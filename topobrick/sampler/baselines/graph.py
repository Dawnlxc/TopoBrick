from __future__ import annotations

import os
from collections import defaultdict

import pandas as pd

CANONICAL_RELATIONS = {"hasPoint", "hasPart", "feeds", "isLocationOf"}
INVERSE_RELATIONS = {
    "isPointOf": "hasPoint",
    "isPartOf": "hasPart",
    "isFedBy": "feeds",
    "hasLocation": "isLocationOf",
}


class KnowledgeGraphIndex:
    def __init__(self, processed_dir: str):
        edges = pd.read_parquet(os.path.join(processed_dir, "kg_edges.parquet"))
        relation_column = (
            "rel_canonical" if "rel_canonical" in edges.columns else "rel"
        )
        self._neighbors: dict[str, set[str]] = defaultdict(set)
        for source, relation, destination in zip(
            edges["src"], edges[relation_column].astype(str), edges["dst"]
        ):
            relation = relation.removeprefix("global_")
            relation = INVERSE_RELATIONS.get(relation, relation)
            if relation not in CANONICAL_RELATIONS:
                continue
            self._neighbors[source].add(destination)
            self._neighbors[destination].add(source)

    def neighbors(self, node: str) -> set[str]:
        return self._neighbors.get(node, set())


def points_within_hops(
    graph: KnowledgeGraphIndex,
    target: str,
    hops: int,
    timeseries_points: set[str],
) -> list[str]:
    visited = {target}
    frontier = {target}
    selected = []
    for _ in range(hops):
        next_frontier = set()
        for node in sorted(frontier):
            for neighbor in sorted(graph.neighbors(node)):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                next_frontier.add(neighbor)
                if neighbor in timeseries_points:
                    selected.append(neighbor)
        frontier = next_frontier
        if not frontier:
            break
    return selected
