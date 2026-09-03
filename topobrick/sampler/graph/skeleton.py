from __future__ import annotations

import json
import os
from collections import Counter, defaultdict, deque

import pandas as pd

from topobrick.sampler.graph.ontology import (
    DEFAULT_ONTOLOGY_PATH,
    FALLBACK_TYPE,
    is_structural_type,
    resolve_top_level_types,
)
from topobrick.utils.paths import resolve_processed_root

STRUCTURAL_RELATIONS = {"feeds", "hasPart", "isLocationOf"}
INVERSE_STRUCTURAL_RELATIONS = {
    "isFedBy": "feeds",
    "isPartOf": "hasPart",
    "hasLocation": "isLocationOf",
}
ASCEND_PRIORITY = {"feeds": 0, "hasPart": 1, "isLocationOf": 2}

class BuildingSkeleton:
    def __init__(self, data: dict):
        self.building: str = data["building"]
        self.roots: set[str] = set(data["roots"])
        self.node_class: dict[str, str] = data["nodes"]
        self.class_type: dict[str, str] = data.get("class_type", {})
        self._children: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self._parents: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for parent, relation, child in data["edges"]:
            self._children[parent].append((relation, child))
            self._parents[child].append((relation, parent))

    @classmethod
    def load(
        cls, dataset: str, processed_root: str | None = None
    ) -> "BuildingSkeleton":
        path = os.path.join(
            resolve_processed_root(processed_root), dataset, "skeleton.json"
        )
        with open(path) as handle:
            return cls(json.load(handle))

    def is_node(self, uri: str) -> bool:
        return uri in self.node_class

    def brick_class(self, uri: str) -> str:
        return self.node_class.get(uri, "")

    def type_of(self, uri: str) -> str:
        return self.class_type.get(self.node_class.get(uri, ""), "")

    def parents(self, uri: str) -> list[tuple[str, str]]:
        return self._parents.get(uri, [])

    def children(self, uri: str) -> list[tuple[str, str]]:
        return self._children.get(uri, [])

    def up_cone(
        self, start: str, max_hops: int = 12
    ) -> tuple[set[str], list[list[str]]]:
        if start not in self.node_class:
            return set(), []
        nodes = {start}
        edges = []
        seen_edges = set()
        pending = deque([(start, 0)])
        while pending:
            current, depth = pending.popleft()
            if depth >= max_hops:
                continue
            for relation, parent in self._parents.get(current, []):
                edge = (parent, relation, current)
                if edge not in seen_edges:
                    seen_edges.add(edge)
                    edges.append(list(edge))
                if parent not in nodes:
                    nodes.add(parent)
                    pending.append((parent, depth + 1))
        return nodes, edges

    def primary_path(self, start: str, max_hops: int = 12) -> list[str]:
        if start not in self.node_class:
            return []
        path = [start]
        current = start
        visited = {start}
        for _ in range(max_hops):
            if current in self.roots:
                break
            parents = [
                (ASCEND_PRIORITY.get(relation, 9), relation, parent)
                for relation, parent in self._parents.get(current, [])
                if parent not in visited
            ]
            if not parents:
                break
            parents.sort()
            current = parents[0][2]
            path.append(current)
            visited.add(current)
        return path

    def subtree(self, start: str, max_depth: int | None = None) -> set[str]:
        if start not in self.node_class:
            return set()
        descendants = {start}
        pending = deque([(start, 0)])
        while pending:
            current, depth = pending.popleft()
            if max_depth is not None and depth >= max_depth:
                continue
            for _, child in self._children.get(current, []):
                if child not in descendants:
                    descendants.add(child)
                    pending.append((child, depth + 1))
        return descendants


def _short_name(uri: str) -> str:
    return uri.split("#", 1)[-1] if "#" in uri else uri.rsplit("/", 1)[-1]


def _canonical_structural_edge(
    source: str, relation: str, destination: str
) -> tuple[str, str, str] | None:
    relation = relation.removeprefix("global_")
    if relation in STRUCTURAL_RELATIONS:
        return source, relation, destination
    canonical_relation = INVERSE_STRUCTURAL_RELATIONS.get(relation)
    if canonical_relation:
        return destination, canonical_relation, source
    return None


def build_skeleton(
    dataset: str,
    *,
    processed_root: str | None = None,
    ontology_path: str = DEFAULT_ONTOLOGY_PATH,
    write: bool = True,
) -> dict:
    processed_dir = os.path.join(resolve_processed_root(processed_root), dataset)
    nodes = pd.read_parquet(os.path.join(processed_dir, "kg_nodes.parquet"))
    edges = pd.read_parquet(os.path.join(processed_dir, "kg_edges.parquet"))

    class_type = resolve_top_level_types(
        nodes["brick_class"].dropna().unique(), ontology_path=ontology_path
    )
    node_class = dict(zip(nodes["node_id"], nodes["brick_class"]))
    timeseries_column = (
        "has_timeseries" if "has_timeseries" in nodes.columns else "has_ts"
    )
    has_timeseries = dict(
        zip(nodes["node_id"], nodes[timeseries_column].fillna(False))
    )

    def top_level_type(uri: str) -> str:
        return class_type.get(node_class.get(uri, ""), FALLBACK_TYPE)

    structural_nodes = {
        uri
        for uri in node_class
        if is_structural_type(top_level_type(uri))
        and not bool(has_timeseries.get(uri, False))
    }
    relation_column = "rel_canonical" if "rel_canonical" in edges.columns else "rel"
    structural_edges = []
    children = set()
    seen_edges = set()
    for source, relation, destination in zip(
        edges["src"], edges[relation_column].astype(str), edges["dst"]
    ):
        edge = _canonical_structural_edge(source, relation, destination)
        if (
            edge is not None
            and edge[0] in structural_nodes
            and edge[2] in structural_nodes
            and edge not in seen_edges
        ):
            seen_edges.add(edge)
            structural_edges.append(list(edge))
            children.add(edge[2])

    roots = sorted(structural_nodes - children)
    skeleton = {
        "building": dataset,
        "roots": roots,
        "class_type": class_type,
        "nodes": {uri: node_class[uri] for uri in sorted(structural_nodes)},
        "edges": structural_edges,
    }

    type_counts = Counter(top_level_type(uri) for uri in structural_nodes)
    relation_counts = Counter(edge[1] for edge in structural_edges)
    connected = structural_nodes & (
        children | {edge[0] for edge in structural_edges}
    )
    root_preview = [_short_name(root) for root in roots[:8]]
    print(f"\n=== {dataset} skeleton ===")
    print(f"  structural nodes : {len(structural_nodes)}  by type {dict(type_counts)}")
    print(
        f"  structural edges : {len(structural_edges)}  "
        f"by relation {dict(relation_counts)}"
    )
    print(
        f"  roots            : {len(roots)}  -> {root_preview}"
        + (" ..." if len(roots) > 8 else "")
    )
    print(f"  isolated nodes   : {len(structural_nodes - connected)}")

    if write:
        output_path = os.path.join(processed_dir, "skeleton.json")
        with open(output_path, "w") as handle:
            json.dump(skeleton, handle)
        print(f"  wrote {output_path} ({os.path.getsize(output_path) / 1024:.0f} KB)")
    return skeleton
