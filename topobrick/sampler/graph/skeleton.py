"""The building's structural graph, loaded from skeleton.json.

Ascend to a node's ancestry (its "you are here"), descend into its subtree (every
zone under an RTU). Structural nodes only — leaves hang off this, they are not in it.
"""
from __future__ import annotations
import json
import os
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

PROCESSED_ROOT = os.environ.get('TOPOBRICK_DATA_ROOT', os.path.expanduser('~/topobrick_data/processed'))

# Canonical structural rels, ordered by ascend priority (lower = preferred).
# isPointOf (leaf->equipment) is the pre-skeleton step handled by the leaf
# resolver, so it is not included here.
ASCEND_PRIORITY = {"feeds": 0, "hasPart": 1, "isLocationOf": 2}


class Skeleton:
    def __init__(self, data: dict):
        self.building: str = data["building"]
        self.roots: Set[str] = set(data["roots"])
        self.node_class: Dict[str, str] = data["nodes"]
        self.class_type: Dict[str, str] = data.get("class_type", {})
        # parent --rel--> child
        self._out: Dict[str, List[Tuple[str, str]]] = defaultdict(list)  # node -> [(rel, child)]
        self._in: Dict[str, List[Tuple[str, str]]] = defaultdict(list)   # node -> [(rel, parent)]
        for s, rel, d in data["edges"]:
            self._out[s].append((rel, d))
            self._in[d].append((rel, s))

    # ---- loading ----
    @classmethod
    def load(cls, dataset: str, root: str = PROCESSED_ROOT) -> "Skeleton":
        with open(os.path.join(root, dataset, "skeleton.json")) as f:
            return cls(json.load(f))

    # ---- node info ----
    def is_node(self, uri: str) -> bool:
        return uri in self.node_class

    def brick_class(self, uri: str) -> str:
        return self.node_class.get(uri, "")

    def type_of(self, uri: str) -> str:
        return self.class_type.get(self.node_class.get(uri, ""), "")

    def parents(self, uri: str) -> List[Tuple[str, str]]:
        """[(rel, parent_uri)] — edges where `uri` is the child/downstream."""
        return self._in.get(uri, [])

    def children(self, uri: str) -> List[Tuple[str, str]]:
        """[(rel, child_uri)] — edges where `uri` is the parent/upstream."""
        return self._out.get(uri, [])

    # ---- ascend: ancestry up-cone ----
    def up_cone(self, start: str, max_hops: int = 12
               ) -> Tuple[Set[str], List[List[str]]]:
        """All ancestors of `start` (toward roots) via structural edges — both
        containment (hasPart/isLocationOf) and feeds. feeds-upstream is kept in
        full: an Electrical_Circuit that feeds an AHU is that AHU's own power
        metering (a legitimate load proxy), not noise. Returns (nodes,
        edges[[p,rel,c]]).
        """
        if start not in self.node_class:
            return set(), []
        nodes: Set[str] = {start}
        edges: List[List[str]] = []
        seen_edge: Set[Tuple[str, str, str]] = set()
        q = deque([(start, 0)])
        while q:
            cur, h = q.popleft()
            if h >= max_hops:
                continue
            for rel, par in self._in.get(cur, []):
                key = (par, rel, cur)
                if key not in seen_edge:
                    seen_edge.add(key)
                    edges.append([par, rel, cur])
                if par not in nodes:
                    nodes.add(par)
                    q.append((par, h + 1))
        return nodes, edges

    def primary_path(self, start: str, max_hops: int = 12) -> List[str]:
        """Single greedy ancestor path (priority feeds<hasPart<isLocationOf,
        then prefer reaching a root). For compact 'you are here' display."""
        if start not in self.node_class:
            return []
        path = [start]
        cur = start
        visited = {start}
        for _ in range(max_hops):
            if cur in self.roots:
                break
            pars = [(ASCEND_PRIORITY.get(rel, 9), rel, p)
                    for rel, p in self._in.get(cur, []) if p not in visited]
            if not pars:
                break
            pars.sort()
            cur = pars[0][2]
            path.append(cur)
            visited.add(cur)
        return path

    # ---- descend: subtree ----
    def subtree(self, start: str, max_depth: Optional[int] = None
                ) -> Set[str]:
        """All structural descendants of `start` (inclusive) via _out edges.
        Used to resolve `expand:subtree` pulls (e.g. every zone under an RTU)."""
        if start not in self.node_class:
            return set()
        out: Set[str] = {start}
        q = deque([(start, 0)])
        while q:
            cur, d = q.popleft()
            if max_depth is not None and d >= max_depth:
                continue
            for _rel, child in self._out.get(cur, []):
                if child not in out:
                    out.add(child)
                    q.append((child, d + 1))
        return out

    def reaches_root(self, start: str) -> Optional[str]:
        """Return a root reachable by ascending from `start`, else None."""
        nodes, _ = self.up_cone(start)
        hit = nodes & self.roots
        return next(iter(hit)) if hit else None
