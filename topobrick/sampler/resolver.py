"""Step 2 of the runtime pipeline: the LeafResolver, which maps the building's
structural nodes to the timeseries leaf points that hang off them.

It builds an in-memory point index from the kg_edges point relations (hasPoint /
global_hasPoint) and exposes it three ways: attach a forecast-target leaf to its
structural parent(s) (the entry point for deriving the target's ancestry
up-cone), resolve an agent's class-filtered pull requests into concrete leaf
URIs, and report which nodes host which point classes for the global-driver
overview. Leaves that have no structural parent are grouped into synthetic
clusters here so they still have structural peers; point data is resolved at
runtime and never stored in skeleton.json.
"""
from __future__ import annotations
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set

import pandas as pd

from topobrick.sampler.skeleton import Skeleton

PROCESSED_ROOT = os.environ.get('TOPOBRICK_DATA_ROOT', os.path.expanduser('~/topobrick_data/processed'))
POINT_RELS = {"hasPoint", "global_hasPoint"}


class LeafResolver:
    def __init__(self, dataset: str, skeleton: Skeleton,
                 root: str = PROCESSED_ROOT):
        self.dataset = dataset
        self.skel = skeleton
        proc = os.path.join(root, dataset)
        nodes = pd.read_parquet(os.path.join(proc, "kg_nodes.parquet"))
        edges = pd.read_parquet(os.path.join(proc, "kg_edges.parquet"))
        tscol = "has_timeseries" if "has_timeseries" in nodes.columns else "has_ts"
        self.has_ts: Dict[str, bool] = dict(zip(nodes["node_id"], nodes[tscol].fillna(False)))
        self.brick_class: Dict[str, str] = dict(zip(nodes["node_id"], nodes["brick_class"]))

        # node -> {brick_class -> [leaf_uri]}   (leaves = has_ts)
        self.points: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        # leaf -> [parent node]
        self.leaf_parent: Dict[str, List[str]] = defaultdict(list)
        rel_col = "rel_canonical" if "rel_canonical" in edges.columns else "rel"
        seen = set()
        for s, rel, d in zip(edges["src"], edges[rel_col].astype(str), edges["dst"]):
            if rel not in POINT_RELS:
                continue
            # orient by has_ts: leaf is the timeseries endpoint, parent the other
            s_leaf, d_leaf = bool(self.has_ts.get(s)), bool(self.has_ts.get(d))
            if d_leaf and not s_leaf:
                parent, leaf = s, d
            elif s_leaf and not d_leaf:
                parent, leaf = d, s
            else:
                continue  # both/neither has_ts — not a clean point attachment
            key = (parent, leaf)
            if key in seen:
                continue
            seen.add(key)
            cls = self.brick_class.get(leaf, "")
            self.points[parent][cls].append(leaf)
            self.leaf_parent[leaf].append(parent)

        # class -> set(host nodes)
        self._class_hosts: Dict[str, Set[str]] = defaultdict(set)
        for node, by_cls in self.points.items():
            for cls in by_cls:
                self._class_hosts[cls].add(node)

        # orphan recovery: leaves with NO skeleton parent get a synthetic
        # cluster so the agent still has structural peers. Two kinds:
        #   - non-skeleton Point parent (e.g. Electrical_*_Meter sub-readings):
        #     cluster = that parent node.
        #   - no parent at all (BTS virtual-equipment points): cluster = the
        #     leaf URI's dotted-path prefix (provider encodes the unmodeled
        #     equipment there). Validated on Site_C: 78 coherent clusters, each
        #     one unmodeled AHU/terminal's full point suite.
        # Only fires for skeleton-orphan leaves; clean (skeleton-reachable)
        # leaves never touch this. LBNL has no orphans so it never activates.
        self._leaf_cluster: Dict[str, str] = {}
        self._cluster_members: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: defaultdict(list))
        for leaf, ts in self.has_ts.items():
            if not ts:
                continue
            parents = self.leaf_parent.get(leaf, [])
            if any(self.skel.is_node(p) for p in parents):
                continue  # skeleton-reachable, not an orphan
            if parents:
                ck = "node:" + parents[0]            # meter-grouped
            else:
                frag = leaf.split("#", 1)[-1] if "#" in leaf else leaf.rsplit("/", 1)[-1]
                segs = frag.split(".")
                ck = "pfx:" + (".".join(segs[:-1]) if len(segs) > 1 else frag)
            self._leaf_cluster[leaf] = ck
            self._cluster_members[ck][self.brick_class.get(leaf, "")].append(leaf)

    @classmethod
    def load(cls, dataset: str, skeleton: Optional[Skeleton] = None,
             root: str = PROCESSED_ROOT) -> "LeafResolver":
        skeleton = skeleton or Skeleton.load(dataset, root=root)
        return cls(dataset, skeleton, root=root)

    # ---- 1. attach ----
    def attach_nodes(self, leaf: str) -> List[str]:
        """Structural parent(s) of a target leaf (its skeleton entry point)."""
        return list(self.leaf_parent.get(leaf, []))

    # ---- 2. pull resolution ----
    def points_under(self, node: str, brick_class: Optional[str] = None,
                     expand: str = "self") -> List[str]:
        """Leaf URIs of `brick_class` (or all if None) attached to `node`
        (expand='self') or anywhere in its structural subtree (expand='subtree')."""
        nodes = self.skel.subtree(node) if expand == "subtree" else {node}
        out: List[str] = []
        seen: Set[str] = set()
        for n in nodes:
            by_cls = self.points.get(n)
            if not by_cls:
                continue
            classes = [brick_class] if brick_class else list(by_cls.keys())
            for c in classes:
                for leaf in by_cls.get(c, []):
                    if leaf not in seen:
                        seen.add(leaf)
                        out.append(leaf)
        return out

    # ---- 3. host lookups (for overview composition) ----
    def host_histogram(self, node: str) -> Dict[str, int]:
        return {c: len(v) for c, v in self.points.get(node, {}).items()}

    def hosts_of_class(self, brick_class: str) -> List[str]:
        return sorted(self._class_hosts.get(brick_class, ()))

    def orphan_leaves(self) -> List[str]:
        """has_ts leaves with no structural parent (need orphan recovery)."""
        return [u for u, ts in self.has_ts.items()
                if ts and not self.leaf_parent.get(u)]

    # ---- 4. orphan recovery ----
    def cluster_of(self, leaf: str) -> Optional[str]:
        """Synthetic cluster key for a skeleton-orphan leaf, else None."""
        return self._leaf_cluster.get(leaf)

    def cluster_peers(self, leaf: str, brick_class: Optional[str] = None
                      ) -> List[str]:
        """Co-cluster leaves (its unmodeled-equipment siblings), optionally
        filtered by brick_class, excluding `leaf` itself."""
        ck = self._leaf_cluster.get(leaf)
        if ck is None:
            return []
        by_cls = self._cluster_members.get(ck, {})
        classes = [brick_class] if brick_class else list(by_cls.keys())
        out: List[str] = []
        for c in classes:
            for m in by_cls.get(c, []):
                if m != leaf:
                    out.append(m)
        return out
