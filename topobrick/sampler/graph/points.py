from __future__ import annotations

import os
from collections import defaultdict

import pandas as pd

from topobrick.sampler.graph.skeleton import BuildingSkeleton
from topobrick.utils.paths import resolve_processed_root

POINT_RELATIONS = {"hasPoint", "isPointOf"}

class PointIndex:
    def __init__(
        self,
        dataset: str,
        skeleton: BuildingSkeleton,
        processed_root: str | None = None,
    ):
        self.skeleton = skeleton
        processed_dir = os.path.join(
            resolve_processed_root(processed_root), dataset
        )
        nodes = pd.read_parquet(os.path.join(processed_dir, "kg_nodes.parquet"))
        edges = pd.read_parquet(os.path.join(processed_dir, "kg_edges.parquet"))
        timeseries_column = (
            "has_timeseries" if "has_timeseries" in nodes.columns else "has_ts"
        )
        self.has_timeseries = dict(
            zip(nodes["node_id"], nodes[timeseries_column].fillna(False))
        )
        self.point_class = dict(zip(nodes["node_id"], nodes["brick_class"]))
        self._points_by_anchor: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._anchors_by_point: dict[str, list[str]] = defaultdict(list)

        relation_column = (
            "rel_canonical" if "rel_canonical" in edges.columns else "rel"
        )
        seen_attachments = set()
        for source, relation, destination in zip(
            edges["src"], edges[relation_column].astype(str), edges["dst"]
        ):
            relation = relation.removeprefix("global_")
            if relation not in POINT_RELATIONS:
                continue
            source_is_point = bool(self.has_timeseries.get(source))
            destination_is_point = bool(self.has_timeseries.get(destination))
            if destination_is_point and not source_is_point:
                anchor, point = source, destination
            elif source_is_point and not destination_is_point:
                anchor, point = destination, source
            else:
                continue
            if (anchor, point) in seen_attachments:
                continue
            seen_attachments.add((anchor, point))
            self._points_by_anchor[anchor][self.class_of(point)].append(point)
            self._anchors_by_point[point].append(anchor)

        self._hosts_by_class: dict[str, set[str]] = defaultdict(set)
        for anchor, classes in self._points_by_anchor.items():
            for point_class in classes:
                self._hosts_by_class[point_class].add(anchor)

        self._cluster_by_point: dict[str, str] = {}
        self._points_by_cluster: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for point, has_timeseries in self.has_timeseries.items():
            if not has_timeseries:
                continue
            anchors = self._anchors_by_point.get(point, [])
            if any(self.skeleton.is_node(anchor) for anchor in anchors):
                continue
            cluster = self._cluster_key(point, anchors)
            self._cluster_by_point[point] = cluster
            self._points_by_cluster[cluster][self.class_of(point)].append(point)

    @classmethod
    def load(
        cls,
        dataset: str,
        skeleton: BuildingSkeleton | None = None,
        processed_root: str | None = None,
    ) -> "PointIndex":
        skeleton = skeleton or BuildingSkeleton.load(dataset, processed_root)
        return cls(dataset, skeleton, processed_root)

    @staticmethod
    def _cluster_key(point: str, anchors: list[str]) -> str:
        if anchors:
            return "node:" + anchors[0]
        fragment = (
            point.split("#", 1)[-1]
            if "#" in point
            else point.rsplit("/", 1)[-1]
        )
        segments = fragment.split(".")
        prefix = ".".join(segments[:-1]) if len(segments) > 1 else fragment
        return "prefix:" + prefix

    def class_of(self, point: str) -> str:
        return self.point_class.get(point, "")

    def anchors(self, point: str) -> list[str]:
        return list(self._anchors_by_point.get(point, []))

    def classes_on(self, anchor: str) -> dict[str, int]:
        return {
            point_class: len(points)
            for point_class, points in self._points_by_anchor.get(anchor, {}).items()
        }

    def points_on(self, anchor: str, point_class: str) -> list[str]:
        return list(self._points_by_anchor.get(anchor, {}).get(point_class, []))

    def has_points(self, anchor: str) -> bool:
        return bool(self._points_by_anchor.get(anchor))

    def points_under(
        self,
        anchor: str,
        point_class: str | None = None,
        *,
        include_subtree: bool = False,
    ) -> list[str]:
        anchors = self.skeleton.subtree(anchor) if include_subtree else {anchor}
        points = []
        seen = set()
        for current in anchors:
            points_by_class = self._points_by_anchor.get(current, {})
            classes = [point_class] if point_class else list(points_by_class)
            for class_name in classes:
                for point in points_by_class.get(class_name, []):
                    if point not in seen:
                        seen.add(point)
                        points.append(point)
        return points

    def hosts_of_class(self, point_class: str) -> list[str]:
        return sorted(self._hosts_by_class.get(point_class, ()))

    def cluster_peers(
        self, point: str, point_class: str | None = None
    ) -> list[str]:
        cluster = self._cluster_by_point.get(point)
        if cluster is None:
            return []
        points_by_class = self._points_by_cluster.get(cluster, {})
        classes = [point_class] if point_class else list(points_by_class)
        return [
            peer
            for class_name in classes
            for peer in points_by_class.get(class_name, [])
            if peer != point
        ]
