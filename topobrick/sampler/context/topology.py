from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

from topobrick.sampler.graph.points import PointIndex
from topobrick.sampler.graph.skeleton import BuildingSkeleton

MAX_DISPLAYED_CLASSES = 12


@dataclass(frozen=True)
class TopologyContext:
    text: str
    anchors: dict[str, str]
    structure: dict


def _short_name(uri: str, length: int = 14) -> str:
    name = uri.split("#", 1)[-1] if "#" in uri else uri.rsplit("/", 1)[-1]
    if "." in name:
        name = name.split(".")[-1]
    return name[:length]


def _anchor_chain(
    target: str, skeleton: BuildingSkeleton, points: PointIndex
) -> dict:
    anchors = [anchor for anchor in points.anchors(target) if skeleton.is_node(anchor)]
    if not anchors:
        return {
            "leaf": target,
            "attach": [],
            "ancestry_nodes": [],
            "ancestry_edges": [],
            "roots_reached": [],
            "primary_path": [],
            "orphan": True,
        }

    ancestry_nodes = set()
    ancestry_edges = []
    seen_edges = set()
    for anchor in anchors:
        nodes, edges = skeleton.up_cone(anchor)
        ancestry_nodes.update(nodes)
        for edge in edges:
            edge_key = tuple(edge)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                ancestry_edges.append(edge)
    return {
        "leaf": target,
        "attach": anchors,
        "ancestry_nodes": sorted(ancestry_nodes),
        "ancestry_edges": ancestry_edges,
        "roots_reached": sorted(ancestry_nodes & skeleton.roots),
        "primary_path": skeleton.primary_path(anchors[0]),
        "orphan": False,
    }


def _ordered_ancestry(structure: dict, skeleton: BuildingSkeleton) -> list[str]:
    anchors = structure.get("attach") or []
    ancestry = set(structure.get("ancestry_nodes") or [])
    if not anchors:
        return sorted(ancestry)
    distance = {anchor: 0 for anchor in anchors}
    pending = deque(anchors)
    while pending:
        current = pending.popleft()
        for _, parent in skeleton.parents(current):
            if parent in ancestry and parent not in distance:
                distance[parent] = distance[current] + 1
                pending.append(parent)
    return sorted(
        ancestry,
        key=lambda node: (distance.get(node, 99), skeleton.brick_class(node)),
    )


def _global_hosts(
    skeleton: BuildingSkeleton, points: PointIndex
) -> list[str]:
    location_roots = [
        root for root in skeleton.roots if skeleton.type_of(root) == "Location"
    ]
    candidates = set(location_roots)
    for root in location_roots:
        candidates.update(child for _, child in skeleton.children(root))
    candidates.update(
        node
        for node in skeleton.node_class
        if skeleton.type_of(node) in {"External", "Collection"}
    )
    return sorted(node for node in candidates if points.has_points(node))


def _format_classes(classes: Counter, limit: int = MAX_DISPLAYED_CLASSES) -> str:
    shown = classes.most_common(limit)
    remaining = len(classes) - len(shown)
    text = ", ".join(f"{name}×{count}" for name, count in shown)
    return text + (f", +{remaining} more" if remaining else "")


def build_topology_context(
    target: str, skeleton: BuildingSkeleton, points: PointIndex
) -> TopologyContext:
    structure = _anchor_chain(target, skeleton, points)
    target_class = points.class_of(target) or "?"
    anchor_table: dict[str, str] = {}
    aliases: dict[str, str] = {}
    lines = [f"TARGET: {target_class}  (id={_short_name(target)})", ""]

    def alias(uri: str) -> str:
        if uri not in aliases:
            name = f"A{len(aliases) + 1}"
            aliases[uri] = name
            anchor_table[name] = uri
        return aliases[uri]

    def point_classes(anchor: str) -> Counter:
        return Counter(points.classes_on(anchor))

    def cohort_summary(anchor: str) -> str:
        peers = [
            point
            for point in points.points_under(
                anchor, target_class, include_subtree=True
            )
            if point != target
        ]
        if not peers:
            return ""
        examples = ", ".join(_short_name(peer) for peer in peers[:5])
        remaining = f" (+{len(peers) - 5})" if len(peers) > 5 else ""
        return f"{len(peers)} {target_class} peers: {examples}{remaining}"

    shown_nodes = set()
    if structure["orphan"]:
        peers = points.cluster_peers(target)
        classes = Counter(points.class_of(peer) or "?" for peer in peers)
        cluster_alias = alias("__cluster__")
        lines.extend(
            [
                "YOU ARE HERE: KG-orphan (no structural parent). Recovered "
                "into an UNMODELED-equipment cluster; its cluster-mates:",
                f"  [{cluster_alias}] cluster (expand=cluster): "
                f"{_format_classes(classes)}",
                "",
            ]
        )
    else:
        target_anchor = structure["attach"][0]
        shown_nodes.add(target_anchor)
        target_alias = alias(target_anchor)
        classes = point_classes(target_anchor)
        class_text = f": self={{{_format_classes(classes)}}}" if classes else ""
        lines.extend(
            [
                f"on target node [{target_alias}] "
                f"{skeleton.brick_class(target_anchor)}"
                f"({_short_name(target_anchor)}){class_text}",
                "",
                "GROUPED-WITH  (each parent branch = a peer cohort + its "
                "upstream; pull a cohort with expand=subtree on its anchor):",
            ]
        )
        for relation, parent in skeleton.parents(target_anchor):
            parent_alias = alias(parent)
            cohort = cohort_summary(parent)
            classes = point_classes(parent)
            line = (
                f"  via {relation} <- [{parent_alias}] "
                f"{skeleton.brick_class(parent)}({_short_name(parent)})"
            )
            if cohort:
                line += f"  = {cohort}"
            if classes:
                line += f"  self={{{_format_classes(classes)}}}"
            lines.append(line)
            shown_nodes.add(parent)

            ancestry, _ = skeleton.up_cone(parent)
            for ancestor in _ordered_ancestry(
                {"attach": [parent], "ancestry_nodes": ancestry}, skeleton
            ):
                if ancestor == parent or ancestor in shown_nodes:
                    continue
                shown_nodes.add(ancestor)
                ancestor_alias = alias(ancestor)
                classes = point_classes(ancestor)
                root_label = " (root)" if ancestor in skeleton.roots else ""
                segment = (
                    f"      ^up [{ancestor_alias}] "
                    f"{skeleton.brick_class(ancestor)}"
                    f"({_short_name(ancestor)}){root_label}"
                )
                if classes:
                    segment += f"  self={{{_format_classes(classes)}}}"
                lines.append(segment)
        lines.append("")

    global_hosts = [
        host for host in _global_hosts(skeleton, points) if host not in shown_nodes
    ]
    if global_hosts:
        lines.append("GLOBAL DRIVERS  (building-wide; weather/utility/plant):")
        for host in global_hosts:
            classes = point_classes(host)
            if classes:
                host_alias = alias(host)
                lines.append(
                    f"  [{host_alias}] {skeleton.brick_class(host)}"
                    f"({_short_name(host)}): {_format_classes(classes)}"
                )
        lines.append("")

    lines.append(
        "BUILDING-WIDE POOL: request any brick_class building-wide "
        "(expand=building) for broad peer pooling."
    )
    return TopologyContext("\n".join(lines), anchor_table, structure)
