"""Renders the egocentric context the picker LLM sees, then resolves its pull
requests back into concrete timeseries leaves — step 4 of the sampler pipeline
(after spine derivation, before the pipeline assembles the subgraph record).

render_context turns (target, skeleton, spine, resolver) into the "you are here"
prompt — the target's position, the peer cohorts and pullable class histograms at
each ancestor, and building-wide global drivers — labelling anchors A1..An that the
picker references. resolve_pulls maps those pull requests back to concrete leaf URIs
via the resolver's primitives (self / subtree / building / cluster).

Deterministic; no LLM call, no timeseries-data inspection beyond has_ts membership.
"""
from __future__ import annotations
from collections import Counter, deque
from typing import Dict, List, Optional, Tuple

from topobrick.sampler.graph.skeleton import Skeleton
from topobrick.sampler.graph.resolver import LeafResolver

# how many leaf classes to show per histogram
HIST_TOPK = 12
SUBTREE_HIST_CAP = 400      # node budget when aggregating a subtree histogram


def _short(uri: str, n: int = 14) -> str:
    s = uri.split("#", 1)[-1] if "#" in uri else uri.rsplit("/", 1)[-1]
    if "." in s:                       # BTS dotted path -> last segment
        s = s.split(".")[-1]
    return s[:n]


def _self_classes(node: str, resolver: LeafResolver) -> Counter:
    """Point classes directly on a node (cheap, never explodes)."""
    return Counter({c: len(v) for c, v in resolver.points.get(node, {}).items()})


def _order_ancestry(spine: dict, skel: Skeleton) -> List[str]:
    """Ancestry nodes ordered nearest-attach -> root (BFS distance from attach)."""
    attach = spine.get("attach") or []
    anc = set(spine.get("ancestry_nodes") or [])
    if not attach:
        return sorted(anc)
    dist = {a: 0 for a in attach}
    q = deque(attach)
    while q:
        cur = q.popleft()
        for _rel, par in skel.parents(cur):
            if par in anc and par not in dist:
                dist[par] = dist[cur] + 1
                q.append(par)
    return sorted(anc, key=lambda n: (dist.get(n, 99), skel.brick_class(n)))


def _global_hosts(skel: Skeleton, resolver: LeafResolver) -> List[str]:
    """Building-wide driver hosts (structural heuristic, NOT a class whitelist):
    LOCATION-typed roots (Building/Site) + their direct children (where weather
    stations / utility meters attach) + External + Collection nodes, restricted
    to those that actually host has_ts leaves.

    Equipment roots (e.g. LBNL RTUs, which are roots only because the KG lacks
    an RTU->Building edge) are NOT global drivers and are excluded — otherwise
    every top-level RTU pollutes the global list.
    """
    cand = set()
    loc_roots = [r for r in skel.roots if skel.type_of(r) == "Location"]
    cand.update(loc_roots)
    for r in loc_roots:
        for _rel, ch in skel.children(r):
            cand.add(ch)
    for u in skel.node_class:
        if skel.type_of(u) in ("External", "Collection"):
            cand.add(u)
    return [u for u in cand if resolver.points.get(u)]


def _fmt_hist(h: Counter, k: int = HIST_TOPK) -> str:
    items = h.most_common(k)
    extra = len(h) - len(items)
    s = ", ".join(f"{c}×{n}" for c, n in items)
    return s + (f", +{extra} more" if extra > 0 else "")


def render_context(target: str, skel: Skeleton, spine: dict,
                   resolver: LeafResolver) -> Tuple[str, Dict[str, str]]:
    """Returns (prompt_text, anchor_map: alias -> node_uri | special-token).

    Topology-preserving layout: the target's own node, then each PARENT branch
    (relation + the same-class peer cohort that branch groups + that parent's
    upstream chain with self-points), then building-wide global drivers. The
    DAG branches and the actual cohorts are kept, not flattened to counts.
    """
    tcls = resolver.brick_class.get(target, "?")
    anchor_map: Dict[str, str] = {}
    node_alias: Dict[str, str] = {}
    lines: List[str] = []
    lines.append(f"TARGET: {tcls}  (id={_short(target)})")
    lines.append("")
    aid = 0

    def alias(uri: str) -> str:
        nonlocal aid
        if uri in node_alias:
            return node_alias[uri]
        aid += 1
        a = f"A{aid}"
        node_alias[uri] = a
        anchor_map[a] = uri
        return a

    def cohort_str(parent: str) -> str:
        coh = [l for l in resolver.points_under(parent, tcls, expand="subtree")
               if l != target]
        if not coh:
            return ""
        ex = ", ".join(_short(c) for c in coh[:5])
        more = f" (+{len(coh)-5})" if len(coh) > 5 else ""
        return f"{len(coh)} {tcls} peers: {ex}{more}"

    used: set = set()

    if spine.get("orphan"):
        peers = resolver.cluster_peers(target)
        hist = Counter(resolver.brick_class.get(p, "?") for p in peers)
        a = alias("__cluster__")
        lines.append("YOU ARE HERE: KG-orphan (no structural parent). Recovered "
                     "into an UNMODELED-equipment cluster; its cluster-mates:")
        lines.append(f"  [{a}] cluster (expand=cluster): {_fmt_hist(hist)}")
        lines.append("")
    else:
        attach = (spine.get("attach") or [target])[0]
        used.add(attach)
        a0 = alias(attach)
        self0 = _self_classes(attach, resolver)
        lines.append(f"on target node [{a0}] {skel.brick_class(attach)}"
                     f"({_short(attach)})"
                     + (f": self={{{_fmt_hist(self0)}}}" if self0 else ""))
        lines.append("")
        lines.append("GROUPED-WITH  (each parent branch = a peer cohort + its "
                     "upstream; pull a cohort with expand=subtree on its anchor):")
        for rel, parent in skel.parents(attach):
            ap = alias(parent)
            cs = cohort_str(parent)
            ps = _self_classes(parent, resolver)
            line = f"  via {rel} <- [{ap}] {skel.brick_class(parent)}({_short(parent)})"
            if cs:
                line += f"  = {cs}"
            if ps:
                line += f"  self={{{_fmt_hist(ps)}}}"
            lines.append(line)
            used.add(parent)
            # parent's upstream chain to root (deduped), with self-points
            anc_nodes, _ = skel.up_cone(parent)
            order = _order_ancestry({"attach": [parent], "ancestry_nodes": anc_nodes}, skel)
            for anc in order:
                if anc == parent or anc in used:
                    continue
                used.add(anc)
                aa = alias(anc)
                hs = _self_classes(anc, resolver)
                tag = " (root)" if anc in skel.roots else ""
                seg = f"      ^up [{aa}] {skel.brick_class(anc)}({_short(anc)}){tag}"
                if hs:
                    seg += f"  self={{{_fmt_hist(hs)}}}"
                lines.append(seg)
        lines.append("")

    # global drivers (skip nodes already shown above)
    ghosts = [g for g in _global_hosts(skel, resolver) if g not in used]
    if ghosts:
        lines.append("GLOBAL DRIVERS  (building-wide; weather/utility/plant):")
        for g in ghosts:
            h = _self_classes(g, resolver)
            if h:
                a = alias(g)
                lines.append(f"  [{a}] {skel.brick_class(g)}({_short(g)}): {_fmt_hist(h)}")
        lines.append("")

    lines.append("BUILDING-WIDE POOL: request any brick_class building-wide "
                 "(expand=building) for broad peer pooling.")
    return "\n".join(lines), anchor_map


def resolve_pulls(pulls: List[dict], anchor_map: Dict[str, str],
                  target: str, skel: Skeleton, resolver: LeafResolver
                  ) -> Tuple[List[str], List[dict]]:
    """Map agent pulls -> leaf URIs. Returns (leaves, resolved_pull_records).

    No per-pull cap: the agent is taught to scope its cohort by reasoning
    (tight serving-equipment vs building-wide), so we don't rigidly trim — a
    code cap is brittle (e.g. it would silently drop a legitimately-large loop).
    """
    leaves: List[str] = []
    seen = {target}
    records: List[dict] = []
    for p in pulls:
        anchor = p.get("anchor", "")
        expand = p.get("expand", "self")
        cls = p.get("point_class") or None
        node = anchor_map.get(anchor, anchor)  # alias -> uri, else raw

        if expand == "cluster" or node == "__cluster__":
            got = resolver.cluster_peers(target, cls)
        elif expand == "building":
            got = []
            for host in resolver.hosts_of_class(cls) if cls else []:
                got.extend(resolver.points.get(host, {}).get(cls, []))
        elif expand == "subtree":
            got = resolver.points_under(node, cls, expand="subtree")
        else:  # self
            got = resolver.points_under(node, cls, expand="self")

        kept = [lf for lf in got if not (lf in seen or seen.add(lf))]
        leaves.extend(kept)
        records.append({**p, "n_resolved": len(kept)})
    return leaves, records
