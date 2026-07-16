"""
Usage:
  python -m topobrick.sampler.build_skeleton --dataset LBNL59
  python -m topobrick.sampler.build_skeleton --all
"""
from __future__ import annotations
import argparse
import json
import os
from collections import Counter

import pandas as pd

from topobrick.sampler.graph import brick as BT

PROCESSED_ROOT = os.environ.get('TOPOBRICK_DATA_ROOT', os.path.expanduser('~/topobrick_data/processed'))
STRUCT_RELS = ("feeds", "hasPart", "isLocationOf")
ALL_DATASETS = ("LBNL59", "BTS_Site_B", "BTS_Site_C", "BTS_Site_A")


def _frag(u: str) -> str:
    s = str(u)
    return s.split("#", 1)[-1] if "#" in s else s.rsplit("/", 1)[-1]


def build_skeleton(dataset: str, ttl_path: str = BT.DEFAULT_TTL,
                   write: bool = True) -> dict:
    proc = os.path.join(PROCESSED_ROOT, dataset)
    nodes = pd.read_parquet(os.path.join(proc, "kg_nodes.parquet"))
    edges = pd.read_parquet(os.path.join(proc, "kg_edges.parquet"))

    # 1. Brick top-type per class, from the ontology + manual overrides (brick.py).
    #    Deliberately NOT the dataset's own `node_type` column, which mislabels
    #    Point subclasses such as Fan_Speed as Equipment.
    class_type = BT.resolve_types(nodes["brick_class"].dropna().unique(),
                                  ttl_path=ttl_path)
    node_class = dict(zip(nodes["node_id"], nodes["brick_class"]))
    tscol = "has_timeseries" if "has_timeseries" in nodes.columns else "has_ts"
    has_ts = dict(zip(nodes["node_id"], nodes[tscol].fillna(False)))

    def ntype(uri: str) -> str:
        return class_type.get(node_class.get(uri, ""), BT.FALLBACK_TYPE)

    # Structural = a structural type AND no timeseries. A Building_Electrical_Meter
    # is typed Equipment but carries a reading, so it is a leaf, not architecture.
    struct_nodes = {u for u in node_class
                    if BT.is_structural(ntype(u)) and not bool(has_ts.get(u, False))}

    # Structural edges, both endpoints structural. `is_inverse` is deliberately NOT
    # filtered: the TTLs write containment as isPartOf, so every hasPart edge is an
    # inverse row, and dropping them would take the whole Building->Floor->Zone tree.
    rel_col = "rel_canonical" if "rel_canonical" in edges.columns else "rel"
    se = edges[edges[rel_col].isin(STRUCT_RELS)]
    skel_edges = []
    seen_edge = set()
    children = set()
    for s, rel, d in zip(se["src"], se[rel_col].astype(str), se["dst"]):
        if s in struct_nodes and d in struct_nodes:
            key = (s, rel, d)
            if key in seen_edge:
                continue
            seen_edge.add(key)
            skel_edges.append([s, rel, d])
            children.add(d)

    # 4. Roots = structural nodes that are never a child (dst) of a structural edge.
    roots = sorted(u for u in struct_nodes if u not in children)

    skeleton = {
        "building": dataset,
        "roots": roots,
        "class_type": class_type,
        "nodes": {u: node_class[u] for u in sorted(struct_nodes)},
        "edges": skel_edges,
    }

    # ---- summary ----
    type_cnt = Counter(ntype(u) for u in struct_nodes)
    rel_cnt = Counter(e[1] for e in skel_edges)
    connected = struct_nodes & (set(children) | {e[0] for e in skel_edges})
    isolated = len(struct_nodes) - len(connected)
    print(f"\n=== {dataset} skeleton ===")
    print(f"  structural nodes : {len(struct_nodes)}  by type {dict(type_cnt)}")
    print(f"  structural edges : {len(skel_edges)}  by rel {dict(rel_cnt)}")
    print(f"  roots            : {len(roots)}  -> {[ _frag(r) for r in roots[:8] ]}"
          + (" ..." if len(roots) > 8 else ""))
    print(f"  isolated nodes   : {isolated}  (structural-typed, no structural edge)")

    if write:
        out = os.path.join(proc, "skeleton.json")
        with open(out, "w") as f:
            json.dump(skeleton, f)
        size_kb = os.path.getsize(out) / 1024
        print(f"  wrote {out}  ({size_kb:.0f} KB)")

    return skeleton


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--ttl", default=BT.DEFAULT_TTL)
    ap.add_argument("--no_write", action="store_true")
    args = ap.parse_args()

    targets = ALL_DATASETS if args.all else [args.dataset]
    if not targets or targets == [None]:
        ap.error("pass --dataset <DS> or --all")
    for ds in targets:
        build_skeleton(ds, ttl_path=args.ttl, write=not args.no_write)


if __name__ == "__main__":
    main()
