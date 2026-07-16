"""
1hop
    — all has_ts leaves within 1 KG hop of the target (BFS over all Brick relations, both directions). Sparse for point targets.
khop
    — all has_ts leaves within K hops (default K=3 — the distance at which sibling points of a point target become reachable).
same_ont
    — all has_ts leaves of the target's own brick_class, building-wide.
random  
    — random has_ts leaves; per-target count matched to that target's khop count (random-vs-structure control at equal size).

Usage:
  python -m topobrick.sampler.ablations --dataset LBNL59 \
      --strategies 1hop khop random same_ont --k 3
"""
from __future__ import annotations
import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Set

import pandas as pd

from topobrick.sampler.sparql_helper import SparqlHelper, ALL_RELS
from topobrick.sampler.runner import forecast_targets

PROCESSED_ROOT = os.environ.get(
    'TOPOBRICK_DATA_ROOT', os.path.expanduser('~/topobrick_data/processed'))


def _load_pools(dataset: str):
    n = pd.read_parquet(os.path.join(PROCESSED_ROOT, dataset, "kg_nodes.parquet"))
    tscol = "has_timeseries" if "has_timeseries" in n.columns else "has_ts"
    has_ts = set(n.loc[n[tscol].fillna(False), "node_id"])
    bclass = dict(zip(n["node_id"], n["brick_class"]))
    by_class: Dict[str, List[str]] = defaultdict(list)
    for u in has_ts:
        by_class[bclass.get(u, "")].append(u)
    return has_ts, bclass, by_class


def khop_leaves(helper: SparqlHelper, target: str, k: int,
                has_ts: Set[str]) -> List[str]:
    """All has_ts leaves within k hops (BFS both directions), nearest-first."""
    visited = {target}
    frontier = {target}
    leaves: List[str] = []
    for _ in range(k):
        nxt: Set[str] = set()
        for node in frontier:
            for nb in helper.neighbors(node, ALL_RELS, direction="both", limit=500):
                u = nb["neighbour"]
                if u in visited:
                    continue
                visited.add(u)
                nxt.add(u)
                if u in has_ts and u != target:
                    leaves.append(u)
        frontier = nxt
        if not frontier:
            break
    return leaves


def _record(t: str, leaves: List[str], bclass: Dict[str, str], strat: str) -> dict:
    return {
        "target_uri": t, "target_class": bclass.get(t, ""),
        "nodes": [t] + leaves, "edges": [],
        "n_leaves": len(leaves), "leaves": leaves,
        "llm_used": False, "sampler": strat,
    }


def run(dataset: str, strategies: List[str], k: int = 3,
        out_dir: str = "outputs/subgraphs_ablation", seed: int = 0):
    targets = forecast_targets(dataset)
    helper = SparqlHelper(os.path.join(PROCESSED_ROOT, dataset))
    has_ts, bclass, by_class = _load_pools(dataset)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[{dataset}] {len(targets)} forecastable targets")

    # khop counts are needed both for the khop/1hop outputs and to size `random`
    khop_cache: Dict[str, List[str]] = {}

    def get_khop(t, kk):
        key = f"{t}|{kk}"
        if key not in khop_cache:
            khop_cache[key] = khop_leaves(helper, t, kk, has_ts)
        return khop_cache[key]

    for strat in strategies:
        rng = random.Random(seed)
        recs = {}
        for t in targets:
            if strat.endswith("hop"):                 # 1hop / 2hop / 3hop / khop
                kk = int(strat[:-3]) if strat[:-3].isdigit() else k
                leaves = get_khop(t, kk)
            elif strat == "same_ont":
                leaves = [u for u in by_class.get(bclass.get(t, ""), []) if u != t]
            elif strat == "random":
                n = len(get_khop(t, 2))               # match the 2-hop baseline size
                pool = [u for u in has_ts if u != t]
                leaves = rng.sample(pool, min(n, len(pool)))
            else:
                raise ValueError(strat)
            recs[t] = _record(t, leaves, bclass, strat)
        sizes = [r["n_leaves"] for r in recs.values()]
        out = os.path.join(out_dir, f"{dataset}_{strat}.json")
        with open(out, "w") as f:
            json.dump({"dataset": dataset, "sampler": strat,
                       "n_targets": len(recs), "subgraphs": recs}, f)
        nz = sum(1 for s in sizes if s == 0)
        print(f"  {strat:9s}: mean_leaves={sum(sizes)/max(1,len(sizes)):.1f} "
              f"median={sorted(sizes)[len(sizes)//2]} zero={nz}  -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--strategies", nargs="+",
                    default=["1hop", "2hop", "3hop", "random", "same_ont"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.dataset, args.strategies, k=args.k, seed=args.seed)


if __name__ == "__main__":
    main()
