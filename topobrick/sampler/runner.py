"""
Usage:
  python -m topobrick.sampler.runner \
            --dataset LBNL59 \
            --model openai/gpt-oss-20b \
            --base_url http://127.0.0.1:8000/v1 \
            --workers 8 \
            --out outputs/subgraphs/LBNL59.json 
"""
from __future__ import annotations
import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI

from topobrick.sampler.graph.skeleton import Skeleton
from topobrick.sampler.graph.resolver import LeafResolver
from topobrick.sampler.pipeline import process_one_target

from topobrick.preprocessing.l3_targets import is_forecast_eligible

PROCESSED_ROOT = os.environ.get('TOPOBRICK_DATA_ROOT', os.path.expanduser('~/topobrick_data/processed'))
DEFAULT_BASE_URL = os.environ.get('OPENAI_BASE_URL', 'http://127.0.0.1:8000/v1')


def forecast_targets(dataset: str) -> list:
    """Authoritative forecastable list of targets
    """
    n = pd.read_parquet(os.path.join(PROCESSED_ROOT, dataset, "kg_nodes.parquet"))
    m = n["is_forecast_target"].fillna(False) & n["brick_class"].apply(is_forecast_eligible)
    if "is_usable" in n.columns:
        m = m & n["is_usable"].fillna(False)
    tscol = "has_timeseries" if "has_timeseries" in n.columns else "has_ts"
    m = m & n[tscol].fillna(False)
    return n.loc[m, "node_id"].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--base_url", default=DEFAULT_BASE_URL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all targets")
    ap.add_argument("--max_leaves", type=int, default=60)
    ap.add_argument("--use_verifier", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sk = Skeleton.load(args.dataset)
    rs = LeafResolver.load(args.dataset, sk)
    targets = forecast_targets(args.dataset)
    if args.limit:
        targets = targets[:args.limit]
    print(f"[{args.dataset}] {len(targets)} forecast targets | workers={args.workers}")

    client = OpenAI(base_url=args.base_url,
                    api_key=os.environ.get("OPENAI_API_KEY", "x"))

    results = {}
    t0 = time.time()

    def work(t):
        return t, process_one_target(t, sk, rs, client, args.model,
                                     max_leaves=args.max_leaves,
                                     use_verifier=args.use_verifier)

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, t) for t in targets]
        for f in as_completed(futs):
            t, rec = f.result()
            results[t] = rec
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(targets)}  ({time.time()-t0:.0f}s)")

    out = args.out or f"outputs/subgraphs/{args.dataset}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"dataset": args.dataset, "sampler": "pull_v1",
                   "n_targets": len(results), "subgraphs": results}, f)

    # quality summary
    nl = [r["n_leaves"] for r in results.values()]
    nl_sorted = sorted(nl)
    cap_hit = sum(1 for r in results.values() if r["n_leaves"] >= args.max_leaves)
    fb = sum(1 for r in results.values() if r["fallback_used"])
    orph = sum(1 for r in results.values() if r["orphan"])
    print(f"\n=== {args.dataset} quality ({len(results)} targets, {time.time()-t0:.0f}s) ===")
    print(f"  leaves/target: min={min(nl)} med={nl_sorted[len(nl)//2]} "
          f"mean={sum(nl)/len(nl):.1f} max={max(nl)}")
    print(f"  hit max_leaves cap ({args.max_leaves}): {cap_hit}")
    print(f"  fallback used: {fb}   orphan: {orph}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
