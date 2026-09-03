from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI

from topobrick.preprocessing.targets.selection import is_forecast_eligible
from topobrick.sampler.graph.points import PointIndex
from topobrick.sampler.graph.skeleton import BuildingSkeleton
from topobrick.sampler.pipeline import sample_target

DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_API_KEY = os.environ.get(
    "OPENAI_API_KEY", "REPLACE_WITH_YOUR_API_KEY"
)


def eligible_targets(dataset: str, processed_root: str) -> list[str]:
    nodes = pd.read_parquet(
        os.path.join(processed_root, dataset, "kg_nodes.parquet")
    )
    eligible = nodes["is_forecast_target"].fillna(False)
    eligible &= nodes["brick_class"].apply(is_forecast_eligible)
    if "is_usable" in nodes.columns:
        eligible &= nodes["is_usable"].fillna(False)
    timeseries_column = (
        "has_timeseries" if "has_timeseries" in nodes.columns else "has_ts"
    )
    eligible &= nodes[timeseries_column].fillna(False)
    return nodes.loc[eligible, "node_id"].tolist()


def _print_summary(
    dataset: str,
    results: dict[str, dict],
    max_points: int,
    elapsed: float,
    output_path: str,
) -> None:
    sizes = sorted(record["n_leaves"] for record in results.values())
    cap_hits = sum(size >= max_points for size in sizes)
    fallback_count = sum(record["fallback_used"] for record in results.values())
    orphan_count = sum(record["orphan"] for record in results.values())
    print(f"\n=== {dataset} sampler ({len(results)} targets, {elapsed:.0f}s) ===")
    print(
        f"  points/target: min={sizes[0]} median={sizes[len(sizes) // 2]} "
        f"mean={sum(sizes) / len(sizes):.1f} max={sizes[-1]}"
    )
    print(f"  reached {max_points}-point cap: {cap_hits}")
    print(f"  fallback: {fallback_count}  orphan: {orphan_count}")
    print(f"  wrote {output_path}")


def run_sampler(
    dataset: str,
    *,
    processed_root: str,
    model: str = "openai/gpt-oss-20b",
    base_url: str = DEFAULT_BASE_URL,
    workers: int = 8,
    max_points: int = 60,
    use_verifier: bool = True,
    output_path: str | None = None,
    api_key: str | None = None,
) -> str:
    skeleton = BuildingSkeleton.load(dataset, processed_root)
    points = PointIndex.load(dataset, skeleton, processed_root)
    targets = eligible_targets(dataset, processed_root)
    if not targets:
        raise ValueError(f"no forecast targets found for {dataset}")
    print(f"[{dataset}] {len(targets)} forecast targets | workers={workers}")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key or DEFAULT_API_KEY,
    )
    started_at = time.time()

    def sample(target: str) -> tuple[str, dict]:
        return target, sample_target(
            target,
            skeleton,
            points,
            client,
            model,
            max_points=max_points,
            use_verifier=use_verifier,
        )

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(sample, target) for target in targets]
        for completed, future in enumerate(as_completed(futures), start=1):
            target, record = future.result()
            results[target] = record
            if completed % 20 == 0:
                print(
                    f"  {completed}/{len(targets)} "
                    f"({time.time() - started_at:.0f}s)"
                )

    output_path = output_path or f"outputs/subgraphs/{dataset}.json"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(
            {
                "dataset": dataset,
                "sampler": "pull_v1",
                "n_targets": len(results),
                "subgraphs": results,
            },
            handle,
        )
    _print_summary(
        dataset,
        results,
        max_points,
        time.time() - started_at,
        output_path,
    )
    return output_path
