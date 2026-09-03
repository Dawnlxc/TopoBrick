from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from collections.abc import Iterable

import pandas as pd

from topobrick.sampler.baselines.graph import KnowledgeGraphIndex, points_within_hops
from topobrick.sampler.runner import eligible_targets
from topobrick.utils.paths import resolve_processed_root

DEFAULT_STRATEGIES = ("2hop", "3hop", "random", "same_ontology")

def _point_pools(
    dataset: str, processed_root: str
) -> tuple[set[str], dict[str, str], dict[str, list[str]]]:
    nodes = pd.read_parquet(
        os.path.join(processed_root, dataset, "kg_nodes.parquet")
    )
    timeseries_column = (
        "has_timeseries" if "has_timeseries" in nodes.columns else "has_ts"
    )
    timeseries_points = set(
        nodes.loc[nodes[timeseries_column].fillna(False), "node_id"]
    )
    point_class = dict(zip(nodes["node_id"], nodes["brick_class"]))
    points_by_class: dict[str, list[str]] = defaultdict(list)
    for point in sorted(timeseries_points):
        points_by_class[point_class.get(point, "")].append(point)
    return timeseries_points, point_class, points_by_class


def _record(
    target: str,
    selected_points: list[str],
    point_class: dict[str, str],
    strategy: str,
) -> dict:
    return {
        "target_uri": target,
        "target_class": point_class.get(target, ""),
        "nodes": [target, *selected_points],
        "edges": [],
        "n_leaves": len(selected_points),
        "leaves": selected_points,
        "llm_used": False,
        "sampler": strategy,
    }


def build_sampler_baselines(
    dataset: str,
    strategies: Iterable[str] = DEFAULT_STRATEGIES,
    *,
    processed_root: str | None = None,
    output_dir: str = "outputs/subgraphs_ablation",
    random_seed: int = 0,
    random_size_hops: int = 2,
) -> list[str]:
    processed_root = resolve_processed_root(processed_root)
    targets = eligible_targets(dataset, processed_root)
    if not targets:
        raise ValueError(f"no forecast targets found for {dataset}")
    graph = KnowledgeGraphIndex(os.path.join(processed_root, dataset))
    timeseries_points, point_class, points_by_class = _point_pools(
        dataset, processed_root
    )
    os.makedirs(output_dir, exist_ok=True)
    hop_cache: dict[tuple[str, int], list[str]] = {}

    def fixed_hop_points(target: str, hops: int) -> list[str]:
        key = (target, hops)
        if key not in hop_cache:
            hop_cache[key] = points_within_hops(
                graph, target, hops, timeseries_points
            )
        return hop_cache[key]

    output_paths = []
    for strategy in strategies:
        generator = random.Random(random_seed)
        records = {}
        for target in targets:
            if strategy.endswith("hop") and strategy[:-3].isdigit():
                selected = fixed_hop_points(target, int(strategy[:-3]))
            elif strategy == "same_ontology":
                selected = [
                    point
                    for point in points_by_class.get(point_class.get(target, ""), [])
                    if point != target
                ]
            elif strategy == "random":
                sample_size = len(fixed_hop_points(target, random_size_hops))
                population = sorted(timeseries_points - {target})
                selected = generator.sample(
                    population, min(sample_size, len(population))
                )
            else:
                raise ValueError(f"unknown sampler baseline: {strategy}")
            records[target] = _record(target, selected, point_class, strategy)

        output_path = os.path.join(output_dir, f"{dataset}_{strategy}.json")
        with open(output_path, "w") as handle:
            json.dump(
                {
                    "dataset": dataset,
                    "sampler": strategy,
                    "n_targets": len(records),
                    "subgraphs": records,
                },
                handle,
            )
        sizes = sorted(record["n_leaves"] for record in records.values())
        zero_count = sum(size == 0 for size in sizes)
        print(
            f"  {strategy}: mean={sum(sizes) / len(sizes):.1f} "
            f"median={sizes[len(sizes) // 2]} zero={zero_count} -> {output_path}"
        )
        output_paths.append(output_path)
    return output_paths
