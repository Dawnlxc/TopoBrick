"""Cache materialized topology subgraphs for forecasting."""

from __future__ import annotations

import gzip
import json
import os

import pandas as pd
import torch


def build_subgraph_entry(
    target_node_id: str,
    node_ids,
    has_timeseries_by_node,
    is_fallback: bool,
    max_nodes: int,
):
    node_ids = [node_id for node_id in node_ids if node_id != target_node_id]
    node_ids = [target_node_id, *node_ids][:max_nodes]
    return {
        "node_ids": node_ids,
        "target_local_index": torch.tensor(0, dtype=torch.long),
        "node_mask": torch.ones(len(node_ids), dtype=torch.bool),
        "has_ts_mask": torch.tensor(
            [bool(has_timeseries_by_node.get(node_id, False)) for node_id in node_ids],
            dtype=torch.bool,
        ),
        "is_fallback": is_fallback,
    }


def build_forecast_cache(
    subgraphs_path: str,
    dataset: str,
    processed_root: str,
    output_path: str,
    *,
    max_nodes: int = 72,
) -> str:
    opener = gzip.open if subgraphs_path.endswith(".gz") else open
    with opener(subgraphs_path, "rt") as handle:
        materialized = json.load(handle)

    processed_dir = os.path.join(processed_root, dataset)
    point_map = pd.read_parquet(
        os.path.join(processed_dir, "point_map.parquet"),
        columns=["kg_node_id"],
    )
    has_timeseries_by_node = {
        str(node_id): True for node_id in point_map["kg_node_id"].dropna().tolist()
    }
    nodes = pd.read_parquet(os.path.join(processed_dir, "kg_nodes.parquet"))
    if "is_usable" in nodes.columns:
        usable_node_ids = set(
            nodes.loc[nodes["is_usable"].fillna(False).astype(bool), "node_id"].astype(
                str
            )
        )
        has_timeseries_by_node = {
            node_id: node_id in usable_node_ids for node_id in has_timeseries_by_node
        }

    subgraphs = {}
    for target_node_id, entry in materialized["subgraphs"].items():
        node_ids = list(entry.get("nodes", []))
        is_fallback = (
            len(node_ids) <= 1
            or int(entry.get("n_leaves", 0)) == 0
            or bool(entry.get("fallback_used", False))
        )
        subgraphs[target_node_id] = build_subgraph_entry(
            target_node_id,
            node_ids,
            has_timeseries_by_node,
            is_fallback,
            max_nodes,
        )

    cache = {
        "version": "topobrick_forecast_v1",
        "max_nodes": max_nodes,
        "subgraphs": subgraphs,
        "source": {
            "json": subgraphs_path,
            "sampler": materialized.get("sampler", "pull"),
        },
    }
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    torch.save(cache, output_path)

    n_fallback = sum(entry["is_fallback"] for entry in subgraphs.values())
    print(
        f"wrote {output_path}: {len(subgraphs)} subgraphs, "
        f"{n_fallback} fallback, max_nodes={max_nodes}"
    )
    return output_path
