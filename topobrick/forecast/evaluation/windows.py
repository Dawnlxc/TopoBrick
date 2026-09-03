"""Select and match evaluation windows."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from topobrick.utils.progress import log


def select_window_keys(
    processed_dir: str,
    windows_filename: str,
    num_windows: int,
    seed: int,
    fixed_path: str | None = None,
):
    if fixed_path and os.path.exists(fixed_path):
        with open(fixed_path) as handle:
            payload = json.load(handle)
        target_node_ids = list(payload["target_node_ids"])
        input_starts = [pd.Timestamp(value) for value in payload["input_starts"]]
        log(f"  loaded {len(target_node_ids)} evaluation windows from {fixed_path}")
        return target_node_ids, input_starts

    windows_path = os.path.join(processed_dir, windows_filename)
    windows = pd.read_parquet(windows_path)
    sample_size = min(num_windows, len(windows))
    random = np.random.default_rng(seed)
    indices = random.choice(len(windows), size=sample_size, replace=False).tolist()
    sample = windows.iloc[indices]
    target_node_ids = sample["target_node_id"].tolist()
    input_starts = pd.to_datetime(sample["input_start"]).tolist()
    log(
        f"  selected {len(target_node_ids)} evaluation windows "
        f"from a pool of {len(windows)} (seed={seed})"
    )

    if fixed_path:
        parent_dir = os.path.dirname(fixed_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(fixed_path, "w") as handle:
            json.dump(
                {
                    "target_node_ids": target_node_ids,
                    "input_starts": [str(value) for value in input_starts],
                },
                handle,
            )
        log(f"  saved evaluation windows to {fixed_path}")
    return target_node_ids, input_starts


def match_window_indices(dataset, target_node_ids, input_starts) -> list[int]:
    timestamps = (
        pd.to_datetime(dataset.windows["input_start"])
        .dt.as_unit("ns")
        .astype("int64")
        .values
    )
    lookup = {
        (target_node_id, int(timestamp)): index
        for index, (target_node_id, timestamp) in enumerate(
            zip(dataset.windows["target_node_id"].values, timestamps)
        )
    }
    return [
        lookup[(target_node_id, int(pd.Timestamp(input_start).value))]
        for target_node_id, input_start in zip(target_node_ids, input_starts)
        if (target_node_id, int(pd.Timestamp(input_start).value)) in lookup
    ]
