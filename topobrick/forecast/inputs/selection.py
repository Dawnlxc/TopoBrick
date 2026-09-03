"""Select exogenous series from a materialized topology subgraph."""

from __future__ import annotations

import torch


VALID_SELECTION_MODES = {"all", "none"}


def select_exogenous_mask(
    subgraph,
    mode: str = "all",
    max_exogenous: int | None = None,
) -> torch.Tensor:
    if mode not in VALID_SELECTION_MODES:
        raise ValueError(
            f"selection mode must be one of {sorted(VALID_SELECTION_MODES)}"
        )

    has_timeseries = subgraph["has_ts_mask"].bool().clone()
    target_index = int(subgraph["target_local_index"])
    if mode == "none":
        selected = torch.zeros_like(has_timeseries)
        selected[target_index] = True
        return selected

    has_timeseries[target_index] = True
    if max_exogenous is None:
        return has_timeseries

    selected = torch.zeros_like(has_timeseries)
    selected[target_index] = True
    candidates = [
        index
        for index in range(has_timeseries.numel())
        if index != target_index and bool(has_timeseries[index])
    ]
    for index in candidates[:max_exogenous]:
        selected[index] = True
    return selected
