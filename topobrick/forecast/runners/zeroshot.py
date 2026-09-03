"""Run target-history zero-shot forecasting baselines."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch

from topobrick.forecast.backends import BACKENDS, MoiraiBackend
from topobrick.forecast.data.dataset import ForecastDataset
from topobrick.forecast.evaluation.metrics import compute_forecast_metrics
from topobrick.forecast.evaluation.windows import (
    match_window_indices,
    select_window_keys,
)
from topobrick.forecast.inputs import (
    DATASET_FREQUENCY_MINUTES,
    PAST_KNOWN_EXOGENOUS,
    TARGET_HISTORY,
    TOPOLOGY_AWARE_EXOGENOUS,
    AvailabilityAwareInputBuilder,
    InputBuilderConfig,
)
from topobrick.preprocessing.targets.variability import compute_per_sensor_train_std
from topobrick.utils.arrays import as_float_array
from topobrick.utils.config import load_config
from topobrick.utils.paths import resolve_path
from topobrick.utils.progress import log


CONTEXT_ALIASES = {
    "target": TARGET_HISTORY,
    "past": PAST_KNOWN_EXOGENOUS,
    "past+future": TOPOLOGY_AWARE_EXOGENOUS,
    TARGET_HISTORY: TARGET_HISTORY,
    PAST_KNOWN_EXOGENOUS: PAST_KNOWN_EXOGENOUS,
    TOPOLOGY_AWARE_EXOGENOUS: TOPOLOGY_AWARE_EXOGENOUS,
}


def collect_inputs(
    dataset,
    indices,
    horizon: int,
    context: str,
    input_builder: AvailabilityAwareInputBuilder,
    brick_class_by_node,
    frequency_minutes: int,
):
    model_inputs = []
    targets = []
    trusted_masks = []
    active_counts = []
    target_classes = []
    sensor_ids = []
    starts = []
    counts = Counter()
    for index in indices:
        prepared = input_builder.build(
            dataset,
            index,
            horizon,
            brick_class_by_node,
        )
        counts.update(prepared.counts)
        selected = input_builder.select_context(prepared, context)
        if torch.is_tensor(selected):
            selected = {"target": selected}
        row = dataset.windows.iloc[index]
        active_count = sum(
            float(np.std(as_float_array(value))) > 1e-3
            for value in selected.get("past_covariates", {}).values()
        )
        model_inputs.append(selected)
        targets.append(prepared.target_future)
        trusted_masks.append(prepared.trusted_mask)
        active_counts.append(active_count)
        target_classes.append(str(brick_class_by_node.get(row["target_node_id"], "?")))
        sensor_ids.append(str(row["sensor_id"]))
        starts.append(
            pd.Period(pd.Timestamp(row["input_start"]), freq=f"{frequency_minutes}min")
        )
    return {
        "model_inputs": model_inputs,
        "target": targets,
        "trusted_mask": trusted_masks,
        "n_active_exogenous": active_counts,
        "target_class": target_classes,
        "sensor_id": sensor_ids,
        "starts": starts,
        "counts": counts,
    }


def run(
    backend,
    dataset,
    indices,
    *,
    horizon: int,
    batch_size: int,
    context: str,
    input_builder: AvailabilityAwareInputBuilder,
    brick_class_by_node,
    frequency_minutes: int,
):
    collected = collect_inputs(
        dataset,
        indices,
        horizon,
        context,
        input_builder,
        brick_class_by_node,
        frequency_minutes,
    )
    if not collected["model_inputs"]:
        raise ValueError("no evaluation windows matched the requested dataset subset")
    started_at = time.time()
    if isinstance(backend, MoiraiBackend):
        predictions = backend.predict_all(
            collected["model_inputs"],
            collected["starts"],
            horizon,
        )
    else:
        chunks = []
        model_inputs = collected["model_inputs"]
        for start in range(0, len(model_inputs), batch_size):
            chunks.append(
                backend.predict(model_inputs[start : start + batch_size], horizon)
            )
            completed = min(start + batch_size, len(model_inputs))
            if completed % (10 * batch_size) == 0 or completed == len(model_inputs):
                log(
                    f"    {completed}/{len(model_inputs)} windows "
                    f"({time.time() - started_at:.1f}s)"
                )
        predictions = np.concatenate(chunks, axis=0)

    return {
        "target": np.stack(collected["target"]),
        "prediction": predictions.astype(np.float32),
        "trusted_mask": np.stack(collected["trusted_mask"]),
        "n_active_exogenous": np.asarray(collected["n_active_exogenous"]),
        "target_class": np.asarray(collected["target_class"]),
        "sensor_id": np.asarray(collected["sensor_id"]),
    }, collected["counts"]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizons", type=int, nargs="+", required=True)
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--windows-pattern", "--windows_pattern", dest="windows_pattern", required=True
    )
    parser.add_argument(
        "--num-windows", "--num_windows", dest="num_windows", type=int, default=999999
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--batch-size", "--batch_size", dest="batch_size", type=int, default=32
    )
    parser.add_argument("--model")
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out-dir", "--out_dir", dest="out_dir", required=True)
    parser.add_argument("--out-suffix", "--out_suffix", dest="out_suffix", default="")
    parser.add_argument(
        "--context",
        nargs="+",
        default=[TARGET_HISTORY],
        choices=CONTEXT_ALIASES,
    )
    parser.add_argument(
        "--exogenous-selection",
        "--exogenous_selection",
        "--cov-selection",
        "--cov_selection",
        dest="exogenous_selection",
        default="all",
        choices=["all", "none"],
    )
    parser.add_argument(
        "--calendar",
        "--add-time-cov",
        "--add_time_cov",
        dest="add_calendar",
        action="store_true",
    )
    parser.add_argument(
        "--past-correlation-threshold",
        "--past_correlation_threshold",
        "--cov-prune-corr",
        "--cov_prune_corr",
        dest="past_correlation_threshold",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--meteorological-future",
        "--meteorological_future",
        dest="meteorological_future",
        choices=["oracle", "persistence", "none"],
        default="none",
    )
    parser.add_argument(
        "--drop-external-future",
        "--drop_external_future",
        dest="drop_external_future",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--external-future-noise-std",
        "--external_future_noise_std",
        dest="external_future_noise_std",
        type=float,
        default=0.0,
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config = load_config(args.config)
    processed_dir = resolve_path(config["paths"]["processed_root"], args.dataset)
    frequency_minutes = DATASET_FREQUENCY_MINUTES.get(args.dataset, 15)
    meteorological_future = (
        "none" if args.drop_external_future else args.meteorological_future
    )
    input_builder = AvailabilityAwareInputBuilder(
        InputBuilderConfig(
            frequency_minutes=frequency_minutes,
            add_calendar=args.add_calendar,
            past_correlation_threshold=args.past_correlation_threshold,
            meteorological_future=meteorological_future,
            meteorological_noise_std=args.external_future_noise_std,
            seed=args.seed,
        )
    )

    nodes = pd.read_parquet(
        os.path.join(processed_dir, "kg_nodes.parquet"),
        columns=["node_id", "brick_class"],
    )
    brick_class_by_node = dict(zip(nodes["node_id"], nodes["brick_class"]))

    by_horizon = {}
    for horizon in args.horizons:
        log(f"\n========== {args.backend} horizon {horizon} ==========")
        input_builder.clear_forecast_cache()
        config["L"] = args.lookback
        windows_filename = args.windows_pattern.format(H=horizon, split=args.split)
        train_std_by_sensor = compute_per_sensor_train_std(
            processed_dir,
            horizon,
            windows_filename=args.windows_pattern.format(H=horizon, split="train"),
        )

        target_node_ids, input_starts = select_window_keys(
            processed_dir,
            windows_filename,
            args.num_windows,
            args.seed,
        )

        dataset = ForecastDataset(
            config,
            dataset_name=args.dataset,
            split=args.split,
            horizon=horizon,
            windows_filename=windows_filename,
            exogenous_selection=args.exogenous_selection,
            target_node_id_subset=list(set(target_node_ids)),
        )
        indices = match_window_indices(dataset, target_node_ids, input_starts)
        log(f"  matched {len(indices)}/{len(target_node_ids)} windows")

        backend_class = BACKENDS[args.backend]
        backend = backend_class(
            args.model or backend_class.default_model,
            args.device,
            args.lookback,
            horizon,
            args.batch_size,
        )
        overall = {}
        for context_argument in args.context:
            context = CONTEXT_ALIASES[context_argument]
            log(f"  --- context={context} ---")
            results, counts = run(
                backend,
                dataset,
                indices,
                horizon=horizon,
                batch_size=args.batch_size,
                context=context,
                input_builder=input_builder,
                brick_class_by_node=brick_class_by_node,
                frequency_minutes=frequency_minutes,
            )
            metrics = compute_forecast_metrics(
                results,
                train_std_by_sensor=train_std_by_sensor,
            )
            log(
                f"    nMAE={metrics['nMAE_overall']:.4f} "
                f"nMSE={metrics['nMSE_overall']:.4f} "
                f"rawMAE={metrics['MAE_overall']:.4f} n={metrics['n']}"
            )
            overall[context] = metrics
            overall[f"exogenous_counts_{context}"] = dict(counts)
        by_horizon[str(horizon)] = {
            "n_windows": len(indices),
            "overall": overall,
        }

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    output_path = os.path.join(
        args.out_dir,
        f"{args.dataset}_{args.backend}{suffix}.json",
    )
    with open(output_path, "w") as handle:
        json.dump(
            {
                "backend": args.backend,
                "model": args.model or BACKENDS[args.backend].default_model,
                "dataset": args.dataset,
                "lookback": args.lookback,
                "split": args.split,
                "context": [CONTEXT_ALIASES[value] for value in args.context],
                "exogenous_selection": args.exogenous_selection,
                "calendar": args.add_calendar,
                "meteorological_future": meteorological_future,
                "by_horizon": by_horizon,
            },
            handle,
            indent=2,
        )
    log(f"wrote {output_path}")


if __name__ == "__main__":
    main()
