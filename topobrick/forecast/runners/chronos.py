"""Run topology-aware zero-shot forecasting with Chronos-2."""

from __future__ import annotations

import json
import os
import random
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline

from topobrick.forecast.data.dataset import ForecastDataset
from topobrick.forecast.inputs import (
    DATASET_FREQUENCY_MINUTES,
    PAST_KNOWN_EXOGENOUS,
    TARGET_HISTORY,
    TOPOLOGY_AWARE_EXOGENOUS,
    AvailabilityAwareInputBuilder,
    InputBuilderConfig,
)
from topobrick.forecast.evaluation.metrics import (
    aggregate_per_sensor,
    compute_forecast_metrics,
)
from topobrick.forecast.evaluation.windows import (
    match_window_indices,
    select_window_keys,
)
from topobrick.preprocessing.targets.variability import compute_per_sensor_train_std
from topobrick.utils.arrays import median_sample_forecast
from topobrick.utils.config import load_config
from topobrick.utils.paths import resolve_path
from topobrick.utils.progress import log


MODE_TO_CONTEXT = {
    "A": TARGET_HISTORY,
    "B": PAST_KNOWN_EXOGENOUS,
    "C": TOPOLOGY_AWARE_EXOGENOUS,
}

MODE_LABEL = {
    "A": "target history",
    "B": "target history + past-known exogenous",
    "C": "topology-aware past/future-known exogenous",
}


def evaluate(
    pipeline,
    dataset,
    indices,
    *,
    horizon: int,
    batch_size: int,
    context: str,
    input_builder: AvailabilityAwareInputBuilder,
    brick_class_by_node,
    cross_learning: bool = False,
):
    targets = []
    predictions = []
    trusted_masks = []
    active_exogenous_counts = []
    target_classes = []
    sensor_ids = []
    input_counts = Counter()
    started_at = time.time()

    for chunk_start in range(0, len(indices), batch_size):
        chunk = indices[chunk_start : chunk_start + batch_size]
        model_inputs = []
        chunk_targets = []
        chunk_masks = []
        chunk_active_counts = []
        chunk_classes = []
        for index in chunk:
            prepared = input_builder.build(
                dataset,
                index,
                horizon,
                brick_class_by_node,
            )
            input_counts.update(prepared.counts)
            model_inputs.append(input_builder.select_context(prepared, context))
            chunk_targets.append(prepared.target_future)
            chunk_masks.append(prepared.trusted_mask)
            chunk_active_counts.append(
                0 if context == TARGET_HISTORY else prepared.counts["n_past"]
            )
            target_node_id = dataset.windows.iloc[index]["target_node_id"]
            chunk_classes.append(str(brick_class_by_node.get(target_node_id, "?")))

        with torch.no_grad():
            chunk_predictions = pipeline.predict(
                model_inputs,
                prediction_length=horizon,
                batch_size=len(model_inputs),
                cross_learning=cross_learning,
            )
        for offset, prediction in enumerate(chunk_predictions):
            predictions.append(median_sample_forecast(prediction, horizon))
            targets.append(chunk_targets[offset])
            trusted_masks.append(chunk_masks[offset])
            active_exogenous_counts.append(chunk_active_counts[offset])
            target_classes.append(chunk_classes[offset])
            sensor_ids.append(str(dataset.windows.iloc[chunk[offset]]["sensor_id"]))

        completed = min(chunk_start + len(chunk), len(indices))
        if completed % (10 * batch_size) == 0 or completed == len(indices):
            log(
                f"    {completed}/{len(indices)} windows ({time.time() - started_at:.1f}s)"
            )

    if not targets:
        raise ValueError("no evaluation windows matched the requested dataset subset")
    return {
        "target": np.stack(targets),
        "prediction": np.stack(predictions),
        "trusted_mask": np.stack(trusted_masks),
        "n_active_exogenous": np.asarray(active_exogenous_counts),
        "target_class": np.asarray(target_classes),
        "sensor_id": np.asarray(sensor_ids),
    }, input_counts


@dataclass
class ChronosRunConfig:
    config: str
    dataset: str
    horizons: list[int] = field(default_factory=lambda: [24, 48, 72, 96])
    split: str = "test"
    num_windows: int = 999999
    batch_size: int = 32
    model: str = "amazon/chronos-2"
    device: str | None = None
    out_dir: str = "outputs/chronos"
    out_suffix: str = ""
    lookback: int | None = None
    windows_pattern: str = "windows_L96_H{H}_{split}_clean.parquet"
    cross_learning: bool = False
    modes: list[str] = field(default_factory=lambda: ["A", "C"])
    exogenous_selection: str = "all"
    max_exogenous: int | None = None
    fixed_eval_windows: str | None = None
    seed: int = 0
    add_calendar: bool = True
    past_correlation_threshold: float = 0.45
    meteorological_future: str = "forecast"
    meteorological_forecaster_calendar: bool = True
    external_future_noise_std: float = 0.0
    external_future_noise_std_norm: float = 0.0


def _resolve_meteorological_future(args) -> str:
    return args.meteorological_future


def _log_overall_metrics(metrics_by_mode) -> None:
    log("  --- overall metrics (raw MAE / nMAE) ---")
    for mode in ("A", "B", "C"):
        if mode not in metrics_by_mode:
            continue
        values = metrics_by_mode[mode]
        deltas = ""
        if mode != "A" and "A" in metrics_by_mode:
            target_only = metrics_by_mode["A"]
            deltas = (
                f"  delta vs A: raw {values['MAE_overall'] - target_only['MAE_overall']:+.4f}, "
                f"normalized {values['nMAE_overall'] - target_only['nMAE_overall']:+.4f}"
            )
        log(
            f"    {mode} {MODE_LABEL[mode]}: rawMAE={values['MAE_overall']:.4f} "
            f"nMAE={values['nMAE_overall']:.4f}{deltas}"
        )


def run_forecast(args: ChronosRunConfig) -> str:
    if any(mode not in MODE_TO_CONTEXT for mode in args.modes):
        raise ValueError(f"modes must be drawn from {tuple(MODE_TO_CONTEXT)}")
    if args.exogenous_selection not in {"all", "none"}:
        raise ValueError("exogenous_selection must be 'all' or 'none'")
    if args.meteorological_future not in {"forecast", "persistence", "none", "oracle"}:
        raise ValueError("unsupported meteorological_future setting")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    processed_dir = resolve_path(config["paths"]["processed_root"], args.dataset)
    nodes = pd.read_parquet(os.path.join(processed_dir, "kg_nodes.parquet"))
    brick_class_by_node = dict(zip(nodes["node_id"], nodes["brick_class"]))

    log(f"loading Chronos {args.model!r} on {args.device}")
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    pipeline = BaseChronosPipeline.from_pretrained(
        args.model,
        device_map=args.device,
        torch_dtype=dtype,
    )

    meteorological_future = _resolve_meteorological_future(args)
    if meteorological_future == "oracle":
        log("  meteorological future: oracle values (diagnostic only)")
    else:
        log(f"  meteorological future: {meteorological_future}")
    if args.add_calendar:
        log("  future-known calendar: hour-of-day sine/cosine")

    builder_config = InputBuilderConfig(
        frequency_minutes=DATASET_FREQUENCY_MINUTES.get(args.dataset, 15),
        add_calendar=args.add_calendar,
        past_correlation_threshold=args.past_correlation_threshold,
        meteorological_future=meteorological_future,
        meteorological_noise_std=args.external_future_noise_std,
        meteorological_noise_std_normalized=args.external_future_noise_std_norm,
        meteorological_forecaster_uses_calendar=args.meteorological_forecaster_calendar,
        seed=args.seed,
    )
    input_builder = AvailabilityAwareInputBuilder(
        builder_config,
        meteorological_forecaster=pipeline
        if meteorological_future == "forecast"
        else None,
    )

    output = {
        "horizons": args.horizons,
        "contexts": MODE_TO_CONTEXT,
        "meteorological_future": meteorological_future,
        "by_horizon": {},
    }
    per_sensor_rows = []
    for horizon in args.horizons:
        log(f"\n========== horizon {horizon} ==========")
        input_builder.clear_forecast_cache()
        if args.lookback is not None:
            config["L"] = int(args.lookback)

        windows_filename = (
            args.windows_pattern.format(H=horizon, split=args.split)
            if args.windows_pattern
            else None
        )
        train_windows_filename = (
            args.windows_pattern.format(H=horizon, split="train")
            if args.windows_pattern
            else None
        )
        train_std_by_sensor = compute_per_sensor_train_std(
            processed_dir,
            horizon,
            windows_filename=train_windows_filename,
        )
        fixed_path = (
            args.fixed_eval_windows.format(DS=args.dataset, H=horizon, SEED=args.seed)
            if args.fixed_eval_windows
            else None
        )
        target_node_ids, input_starts = select_window_keys(
            processed_dir,
            windows_filename or f"windows_H{horizon}_{args.split}.parquet",
            args.num_windows,
            args.seed,
            fixed_path,
        )
        target_subset = list(set(target_node_ids))
        log(f"  loading {len(target_subset)} target subgraphs")
        dataset = ForecastDataset(
            config,
            dataset_name=args.dataset,
            split=args.split,
            horizon=horizon,
            windows_filename=windows_filename,
            exogenous_selection=args.exogenous_selection,
            max_exogenous=args.max_exogenous,
            target_node_id_subset=target_subset,
        )
        indices = match_window_indices(dataset, target_node_ids, input_starts)
        log(f"  matched {len(indices)}/{len(target_node_ids)} evaluation windows")

        results_by_mode = {}
        metrics_by_mode = {}
        cross_learning_label = " [cross-learning]" if args.cross_learning else ""
        for mode in args.modes:
            log(f"  Mode {mode}: {MODE_LABEL[mode]}{cross_learning_label}")
            result, counts = evaluate(
                pipeline,
                dataset,
                indices,
                horizon=horizon,
                batch_size=args.batch_size,
                context=MODE_TO_CONTEXT[mode],
                input_builder=input_builder,
                brick_class_by_node=brick_class_by_node,
                cross_learning=args.cross_learning,
            )
            results_by_mode[mode] = (result, counts)
            metrics_by_mode[mode] = compute_forecast_metrics(
                result,
                train_std_by_sensor=train_std_by_sensor,
            )

        _log_overall_metrics(metrics_by_mode)
        if "C" in results_by_mode:
            counts = results_by_mode["C"][1]
            log("  --- topology-aware exogenous counts ---")
            log(f"    operational schedules: {counts['n_operational_schedule']}")
            log(f"    meteorological:        {counts['n_meteorological']}")
            log(f"    past-known:            {counts['n_past_known']}")

        anchor = next(iter(results_by_mode.values()))[0]
        class_counts = Counter(anchor["target_class"].tolist())
        per_class = {}
        for brick_class, count in class_counts.most_common():
            selection = anchor["target_class"] == brick_class
            entry = {"n": int(count)}
            for mode, (result, _) in results_by_mode.items():
                entry[mode] = compute_forecast_metrics(
                    result,
                    selection,
                    train_std_by_sensor,
                )
            per_class[brick_class] = entry

        log("  --- target classes (top 5 by count) ---")
        for brick_class, count in class_counts.most_common(5):
            fields = "".join(
                f"  {mode}_raw={per_class[brick_class][mode]['MAE_overall']:.3f} "
                f"{mode}_n={per_class[brick_class][mode]['nMAE_overall']:.3f}"
                for mode in ("A", "B", "C")
                if mode in per_class[brick_class]
            )
            log(f"    {brick_class:40s} {count:>4d}{fields}")

        output["by_horizon"][str(horizon)] = {
            "n_windows": len(indices),
            "overall": metrics_by_mode,
            "exogenous_counts": (
                dict(results_by_mode["C"][1]) if "C" in results_by_mode else None
            ),
            "per_class": per_class,
        }
        for mode, (result, _) in results_by_mode.items():
            per_sensor = aggregate_per_sensor(result, train_std_by_sensor)
            per_sensor.insert(0, "context", MODE_TO_CONTEXT[mode])
            per_sensor.insert(0, "mode", mode)
            per_sensor.insert(0, "horizon", horizon)
            per_sensor_rows.append(per_sensor)

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    if args.cross_learning:
        suffix = f"{suffix}_xlearn"
    output_path = os.path.join(
        args.out_dir,
        f"{args.dataset}_{args.split}_future_cov{suffix}.json",
    )
    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=2)
    log(f"\nresults -> {output_path}")

    if per_sensor_rows:
        per_sensor_path = os.path.join(
            args.out_dir,
            f"{args.dataset}_{args.split}_future_cov{suffix}_persensor.parquet",
        )
        pd.concat(per_sensor_rows, ignore_index=True).to_parquet(
            per_sensor_path, index=False
        )
        log(f"per-sensor aggregates -> {per_sensor_path}")
    return output_path
