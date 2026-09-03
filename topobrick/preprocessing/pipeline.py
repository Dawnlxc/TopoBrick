"""Run the preprocessing stages for one building, in order."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from topobrick.preprocessing.ingest.align import align_timeseries_to_kg
from topobrick.preprocessing.ingest.brick_kg import load_bts_kg, load_lbnl59_kg
from topobrick.preprocessing.ingest.bts_stream import stream_bts_to_canonical
from topobrick.preprocessing.ingest.canonical import (
    recompute_outage_with_classes,
    resample_and_clean,
)
from topobrick.preprocessing.ingest.tag_points import filter_points_and_nodes
from topobrick.preprocessing.ingest.timeseries import load_lbnl59_timeseries
from topobrick.preprocessing.quality import observations as observation_quality
from topobrick.preprocessing.quality import sensors as sensor_usability
from topobrick.preprocessing.targets import variability as target_variability
from topobrick.preprocessing.targets import windows as window_validation
from topobrick.preprocessing.targets.selection import materialize_forecast_targets
from topobrick.preprocessing.units.infer import infer_units
from topobrick.preprocessing.units.resolve import resolve_units
from topobrick.preprocessing.units.source import (
    extract_and_audit_source_units,
    print_unit_audit,
)
from topobrick.utils.config import load_config
from topobrick.utils.paths import (
    ensure_directory,
    processed_data_directory,
    raw_data_directory,
)
from topobrick.utils.progress import log

STAGES = ("ingest", "units", "l1", "l2", "l3", "l4")
QC_MARKER = "qc_layer1_audit.csv"  # written by L1; its presence means L1 already ran


@dataclass
class PreprocessingOptions:
    start_date: str | None = None
    end_date: str | None = None
    n_train_months: int | None = None
    n_val_months: int | None = None
    n_test_months: int | None = None
    lookback: int | None = None
    horizons: list[int] | None = None
    stride: int | None = None
    min_train_std: float = 0.1


def run_ingest(cfg, dataset: str, proc: str) -> None:
    raw = raw_data_directory(cfg, dataset)
    series_path = os.path.join(proc, "series.parquet")

    if dataset == "BTS":
        # Streaming: the BTS sensor set does not fit in memory at once. Only
        # sensor-level metadata is returned; series.parquet is written directly.
        sensor_meta, _ = stream_bts_to_canonical(cfg, proc)
        log(f"streamed {series_path}; {len(sensor_meta)} sensors kept")
        series_df = sensor_meta.assign(
            timestamp=pd.NaT,
            value=np.float32(0.0),
            unit=None,
            kg_node_id=pd.NA,
            brick_class=pd.NA,
            observed_mask=False,
            imputed_mask=False,
            outage_mask=False,
        )
        kg_nodes, kg_edges = load_bts_kg(raw, cfg)
    elif dataset == "LBNL59":
        series_df, _ = resample_and_clean(load_lbnl59_timeseries(raw, cfg), cfg)
        kg_nodes, kg_edges = load_lbnl59_kg(raw, cfg)
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    point_map, kg_nodes, series_df = align_timeseries_to_kg(
        series_df, kg_nodes, kg_edges, cfg
    )
    kg_nodes = filter_points_and_nodes(kg_nodes, kg_edges, point_map, cfg)

    if dataset != "BTS":
        # Recompute outages after KG alignment supplies control-point classes.
        nid2bc = dict(zip(kg_nodes["node_id"], kg_nodes["brick_class"]))
        node_col = "kg_node_id" if "kg_node_id" in point_map.columns else "node_id"
        sid2bc = {
            str(sid): nid2bc.get(str(nid), "")
            for sid, nid in zip(point_map["sensor_id"], point_map[node_col])
        }
        series_df = recompute_outage_with_classes(series_df, sid2bc, cfg)

    if dataset != "BTS":
        # One write, after alignment has attached kg_node_id. For BTS the file is
        # already on disk and kg_node_id stays null — join through point_map.
        series_df.to_parquet(series_path, index=False)
        log(f"wrote {series_path} ({len(series_df):,} rows)")

    kg_nodes.to_parquet(os.path.join(proc, "kg_nodes.parquet"), index=False)
    kg_edges.to_parquet(os.path.join(proc, "kg_edges.parquet"), index=False)
    point_map.to_parquet(os.path.join(proc, "point_map.parquet"), index=False)
    log(f"wrote kg_nodes / kg_edges / point_map to {proc}")


def run_units(cfg, dataset: str, proc: str) -> None:
    infer_units(proc).to_csv(os.path.join(proc, "inferred_units.csv"), index=False)
    log("wrote inferred_units.csv")

    source_dataset = "BTS" if dataset.startswith("BTS") else dataset
    audit = extract_and_audit_source_units(cfg, source_dataset, proc)
    print_unit_audit(audit, proc)
    resolve_units(proc)


def run_l1(proc: str, force: bool) -> None:
    if os.path.exists(os.path.join(proc, QC_MARKER)) and not force:
        raise SystemExit(
            f"refusing to re-run L1: {QC_MARKER} already exists in {proc}.\n"
            "L1's MAD test is not idempotent — a second pass masks more readings.\n"
            "Rebuild from raw, or pass --force if you know the series is unfiltered."
        )
    root, name = os.path.split(proc.rstrip("/"))
    observation_quality.run_filter(name, processed_root=root, dry_run=False)


def run_l2(proc: str) -> None:
    root, name = os.path.split(proc.rstrip("/"))
    sensor_usability.run_audit(name, processed_root=root, dry_run=False)


def run_l3(proc: str) -> None:
    materialize_forecast_targets(proc)


def run_l4(proc: str, options: PreprocessingOptions, cfg: dict) -> None:
    """Enumerate windows, keep the valid ones, then drop degenerate sensors."""
    if not (options.start_date and options.end_date):
        raise ValueError("L4 requires start_date and end_date")
    root, name = os.path.split(proc.rstrip("/"))
    for horizon in options.horizons or []:
        window_validation.build_validated_windows(
            dataset=name,
            start_date=options.start_date,
            end_date=options.end_date,
            train_months=options.n_train_months,
            validation_months=options.n_val_months,
            test_months=options.n_test_months,
            lookback=options.lookback,
            horizon=horizon,
            stride=options.stride,
            max_horizon_invalid_fraction=cfg["preprocessing"][
                "max_horizon_invalid_frac"
            ],
            processed_root=root,
        )
        target_variability.drop_low_variability_targets(
            name,
            processed_root=root,
            horizons=(horizon,),
            lookback=options.lookback,
            min_train_std=options.min_train_std,
        )


def apply_config_defaults(options: PreprocessingOptions, cfg: dict) -> None:
    split = cfg["split"]
    if options.n_train_months is None:
        options.n_train_months = int(split["train_months"])
    if options.n_val_months is None:
        options.n_val_months = int(split["val_months"])
    if options.n_test_months is None:
        options.n_test_months = int(split["test_months"])
    if options.lookback is None:
        options.lookback = int(cfg["L"])
    if options.horizons is None:
        options.horizons = [int(horizon) for horizon in cfg["horizons"]]
    if options.stride is None:
        options.stride = int(split["stride"])


def run_preprocessing(
    config_path: str,
    *,
    raw_dataset: str | None = None,
    stop_after: str = "l4",
    force: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    horizons: list[int] | None = None,
    lookback: int | None = None,
    min_train_std: float = 0.1,
) -> str:
    if stop_after not in STAGES:
        raise ValueError(f"stop_after must be one of {STAGES}")

    cfg = load_config(config_path)
    dataset = raw_dataset or cfg["dataset"]["name"]
    options = PreprocessingOptions(
        start_date=start_date or cfg["split"].get("start_date"),
        end_date=end_date or cfg["split"].get("end_date"),
        horizons=horizons,
        lookback=lookback,
        min_train_std=min_train_std,
    )
    apply_config_defaults(options, cfg)
    proc = ensure_directory(processed_data_directory(cfg))
    log(f"=== preprocessing {dataset} -> {proc} ===")

    for stage in STAGES:
        if stage == "ingest":
            run_ingest(cfg, dataset, proc)
        elif stage == "units":
            run_units(cfg, dataset, proc)
        elif stage == "l1":
            run_l1(proc, force)
        elif stage == "l2":
            run_l2(proc)
        elif stage == "l3":
            run_l3(proc)
        elif stage == "l4":
            run_l4(proc, options, cfg)
        log(f"--- stage {stage} done")
        if stage == stop_after:
            break

    log(f"=== {dataset} complete ===")
    return proc
