"""Run the complete TopoBrick workflow from one entry point."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os

from topobrick.utils.config import load_config
from topobrick.utils.paths import resolve_path
from topobrick.utils.progress import log


WORKFLOW_STAGES = ("preprocess", "skeleton", "sample", "cache", "forecast")
RELEASED_SUBGRAPH_SHA256 = {
    "LBNL59": "d5f018efb4106858747790b631c0ba979d8903c848978c30d1e3795f3e484c17",
    "BTS_Site_B": "733f3d923eef967eb43d4808820738275fc5323a2b99f1e624ce5216d3ef1940",
    "BTS_Site_C": "ef1b4c9d5e508ddc23b8dc315fb23a61792e557b19908280612d7e4c6b7ee017",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--start-at",
        choices=WORKFLOW_STAGES,
        default="preprocess",
        help="first pipeline stage to run",
    )
    parser.add_argument(
        "--stop-after",
        choices=WORKFLOW_STAGES,
        default="forecast",
        help="last pipeline stage to run",
    )
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--horizons", type=int, nargs="+")
    parser.add_argument("--lookback", type=int)
    parser.add_argument("--force-preprocessing", action="store_true")

    sampler = parser.add_argument_group("agentic sampler")
    sampler.add_argument("--llm-model", default="openai/gpt-oss-20b")
    sampler.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    sampler.add_argument("--workers", type=int, default=8)
    sampler.add_argument("--max-points", type=int, default=60)
    sampler.add_argument("--sampler-baselines", action="store_true")
    sampler.add_argument(
        "--released-subgraphs",
        action="store_true",
        help=(
            "use and verify the released sampler artifact, skipping the "
            "skeleton and sample stages"
        ),
    )
    sampler.add_argument(
        "--verifier", action=argparse.BooleanOptionalAction, default=True
    )

    forecast = parser.add_argument_group("forecast")
    forecast.add_argument("--forecast-model", default="amazon/chronos-2")
    forecast.add_argument("--device")
    forecast.add_argument("--seed", type=int, default=0)
    forecast.add_argument(
        "--modes", nargs="+", choices=("A", "B", "C"), default=["A", "C"]
    )
    forecast.add_argument("--num-windows", type=int, default=999999)
    forecast.add_argument("--batch-size", type=int, default=32)
    forecast.add_argument("--max-nodes", type=int, default=72)
    forecast.add_argument("--past-correlation-threshold", type=float, default=0.45)
    forecast.add_argument(
        "--meteorological-future",
        choices=("forecast", "persistence", "none", "oracle"),
        default="forecast",
    )
    forecast.add_argument(
        "--calendar", action=argparse.BooleanOptionalAction, default=True
    )
    forecast.add_argument(
        "--meteorological-forecaster-calendar",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    forecast.add_argument("--output-root", default="outputs")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def selected_stages(start_at: str, stop_after: str) -> tuple[str, ...]:
    start = WORKFLOW_STAGES.index(start_at)
    stop = WORKFLOW_STAGES.index(stop_after)
    if start > stop:
        raise ValueError("--start-at must not come after --stop-after")
    return WORKFLOW_STAGES[start : stop + 1]


def effective_stages(
    start_at: str,
    stop_after: str,
    released_subgraphs: bool = False,
) -> tuple[str, ...]:
    stages = selected_stages(start_at, stop_after)
    if released_subgraphs:
        stages = tuple(
            stage for stage in stages if stage not in {"skeleton", "sample"}
        )
    return stages


def subgraphs_path(
    output_root: str,
    dataset: str,
    released: bool = False,
) -> str:
    base_path = os.path.join(output_root, "subgraphs", dataset)
    suffix = ".json.gz" if released else ".json"
    return f"{base_path}{suffix}"


def verify_released_subgraphs(path: str, dataset: str) -> None:
    expected = RELEASED_SUBGRAPH_SHA256.get(dataset)
    if expected is None:
        raise ValueError(f"no released sampler checksum is registered for {dataset}")
    digest = hashlib.sha256()
    try:
        with gzip.open(path, "rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"released sampler artifact not found: {path}") from error
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(
            f"released sampler checksum mismatch for {dataset}: "
            f"expected {expected}, found {actual}"
        )


def run_workflow(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    raw_dataset = config["dataset"]["name"]
    dataset = config["dataset"]["processed_subdir"]
    horizons = args.horizons or [int(value) for value in config["horizons"]]
    lookback = args.lookback or int(config["L"])
    start_date = args.start_date or config["split"].get("start_date")
    end_date = args.end_date or config["split"].get("end_date")
    processed_root = resolve_path(config["paths"]["processed_root"])
    output_root = resolve_path(args.output_root)
    stages = effective_stages(
        args.start_at,
        args.stop_after,
        released_subgraphs=args.released_subgraphs,
    )
    sampled_subgraphs = subgraphs_path(
        output_root,
        dataset,
        released=args.released_subgraphs,
    )
    cache_path = os.path.join(processed_root, dataset, "kg_cache_pull.pt")
    forecast_dir = os.path.join(output_root, "forecast", dataset)

    log(f"dataset: {dataset} (raw loader: {raw_dataset})")
    log(f"stages: {' -> '.join(stages)}")
    log(f"date range: {start_date} to {end_date} (exclusive)")
    log(f"lookback: {lookback}; horizons: {horizons}")
    log(f"processed data: {os.path.join(processed_root, dataset)}")
    log(f"subgraphs: {sampled_subgraphs}")
    log(f"forecast output: {forecast_dir}")
    if args.dry_run:
        return

    if args.released_subgraphs:
        verify_released_subgraphs(sampled_subgraphs, dataset)
        log("released sampler checksum: verified")

    if "preprocess" in stages:
        if not start_date or not end_date:
            raise ValueError(
                "preprocessing needs split.start_date and split.end_date in the config "
                "or the --start-date and --end-date options"
            )
        from topobrick.preprocessing.pipeline import run_preprocessing

        run_preprocessing(
            args.config,
            raw_dataset=raw_dataset,
            stop_after="l4",
            force=args.force_preprocessing,
            start_date=start_date,
            end_date=end_date,
            horizons=horizons,
            lookback=lookback,
        )

    if "skeleton" in stages:
        from topobrick.sampler.graph.skeleton import build_skeleton

        build_skeleton(dataset, processed_root=processed_root)

    if "sample" in stages:
        from topobrick.sampler.runner import run_sampler

        run_sampler(
            dataset,
            processed_root=processed_root,
            model=args.llm_model,
            base_url=args.base_url,
            workers=args.workers,
            max_points=args.max_points,
            use_verifier=args.verifier,
            output_path=sampled_subgraphs,
        )
        if args.sampler_baselines:
            from topobrick.sampler.baselines.runner import build_sampler_baselines

            build_sampler_baselines(
                dataset,
                processed_root=processed_root,
                output_dir=os.path.join(output_root, "subgraphs_ablation"),
            )

    if "cache" in stages:
        from topobrick.forecast.data.cache import build_forecast_cache

        build_forecast_cache(
            sampled_subgraphs,
            dataset,
            processed_root,
            cache_path,
            max_nodes=args.max_nodes,
        )

    if "forecast" in stages:
        from topobrick.forecast.runners.chronos import (
            ChronosRunConfig,
            run_forecast,
        )

        run_forecast(
            ChronosRunConfig(
                config=args.config,
                dataset=dataset,
                horizons=horizons,
                num_windows=args.num_windows,
                batch_size=args.batch_size,
                model=args.forecast_model,
                device=args.device,
                out_dir=forecast_dir,
                lookback=lookback,
                windows_pattern=f"windows_L{lookback}_H{{H}}_{{split}}_clean.parquet",
                modes=args.modes,
                seed=args.seed,
                add_calendar=args.calendar,
                past_correlation_threshold=args.past_correlation_threshold,
                meteorological_future=args.meteorological_future,
                meteorological_forecaster_calendar=(
                    args.meteorological_forecaster_calendar
                ),
            )
        )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        stages = effective_stages(
            args.start_at,
            args.stop_after,
            released_subgraphs=args.released_subgraphs,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.released_subgraphs and "cache" not in stages:
        parser.error("--released-subgraphs requires the cache stage")
    if args.released_subgraphs and args.sampler_baselines:
        parser.error("--sampler-baselines cannot be used with --released-subgraphs")
    run_workflow(args)


if __name__ == "__main__":
    main()
