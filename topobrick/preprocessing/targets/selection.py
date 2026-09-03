"""L3: semantic and empirical forecast-target filtering."""

from __future__ import annotations

import os

import pandas as pd

from topobrick.utils.progress import log

FORECAST_EXCLUDE_SUBSTRINGS = (
    "Parameter",
    "Gain",
    "Limit",
    "Setpoint",
    "Status",
    "Command",
    "Mode",
    "Alarm",
    "Enable",
    "Stages",
    "Position",
    "Damper",
    "Valve",
    "Fan_Speed",
    "Motor_Speed",
    "Angle",
    "Voltage",
    "Current",
    "Frequency",
    "Reactive_Power",
    "Run_Time",
    "On_Timer",
    "Duration",
    "Timer",
    "Peak_Power_Demand",
    "Energy",
    "Sensor_Ext",
)
FORECAST_EXCLUDE_EXACT = {"Sensor", "Speed_Sensor"}


def is_forecast_eligible(brick_class: str | None) -> bool:
    """Return whether a Brick class is suitable for continuous forecasting."""
    if brick_class is None or pd.isna(brick_class) or brick_class == "":
        return False
    if brick_class in FORECAST_EXCLUDE_EXACT:
        return False
    return not any(part in brick_class for part in FORECAST_EXCLUDE_SUBSTRINGS)


def select_forecast_targets(processed_directory: str) -> pd.DataFrame:
    """Apply the paper's L3 target predicate to processed sensor metadata."""
    nodes = pd.read_parquet(os.path.join(processed_directory, "kg_nodes.parquet"))
    required = {"node_id", "brick_class", "is_forecast_target", "is_usable"}
    missing = required.difference(nodes.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"L3 requires columns [{missing_columns}]; run L2 first")

    eligible = nodes["brick_class"].apply(is_forecast_eligible)
    selected = (
        nodes["is_forecast_target"].fillna(False).astype(bool)
        & nodes["is_usable"].fillna(False).astype(bool)
        & eligible
    )
    target_nodes = nodes.loc[selected, ["node_id", "brick_class"]].rename(
        columns={"node_id": "target_node_id"}
    )

    point_map = pd.read_parquet(
        os.path.join(processed_directory, "point_map.parquet"),
        columns=["sensor_id", "kg_node_id"],
    )
    point_map = point_map.drop_duplicates("kg_node_id", keep="last")
    targets = target_nodes.merge(
        point_map,
        left_on="target_node_id",
        right_on="kg_node_id",
        how="inner",
    ).drop(columns="kg_node_id")
    return targets.reset_index(drop=True)


def materialize_forecast_targets(processed_directory: str) -> pd.DataFrame:
    targets = select_forecast_targets(processed_directory)
    output_path = os.path.join(processed_directory, "forecast_targets.parquet")
    targets.to_parquet(output_path, index=False)
    log(f"L3 selected {len(targets)} forecast targets -> {output_path}")
    return targets


def load_forecast_targets(processed_directory: str) -> pd.DataFrame:
    path = os.path.join(processed_directory, "forecast_targets.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} is missing; run preprocessing through L3")
    return pd.read_parquet(path)
