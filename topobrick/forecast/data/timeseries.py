"""Load aligned target and exogenous time-series windows."""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from topobrick.forecast.data.units import (
    canonicalise_unit,
    temperature_to_canonical,
)
from topobrick.utils.progress import log


def _ffill_masked(x: np.ndarray, trust: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    if trust.all():
        return x.astype(np.float32, copy=False)

    out = x.astype(np.float32, copy=True)
    trusted_indices = np.flatnonzero(trust)
    if not trusted_indices.size:
        out[:] = 0.0
        return out

    first_trusted = int(trusted_indices[0])
    out[:first_trusted] = out[first_trusted]
    last_value = out[first_trusted]
    for index in range(first_trusted + 1, len(out)):
        if trust[index]:
            last_value = out[index]
        else:
            out[index] = last_value
    return out


class TimeSeriesSlicer:
    def __init__(
        self,
        processed_dir: str,
        target_sensor_ids: set[str],
        exogenous_node_ids: set[str],
        lookback: int,
        horizon: int,
    ):
        self.lookback = int(lookback)
        self.horizon = int(horizon)

        nodes = pd.read_parquet(os.path.join(processed_dir, "kg_nodes.parquet"))
        if "resolved_unit" in nodes.columns:
            self._unit_by_node = {
                node_id: canonicalise_unit(unit)
                for node_id, unit in zip(nodes["node_id"], nodes["resolved_unit"])
            }
        else:
            self._unit_by_node = {}

        series_path = os.path.join(processed_dir, "series.parquet")
        columns = [
            "sensor_id",
            "kg_node_id",
            "timestamp",
            "value",
            "observed_mask",
        ]
        try:
            schema = pq.ParquetFile(series_path).schema_arrow
            if any(field.name == "outage_mask" for field in schema):
                columns.append("outage_mask")
        except Exception:
            pass

        sensor_id_by_node: Dict[str, str] = {}
        try:
            point_map = pd.read_parquet(
                os.path.join(processed_dir, "point_map.parquet")
            )
            node_column = (
                "kg_node_id" if "kg_node_id" in point_map.columns else "node_id"
            )
            for _, row in point_map.iterrows():
                node_id = row.get(node_column)
                sensor_id = row.get("sensor_id")
                if pd.notna(node_id) and pd.notna(sensor_id):
                    sensor_id_by_node[str(node_id)] = str(sensor_id)
        except Exception as error:
            log(f"  TimeSeriesSlicer: point_map load failed: {error}")
        mapped_exogenous_sensor_ids = {
            sensor_id_by_node[node_id]
            for node_id in exogenous_node_ids
            if node_id in sensor_id_by_node
        }

        try:
            sensor_ids = set(target_sensor_ids) | mapped_exogenous_sensor_ids
            node_ids = set(exogenous_node_ids)
            row_filter = None
            if sensor_ids:
                row_filter = pads.field("sensor_id").isin(list(sensor_ids))
            if node_ids and "kg_node_id" in columns:
                node_filter = pads.field("kg_node_id").isin(list(node_ids))
                row_filter = (
                    node_filter if row_filter is None else row_filter | node_filter
                )
            parquet = pads.dataset(series_path, format="parquet")
            table = parquet.to_table(columns=columns, filter=row_filter)
            frame = table.to_pandas().reset_index(drop=True)
            del table
        except Exception as error:
            log(
                f"  TimeSeriesSlicer: pyarrow filter failed ({error}); "
                "falling back to full read"
            )
            frame = pd.read_parquet(series_path, columns=columns)
            keep_target_sensor = frame["sensor_id"].isin(target_sensor_ids)
            keep_exogenous_sensor = (
                frame["sensor_id"].isin(mapped_exogenous_sensor_ids)
                if mapped_exogenous_sensor_ids
                else False
            )
            keep_exogenous_node = (
                frame["kg_node_id"].isin(exogenous_node_ids)
                if exogenous_node_ids
                else False
            )
            if isinstance(keep_exogenous_sensor, bool) and isinstance(
                keep_exogenous_node, bool
            ):
                frame = frame[keep_target_sensor].reset_index(drop=True)
            else:
                mask = keep_target_sensor
                if not isinstance(keep_exogenous_sensor, bool):
                    mask = mask | keep_exogenous_sensor
                if not isinstance(keep_exogenous_node, bool):
                    mask = mask | keep_exogenous_node
                frame = frame[mask].reset_index(drop=True)
        log(
            f"  TimeSeriesSlicer: filtered series to {len(frame):,} rows "
            f"({len(target_sensor_ids)} targets + "
            f"{len(exogenous_node_ids)} exogenous nodes; "
            f"{len(mapped_exogenous_sensor_ids)} via point map)"
        )
        self.series_by_sensor = {
            sensor_id: group
            for sensor_id, group in frame.groupby("sensor_id", sort=False)
        }

        self.series_by_node: Dict[str, Any] = {}
        if frame["kg_node_id"].notna().any():
            groups = frame.dropna(subset=["kg_node_id"]).groupby(
                "kg_node_id",
                sort=False,
            )
            for node_id, group in groups:
                self.series_by_node[str(node_id)] = group
        point_map_additions = 0
        for node_id in exogenous_node_ids:
            if node_id in self.series_by_node:
                continue
            sensor_id = sensor_id_by_node.get(str(node_id))
            if sensor_id is not None and sensor_id in self.series_by_sensor:
                self.series_by_node[str(node_id)] = self.series_by_sensor[sensor_id]
                point_map_additions += 1
        if point_map_additions:
            log(
                f"  TimeSeriesSlicer: filled {point_map_additions} exogenous nodes "
                f"from point_map ({len(self.series_by_node)} node series)"
            )
        self._aligned_node_series: Dict[str, tuple | None] = {}

        self._unit_by_sensor: Dict[str, str] = {}
        try:
            point_map = pd.read_parquet(
                os.path.join(processed_dir, "point_map.parquet")
            )
            for _, row in point_map.iterrows():
                node_id = row.get("kg_node_id")
                sensor_id = row.get("sensor_id")
                if pd.isna(node_id) or pd.isna(sensor_id):
                    continue
                unit = self._unit_by_node.get(str(node_id))
                if unit is not None:
                    self._unit_by_sensor[str(sensor_id)] = unit
        except Exception as error:
            log(f"  TimeSeriesSlicer: unit lookup skipped: {error}")
        self._temp_to_canonical = temperature_to_canonical

    @staticmethod
    def _pad_left(a: np.ndarray, target_len: int, fill) -> np.ndarray:
        if len(a) >= target_len:
            return a[-target_len:]
        pad = target_len - len(a)
        return np.concatenate([np.full(pad, fill, dtype=a.dtype), a])

    @staticmethod
    def _pad_right(a: np.ndarray, target_len: int, fill) -> np.ndarray:
        if len(a) >= target_len:
            return a[:target_len]
        pad = target_len - len(a)
        return np.concatenate([a, np.full(pad, fill, dtype=a.dtype)])

    def get_target_window(
        self,
        sensor_id: str,
        input_start,
        input_end,
        target_start,
        target_end,
    ):
        frame = self.series_by_sensor.get(sensor_id)
        if frame is None:
            history = np.zeros(self.lookback, dtype=np.float32)
            future = np.zeros(self.horizon, dtype=np.float32)
            history_mask = np.zeros(self.lookback, dtype=bool)
            future_mask = np.zeros(self.horizon, dtype=bool)
            return history, history_mask, history_mask, future, future_mask
        frame = frame.sort_values("timestamp")
        timestamps = pd.DatetimeIndex(frame["timestamp"].values)
        values = frame["value"].values.astype(np.float32)
        unit = self._unit_by_sensor.get(str(sensor_id))
        if unit is not None:
            values = self._temp_to_canonical(values, unit).astype(np.float32)
        observed = frame["observed_mask"].values.astype(bool)
        outage = (
            frame["outage_mask"].values.astype(bool)
            if "outage_mask" in frame.columns
            else np.zeros(len(frame), dtype=bool)
        )
        trustworthy = observed & (~outage)

        input_start_index = np.searchsorted(timestamps, input_start)
        input_end_index = np.searchsorted(timestamps, input_end, side="right")
        history = values[input_start_index:input_end_index]
        history_observed = observed[input_start_index:input_end_index]
        history_trusted = trustworthy[input_start_index:input_end_index]

        history = _ffill_masked(history, history_trusted)

        target_start_index = np.searchsorted(timestamps, target_start)
        target_end_index = np.searchsorted(timestamps, target_end, side="right")
        future = values[target_start_index:target_end_index]
        future_trusted = trustworthy[target_start_index:target_end_index]

        history = self._pad_left(history, self.lookback, 0.0)
        history_observed = self._pad_left(history_observed, self.lookback, False)
        history_trusted = self._pad_left(history_trusted, self.lookback, False)
        future = self._pad_right(future, self.horizon, 0.0)
        future_trusted = self._pad_right(future_trusted, self.horizon, False)
        return (
            history,
            history_observed,
            history_trusted,
            future,
            future_trusted,
        )

    def get_context_window(self, nid: str, input_start, input_end):
        series = self._node_series(nid)
        if series is None:
            return (
                np.zeros(self.lookback, dtype=np.float32),
                np.zeros(self.lookback, dtype=bool),
                False,
            )
        ts, v, obs, outage_col = series
        trust_full = obs & (~outage_col)
        lo = np.searchsorted(ts, input_start)
        hi = np.searchsorted(ts, input_end, side="right")
        x = v[lo:hi]
        x_obs = obs[lo:hi]
        x_trust = trust_full[lo:hi]
        x = _ffill_masked(x, x_trust)
        if len(x) < self.lookback:
            pad = self.lookback - len(x)
            x = np.concatenate([np.zeros(pad, dtype=np.float32), x])
            x_obs = np.concatenate([np.zeros(pad, dtype=bool), x_obs])
        else:
            x = x[-self.lookback :]
            x_obs = x_obs[-self.lookback :]
        nan_mask = np.isnan(x)
        if nan_mask.any():
            x[nan_mask] = 0.0
            x_obs[nan_mask] = False
        ts_valid = bool(x_obs.any())
        return x, x_obs, ts_valid

    def get_future_window(self, nid: str, target_start, target_end) -> np.ndarray:
        series = self._node_series(nid)
        if series is None:
            return np.zeros(self.horizon, dtype=np.float32)
        timestamps, values, observed, _ = series
        start = np.searchsorted(timestamps, target_start)
        end = np.searchsorted(timestamps, target_end, side="right")
        future = values[start:end].copy()
        future_observed = observed[start:end]
        if (~future_observed).any():
            last_value = (
                float(future[future_observed][-1]) if future_observed.any() else 0.0
            )
            future = np.where(future_observed, future, last_value).astype(np.float32)
        future = self._pad_right(
            future,
            self.horizon,
            float(future[-1]) if len(future) else 0.0,
        )
        future[np.isnan(future)] = 0.0
        return future.astype(np.float32)

    def observed_std_for_node(self, nid: str) -> float:
        series = self._node_series(nid)
        if series is None:
            return 1.0
        _, values, observed, _ = series
        observed_values = values[observed]
        observed_values = observed_values[~np.isnan(observed_values)]
        return float(observed_values.std()) + 1e-6 if observed_values.size >= 8 else 1.0

    def _node_series(self, nid: str):
        if nid in self._aligned_node_series:
            return self._aligned_node_series[nid]
        frame = self.series_by_node.get(nid)
        if frame is None or len(frame) == 0:
            self._aligned_node_series[nid] = None
            return None
        frame = frame.sort_values("timestamp")
        timestamps = pd.DatetimeIndex(frame["timestamp"].values)
        values = frame["value"].values.astype(np.float32)
        unit = self._unit_by_node.get(nid)
        if unit is not None:
            values = self._temp_to_canonical(values, unit).astype(np.float32)
        observed = frame["observed_mask"].values.astype(bool)
        outage = (
            frame["outage_mask"].values.astype(bool)
            if "outage_mask" in frame.columns
            else np.zeros(len(frame), dtype=bool)
        )
        self._aligned_node_series[nid] = (timestamps, values, observed, outage)
        return self._aligned_node_series[nid]

    REVIN_STD_FLOOR = 0.01

    @staticmethod
    def _safe_std(ref: np.ndarray, mean: float) -> float:
        std = float(ref.std()) if ref.size else 0.0
        return max(
            std,
            TimeSeriesSlicer.REVIN_STD_FLOOR,
            abs(mean) * TimeSeriesSlicer.REVIN_STD_FLOOR,
        )

    @staticmethod
    def revin_target(
        x: np.ndarray,
        x_trust: np.ndarray,
        x_obs: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        ref = x[x_trust] if x_trust.any() else (x[x_obs] if x_obs.any() else x)
        mean = float(ref.mean()) if ref.size else 0.0
        std = TimeSeriesSlicer._safe_std(ref, mean)
        return (x - mean) / std, (y - mean) / std, mean, std

    @staticmethod
    def revin_context(cx: np.ndarray, cxo: np.ndarray) -> np.ndarray:
        ref = cx[cxo] if cxo.any() else cx
        mean = float(ref.mean()) if ref.size else 0.0
        std = TimeSeriesSlicer._safe_std(ref, mean)
        return (cx - mean) / std
