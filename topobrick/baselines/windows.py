"""Materialise the L4 evaluation windows into flat tensors, once per (dataset, horizon)."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch

from topobrick.forecast.data.timeseries import TimeSeriesSlicer
from topobrick.preprocessing.targets.variability import compute_per_sensor_train_std
from topobrick.utils.config import load_config
from topobrick.utils.paths import processed_data_directory
from topobrick.utils.progress import log

DEFAULT_PATTERN = "windows_L{L}_H{H}_{split}_clean.parquet"
SPLITS = ("train", "val", "test")


def _windows(proc: str, fname: str) -> pd.DataFrame:
    win = pd.read_parquet(os.path.join(proc, fname))
    win["sensor_id"] = win["sensor_id"].astype(str)
    for c in ("input_start", "input_end", "target_start", "target_end"):
        win[c] = pd.to_datetime(win[c])
    return win


def build_ci(proc: str, fname: str, L: int, H: int) -> dict:
    win = _windows(proc, fname)
    slicer = TimeSeriesSlicer(
        proc,
        target_sensor_ids=set(win["sensor_id"]),
        exogenous_node_ids=set(),
        lookback=L,
        horizon=H,
    )
    n = len(win)
    X = np.zeros((n, L), dtype=np.float32)
    Y = np.zeros((n, H), dtype=np.float32)
    M = np.zeros((n, H), dtype=bool)
    RM = np.zeros(n, dtype=np.float32)
    RS = np.ones(n, dtype=np.float32)
    sids = []
    for i, r in enumerate(win.itertuples(index=False)):
        x, x_obs, x_trust, y, y_trust = slicer.get_target_window(
            r.sensor_id, r.input_start, r.input_end, r.target_start, r.target_end
        )
        xn, yn, mean, std = TimeSeriesSlicer.revin_target(x, x_trust, x_obs, y)
        X[i] = xn
        Y[i] = yn
        M[i] = y_trust
        RM[i] = mean
        RS[i] = std
        sids.append(str(r.sensor_id))
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "layout": "ci",
        "x": torch.from_numpy(X),
        "y": torch.from_numpy(Y),
        "mask": torch.from_numpy(M),
        "revin_mean": torch.from_numpy(RM),
        "revin_std": torch.from_numpy(RS),
        "sid": sids,
        "n_windows": n,
        "n_scored_cells": int(M.sum()),
    }


def build_cd(
    proc: str,
    fname: str,
    channels: list[str],
    L: int,
    H: int,
    slicer: TimeSeriesSlicer,
) -> dict:
    win = _windows(proc, fname)
    anchors = np.sort(win["input_start"].unique())
    anchor_index = {timestamp: index for index, timestamp in enumerate(anchors)}
    channel_index = {channel: index for index, channel in enumerate(channels)}
    n_anchors, n_channels = len(anchors), len(channels)

    timestamps = (
        win.drop_duplicates("input_start")
        .set_index("input_start")[["input_end", "target_start", "target_end"]]
        .loc[anchors]
    )
    X = np.zeros((n_anchors, L, n_channels), dtype=np.float32)
    Y = np.zeros((n_anchors, H, n_channels), dtype=np.float32)
    TRUST = np.zeros((n_anchors, H, n_channels), dtype=bool)
    RM = np.zeros((n_anchors, n_channels), dtype=np.float32)
    RS = np.ones((n_anchors, n_channels), dtype=np.float32)

    for anchor, input_start in enumerate(anchors):
        input_end, target_start, target_end = timestamps.iloc[anchor]
        for channel, channel_number in channel_index.items():
            x, x_obs, x_trust, y, y_trust = slicer.get_target_window(
                channel, input_start, input_end, target_start, target_end
            )
            xn, yn, mean, std = TimeSeriesSlicer.revin_target(
                x, x_trust, x_obs, y
            )
            X[anchor, :, channel_number] = xn
            Y[anchor, :, channel_number] = yn
            TRUST[anchor, :, channel_number] = y_trust
            RM[anchor, channel_number] = mean
            RS[anchor, channel_number] = std

    present = np.zeros((n_anchors, n_channels), dtype=bool)
    for sensor_id, input_start in zip(win["sensor_id"], win["input_start"]):
        channel_number = channel_index.get(sensor_id)
        if channel_number is not None:
            present[anchor_index[input_start], channel_number] = True

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    X *= present[:, None, :]
    Y *= present[:, None, :]
    mask = TRUST & present[:, None, :]

    return {
        "layout": "cd",
        "x": torch.from_numpy(X),
        "y": torch.from_numpy(Y),
        "mask": torch.from_numpy(mask),
        "present": torch.from_numpy(present),
        "revin_mean": torch.from_numpy(RM),
        "revin_std": torch.from_numpy(RS),
        "n_anchors": n_anchors,
        "n_channels": n_channels,
        "n_windows": int(present.sum()),
        "n_scored_cells": int(mask.sum()),
    }


def build(
    cfg,
    dataset: str,
    horizon: int,
    lookback: int,
    layout: str,
    pattern: str = DEFAULT_PATTERN,
) -> dict:
    proc = processed_data_directory(cfg, dataset)

    def fname(split):
        return pattern.format(L=lookback, H=horizon, split=split)

    out = {
        "dataset": dataset,
        "horizon": horizon,
        "lookback": lookback,
        "layout": layout,
        "processed_root": cfg["paths"]["processed_root"],
    }

    if layout == "cd":
        channels: set[str] = set()
        for split in SPLITS:
            split_channels = pd.read_parquet(
                os.path.join(proc, fname(split)), columns=["sensor_id"]
            )
            channels.update(split_channels["sensor_id"].dropna().astype(str))
        ordered_channels = sorted(channels)
        out["channels"] = ordered_channels
        slicer = TimeSeriesSlicer(
            proc,
            target_sensor_ids=set(ordered_channels),
            exogenous_node_ids=set(),
            lookback=lookback,
            horizon=horizon,
        )
        for split in SPLITS:
            out[split] = build_cd(
                proc,
                fname(split),
                ordered_channels,
                lookback,
                horizon,
                slicer,
            )
            log(
                f"  {split}: anchors={out[split]['n_anchors']:,} "
                f"channels={out[split]['n_channels']} "
                f"windows={out[split]['n_windows']:,} "
                f"scored={out[split]['n_scored_cells']:,}"
            )
    else:
        for split in SPLITS:
            out[split] = build_ci(proc, fname(split), lookback, horizon)
            log(
                f"  {split}: windows={out[split]['n_windows']:,} "
                f"scored={out[split]['n_scored_cells']:,}"
            )

    out["sid2std"] = compute_per_sensor_train_std(
        proc, horizon, windows_filename=fname("train")
    )
    nodes = pd.read_parquet(
        os.path.join(proc, "kg_nodes.parquet"), columns=["node_id", "brick_class"]
    )
    nid2bc = dict(zip(nodes["node_id"], nodes["brick_class"]))
    pmap = pd.read_parquet(
        os.path.join(proc, "point_map.parquet"), columns=["sensor_id", "kg_node_id"]
    )
    out["sid2bc"] = {
        str(s): nid2bc.get(str(n), "?")
        for s, n in zip(pmap["sensor_id"], pmap["kg_node_id"])
        if pd.notna(s) and pd.notna(n)
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True, help="processed-tree directory name")
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--lookback", type=int, default=96)
    ap.add_argument("--layout", choices=("ci", "cd"), default="ci")
    ap.add_argument(
        "--windows-pattern",
        "--windows_pattern",
        dest="windows_pattern",
        default=DEFAULT_PATTERN,
    )
    ap.add_argument("--out", required=True, help="destination .pt")
    a = ap.parse_args()

    if os.path.exists(a.out):
        log(f"exists, skipping: {a.out}")
        return
    cfg = load_config(a.config)
    cfg["L"] = a.lookback
    log(f"{a.dataset} H={a.horizon} layout={a.layout}")
    payload = build(
        cfg,
        a.dataset,
        a.horizon,
        a.lookback,
        a.layout,
        a.windows_pattern,
    )
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(payload, a.out)
    log(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
