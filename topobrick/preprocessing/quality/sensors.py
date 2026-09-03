"""L2: sensor-level usability auditing."""

from __future__ import annotations

import os
import shutil
from collections import Counter
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from topobrick.utils.paths import resolve_processed_root

PROCESSED_ROOT = resolve_processed_root()


def _mean_run_length(values: np.ndarray, observed: np.ndarray) -> float:
    """Average length of runs of identical observed values."""
    if observed.sum() < 2:
        return 0.0
    o_idx = np.where(observed)[0]
    if len(o_idx) < 2:
        return 0.0
    v = values[o_idx]
    diffs = np.diff(v) != 0
    bounds = np.concatenate([[0], np.where(diffs)[0] + 1, [len(v)]])
    runs = np.diff(bounds)
    return float(runs.mean()) if len(runs) else 0.0


def _is_cumulative_monotone(
    values: np.ndarray, observed: np.ndarray, monotone_pct: float = 0.98
) -> bool:
    """Detect cumulative meters from monotonicity, endpoint, and dynamic range."""
    if observed.sum() < 10:
        return False
    v = values[observed]
    if len(v) < 10:
        return False
    diffs = np.diff(v)
    non_dec = (diffs >= -1e-9).sum() / max(1, len(diffs))
    if non_dec < monotone_pct:
        return False
    vmin, vmax = float(np.min(v)), float(np.max(v))
    rng = vmax - vmin
    if rng < 1e-6:
        return False  # constant, not cumulative
    # cumulative meter ends near its peak (only ever increases)
    ends_high = v[-1] >= vmin + 0.9 * rng
    return ends_high


def audit_one_sensor(sub: pd.DataFrame) -> Dict:
    """Compute quality metrics for one sensor."""
    n_total = len(sub)
    v = sub["value"].to_numpy(dtype=np.float64)
    obs = sub["observed_mask"].to_numpy(dtype=bool)
    n_obs = int(obs.sum())
    if n_obs < 2:
        return {
            "n_total": n_total,
            "n_obs": n_obs,
            "obs_rate": n_obs / n_total if n_total else 0,
            "n_unique_obs": 0,
            "std_obs": 0.0,
            "min_obs": np.nan,
            "max_obs": np.nan,
            "median_obs": np.nan,
            "pct_zero_obs": 0.0,
            "mean_run_length_identical": 0.0,
            "is_cumulative_monotone": False,
        }
    v_obs = v[obs]
    return {
        "n_total": n_total,
        "n_obs": n_obs,
        "obs_rate": n_obs / n_total,
        "n_unique_obs": int(len(np.unique(v_obs))),
        "std_obs": float(np.std(v_obs)),
        "min_obs": float(np.min(v_obs)),
        "max_obs": float(np.max(v_obs)),
        "median_obs": float(np.median(v_obs)),
        "pct_zero_obs": float((v_obs == 0).sum() / max(1, len(v_obs))),
        "mean_run_length_identical": _mean_run_length(v, obs),
        "is_cumulative_monotone": _is_cumulative_monotone(v, obs),
    }


CONTROL_SIGNAL_SUBSTR = (
    "Setpoint",
    "Command",
    "Status",
    "Mode",
    "Enable",
    "Stages",
    "Fan_Speed",
    "Position",
    "Damper",
    "Valve",
    "Parameter",
    "Limit",
)


def _is_control_signal(brick_class: str) -> bool:
    return bool(brick_class) and any(s in brick_class for s in CONTROL_SIGNAL_SUBSTR)


def decide_is_usable(
    m: Dict,
    brick_class: str = "",
    min_obs_rate: float = 0.40,
    min_unique: int = 5,
    min_std: float = 0.01,
    max_pct_zero: float = 0.50,
    max_mean_run: float = 96.0,
    exclude_cumulative: bool = True,
) -> Tuple[bool, str]:
    """Returns (is_usable, reason)."""
    if m["obs_rate"] < min_obs_rate:
        return False, f"obs_rate={m['obs_rate']:.2f}<{min_obs_rate}"
    if m["n_obs"] < 100:
        return False, f"n_obs={m['n_obs']}<100"

    is_ctrl = _is_control_signal(brick_class)
    if is_ctrl:
        # A setpoint that only ever steps up looks monotone, but it is not a
        # cumulative meter. Reject only a signal that never changes at all.
        if m["n_unique_obs"] < 2:
            return False, "control_signal_constant(n_unique<2)"
        return True, ""

    if exclude_cumulative and m["is_cumulative_monotone"]:
        return False, "cumulative_meter"

    if m["n_unique_obs"] < min_unique:
        return False, f"n_unique={m['n_unique_obs']}<{min_unique}"
    if m["std_obs"] < min_std:
        return False, f"std={m['std_obs']:.4f}<{min_std}"
    if m["pct_zero_obs"] > max_pct_zero:
        return False, f"pct_zero={m['pct_zero_obs']:.2f}>{max_pct_zero}"
    if m["mean_run_length_identical"] > max_mean_run:
        return False, f"mean_run={m['mean_run_length_identical']:.1f}>{max_mean_run}"
    return True, ""


def run_audit(
    dataset: str,
    processed_root: str = PROCESSED_ROOT,
    dry_run: bool = False,
    min_obs_rate: float = 0.40,
    min_unique: int = 5,
    min_std: float = 0.01,
    max_pct_zero: float = 0.50,
    max_mean_run: float = 96.0,
    exclude_cumulative: bool = True,
) -> Dict:
    proc = os.path.join(processed_root, dataset)
    series_path = os.path.join(proc, "series.parquet")
    nodes_path = os.path.join(proc, "kg_nodes.parquet")

    print(f"[load] kg_nodes from {nodes_path}")
    nodes = pd.read_parquet(nodes_path)
    print(
        f"  total: {len(nodes):,} nodes, "
        f"has_ts=True: {nodes['has_timeseries'].fillna(False).astype(bool).sum():,}"
    )

    # Build kg_node_id → [sensor_id] maps via point_map.parquet (authoritative;
    # kg_nodes.ts_refs is empty in some datasets).
    pmap = pd.read_parquet(
        os.path.join(proc, "point_map.parquet"),
        columns=["sensor_id", "kg_node_id", "brick_class"],
    )
    nid_to_sids: Dict[str, List[str]] = {}
    nid_to_bc: Dict[str, str] = {}
    for _, r in pmap.iterrows():
        nid = str(r["kg_node_id"])
        sid = str(r["sensor_id"])
        nid_to_sids.setdefault(nid, []).append(sid)
        if pd.notna(r["brick_class"]):
            nid_to_bc[nid] = str(r["brick_class"])

    print("[load] series.parquet")
    # Read repeated sensor IDs as dictionary-encoded categories.
    s = pq.read_table(
        series_path,
        columns=["sensor_id", "value", "observed_mask"],
        read_dictionary=["sensor_id"],
    ).to_pandas()
    print(f"  rows: {len(s):,}")

    # Audited per sensor_id; a KG node backed by several sensors takes the worst
    # of each statistic, so one bad stream is enough to retire the node.
    print("[audit] processing sensors ...")
    per_sid_metrics: Dict[str, Dict] = {}
    n_processed = 0
    for sid, grp in s.groupby("sensor_id", sort=False, observed=True):
        per_sid_metrics[str(sid)] = audit_one_sensor(grp)
        n_processed += 1
        if n_processed % 500 == 0:
            print(f"  {n_processed} sensors processed")
    print(f"  done. {n_processed} sensors audited.")

    print("[aggregate] sid → node ...")
    per_nid: List[Dict] = []
    for nid, sids in nid_to_sids.items():
        bc = nid_to_bc.get(nid, "")
        ms = [per_sid_metrics[sid] for sid in sids if sid in per_sid_metrics]
        if not ms:
            continue
        n_total = sum(m["n_total"] for m in ms)
        n_obs = sum(m["n_obs"] for m in ms)
        agg = {
            "node_id": nid,
            "brick_class": bc,
            "n_sids": len(ms),
            "n_total": n_total,
            "n_obs": n_obs,
            "obs_rate": n_obs / n_total if n_total else 0,
            "n_unique_obs": int(np.min([m["n_unique_obs"] for m in ms])),
            "std_obs": float(np.min([m["std_obs"] for m in ms])),
            "min_obs": float(np.nanmin([m["min_obs"] for m in ms]))
            if any(not np.isnan(m["min_obs"]) for m in ms)
            else np.nan,
            "max_obs": float(np.nanmax([m["max_obs"] for m in ms]))
            if any(not np.isnan(m["max_obs"]) for m in ms)
            else np.nan,
            "median_obs": float(np.nanmean([m["median_obs"] for m in ms]))
            if any(not np.isnan(m["median_obs"]) for m in ms)
            else np.nan,
            "pct_zero_obs": float(np.max([m["pct_zero_obs"] for m in ms])),
            "mean_run_length_identical": float(
                np.max([m["mean_run_length_identical"] for m in ms])
            ),
            "is_cumulative_monotone": any(m["is_cumulative_monotone"] for m in ms),
        }
        usable, reason = decide_is_usable(
            agg,
            brick_class=bc,
            min_obs_rate=min_obs_rate,
            min_unique=min_unique,
            min_std=min_std,
            max_pct_zero=max_pct_zero,
            max_mean_run=max_mean_run,
            exclude_cumulative=exclude_cumulative,
        )
        agg["is_usable"] = usable
        agg["reason"] = reason
        per_nid.append(agg)

    df_q = pd.DataFrame(per_nid)
    n_usable = int(df_q["is_usable"].sum())
    print(f"\n[result] {n_usable:,} / {len(df_q):,} sensors marked is_usable=True")

    print("\n[summary] per Brick class (top 30):")
    cls_summary = (
        df_q.groupby("brick_class")
        .agg(
            N=("node_id", "count"),
            usable=("is_usable", "sum"),
        )
        .sort_values("N", ascending=False)
    )
    cls_summary["unusable"] = cls_summary["N"] - cls_summary["usable"]
    print(cls_summary.head(30).to_string())

    print("\n[summary] reasons for is_usable=False (top 10):")
    reason_counts = Counter(df_q.loc[~df_q["is_usable"], "reason"])
    for r, c in reason_counts.most_common(10):
        print(f"  {r:40s}: {c}")

    if dry_run:
        print("\n[dry_run] no writes.")
        return {"n_usable": n_usable, "n_total": len(df_q), "df_q": df_q}

    audit_path = os.path.join(proc, "sensor_quality.parquet")
    df_q.to_parquet(audit_path, index=False)
    print(f"\n[write] {audit_path}")

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    bak = nodes_path + f".pre_is_usable_{ts}.bak"
    if not os.path.exists(bak):
        shutil.copyfile(nodes_path, bak)
        print(f"[backup] {bak}")
    nid_to_usable = dict(zip(df_q["node_id"], df_q["is_usable"]))
    # default False for has_ts=False sensors / unprocessed
    nodes["is_usable"] = nodes["node_id"].map(nid_to_usable).fillna(False).astype(bool)
    nodes.to_parquet(nodes_path, index=False)
    print("[write] kg_nodes.parquet  (added column `is_usable`)")
    return {"n_usable": n_usable, "n_total": len(df_q), "df_q": df_q}
