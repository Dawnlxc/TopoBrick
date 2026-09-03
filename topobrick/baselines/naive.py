"""Evaluate persistence and seasonal-naive baselines on an L4 cache."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from topobrick.utils.progress import log


def predict(x: np.ndarray, H: int, method: str, P: int) -> np.ndarray:
    """x: [n, L] normalised lookback -> [n, H]."""
    L = x.shape[1]
    if method == "persistence" or L < P:
        return np.repeat(x[:, -1:], H, axis=1)
    idx = (L - P) + (np.arange(H) % P)
    return x[:, idx]


def score(d, H, method, P, sid2std, sid2bc=None):
    x = d["x"].numpy()
    yhat_n = predict(x, H, method, P)
    rm = d["revin_mean"].numpy()[:, None]
    rs = d["revin_std"].numpy()[:, None]
    Y = d["y"].numpy() * rs + rm
    P_ = yhat_n * rs + rm
    M = d["mask"].numpy()
    diff = (P_ - Y) * M
    denom = max(M.sum(), 1)
    out = {
        "n_scored_cells": int(M.sum()),
        "n_windows": int(x.shape[0]),
        "raw_MAE": float(np.abs(diff).sum() / denom),
        "raw_MSE": float((diff**2).sum() / denom),
    }
    stds = np.array(
        [max(float(sid2std.get(s, 1.0)), 0.01) for s in d["sid"]], dtype=np.float32
    )[:, None]
    dn = diff / stds
    out["nMAE"] = float(np.abs(dn).sum() / denom)
    out["nMSE"] = float((dn**2).sum() / denom)
    if sid2bc is not None:
        per, cls = {}, np.array([str(sid2bc.get(s, "?")) for s in d["sid"]])
        for c in sorted(set(cls.tolist())):
            mk = cls == c
            m = M[mk]
            if m.sum() == 0:
                continue
            per[c] = {
                "n": int(mk.sum()),
                "raw_MAE": float(np.abs(diff[mk]).sum() / m.sum()),
                "nMAE": float(np.abs(dn[mk]).sum() / m.sum()),
            }
        out["per_class"] = per
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--cache", required=True, help="a `ci` cache from topobrick.baselines.windows"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--seasonal-period-hours",
        "--seasonal_period_hours",
        dest="seasonal_period_hours",
        type=float,
        default=24.0,
    )
    ap.add_argument(
        "--freq-minutes",
        "--freq_minutes",
        dest="freq_minutes",
        type=int,
        default=15,
    )
    a = ap.parse_args()

    if os.path.exists(a.out):
        log(f"exists, skipping: {a.out}")
        return
    blob = torch.load(a.cache, weights_only=False)
    if blob.get("layout") != "ci":
        raise SystemExit("naive baselines need a `ci` cache (one row per window)")
    H, L, ds = blob["horizon"], blob["lookback"], blob["dataset"]
    P = int(a.seasonal_period_hours * 60 / a.freq_minutes)
    te = blob["test"]
    log(
        f"[naive {ds} H={H}] windows={te['n_windows']:,} "
        f"scored cells={te['n_scored_cells']:,} seasonal period={P}"
    )

    res = {
        m: score(te, H, m, P, blob["sid2std"], blob.get("sid2bc"))
        for m in ("persistence", "seasonal_naive")
    }
    for m, r in res.items():
        log(
            f"  {m:15s} nMAE={r['nMAE']:.4f} nMSE={r['nMSE']:.4f} rawMAE={r['raw_MAE']:.4f}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(
            {
                "dataset": ds,
                "horizon": H,
                "lookback": L,
                "seasonal_period": P,
                "deterministic": True,
                "processed_root": blob["processed_root"],
                "cache": os.path.basename(a.cache),
                "n_windows": te["n_windows"],
                "n_scored_cells": te["n_scored_cells"],
                "test": res,
            },
            f,
            indent=2,
        )
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
