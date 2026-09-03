"""Train one full-shot baseline on a cache from `topobrick.baselines.windows`."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import torch

from topobrick.baselines.masked_cd import wrap
from topobrick.utils.progress import log


def _import_models(models_root: str | None):
    root = models_root or os.environ.get("TOPOBRICK_BASELINE_MODELS")
    if not root:
        raise SystemExit(
            "No model source. Pass --models-root or set $TOPOBRICK_BASELINE_MODELS "
            "to a checkout exposing "
            "models/{PatchTST,iTransformer,FITS,DLinear,TimeXer}.py."
        )
    if not os.path.isdir(os.path.join(root, "models")):
        raise SystemExit(f"{root} has no models/ directory")
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def build_model(name: str, seq_len: int, pred_len: int, n_channels: int, hp: dict):
    """Instantiate a baseline model from the configured checkout."""
    if name == "patchtst":
        Model = import_module("models.PatchTST").Model
        cfg = SimpleNamespace(
            enc_in=n_channels,
            seq_len=seq_len,
            pred_len=pred_len,
            e_layers=hp["e_layers"],
            n_heads=hp["n_heads"],
            d_model=hp["d_model"],
            d_ff=hp["d_ff"],
            dropout=hp["dropout"],
            fc_dropout=hp["fc_dropout"],
            head_dropout=hp["head_dropout"],
            individual=0,
            patch_len=hp["patch_len"],
            stride=hp["stride"],
            padding_patch="end",
            revin=hp["revin"],
            affine=0,
            subtract_last=0,
            decomposition=0,
            kernel_size=25,
        )
        return Model(cfg), "plain"
    if name == "itransformer":
        Model = import_module("models.iTransformer").Model
        cfg = SimpleNamespace(
            seq_len=seq_len,
            pred_len=pred_len,
            output_attention=False,
            use_revin=hp["use_revin"],
            d_model=hp["d_model"],
            d_ff=hp["d_ff"],
            embed="timeF",
            freq="t",
            dropout=hp["dropout"],
            factor=1,
            n_heads=hp["n_heads"],
            activation="gelu",
            e_layers=hp["e_layers"],
        )
        return Model(cfg), "itr"
    if name == "fits":
        Model = import_module("models.FITS").Model
        cfg = SimpleNamespace(
            seq_len=seq_len,
            pred_len=pred_len,
            individual=hp["individual"],
            enc_in=n_channels,
            cut_freq=hp["cut_freq"],
        )
        return Model(cfg), "plain"
    if name == "dlinear":
        Model = import_module("models.DLinear").Model
        cfg = SimpleNamespace(
            seq_len=seq_len,
            pred_len=pred_len,
            individual=hp["individual"],
            enc_in=n_channels,
            moving_avg=hp["moving_avg"],
        )
        return Model(cfg), "plain"
    if name == "timexer":
        Model = import_module("models.TimeXer").Model
        cfg = SimpleNamespace(
            features="M",
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=n_channels,
            use_revin=hp["use_revin"],
            patch_len=hp["patch_len"],
            d_model=hp["d_model"],
            d_ff=hp["d_ff"],
            e_layers=hp["e_layers"],
            n_heads=hp["n_heads"],
            dropout=hp["dropout"],
            factor=1,
            embed="timeF",
            freq="t",
            activation="gelu",
        )
        return Model(cfg), "itr"
    raise ValueError(name)


def forward(model, kind, x, mode, present=None):
    """Run one channel-independent or channel-dependent batch."""
    model_input = x.unsqueeze(-1) if mode == "ci" else x
    if kind == "masked":
        out = model(model_input, None, None, None, present=present)
    elif kind == "itr":
        out = model(model_input, None, None, None)
    else:
        out = model(model_input)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out[..., 0] if mode == "ci" and out.dim() == 3 else out


SIGMA_WEIGHT_CAP = 50.0


def masked_mse(
    y_hat, y, mask, rev_std, mode, min_revin_std=0.1, sigma=None
):
    """Compute masked MSE, excluding low-variance lookback windows."""
    ok = (rev_std >= min_revin_std).float()
    variance_axis = -1 if mode == "ci" else 1
    valid = mask.float() * ok.unsqueeze(variance_axis)
    squared_error = (y_hat - y) ** 2
    if sigma is not None:
        weights = (rev_std / sigma).clamp(max=SIGMA_WEIGHT_CAP) ** 2
        squared_error = squared_error * weights.unsqueeze(variance_axis)
    return (squared_error * valid).sum() / valid.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(
    model,
    kind,
    data,
    device,
    batch_size,
    mode,
    sid2std=None,
    channels=None,
    sid2bc=None,
):
    model.eval()
    n = data["x"].shape[0]
    prediction = torch.empty_like(data["y"])
    present = data.get("present")
    for start in range(0, n, batch_size):
        stop = start + batch_size
        present_batch = (
            None if present is None else present[start:stop].to(device)
        )
        prediction[start:stop] = (
            forward(
                model,
                kind,
                data["x"][start:stop].to(device),
                mode,
                present_batch,
            )
            .float()
            .cpu()
        )
    axis = -1 if mode == "ci" else 1
    revin_mean = data["revin_mean"].unsqueeze(axis)
    revin_std = data["revin_std"].unsqueeze(axis)
    target = (data["y"] * revin_std + revin_mean).numpy()
    predicted = (prediction * revin_std + revin_mean).numpy()
    mask = data["mask"].numpy()
    difference = (predicted - target) * mask
    denominator = max(mask.sum(), 1)
    out = {
        "n_scored_cells": int(mask.sum()),
        "raw_MAE": float(np.abs(difference).sum() / denominator),
        "raw_MSE": float((difference**2).sum() / denominator),
    }
    if sid2std is None:
        return out
    if mode == "ci":
        stds = np.array(
            [max(float(sid2std.get(s, 1.0)), 0.01) for s in data["sid"]],
            dtype=np.float32,
        )[:, None]
    else:
        stds = np.array(
            [max(float(sid2std.get(s, 1.0)), 0.01) for s in channels],
            dtype=np.float32,
        )[None, None, :]
    normalized_difference = difference / stds
    out["nMAE"] = float(np.abs(normalized_difference).sum() / denominator)
    out["nMSE"] = float((normalized_difference**2).sum() / denominator)
    if mode == "ci" and sid2bc is not None:
        per, cls = {}, np.array(
            [str(sid2bc.get(s, "?")) for s in data["sid"]]
        )
        for c in sorted(set(cls.tolist())):
            mk = cls == c
            m = mask[mk]
            if m.sum() == 0:
                continue
            per[c] = {
                "n": int(mk.sum()),
                "raw_MAE": float(np.abs(difference[mk]).sum() / m.sum()),
                "nMAE": float(
                    np.abs(normalized_difference[mk]).sum() / m.sum()
                ),
            }
        out["per_class"] = per
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--cache", required=True, help=".pt from topobrick.baselines.windows"
    )
    ap.add_argument(
        "--model",
        required=True,
        choices=["patchtst", "itransformer", "fits", "timexer", "dlinear"],
    )
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True, help="destination metrics .json")
    ap.add_argument("--models-root", "--models_root", dest="models_root", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument(
        "--lradj", choices=["type1", "type3", "TST", "constant"], default=None
    )
    for k in ("d_model", "d_ff", "n_heads", "e_layers"):
        ap.add_argument(f"--{k.replace('_', '-')}", f"--{k}", dest=k, type=int)
    ap.add_argument(
        "--min-revin-std",
        "--min_revin_std",
        dest="min_revin_std",
        type=float,
        default=0.1,
    )
    ap.add_argument(
        "--loss-scale",
        "--loss_scale",
        dest="loss_scale",
        choices=["revin", "sigma"],
        default="sigma",
        help="training-loss scale; the paper uses 'sigma'",
    )
    ap.add_argument(
        "--no-variate-mask",
        dest="variate_mask",
        action="store_false",
        help="include absent zero-filled channels in multivariate attention",
    )
    ap.set_defaults(variate_mask=True)
    a = ap.parse_args()

    if os.path.exists(a.out):
        log(f"exists, skipping: {a.out}")
        return
    _import_models(a.models_root)

    blob = torch.load(a.cache, weights_only=False)
    mode = blob.get("layout")
    if mode not in {"ci", "cd"}:
        raise SystemExit(f"unsupported cache layout: {mode!r}")
    if mode == "cd" and a.model not in {"itransformer", "timexer"}:
        raise SystemExit("cd caches are supported by iTransformer and TimeXer")
    L, H, ds = blob["lookback"], blob["horizon"], blob["dataset"]
    channels = blob.get("channels")
    N = len(channels) if mode == "cd" else 1

    hp = DEFAULTS[a.model].copy()
    if a.model == "fits":
        hp["cut_freq"] = hp["cut_freq"] or max(2, L // 96 * 6 + 1)
    for k, v in (
        ("lr", a.lr),
        ("bs", a.batch_size),
        ("epochs", a.epochs),
        ("patience", a.patience),
        ("lradj", a.lradj),
        ("d_model", a.d_model),
        ("d_ff", a.d_ff),
        ("n_heads", a.n_heads),
        ("e_layers", a.e_layers),
    ):
        if v is not None:
            hp[k] = v

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    device = a.device if torch.cuda.is_available() else "cpu"

    tr, va, te = blob["train"], blob["val"], blob["test"]
    log(
        f"[{a.model} {ds} H={H} {mode} seed={a.seed}] "
        f"scored cells train={tr['n_scored_cells']:,} test={te['n_scored_cells']:,} hp={hp}"
    )

    model, kind = build_model(a.model, L, H, N, hp)
    if mode == "cd" and a.variate_mask:
        if "present" not in tr:
            raise SystemExit("cd cache has no per-anchor channel-presence mask")
        model = wrap(a.model, model)
        kind = "masked"
    model = model.to(device)
    n_par = sum(p.numel() for p in model.parameters())

    X = tr["x"].to(device)
    Y = tr["y"].to(device)
    M = tr["mask"].to(device)
    RS = tr["revin_std"].to(device)
    PR = tr["present"].to(device) if mode == "cd" and a.variate_mask else None
    SIG = None
    if a.loss_scale == "sigma":
        sigma_keys = channels if mode == "cd" else tr["sid"]
        SIG = torch.tensor(
            [
                max(float(blob["sid2std"].get(key, 1.0)), 0.01)
                for key in sigma_keys
            ],
            dtype=torch.float32,
            device=device,
        )
    n = X.shape[0]
    steps = (n + hp["bs"] - 1) // hp["bs"]
    optim = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    sched = None
    if hp["lradj"] == "TST":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optim,
            max_lr=hp["lr"],
            epochs=hp["epochs"],
            steps_per_epoch=steps,
            pct_start=0.3,
        )
    g = torch.Generator(device="cpu")
    g.manual_seed(a.seed)

    selection_metric = "raw_MAE"
    best = {"epoch": -1, "metric": float("inf"), "state": None}
    no_imp, hist, t0 = 0, [], time.time()
    for ep in range(1, hp["epochs"] + 1):
        model.train()
        perm = torch.randperm(n, generator=g).to(device)
        losses = []
        for i in range(0, n, hp["bs"]):
            idx = perm[i : i + hp["bs"]]
            optim.zero_grad(set_to_none=True)
            loss = masked_mse(
                forward(
                    model,
                    kind,
                    X[idx],
                    mode,
                    None if PR is None else PR[idx],
                ),
                Y[idx],
                M[idx],
                RS[idx],
                mode,
                a.min_revin_std,
                None
                if SIG is None
                else (SIG[idx] if mode == "ci" else SIG),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            if sched is not None:
                sched.step()
            losses.append(float(loss.item()))
        if hp["lradj"] == "type1":  # halve each epoch
            for pg in optim.param_groups:
                pg["lr"] = hp["lr"] * (0.5**ep)
        elif hp["lradj"] == "type3":  # flat for 2 epochs, then x0.8
            nxt = ep + 1
            for pg in optim.param_groups:
                pg["lr"] = hp["lr"] if nxt < 3 else hp["lr"] * (0.8 ** (nxt - 3))
        v = evaluate(
            model,
            kind,
            va,
            device,
            4096,
            mode,
            blob["sid2std"],
            channels,
        )
        hist.append(
            {
                "epoch": ep,
                "train_loss": float(np.mean(losses)),
                "val_raw_MAE": v["raw_MAE"],
                "val_nMAE": v.get("nMAE"),
            }
        )
        if v[selection_metric] < best["metric"] - 1e-9:
            best = {
                "epoch": ep,
                "metric": v[selection_metric],
                "state": {
                    k: t.detach().cpu().clone() for k, t in model.state_dict().items()
                },
            }
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= hp["patience"]:
                log(f"  early stop at epoch {ep}")
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    test = evaluate(
        model,
        kind,
        te,
        device,
        4096,
        mode,
        blob["sid2std"],
        channels,
        blob.get("sid2bc"),
    )
    log(
        f"  test nMAE={test['nMAE']:.4f} nMSE={test['nMSE']:.4f} "
        f"rawMAE={test['raw_MAE']:.4f} cells={test['n_scored_cells']:,} "
        f"best_epoch={best['epoch']} ({time.time() - t0:.0f}s)"
    )

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(
            {
                "model": a.model,
                "mode": mode,
                "dataset": ds,
                "horizon": H,
                "lookback": L,
                "seed": a.seed,
                "n_channels": N,
                "processed_root": blob["processed_root"],
                "cache": os.path.basename(a.cache),
                "hyperparams": hp,
                "n_params": n_par,
                "n_scored_cells": {
                    s: blob[s]["n_scored_cells"] for s in ("train", "val", "test")
                },
                "best_epoch": best["epoch"],
                "selection_metric": selection_metric,
                "best_val_metric": best["metric"],
                "epochs_run": len(hist),
                "test": test,
                "history": hist,
            },
            f,
            indent=2,
        )
    log(f"wrote {a.out}")


DEFAULTS = {
    "patchtst": dict(
        lr=1e-4,
        bs=128,
        epochs=30,
        patience=5,
        d_model=128,
        d_ff=256,
        n_heads=16,
        e_layers=3,
        dropout=0.2,
        fc_dropout=0.2,
        head_dropout=0.0,
        patch_len=16,
        stride=8,
        revin=1,
        lradj="TST",
    ),
    "itransformer": dict(
        lr=1e-4,
        bs=32,
        epochs=30,
        patience=5,
        d_model=128,
        d_ff=128,
        n_heads=8,
        e_layers=2,
        dropout=0.1,
        use_revin=1,
        lradj="type1",
    ),
    "fits": dict(
        lr=5e-4, bs=64, epochs=30, patience=5, cut_freq=0, individual=0, lradj="type1"
    ),
    "dlinear": dict(
        lr=5e-3,
        bs=32,
        epochs=30,
        patience=5,
        individual=0,
        moving_avg=25,
        lradj="type1",
    ),
    "timexer": dict(
        lr=1e-3,
        bs=256,
        epochs=30,
        patience=5,
        d_model=256,
        d_ff=256,
        e_layers=1,
        n_heads=8,
        patch_len=16,
        dropout=0.1,
        use_revin=1,
        lradj="type3",
    ),
}

if __name__ == "__main__":
    main()
