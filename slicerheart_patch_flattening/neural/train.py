"""Train / evaluate the outline surrogate; scaling study over training-set size.

    python -m slicerheart_patch_flattening.neural.train --data dir1 dir2 --out output/surrogate \
        --sizes 250 500 1000 2000 --epochs 80

Reports, per training-set size: test outline RMS / max error vs the ARAP label (mm),
against two baselines on the same test set: orthographic projection (residual = 0) and
LSCM (the Baffle Planner's method, aligned to the ARAP outline). Also inference time.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .data import load_raw, build_arrays
from .model import OutlineSurrogate


def split(n, test_frac=0.15, seed=0):
    rng = np.random.default_rng(seed); idx = rng.permutation(n); nt = int(n * test_frac)
    return idx[nt:], idx[:nt]


def to_t(a):
    return torch.tensor(a, dtype=torch.float32)


def outline_errors(resid_pred, resid_true):
    d = np.linalg.norm(resid_pred - resid_true, axis=2)          # (N,K)
    return dict(rms_mm=float(np.sqrt((d**2).mean())), max_mm=float(d.max()),
                per_shape_rms=np.sqrt((d**2).mean(1)), per_shape_max=d.max(1))


def shift_invariant_mse(pred, target):
    """Min over cyclic shifts of the boundary index of the per-point MSE (start-point ambiguity)."""
    K = pred.shape[1]
    best = None
    for s in range(K):
        e = ((pred - torch.roll(target, s, dims=1)) ** 2).mean(dim=(1, 2))
        best = e if best is None else torch.minimum(best, e)
    return best.mean()


def train_one(arr, train_idx, test_idx, epochs=80, lr=2e-3, batch=32, seed=0, verbose=False,
              device="cpu", amp=False, shift_invariant=False):
    torch.manual_seed(seed)
    model = OutlineSurrogate().to(device)
    Xb, Xs = to_t(arr["boundary"]).to(device), to_t(arr["surface"]).to(device)
    Y, S = to_t(arr["residual"]).to(device), to_t(arr["scalars"]).to(device)
    s_scale = S[train_idx].std(0) + 1e-6
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * max(1, len(train_idx) // batch + 1))
    tr = np.array(train_idx)
    t0 = time.time()
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(tr); tot = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            with torch.autocast(device_type=device if device != "cpu" else "cpu", enabled=amp, dtype=torch.bfloat16):
                resid, sc = model(Xb[b], Xs[b])
                main = shift_invariant_mse(resid.float(), Y[b]) if shift_invariant else ((resid.float() - Y[b])**2).mean()
                loss = main + 0.05 * (((sc.float() - S[b]) / s_scale)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(b)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"    epoch {ep:3d} train loss {tot / len(tr):.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        resid, sc = model(Xb[test_idx], Xs[test_idx])
    err = outline_errors(resid.float().cpu().numpy(), arr["residual"][test_idx])
    sc_err = np.abs(sc.float().cpu().numpy() - arr["scalars"][test_idx]).mean(0)
    with torch.no_grad():
        t1 = time.time(); model(Xb[test_idx[:1]], Xs[test_idx[:1]]); dt = time.time() - t1
    return model, dict(train_time_s=time.time() - t0, infer_ms=1000 * dt, n_train=int(len(train_idx)),
                       outline_rms_mm=err["rms_mm"], outline_max_mm=err["max_mm"],
                       outline_rms_p95_mm=float(np.percentile(err["per_shape_rms"], 95)),
                       suture_mismatch_mae_mm=float(sc_err[0]), totalK_mae_deg=float(sc_err[1]), ndarts_mae=float(sc_err[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="*", default=[], help="directories of sample_*.npz (legacy)")
    ap.add_argument("--out", default="output/surrogate")
    ap.add_argument("--sizes", nargs="*", type=int, default=[250, 500, 1000, 2000])
    ap.add_argument("--epochs", type=int, default=None, help="epochs per run (default: budget / n_train)")
    ap.add_argument("--budget", type=int, default=24000, help="sample-epochs per run when --epochs is not given")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast")
    ap.add_argument("--shift-invariant", action="store_true", help="min over cyclic shifts of the outline loss")
    ap.add_argument("--shards", nargs="*", default=None, help="shard zips/dirs (dataset v1) instead of --data sample dirs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.shards:
        from .dataset_io import load_shards
        recs = load_shards(a.shards, limit=a.max_samples)
    else:
        recs = load_raw(a.data)
    if a.max_samples:
        recs = recs[:a.max_samples]
    print(f"{len(recs)} samples", flush=True)
    arr = build_arrays(recs)
    tr_all, te = split(len(recs))
    # baselines on the test set
    proj_err = outline_errors(np.zeros_like(arr["residual"][te]), arr["residual"][te])
    results = dict(n_total=len(recs), n_test=len(te), baseline_projection_rms_mm=proj_err["rms_mm"],
                   baseline_projection_max_mm=proj_err["max_mm"], runs=[])
    # LSCM baseline (what the Baffle Planner does today) vs ARAP labels, on a subset for time
    from ..geometric.methods import flatten
    from ..mesh import TriMesh
    from .data import rigid_align_2d
    from ..physics_ml.synthetic_dataset import resample_loop
    lscm_d = []
    for i in te[:60]:
        r = recs[i]; m = TriMesh(r["shape_xyz"], r["faces"]); loop = r["boundary_loop"]
        o_l = resample_loop(flatten(m, "lscm").uv[loop], len(r["arap_outline"]))
        o_a = np.asarray(r["arap_outline"])
        lscm_d.append(np.linalg.norm(rigid_align_2d(o_l, o_a) - o_a, axis=1))
    lscm_d = np.stack(lscm_d)
    results["baseline_lscm_vs_arap_rms_mm"] = float(np.sqrt((lscm_d**2).mean())); results["baseline_lscm_vs_arap_max_mm"] = float(lscm_d.max())
    print(f"test {len(te)}: projection baseline rms {proj_err['rms_mm']:.3f} max {proj_err['max_mm']:.3f} mm; "
          f"LSCM-vs-ARAP rms {results['baseline_lscm_vs_arap_rms_mm']:.3f} max {results['baseline_lscm_vs_arap_max_mm']:.3f} mm", flush=True)
    best = None
    for n in a.sizes:
        if n > len(tr_all):
            continue
        epochs = a.epochs or max(10, a.budget // n)
        model, res = train_one(arr, tr_all[:n], te, epochs=epochs, verbose=True, device=a.device, amp=a.amp,
                               shift_invariant=a.shift_invariant)
        res["epochs"] = epochs
        print(f"  n_train={n}: outline rms {res['outline_rms_mm']:.3f} mm (p95 of per-shape rms {res['outline_rms_p95_mm']:.3f}), "
              f"max {res['outline_max_mm']:.3f} mm; suture-mismatch MAE {res['suture_mismatch_mae_mm']:.2f} mm; "
              f"train {res['train_time_s']:.0f} s, infer {res['infer_ms']:.1f} ms", flush=True)
        results["runs"].append(res); best = model
    torch.save(best.cpu().state_dict(), os.path.join(a.out, "outline_surrogate.pt"))
    with open(os.path.join(a.out, "scaling.json"), "w") as f:
        json.dump(results, f, indent=1)
    np.savez(os.path.join(a.out, "test_split.npz"), test_idx=te)


if __name__ == "__main__":
    main()
