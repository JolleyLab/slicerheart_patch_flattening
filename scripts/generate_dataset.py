"""Generate a sharded synthetic baffle dataset (geometry + ARAP/LSCM labels, optional physics labels).

    python scripts/generate_dataset.py --out output/dataset --shard 3 --n 10000 --workers 4
    python scripts/generate_dataset.py --out output/dataset --shard 0 --n 2500 --physics --workers 4

Shard s with n samples uses seeds (base_seed + 1000*s + worker) and sample ids s*n .. s*n+n-1,
so any shard can be regenerated independently and bit-for-bit. Output: <out>/shard_<s>/sample_*.npz
+ index.json, and <out>/shard_<s>.zip when --zip is given. Record format: neural/DATASET.md.
"""
import argparse
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PHYSICS_PRESSURES = [15.0, 30.0]                  # mmHg (normal / high PA pressure, Kizilski et al. 2025)
PHYSICS_MATERIALS = [                             # placeholders until measured data are fitted
    dict(model="stvk", E=10.0, nu=0.3, thickness=0.3, name="pericardium_like"),
    dict(model="stvk", E=40.0, nu=0.3, thickness=0.4, name="eptfe_like"),
]


def _work(args):
    out_dir, n, seed, start, physics, h = args
    from slicerheart_patch_flattening.physics_ml.synthetic_dataset import make_dataset
    mats = None; pressures = None
    if physics:
        from slicerheart_patch_flattening.physics_ml.warp_membrane import Material
        mats = [Material(**{k: v for k, v in m.items() if k != "name"}) for m in PHYSICS_MATERIALS]
        pressures = PHYSICS_PRESSURES
    return make_dataset(n, out_dir, seed=seed, pressure_mmhg=pressures, material=mats, h=h,
                        start_index=start, verbose_every=500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/dataset")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n", type=int, default=10000, help="samples in this shard")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--base-seed", type=int, default=20261030)
    ap.add_argument("--h", type=float, default=1.0, help="nominal mesh edge length (mm); randomised x0.75-1.3 per sample")
    ap.add_argument("--physics", action="store_true", help="add inflated states for the pressure/material grid")
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()
    shard_dir = os.path.join(a.out, f"shard_{a.shard:02d}" + ("_physics" if a.physics else ""))
    os.makedirs(shard_dir, exist_ok=True)
    per = [a.n // a.workers + (1 if w < a.n % a.workers else 0) for w in range(a.workers)]
    starts = [a.shard * a.n + sum(per[:w]) for w in range(a.workers)]
    jobs = [(shard_dir, per[w], a.base_seed + 1000 * a.shard + w, starts[w], a.physics, a.h) for w in range(a.workers) if per[w] > 0]
    t0 = time.time()
    with Pool(len(jobs)) as pool:
        parts = pool.map(_work, jobs)
    index = [r for p in parts for r in p]
    manifest = dict(shard=a.shard, n=len(index), base_seed=a.base_seed, workers=a.workers, h=a.h, physics=a.physics,
                    pressures_mmhg=PHYSICS_PRESSURES if a.physics else None, materials=PHYSICS_MATERIALS if a.physics else None,
                    generator="slicerheart_patch_flattening.physics_ml.synthetic_dataset.make_sample (wide=True)",
                    seconds=time.time() - t0, index=index)
    with open(os.path.join(shard_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    for w in range(a.workers):      # per-worker index files are superseded by the manifest
        p = os.path.join(shard_dir, "index.json")
        if os.path.exists(p):
            os.remove(p)
    print(f"shard {a.shard}: {len(index)} samples in {time.time() - t0:.0f} s -> {shard_dir}")
    if a.zip:
        z = shutil.make_archive(shard_dir, "zip", root_dir=a.out, base_dir=os.path.basename(shard_dir))
        print(f"zipped: {z} ({os.path.getsize(z) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
