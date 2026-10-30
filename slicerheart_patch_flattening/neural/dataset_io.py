"""Reading the sharded dataset and serving it to PyTorch.

    from slicerheart_patch_flattening.neural.dataset_io import BaffleDataset, load_shards
    recs = load_shards(["shard_00.zip", "shard_01.zip"])            # or extracted directories
    ds = BaffleDataset(recs, split="train")                          # featurised, cached in RAM
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)

Splits are deterministic by sample id hash (80/10/10), so train/val/test are the same on
every machine and every shard subset. Records come straight from the npz files; featurisation
(canonical frame, residual target) is applied lazily by `BaffleDataset`, so changing the
featurisation never requires regenerating the data.
"""
from __future__ import annotations

import glob
import hashlib
import io
import json
import os
import zipfile

import numpy as np

from .data import featurize, M_SURFACE


def _read_npz(data: bytes | str, name: str = "") -> dict:
    z = np.load(io.BytesIO(data) if isinstance(data, bytes) else data, allow_pickle=False)
    rec = {k: z[k] for k in z.files if k != "meta"}
    rec.update(json.loads(str(z["meta"])))
    if "id" not in rec:                      # older files: id from the file name
        stem = os.path.splitext(os.path.basename(name or str(data)))[0]
        rec["id"] = int(stem.split("_")[-1])
    return rec


def load_shards(paths: list[str], limit: int | None = None) -> list[dict]:
    """Load records from shard zip files and/or extracted shard directories."""
    recs = []
    for p in paths:
        if p.endswith(".zip"):
            with zipfile.ZipFile(p) as zf:
                names = sorted(n for n in zf.namelist() if n.endswith(".npz"))
                for n in names:
                    recs.append(_read_npz(zf.read(n), n))
                    if limit and len(recs) >= limit:
                        return recs
        else:
            for f in sorted(glob.glob(os.path.join(p, "**", "sample_*.npz"), recursive=True)):
                recs.append(_read_npz(f, f))
                if limit and len(recs) >= limit:
                    return recs
    return recs


def split_of(sample_id: int, seed_tag: str = "dataset-v1") -> str:
    """Deterministic 80/10/10 split by hash of the sample id."""
    h = int(hashlib.sha1(f"{seed_tag}:{sample_id}".encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


class BaffleDataset:
    """Featurised samples as torch tensors (boundary (K,3), surface (M,6), residual (K,2), scalars (3))."""

    def __init__(self, recs: list[dict], split: str | None = None, M: int = M_SURFACE, seed: int = 0,
                 sample_id_key: str = "id"):
        import torch
        rng = np.random.default_rng(seed)
        self.items = []
        for i, r in enumerate(recs):
            sid = int(r.get(sample_id_key, i))
            if split is not None and split_of(sid) != split:
                continue
            f = featurize(r, rng, M)
            self.items.append(dict(boundary=torch.tensor(f["boundary"]), surface=torch.tensor(f["surface"]),
                                   residual=torch.tensor(f["residual"]), proj=torch.tensor(f["proj"]),
                                   scalars=torch.tensor(f["scalars"]), family=f["family"], id=sid))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        return it["boundary"], it["surface"], it["residual"], it["scalars"]
