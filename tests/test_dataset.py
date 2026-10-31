"""Synthetic dataset generation (the GitHub Actions dataset workflow) and the shard loader.

The neural surrogate itself (neural/model.py, neural/train.py) is work in progress and has no
tests yet; only the data pipeline it will consume is tested here."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from slicerheart_patch_flattening.neural.dataset_io import load_shards, split_of

pytestmark = pytest.mark.dataset

HERE = os.path.dirname(__file__)
SAMPLE_100 = os.path.join(HERE, "..", "slicerheart_patch_flattening", "neural", "data", "sample_100")
GENERATE_SCRIPT = os.path.join(HERE, "..", "scripts", "generate_dataset.py")


def test_make_dataset(tmp_path):
    from slicerheart_patch_flattening.physics_ml.synthetic_dataset import make_dataset
    idx = make_dataset(3, str(tmp_path / "ds"), seed=1, h=1.3)
    assert len(idx) == 3 and os.path.exists(tmp_path / "ds" / "sample_000002.npz")
    d = np.load(tmp_path / "ds" / "sample_000000.npz")
    assert d["arap_outline"].shape == (128, 2) and d["boundary_resampled"].shape == (128, 3)
    assert np.isfinite(d["arap_outline"]).all()
    # deterministic for a given seed
    again = make_dataset(1, str(tmp_path / "ds2"), seed=1, h=1.3)
    d2 = np.load(tmp_path / "ds2" / "sample_000000.npz")
    assert np.array_equal(d["arap_outline"], d2["arap_outline"])


def test_generate_dataset_script(tmp_path):
    """The sharded generator used by the generate-dataset workflow: shard 1 of 4 samples holds ids 4..7."""
    subprocess.run([sys.executable, GENERATE_SCRIPT, "--out", str(tmp_path), "--shard", "1", "--n", "4",
                    "--workers", "2"], check=True, timeout=600)
    man = json.load(open(tmp_path / "shard_01" / "manifest.json"))
    assert man["n"] == 4 and sorted(r["id"] for r in man["index"]) == [4, 5, 6, 7]
    files = sorted(p for p in os.listdir(tmp_path / "shard_01") if p.endswith(".npz"))
    assert len(files) == 4


def test_sample_shard_and_splits():
    recs = load_shards([SAMPLE_100], limit=20)
    assert len(recs) == 20
    assert {"arap_uv", "lscm_uv", "vertex_stretch_s1", "faces", "metrics", "id"} <= set(recs[0])
    r = recs[0]
    assert r["arap_uv"].shape[1] == 2 and r["faces"].shape[1] == 3 and np.isfinite(r["arap_uv"]).all()
    assert r["faces"].max() < len(r["arap_uv"])
    assert split_of(0) == split_of(0) and {split_of(i) for i in range(200)} == {"train", "val", "test"}
    assert split_of(7, "other-tag") in ("train", "val", "test")


def test_torch_dataset():
    pytest.importorskip("torch")
    from slicerheart_patch_flattening.neural.dataset_io import BaffleDataset
    ds = BaffleDataset(load_shards([SAMPLE_100], limit=5))
    assert len(ds) == 5
    b, s, r, sc = ds[0]
    assert tuple(b.shape) == (128, 3) and tuple(s.shape) == (256, 6) and tuple(r.shape) == (128, 2)
