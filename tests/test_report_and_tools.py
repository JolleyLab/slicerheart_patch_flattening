"""Tests for the benchmark harness, report generation, export, pipeline and dataset generator."""
import os

import numpy as np
import pytest

from slicerheart_patch_flattening.library import build_library
from slicerheart_patch_flattening.evaluate import run_benchmark
from slicerheart_patch_flattening.report import write_report
from slicerheart_patch_flattening import export, flatten_recommended
from slicerheart_patch_flattening.geometric import flatten
from slicerheart_patch_flattening.library import generators as G


@pytest.fixture(scope="module")
def shapes():
    s = build_library(h=1.2)
    return [s[0], s[3], s[6]]          # developable control, sphere cap, one tunnel


def test_benchmark_and_report(tmp_path, shapes):
    df = run_benchmark(shapes, ["lscm", "arap"], str(tmp_path / "eval"), do_darts=True, do_refinement=False,
                       do_figures=True, do_templates=False, verbose=False)
    assert len(df) == 6 and set(df["method"]) == {"lscm", "arap"}
    assert (df[df["developable"]]["edge_rel_err_max_abs_pct"] < 0.3).all()
    md, pdf = write_report(str(tmp_path / "eval"))
    assert os.path.getsize(md) > 500 and os.path.getsize(pdf) > 1000
    text = open(md).read()
    assert "Results of the benchmark" in text and "arap" in text


def test_export_formats(tmp_path, shapes):
    m = shapes[2]
    uv = flatten(m, "arap").uv
    outline = export.template_outline(m, uv)
    export.write_svg(str(tmp_path / "t.svg"), outline, "t", suture_offset_mm=1.0)
    export.write_dxf(str(tmp_path / "t.dxf"), outline, suture_offset_mm=1.0)
    export.write_pdf(str(tmp_path / "t.pdf"), outline, "t")
    svg = open(tmp_path / "t.svg").read()
    assert 'width="' in svg and "mm" in svg and "<path" in svg
    assert "LWPOLYLINE" in open(tmp_path / "t.dxf").read()
    assert os.path.getsize(tmp_path / "t.pdf") > 500


def test_pipeline_recommended(shapes):
    r = flatten_recommended(shapes[2])
    assert r.method == "arap" and r.metrics["flipped_triangles"] == 0
    assert "suture-line mismatch" in r.summary()
    if r.needs_darts:
        assert r.darted_metrics["boundary_cumulative_mismatch_mm"] < r.metrics["boundary_cumulative_mismatch_mm"]


def test_synthetic_dataset(tmp_path):
    from slicerheart_patch_flattening.physics_ml.synthetic_dataset import make_dataset
    idx = make_dataset(3, str(tmp_path / "ds"), seed=1, h=1.3)
    assert len(idx) == 3 and os.path.exists(tmp_path / "ds" / "sample_00002.npz")
    d = np.load(tmp_path / "ds" / "sample_00000.npz")
    assert d["arap_outline"].shape == (128, 2) and d["boundary_resampled"].shape == (128, 3)


def test_baffle_planner_tps_generator():
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    b3 = np.c_[10 * np.cos(t), 8 * np.sin(t), 2 * np.sin(2 * t)]
    m = G.baffle_planner_patch(b3, 1.2)
    assert m.n_vertices > 50 and len(m.boundary_loop()) > 30
