"""The benchmark harness and its Markdown/PDF report on a few shapes (the full benchmark runs in
the benchmark workflow)."""
import os

import pytest

from slicerheart_patch_flattening.evaluate import run_benchmark
from slicerheart_patch_flattening.report import write_report, summary_tables

pytestmark = pytest.mark.benchmark


@pytest.fixture(scope="module")
def eval_dir(tmp_path_factory, library):
    out = tmp_path_factory.mktemp("eval")
    shapes = [library[0], library[3], library[6]]          # developable control, sphere cap, one tunnel
    df = run_benchmark(shapes, ["lscm", "arap"], str(out), do_darts=True, do_refinement=True,
                       do_figures=True, do_templates=True, verbose=False)
    return out, df


def test_benchmark_table(eval_dir):
    out, df = eval_dir
    assert len(df) == 6 and set(df["method"]) == {"lscm", "arap"}
    assert (df[df["developable"]]["edge_rel_err_max_abs_pct"] < 0.3).all()
    assert (df["flipped_triangles"] == 0).all()
    assert {"shape", "family", "stretch_Linf", "boundary_cumulative_mismatch_mm", "time_s"} <= set(df.columns)
    csvs = [f for f in os.listdir(out) if f.endswith(".csv")]
    assert csvs and os.path.isdir(out / "figures") and os.listdir(out / "figures")
    tables = summary_tables(df)
    assert tables and all(len(t) > 0 for t in tables.values())


def test_report(eval_dir):
    out, _ = eval_dir
    md, pdf = write_report(str(out))
    assert os.path.getsize(md) > 500 and os.path.getsize(pdf) > 1000
    text = open(md).read()
    assert "Results of the benchmark" in text and "arap" in text and "lscm" in text
