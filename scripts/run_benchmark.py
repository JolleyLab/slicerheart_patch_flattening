"""Build the synthetic library, run every method, write results + report (Markdown and PDF).

    python scripts/run_benchmark.py --out output/benchmark [--methods arap lscm ...] [--quick]

--quick runs 6 shapes x 4 methods without darts/refinement (CI smoke run, ~30 s).
Outputs in --out: synthetic_library/, eval/ (results_long.csv, summary_by_family.csv,
uncertainty.csv, darts.csv, figures/, templates/), report.md, report.pdf.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from slicerheart_patch_flattening.library import build_library, write_library
from slicerheart_patch_flattening.evaluate import run_benchmark, DEFAULT_METHODS
from slicerheart_patch_flattening.report import write_report
from slicerheart_patch_flattening import viz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/benchmark")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-templates", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    shapes = build_library()
    methods = a.methods or DEFAULT_METHODS
    if a.quick:
        shapes = shapes[:6]
        methods = a.methods or ["lscm", "arap", "membrane_stvk", "pca_projection"]
    write_library(os.path.join(a.out, "synthetic_library"), shapes)
    eval_dir = os.path.join(a.out, "eval")
    run_benchmark(shapes, methods, eval_dir, do_darts=not a.quick, do_refinement=not a.quick,
                  do_figures=True, do_templates=not (a.quick or a.no_templates))
    # library gallery for the report
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = 6; nrow = (len(shapes) + ncol - 1) // ncol
    fig = plt.figure(figsize=(3.3 * ncol, 3.2 * nrow))
    for i, m in enumerate(shapes):
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d"); viz.plot_shape_3d(ax, m, title=m.name)
    fig.tight_layout(); os.makedirs(os.path.join(eval_dir, "figures"), exist_ok=True)
    fig.savefig(os.path.join(eval_dir, "figures", "library_gallery.png"), dpi=70); plt.close(fig)
    md, pdf = write_report(eval_dir, os.path.join(a.out, "report.md"), os.path.join(a.out, "report.pdf"))
    print(f"report: {md}\n        {pdf}\ntotal {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
