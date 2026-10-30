"""Benchmark harness: every shape x every method -> metrics table, figures, templates.

Usage (CLI):  python -m slicerheart_patch_flattening.evaluate --library output/synthetic_library --out output/eval
Programmatic: run_benchmark(shapes, methods, out_dir)

Outputs
  results_long.csv       one row per (shape, method) with all metrics + timing
  summary_by_family.csv  median / max per family x method for key metrics
  uncertainty.csv        per-shape method spread and refinement sensitivity
  darts.csv              recommended darts and the metrics before/after cutting (ARAP)
  figures/               per-shape template rows and library-level summaries
  templates/<shape>/     SVG/DXF/PDF templates for the recommended method (ARAP) and LSCM
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from .mesh import TriMesh
from .algorithms.methods import flatten, METHODS
from .metrics import distortion_metrics, KEY_METRICS
from .uncertainty import method_spread, refinement_sensitivity
from .darts import recommend_darts, flatten_with_darts
from . import export, viz

DEFAULT_METHODS = ["pca_projection", "tutte", "harmonic", "mean_value", "lscm", "lscm_edge", "lscm_perimeter",
                   "asap", "arap", "arap_hybrid", "spring", "membrane_stvk",
                   "symmetric_dirichlet", "membrane_suture", "bff_lite", "cetm", "exp_map"]
SPREAD_METHODS = ["lscm", "arap", "membrane_stvk", "asap", "arap_hybrid", "symmetric_dirichlet"]   # isometric/conformal free-boundary methods
TEMPLATE_METHODS = ["arap", "lscm"]


def run_benchmark(shapes: list[TriMesh], methods: list[str] | None = None, out_dir: str = "output/eval",
                  do_darts: bool = True, do_refinement: bool = True, do_figures: bool = True,
                  do_templates: bool = True, verbose: bool = True) -> pd.DataFrame:
    methods = methods or DEFAULT_METHODS
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures"); os.makedirs(fig_dir, exist_ok=True)
    rows, unc_rows, dart_rows = [], [], []
    for m in shapes:
        if verbose:
            print(f"[{m.name}] nV={m.n_vertices} family={m.meta.get('family')}", flush=True)
        results, metrics = {}, {}
        for meth in methods:
            res = flatten(m, meth)
            M = distortion_metrics(m, res.uv)
            results[meth] = res; metrics[meth] = M
            rows.append(dict(shape=m.name, family=m.meta.get("family"), developable=m.meta.get("developable"),
                             n_vertices=m.n_vertices, method=meth, time_s=res.time_s, iterations=res.iterations,
                             converged=res.converged, **M))
        # ---- uncertainty
        u = {"shape": m.name, "family": m.meta.get("family")}
        sp = method_spread(m, {k: results[k].uv for k in SPREAD_METHODS if k in results}, reference="arap")
        u["method_spread_rms_mm"] = sp["boundary_spread_rms_mm"]; u["method_spread_max_mm"] = sp["boundary_spread_max_mm"]
        for k, v in sp["per_method_max_mm"].items():
            u[f"spread_max_{k}_mm"] = v
        if do_refinement:
            u.update(refinement_sensitivity(m, "arap"))
        unc_rows.append(u)
        # ---- darts (ARAP)
        darts_result = None
        if do_darts:
            darts = recommend_darts(m)
            if darts:
                res_c, cut, applied, = flatten_with_darts(m, "arap", darts)
                Mc = distortion_metrics(cut, res_c.uv)
                darts_result = (res_c, cut, applied, Mc)
                for d in applied:
                    dart_rows.append(dict(shape=m.name, family=m.meta.get("family"), kind=d["kind"],
                                          angle_deg=d["angle_deg"], length_mm=d["length_mm"],
                                          n_darts_applied=len(applied),
                                          before_stretch_Linf=metrics["arap"]["stretch_Linf"], after_stretch_Linf=Mc["stretch_Linf"],
                                          before_stretch_min=metrics["arap"]["stretch_min"], after_stretch_min=Mc["stretch_min"],
                                          before_edge_p95_pct=metrics["arap"]["edge_rel_err_p95_abs_pct"], after_edge_p95_pct=Mc["edge_rel_err_p95_abs_pct"],
                                          before_suture_mismatch_mm=metrics["arap"]["boundary_cumulative_mismatch_mm"],
                                          after_suture_mismatch_mm=Mc["boundary_cumulative_mismatch_mm"],
                                          after_flips=Mc["flipped_triangles"]))
        # ---- figures
        if do_figures:
            viz.shape_figure(m, results, metrics, os.path.join(fig_dir, f"{m.name}.png"),
                             methods=[x for x in ["lscm", "arap", "membrane_stvk", "cetm", "membrane_suture"] if x in results],
                             darts_result=darts_result)
        # ---- templates
        if do_templates:
            tdir = os.path.join(out_dir, "templates", m.name); os.makedirs(tdir, exist_ok=True)
            for meth in TEMPLATE_METHODS:
                if meth not in results:
                    continue
                outline = export.template_outline(m, results[meth].uv)
                export.write_svg(os.path.join(tdir, f"{m.name}_{meth}.svg"), outline, f"{m.name} {meth}", suture_offset_mm=1.0)
                export.write_dxf(os.path.join(tdir, f"{m.name}_{meth}.dxf"), outline, suture_offset_mm=1.0)
                export.write_pdf(os.path.join(tdir, f"{m.name}_{meth}.pdf"), outline, f"{m.name} {meth}", suture_offset_mm=1.0)
                results[meth].as_mesh(m).save(os.path.join(tdir, f"{m.name}_{meth}_flat.vtp"))
            if darts_result:
                res_c, cut, applied, Mc = darts_result
                outline = export.template_outline(cut, res_c.uv)
                export.write_svg(os.path.join(tdir, f"{m.name}_arap_darted.svg"), outline, f"{m.name} arap darted",
                                 suture_offset_mm=1.0, darts=applied)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "results_long.csv"), index=False)
    pd.DataFrame(unc_rows).to_csv(os.path.join(out_dir, "uncertainty.csv"), index=False)
    if dart_rows:
        pd.DataFrame(dart_rows).to_csv(os.path.join(out_dir, "darts.csv"), index=False)
    # summary by family x method
    keys = [k for k in KEY_METRICS if k in df.columns and k != "passes_size_tolerance"]
    g = df.groupby(["family", "method"])
    summ = g[keys].median().add_suffix("_median").join(g[keys].max().add_suffix("_max"))
    summ["pass_rate_size_tol"] = g["passes_size_tolerance"].mean()
    summ["time_s_median"] = g["time_s"].median()
    summ.reset_index().to_csv(os.path.join(out_dir, "summary_by_family.csv"), index=False)
    if do_figures:
        ms = [x for x in viz.METHOD_ORDER if x in methods]
        viz.summary_figure(df, os.path.join(fig_dir, "summary_stretch_Linf.png"), "stretch_Linf", "max principal stretch σ₁", ms, hline=1.0)
        viz.summary_figure(df, os.path.join(fig_dir, "summary_edge_p95.png"), "edge_rel_err_p95_abs_pct", "edge length error p95 (%)", ms, log=True)
        viz.summary_figure(df, os.path.join(fig_dir, "summary_suture_mismatch.png"), "boundary_cumulative_mismatch_mm", "cumulative suture-line mismatch (mm)", ms, log=True)
        viz.summary_figure(df, os.path.join(fig_dir, "summary_area_error.png"), "area_total_error_pct", "total area error (%)", ms, hline=0.0)
        viz.summary_figure(df, os.path.join(fig_dir, "summary_time.png"), "time_s", "solve time (s)", ms, log=True)
        viz.scatter_vs_curvature(df[df["method"].isin(SPREAD_METHODS)], os.path.join(fig_dir, "edge_p95_vs_curvature.png"),
                                 "edge_rel_err_p95_abs_pct", "edge length error p95 (%)", SPREAD_METHODS)
        viz.scatter_vs_curvature(df[df["method"].isin(SPREAD_METHODS)], os.path.join(fig_dir, "suture_vs_curvature.png"),
                                 "boundary_cumulative_mismatch_mm", "cumulative suture-line mismatch (mm)", SPREAD_METHODS)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default="output/synthetic_library")
    ap.add_argument("--out", default="output/eval")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--no-darts", action="store_true"); ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--no-figures", action="store_true"); ap.add_argument("--no-templates", action="store_true")
    a = ap.parse_args()
    from .library import load_library
    shapes = load_library(a.library)
    run_benchmark(shapes, a.methods, a.out, not a.no_darts, not a.no_refine, not a.no_figures, not a.no_templates)


if __name__ == "__main__":
    main()
