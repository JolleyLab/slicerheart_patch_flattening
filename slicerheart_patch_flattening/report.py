"""Benchmark report: the "Results of the benchmark" tables as Markdown and a multi-page PDF.

Consumes the outputs of evaluate.run_benchmark (results_long.csv, uncertainty.csv,
darts.csv, figures/) and writes report.md and report.pdf. The PDF is built with
matplotlib only (no LaTeX / weasyprint), so it runs on a bare CI runner and can be
attached to a GitHub release.
"""
from __future__ import annotations

import datetime as _dt
import os
import platform
import subprocess
import textwrap

import numpy as np
import pandas as pd

CLINICAL_FAMILIES = ["dorv_tunnel", "baffle_planner_tps", "papvr"]
TABLE_METHODS = ["lscm", "lscm_perimeter", "asap", "arap", "arap_hybrid", "symmetric_dirichlet",
                 "membrane_stvk", "spring", "membrane_suture", "bff_lite", "cetm", "exp_map",
                 "harmonic", "mean_value", "tutte", "pca_projection", "lscm_edge"]
COLS = [("stretch_Linf", "max stretch", "{:.3f}"), ("stretch_min", "min stretch", "{:.3f}"),
        ("edge_rel_err_p95_abs_pct", "edge err p95 (%)", "{:.1f}"),
        ("boundary_edge_rel_err_max_abs_pct", "boundary edge err max (%)", "{:.1f}"),
        ("boundary_cumulative_mismatch_mm", "suture mismatch (mm)", "{:.2f}"),
        ("angle_err_deg_p95", "angle err p95 (deg)", "{:.1f}"),
        ("area_total_error_pct", "area err (%)", "{:+.1f}"),
        ("time_s", "median time (s)", "{:.2f}")]


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or "n/a"
    except Exception:
        return "n/a"


def _versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "scipy", "vtk", "shapely", "pygeodesic", "igl", "warp", "torch"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "present") if mod != "vtk" else m.vtkVersion.GetVTKVersion()
        except Exception:
            out[mod] = "absent"
    return out


def summary_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Median tables per method: clinical non-developable shapes, developable controls (worst case), per family."""
    methods = [m for m in TABLE_METHODS if m in set(df["method"])]
    tables = {}
    clin = df[df["family"].isin(CLINICAL_FAMILIES) & (~df["developable"].astype(bool))]
    if len(clin):
        g = clin.groupby("method")
        t = pd.DataFrame({label: g[col].median() for col, label, _ in COLS})
        t["shapes with folds"] = g["flipped_triangles"].apply(lambda x: int((x > 0).sum()))
        t["n shapes"] = g.size()
        tables[f"Median over the {clin['shape'].nunique()} non-developable clinical-type shapes ({', '.join(CLINICAL_FAMILIES)})"] = t.loc[methods]
    dev = df[df["developable"].astype(bool)]
    if len(dev):
        g = dev.groupby("method")
        t = pd.DataFrame({"max edge err (%)": g["edge_rel_err_max_abs_pct"].max(),
                          "max suture mismatch (mm)": g["boundary_cumulative_mismatch_mm"].max(),
                          "size-tolerance pass rate": g["passes_size_tolerance"].mean(),
                          "folded triangles (total)": g["flipped_triangles"].sum()})
        tables[f"Developable shapes ({dev['shape'].nunique()}), worst case over shapes"] = t.loc[methods]
    for fam, sub in df.groupby("family"):
        g = sub.groupby("method")
        t = pd.DataFrame({label: g[col].median() for col, label, _ in COLS[:5]})
        tables[f"Family {fam} ({sub['shape'].nunique()} shapes), medians"] = t.loc[[m for m in methods if m in g.groups]]
    return tables


def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    fmt = {label: f for _, label, f in COLS}
    fmt.update({"size-tolerance pass rate": "{:.2f}", "max edge err (%)": "{:.2f}", "max suture mismatch (mm)": "{:.3f}"})
    out = df.copy()
    for c in out.columns:
        if c in fmt:
            out[c] = out[c].map(lambda v: fmt[c].format(v))
        elif out[c].dtype.kind == "f":
            out[c] = out[c].map(lambda v: f"{v:.3g}")
    return out


def write_report(eval_dir: str, out_md: str | None = None, out_pdf: str | None = None, title: str | None = None,
                 n_shape_figures: int = 6) -> tuple[str, str]:
    df = pd.read_csv(os.path.join(eval_dir, "results_long.csv"))
    unc = pd.read_csv(os.path.join(eval_dir, "uncertainty.csv")) if os.path.exists(os.path.join(eval_dir, "uncertainty.csv")) else None
    darts = pd.read_csv(os.path.join(eval_dir, "darts.csv")) if os.path.exists(os.path.join(eval_dir, "darts.csv")) else None
    out_md = out_md or os.path.join(eval_dir, "report.md")
    out_pdf = out_pdf or os.path.join(eval_dir, "report.pdf")
    n_methods, n_shapes = df["method"].nunique(), df["shape"].nunique()
    title = title or f"Results of the benchmark (synthetic library, {n_shapes} shapes, all {n_methods} methods)"
    meta = dict(date=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), commit=_git_rev(), **_versions())
    tables = summary_tables(df)

    # ------------------------------------------------------------------ markdown
    lines = [f"# {title}", "",
             f"Generated {meta['date']}, commit {meta['commit']}, Python {meta['python']}, numpy {meta['numpy']}, scipy {meta['scipy']}, VTK {meta['vtk']}, warp {meta['warp']}.",
             f"Total solve time {df['time_s'].sum():.0f} s for {len(df)} (shape, method) runs.", ""]
    for name, t in tables.items():
        lines += [f"## {name}", "", _fmt(t).to_markdown(disable_numparse=True), ""]
    if unc is not None and len(unc):
        lines += ["## Uncertainty", "",
                  f"Boundary disagreement between the isometric/conformal free-boundary methods after rigid alignment: "
                  f"median RMS {unc['method_spread_rms_mm'].median():.3f} mm, max {unc['method_spread_max_mm'].max():.3f} mm over shapes.", ""]
        if "refine_boundary_max_mm" in unc:
            lines += [f"Outline change under one mesh subdivision (ARAP): max {unc['refine_boundary_max_mm'].max():.4f} mm.", ""]
    if darts is not None and len(darts):
        d1 = darts.groupby("shape").first()
        lines += ["## Darting (ARAP, recommended darts/gussets)", "",
                  f"{len(d1)} shapes received darts. Cumulative suture-line mismatch before -> after: "
                  f"median {d1['before_suture_mismatch_mm'].median():.2f} -> {d1['after_suture_mismatch_mm'].median():.2f} mm; "
                  f"fold-overs after cutting: {int(d1['after_flips'].sum())}.", ""]
    fig_dir = os.path.join(eval_dir, "figures")
    figs = []
    if os.path.isdir(fig_dir):
        summary = [f for f in ("library_gallery.png", "summary_edge_p95.png", "summary_suture_mismatch.png",
                               "summary_stretch_Linf.png", "summary_area_error.png", "summary_time.png",
                               "edge_p95_vs_curvature.png", "suture_vs_curvature.png") if os.path.exists(os.path.join(fig_dir, f))]
        shapes = sorted(f for f in os.listdir(fig_dir) if f.endswith(".png") and not f.startswith("summary") and "vs_curvature" not in f and f != "library_gallery.png")
        figs = summary + shapes[:n_shape_figures]
        lines += ["## Figures", ""] + [f"![{f}](figures/{f})" for f in figs] + [""]
    with open(out_md, "w") as f:
        f.write("\n".join(lines))

    # ------------------------------------------------------------------ pdf
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out_pdf) as pdf:
        fig = plt.figure(figsize=(11, 8.5)); fig.text(0.05, 0.9, title, fontsize=16, weight="bold")
        body = "\n".join(textwrap.wrap(lines[2], 120)) + "\n" + lines[3]
        fig.text(0.05, 0.84, body, fontsize=9, va="top")
        y = 0.72
        for name, t in tables.items():
            if y < 0.15:
                pdf.savefig(fig); plt.close(fig)
                fig = plt.figure(figsize=(11, 8.5)); y = 0.92
            ft = _fmt(t).reset_index().rename(columns={"method": "method", "index": "method"})
            fig.text(0.05, y, name, fontsize=11, weight="bold"); y -= 0.02
            h = 0.022 * (len(ft) + 1)
            ax = fig.add_axes([0.05, max(y - h, 0.02), 0.9, h]); ax.axis("off")
            tb = ax.table(cellText=ft.values, colLabels=ft.columns, loc="upper left", cellLoc="center")
            tb.auto_set_font_size(False); tb.set_fontsize(7); tb.scale(1, 1.1)
            y -= h + 0.05
        pdf.savefig(fig); plt.close(fig)
        for f in figs:
            img = plt.imread(os.path.join(fig_dir, f))
            fh, fw = img.shape[:2]
            fig = plt.figure(figsize=(11, 8.5)); ax = fig.add_axes([0.03, 0.05, 0.94, 0.88]); ax.imshow(img); ax.axis("off")
            fig.text(0.05, 0.95, f, fontsize=10); pdf.savefig(fig); plt.close(fig)
        d = pdf.infodict(); d["Title"] = title; d["Subject"] = f"slicerheart_patch_flattening benchmark, commit {meta['commit']}"
    return out_md, out_pdf
