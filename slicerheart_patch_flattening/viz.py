"""Figures for the evaluation report (matplotlib, headless).

Colour rules: stretch maps use a diverging two-hue scale centred on 1.0
(compression blue, tension red, neutral at 1); methods use the Okabe-Ito
colour-blind-safe categorical palette in a fixed order; shape curvature uses
the same diverging scale centred on zero (saddle vs dome).
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.tri import Triangulation

from .mesh import TriMesh
from .metrics import per_vertex_stretch

OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]
METHOD_ORDER = ["lscm", "arap", "membrane_stvk", "symmetric_dirichlet", "cetm", "membrane_suture", "asap",
                "arap_hybrid", "spring", "lscm_edge", "lscm_perimeter", "bff_lite", "exp_map",
                "harmonic", "mean_value", "tutte", "pca_projection"]


def method_color(name: str) -> str:
    if name in METHOD_ORDER[:6]:
        return OKABE_ITO[METHOD_ORDER.index(name)]
    return "#aaaaaa"


def plot_shape_3d(ax, mesh: TriMesh, scalar=None, title="", vmax=None):
    K = mesh.gaussian_curvature() if scalar is None else scalar
    if vmax is None:
        vmax = max(np.percentile(np.abs(K), 98), 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    col = plt.cm.RdBu_r(norm(K[mesh.F].mean(axis=1)))
    p = ax.plot_trisurf(mesh.V[:, 0], mesh.V[:, 1], mesh.V[:, 2], triangles=mesh.F, linewidth=0.05,
                        edgecolor=(0, 0, 0, 0.15), antialiased=True, shade=False)
    p.set_facecolor(col)
    c = mesh.V.mean(0); r = (mesh.V.max(0) - mesh.V.min(0)).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_axis_off(); ax.set_title(title, fontsize=9)


def plot_template(ax, mesh: TriMesh, uv: np.ndarray, title: str = "", field: str = "s1",
                  vlim: float = 0.15, darts: list[dict] | None = None, show_mesh: bool = False):
    """2D template coloured by local stretch (sigma_1 or area ratio) on a diverging scale about 1."""
    pv = per_vertex_stretch(mesh, uv)
    val = pv[field]
    tri = Triangulation(uv[:, 0], uv[:, 1], mesh.F)
    norm = TwoSlopeNorm(vmin=1 - vlim, vcenter=1.0, vmax=1 + vlim)
    tc = ax.tripcolor(tri, np.clip(val, 1 - vlim, 1 + vlim), cmap="RdBu_r", norm=norm, shading="gouraud")
    if show_mesh:
        ax.triplot(tri, color=(0, 0, 0, 0.12), lw=0.3)
    loop = mesh.boundary_loop(); B = uv[np.r_[loop, loop[0]]]
    ax.plot(B[:, 0], B[:, 1], "k-", lw=0.8)
    for d in (darts or []):
        for side in range(2):
            pts = np.array([p[min(side, len(p) - 1)] for p in d["uv_path"]])
            ax.plot(pts[:, 0], pts[:, 1], "-", color="#000000", lw=1.0)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7); ax.set_xlabel("mm", fontsize=7)
    return tc


def shape_figure(mesh: TriMesh, results: dict, metrics: dict, path: str, field: str = "s1",
                 methods: list[str] | None = None, darts_result=None):
    """One row: 3D shape coloured by Gaussian curvature, then each method's template."""
    methods = methods or [m for m in METHOD_ORDER if m in results]
    ncol = len(methods) + 1 + (1 if darts_result else 0)
    fig = plt.figure(figsize=(3.2 * ncol, 3.6))
    ax = fig.add_subplot(1, ncol, 1, projection="3d")
    md = next(iter(metrics.values()))
    plot_shape_3d(ax, mesh, title=f"{mesh.name}\n|K| total {md['shape_total_abs_K_deg']:.0f}°, area {md['shape_area_mm2']:.0f} mm²")
    tc = None
    for i, m in enumerate(methods):
        axi = fig.add_subplot(1, ncol, i + 2)
        M = metrics[m]
        ttl = (f"{m}\nσmax {M['stretch_Linf']:.3f}  σmin {M['stretch_min']:.3f}\n"
               f"edge p95 {M['edge_rel_err_p95_abs_pct']:.1f}%  suture Δ {M['boundary_cumulative_mismatch_mm']:.2f} mm")
        tc = plot_template(axi, mesh, results[m].uv, ttl, field=field)
    if darts_result:
        res_c, cut, darts, Mc = darts_result
        axd = fig.add_subplot(1, ncol, ncol)
        ttl = (f"arap + {len(darts)} dart/gusset\nσmax {Mc['stretch_Linf']:.3f}  σmin {Mc['stretch_min']:.3f}\n"
               f"edge p95 {Mc['edge_rel_err_p95_abs_pct']:.1f}%  suture Δ {Mc['boundary_cumulative_mismatch_mm']:.2f} mm")
        tc = plot_template(axd, cut, res_c.uv, ttl, field=field, darts=darts)
    if tc is not None:
        cb = fig.colorbar(tc, ax=fig.axes[1:], fraction=0.015, pad=0.01)
        cb.set_label("σ₁ (principal stretch, 1 = isometric)" if field == "s1" else "area ratio", fontsize=8)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def summary_figure(df, path: str, metric: str, ylabel: str, methods: list[str], log: bool = False,
                   hline: float | None = None):
    """Per-family box plots of one metric, one box per method, fixed colours."""
    families = list(dict.fromkeys(df["family"]))
    fig, axes = plt.subplots(1, len(families), figsize=(3.0 * len(families), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, families):
        sub = df[df["family"] == fam]
        data = [sub[sub["method"] == m][metric].values for m in methods]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True,
                        medianprops=dict(color="black", lw=1.2), flierprops=dict(markersize=3))
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(method_color(m)); patch.set_alpha(0.75); patch.set_linewidth(0.6)
        ax.set_xticks(range(1, len(methods) + 1)); ax.set_xticklabels(methods, rotation=60, ha="right", fontsize=7)
        ax.set_title(fam, fontsize=9); ax.grid(axis="y", color="#dddddd", lw=0.5); ax.set_axisbelow(True)
        if hline is not None:
            ax.axhline(hline, color="#555555", lw=0.8, ls="--")
        if log:
            ax.set_yscale("log")
    axes[0].set_ylabel(ylabel, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def scatter_vs_curvature(df, path: str, metric: str, ylabel: str, methods: list[str]):
    """Metric vs shape developability (total |K|), one series per method."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m in methods:
        sub = df[df["method"] == m]
        ax.scatter(sub["shape_total_abs_K_deg"], sub[metric], s=18, color=method_color(m), label=m, alpha=0.85)
    ax.set_xlabel("shape total |Gaussian curvature| (degrees)"); ax.set_ylabel(ylabel)
    ax.grid(color="#dddddd", lw=0.5); ax.set_axisbelow(True); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
