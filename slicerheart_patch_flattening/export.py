"""Export a flattened template for printing or laser cutting, at 1:1 scale in mm.

Formats
  SVG  - vector outline with mm units (viewBox in mm), optional suture-offset outline,
         dart lines, orientation marks and a 10 mm scale bar. Opens in Inkscape /
         browsers / most laser-cutter front ends.
  DXF  - minimal ASCII DXF (R12-style LWPOLYLINE on a layer per element, $INSUNITS=4 mm),
         the usual laser-cutter input.
  PDF  - matplotlib page at exact physical size for a paper printer (print at 100 %).

The suture-bite offset (Kizilski et al. 2025 used 1-1.25 mm for 7-0 Prolene) is a
uniform outward offset of the outline computed with shapely.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from .mesh import TriMesh


def template_outline(mesh: TriMesh, uv: np.ndarray) -> np.ndarray:
    """Ordered 2D boundary polygon of the flattened template."""
    return uv[mesh.boundary_loop()]


def offset_outline(outline: np.ndarray, offset_mm: float) -> np.ndarray:
    if offset_mm == 0:
        return outline
    poly = Polygon(outline).buffer(offset_mm, join_style=1)
    return np.asarray(poly.exterior.coords)[:-1]


def _path_d(P: np.ndarray, close: bool = True) -> str:
    s = "M " + " L ".join(f"{x:.3f},{y:.3f}" for x, y in P)
    return s + (" Z" if close else "")


def write_svg(path: str, outline: np.ndarray, title: str = "", suture_offset_mm: float = 0.0,
              darts: list[dict] | None = None, marks: dict[str, np.ndarray] | None = None,
              margin_mm: float = 10.0):
    """SVG in millimetres; y is flipped so the template reads like the 3D view from above."""
    O = outline.copy(); O[:, 1] *= -1
    elems = [O]
    off = None
    if suture_offset_mm:
        off = offset_outline(outline, suture_offset_mm); off[:, 1] *= -1; elems.append(off)
    allp = np.vstack(elems)
    mn = allp.min(0) - margin_mm; mx = allp.max(0) + margin_mm
    w, h = mx - mn
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.2f}mm" height="{h:.2f}mm" '
             f'viewBox="{mn[0]:.3f} {mn[1]:.3f} {w:.3f} {h:.3f}">',
             f'<title>{title}</title>',
             f'<path d="{_path_d(O)}" fill="none" stroke="black" stroke-width="0.25"/>']
    if off is not None:
        lines.append(f'<path d="{_path_d(off)}" fill="none" stroke="red" stroke-width="0.25" stroke-dasharray="1,1"/>')
    for d in (darts or []):
        for side in range(2):
            pts = np.array([p[min(side, len(p) - 1)] for p in d["uv_path"]]); pts[:, 1] *= -1
            lines.append(f'<path d="{_path_d(pts, False)}" fill="none" stroke="blue" stroke-width="0.25"/>')
        apex = np.array(d["uv_path"][-1][0]); apex[1] *= -1
        lines.append(f'<text x="{apex[0]:.2f}" y="{apex[1]:.2f}" font-size="2.5" fill="blue">{d["kind"]} {abs(d["angle_deg"]):.0f}&#176;</text>')
    for name, p in (marks or {}).items():
        x, y = p[0], -p[1]
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="0.8" fill="none" stroke="green" stroke-width="0.25"/>')
        lines.append(f'<text x="{x+1.2:.2f}" y="{y-1.2:.2f}" font-size="2.5" fill="green">{name}</text>')
    # scale bar 10 mm and label
    x0, y0 = mn[0] + 2, mx[1] - 3
    lines.append(f'<path d="M {x0:.2f},{y0:.2f} L {x0+10:.2f},{y0:.2f}" stroke="black" stroke-width="0.4"/>')
    lines.append(f'<text x="{x0:.2f}" y="{y0-1:.2f}" font-size="2.5">10 mm  {title}</text>')
    lines.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_dxf(path: str, outline: np.ndarray, suture_offset_mm: float = 0.0, darts: list[dict] | None = None):
    """Minimal DXF with closed LWPOLYLINE entities (mm)."""
    def lwpoly(P, layer, closed=True):
        s = ["0", "LWPOLYLINE", "8", layer, "90", str(len(P)), "70", "1" if closed else "0"]
        for x, y in P:
            s += ["10", f"{x:.4f}", "20", f"{y:.4f}"]
        return s
    body = ["0", "SECTION", "2", "HEADER", "9", "$INSUNITS", "70", "4", "0", "ENDSEC",
            "0", "SECTION", "2", "ENTITIES"]
    body += lwpoly(outline, "CUT")
    if suture_offset_mm:
        body += lwpoly(offset_outline(outline, suture_offset_mm), "SUTURE_OFFSET")
    for d in (darts or []):
        for side in range(2):
            pts = np.array([p[min(side, len(p) - 1)] for p in d["uv_path"]])
            body += lwpoly(pts, "DART", closed=False)
    body += ["0", "ENDSEC", "0", "EOF"]
    with open(path, "w") as f:
        f.write("\n".join(body) + "\n")


def write_pdf(path: str, outline: np.ndarray, title: str = "", suture_offset_mm: float = 0.0,
              darts: list[dict] | None = None, margin_mm: float = 15.0):
    """Exact-size PDF: figure size = bounding box in inches, data units = mm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mn = outline.min(0) - margin_mm; mx = outline.max(0) + margin_mm
    w, h = (mx - mn) / 25.4
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1]); ax.set_aspect("equal"); ax.axis("off")
    O = np.vstack([outline, outline[:1]])
    ax.plot(O[:, 0], O[:, 1], "k-", lw=0.6)
    if suture_offset_mm:
        Q = offset_outline(outline, suture_offset_mm); Q = np.vstack([Q, Q[:1]])
        ax.plot(Q[:, 0], Q[:, 1], "r--", lw=0.5)
    for d in (darts or []):
        for side in range(2):
            pts = np.array([p[min(side, len(p) - 1)] for p in d["uv_path"]])
            ax.plot(pts[:, 0], pts[:, 1], "b-", lw=0.5)
    ax.plot([mn[0] + 3, mn[0] + 13], [mn[1] + 3, mn[1] + 3], "k-", lw=1.2)
    ax.text(mn[0] + 3, mn[1] + 4.5, f"10 mm   {title}", fontsize=6)
    fig.savefig(path)
    plt.close(fig)
