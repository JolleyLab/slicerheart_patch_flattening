"""Template export (SVG, DXF, PDF) and the recommended-template pipeline."""
import os
import re

import numpy as np

from slicerheart_patch_flattening import export, flatten_recommended
from slicerheart_patch_flattening.geometric import flatten
from slicerheart_patch_flattening.geometric.darts import recommend_darts, flatten_with_darts
from slicerheart_patch_flattening.mesh import TriMesh


def polygon_area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_perimeter(P):
    return np.linalg.norm(np.roll(P, -1, 0) - P, axis=1).sum()


def test_template_outline(planar_disk):
    outline = export.template_outline(planar_disk, planar_disk.V[:, :2])
    assert len(outline) == len(planar_disk.boundary_loop())
    assert abs(polygon_perimeter(outline) - planar_disk.perimeter()) < 1e-9


def test_offset_outline(planar_disk):
    """A seam allowance of d around a convex outline adds 2 pi d to the perimeter and
    d * perimeter + pi d^2 to the area."""
    outline = export.template_outline(planar_disk, planar_disk.V[:, :2])
    d = 1.5
    off = export.offset_outline(outline, d)
    assert off.shape[1] == 2 and len(off) >= len(outline)
    assert abs(polygon_perimeter(off) - polygon_perimeter(outline) - 2 * np.pi * d) < 0.02 * 2 * np.pi * d
    expected = polygon_area(outline) + d * polygon_perimeter(outline) + np.pi * d**2
    assert abs(polygon_area(off) - expected) < 0.01 * expected
    assert export.offset_outline(outline, 0.0) is outline


def test_write_formats(tmp_path, sphere_cap):
    uv = flatten(sphere_cap, "arap").uv
    outline = export.template_outline(sphere_cap, uv)
    darts = recommend_darts(sphere_cap, max_darts=1)
    _, _, applied = flatten_with_darts(sphere_cap, "arap", darts)
    marks = {"A": uv[0], "B": uv[10]}
    export.write_svg(str(tmp_path / "t.svg"), outline, "cap", suture_offset_mm=1.0, darts=applied, marks=marks)
    export.write_dxf(str(tmp_path / "t.dxf"), outline, suture_offset_mm=1.0, darts=applied)
    export.write_pdf(str(tmp_path / "t.pdf"), outline, "cap", suture_offset_mm=1.0, darts=applied)

    svg = open(tmp_path / "t.svg").read()
    assert svg.count("<path") >= 4 and "<title>cap</title>" in svg and "10 mm" in svg
    # the SVG is in true millimetres: its width is the outline's width plus the margins
    w = float(re.search(r'width="([\d.]+)mm"', svg).group(1))
    off = export.offset_outline(outline, 1.0)
    assert abs(w - (off[:, 0].max() - off[:, 0].min() + 20.0)) < 0.1
    assert "dart" in svg and "A</text>" in svg

    dxf = open(tmp_path / "t.dxf").read().split("\n")
    assert dxf.count("LWPOLYLINE") == 4                          # cut, offset, two dart sides
    assert "CUT" in dxf and "SUTURE_OFFSET" in dxf and "DART" in dxf
    assert dxf[dxf.index("$INSUNITS") + 2] == "4"                # units: millimetres
    n_vertices = int(dxf[dxf.index("LWPOLYLINE") + 4])
    assert n_vertices == len(outline)

    assert os.path.getsize(tmp_path / "t.pdf") > 500
    assert open(tmp_path / "t.pdf", "rb").read(5) == b"%PDF-"


def test_pipeline_recommended_curved(sphere_cap):
    r = flatten_recommended(sphere_cap, suture_exact=True)
    assert r.method == "arap" and r.metrics["flipped_triangles"] == 0
    assert r.needs_darts and r.darted_metrics is not None
    assert r.darted_metrics["boundary_cumulative_mismatch_mm"] < r.metrics["boundary_cumulative_mismatch_mm"]
    assert isinstance(r.darted_mesh, TriMesh) and r.darted_mesh.n_vertices > sphere_cap.n_vertices
    assert r.suture_exact_result.method == "cetm"
    assert r.suture_exact_metrics["boundary_edge_rel_err_max_abs_pct"] < 0.1
    s = r.summary()
    assert "suture-line mismatch" in s and "darts proposed" in s and "size check" in s


def test_pipeline_recommended_developable(half_cylinder):
    r = flatten_recommended(half_cylinder)
    assert not r.needs_darts and r.darted_result is None and r.suture_exact_result is None
    assert r.metrics["passes_size_tolerance"]
    assert "pass" in r.summary() and "darts" not in r.summary()


def test_pipeline_other_method(sphere_cap):
    r = flatten_recommended(sphere_cap, method="lscm", dart_threshold_mm=1e6)
    assert r.method == "lscm" and r.result.method == "lscm" and not r.needs_darts
