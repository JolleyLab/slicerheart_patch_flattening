"""Thin VTK-level interface matching what the SlicerHeart Baffle Planner needs.

The Baffle Planner (BafflePlanner.py, `flattenOutputBaffleModel`) currently
calls the ConformalTextureMapping CLI (Levy's LSCM) with the thin baffle
surface and two user "fixed points", then rescales the result by
sqrt(area3D / area2D) and copies the two fixed points onto the flattened model.

`flatten_polydata` provides the same contract without Slicer: an open
vtkPolyData in, a vtkPolyData with identical connectivity and z = 0 out, in
true millimetres, plus the vertex ids of the fixed points. The module has no
Slicer dependency; inside Slicer the caller passes
``modelNode.GetPolyData()`` and world coordinates of the fixed points.

Only the patch surface is considered: the heart model, clipping and markups
stay in the Baffle Planner.
"""
from __future__ import annotations

import numpy as np
import vtk
from vtk.util import numpy_support as vnp

from .mesh import TriMesh
from .geometric.methods import flatten, METHODS
from .geometric.common import FlatResult
from .metrics import distortion_metrics, per_vertex_stretch


def closest_vertices(mesh: TriMesh, points_world: np.ndarray) -> np.ndarray:
    """Vertex ids nearest to each world point (same rule as the Baffle Planner)."""
    P = np.atleast_2d(points_world)
    d = np.linalg.norm(mesh.V[None] - P[:, None], axis=2)
    return d.argmin(axis=1)


def flatten_polydata(polydata: vtk.vtkPolyData, method: str = "arap",
                     fixed_points_world: np.ndarray | None = None,
                     with_metrics: bool = True, **kw):
    """Flatten an open surface.

    Returns (flat_polydata, result, metrics). `flat_polydata` carries point data
    arrays 'stretch_s1', 'stretch_s2', 'area_ratio' for display in Slicer
    (e.g. colour the flattened model by local stretch), and a field array
    'fixed_point_ids' when fixed points were given.

    For `method='lscm'` the fixed points are used as the LSCM pins, exactly like
    the current CLI; for the other methods they only define the template's
    orientation (first point at the origin, second on the +u axis) so that the
    printed template keeps the surgeon's chosen landmarks.
    """
    mesh = TriMesh.from_vtk(polydata, name="baffle").orient_consistently()
    pins = None
    if fixed_points_world is not None:
        pins = closest_vertices(mesh, np.asarray(fixed_points_world, dtype=float))[:2]
    if method == "lscm" and pins is not None and len(pins) == 2:
        res: FlatResult = flatten(mesh, "lscm", pins=pins, **kw)
    else:
        res = flatten(mesh, method, **kw)
    uv = res.uv
    if pins is not None and len(pins) >= 2:
        # orient: pin0 at origin, pin1 along +u
        uv = uv - uv[pins[0]]
        d = uv[pins[1]]
        ang = np.arctan2(d[1], d[0])
        c, s = np.cos(-ang), np.sin(-ang)
        uv = uv @ np.array([[c, -s], [s, c]]).T
        res.uv = uv
    out = res.as_mesh(mesh).to_vtk()
    pv = per_vertex_stretch(mesh, uv)
    for k in ("s1", "s2", "area_ratio"):
        arr = vnp.numpy_to_vtk(np.ascontiguousarray(pv[k]), deep=True)
        arr.SetName("stretch_" + k if k != "area_ratio" else k)
        out.GetPointData().AddArray(arr)
    if pins is not None:
        ids = vnp.numpy_to_vtkIdTypeArray(np.ascontiguousarray(pins.astype(np.int64)), deep=True)
        ids.SetName("fixed_point_ids")
        out.GetFieldData().AddArray(ids)
    metrics = distortion_metrics(mesh, uv) if with_metrics else None
    return out, res, metrics


def available_methods() -> list[str]:
    return sorted(METHODS)
