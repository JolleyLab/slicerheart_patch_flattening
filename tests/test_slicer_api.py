"""The vtkPolyData entry point used by the SlicerHeart Baffle Planner."""
import numpy as np
import pytest
import vtk
from vtk.util import numpy_support as vnp

from slicerheart_patch_flattening.slicer_api import flatten_polydata, available_methods, closest_vertices
from slicerheart_patch_flattening.geometric import METHODS, flatten
from slicerheart_patch_flattening.mesh import TriMesh
from conftest import rigid_distance


def test_available_methods():
    assert available_methods() == sorted(METHODS)
    assert {"lscm", "arap", "cetm"} <= set(available_methods())


def test_closest_vertices(sphere_cap):
    pts = sphere_cap.V[[3, 40]] + 0.01
    assert closest_vertices(sphere_cap, pts).tolist() == [3, 40]


@pytest.mark.parametrize("method", ["lscm", "arap", "cetm"])
def test_roundtrip(half_cylinder, method):
    pd = half_cylinder.to_vtk()
    out, res, M = flatten_polydata(pd, method)
    assert out.GetNumberOfPoints() == pd.GetNumberOfPoints() and out.GetNumberOfCells() == pd.GetNumberOfCells()
    for name in ("stretch_s1", "stretch_s2", "area_ratio"):
        arr = out.GetPointData().GetArray(name)
        assert arr is not None and arr.GetNumberOfTuples() == pd.GetNumberOfPoints()
        assert np.isfinite(vnp.vtk_to_numpy(arr)).all()
    assert np.abs(vnp.vtk_to_numpy(out.GetPointData().GetArray("stretch_s1")) - 1).max() < 0.01   # developable
    P = vnp.vtk_to_numpy(out.GetPoints().GetData())
    assert np.all(P[:, 2] == 0) and np.allclose(P[:, :2], res.uv)
    assert M["flipped_triangles"] == 0 and M["passes_size_tolerance"]
    assert out.GetFieldData().GetArray("fixed_point_ids") is None
    assert res.method == method


def test_fixed_points_orient_template(half_cylinder):
    """With two fixed points (the surgeon's landmarks): first at the origin, second on the +u axis."""
    i0, i1 = 0, half_cylinder.n_vertices // 2
    fixed = half_cylinder.V[[i0, i1]]
    for method in ("lscm", "arap"):
        out, res, M = flatten_polydata(half_cylinder.to_vtk(), method, fixed_points_world=fixed)
        assert np.allclose(res.uv[i0], 0, atol=1e-9)
        assert abs(res.uv[i1][1]) < 1e-9 and res.uv[i1][0] > 0
        ids = vnp.vtk_to_numpy(out.GetFieldData().GetArray("fixed_point_ids"))
        assert ids.tolist() == [i0, i1]
        # the orientation does not change the template itself (for LSCM the fixed points are also the
        # pins, which on a developable surface moves the solution by far less than the chord error)
        assert rigid_distance(res.uv, flatten(half_cylinder, method).uv) < (1e-3 if method == "lscm" else 1e-6)


def test_lscm_fixed_points_are_pins(sphere_cap):
    loop = sphere_cap.boundary_loop()
    pins = [int(loop[0]), int(loop[len(loop) // 2])]
    _, res, _ = flatten_polydata(sphere_cap.to_vtk(), "lscm", fixed_points_world=sphere_cap.V[pins])
    assert res.params["pins"] == pins


def test_without_metrics(sphere_cap):
    out, res, M = flatten_polydata(sphere_cap.to_vtk(), "arap", with_metrics=False)
    assert M is None and out.GetNumberOfPoints() == sphere_cap.n_vertices


def test_accepts_non_triangle_polydata(sphere_cap):
    """Quads or polygons from a segmentation are triangulated on the way in."""
    V = sphere_cap.V
    F = sphere_cap.F
    pts = vtk.vtkPoints()
    pts.SetData(vnp.numpy_to_vtk(np.ascontiguousarray(V), deep=True))
    cells = vtk.vtkCellArray()
    # merge the first two faces that share an edge into one quad, keep the rest as triangles
    shared = next((a, b) for a in range(len(F)) for b in range(a + 1, len(F)) if len(set(F[a]) & set(F[b])) == 2)
    a, b = shared
    common = list(set(F[a]) & set(F[b]))
    only_a = (set(F[a]) - set(common)).pop()
    only_b = (set(F[b]) - set(common)).pop()
    cells.InsertNextCell(4, [int(only_a), int(common[0]), int(only_b), int(common[1])])
    for i, f in enumerate(F):
        if i not in (a, b):
            cells.InsertNextCell(3, [int(x) for x in f])
    pd = vtk.vtkPolyData(); pd.SetPoints(pts); pd.SetPolys(cells)
    out, res, M = flatten_polydata(pd, "arap")
    assert out.GetNumberOfCells() == sphere_cap.n_faces and M["flipped_triangles"] == 0


def test_unknown_method(sphere_cap):
    with pytest.raises(KeyError):
        flatten_polydata(sphere_cap.to_vtk(), "no_such_method")
