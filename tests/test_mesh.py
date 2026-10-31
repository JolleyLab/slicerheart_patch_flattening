"""TriMesh: topology, discrete curvature, I/O, geodesics."""
import numpy as np
import pytest

from slicerheart_patch_flattening.mesh import TriMesh
from conftest import euler_characteristic


def test_is_a_disk(analytic_shape):
    """Every test shape is a topological disk with consistently oriented, non-degenerate faces."""
    name, m = analytic_shape
    assert euler_characteristic(m) == 1, name
    loop = m.boundary_loop()
    assert len(loop) == len(m.boundary_edges()) == len(np.unique(loop))
    assert m.check_orientation()
    assert (m.face_areas() > 1e-6).all()
    assert m.is_boundary_vertex().sum() == len(loop)
    assert m.remove_unreferenced().n_vertices == m.n_vertices   # no stray vertices


def test_boundary_loop_is_closed_edge_path(sphere_cap):
    loop = sphere_cap.boundary_loop()
    edges = {tuple(sorted(e)) for e in sphere_cap.boundary_edges()}
    for a, b in zip(loop, np.roll(loop, -1)):
        assert tuple(sorted((a, b))) in edges


def test_gauss_bonnet(analytic_shape):
    """Sum of angle defects (interior defects + boundary geodesic curvature) = 2 pi chi, chi = 1 for a disk."""
    name, m = analytic_shape
    assert abs(m.angle_defects().sum() - 2 * np.pi) < 1e-9, name


def test_sphere_cap_curvature(sphere_cap):
    """Integrated angle defect over the interior = (interior area) / R^2 for R = 10."""
    interior = ~sphere_cap.is_boundary_vertex()
    K_int = sphere_cap.angle_defects()[interior].sum()
    A_int = sphere_cap.vertex_areas()[interior].sum()
    assert abs(K_int - A_int / 100.0) / K_int < 0.05
    assert (sphere_cap.gaussian_curvature()[interior] > 0).all()


def test_saddle_has_negative_curvature(saddle):
    interior = ~saddle.is_boundary_vertex()
    K = saddle.gaussian_curvature()[interior]
    assert np.median(K) < 0 and (K < 0).mean() > 0.9


def test_developable_shapes_are_flat(half_cylinder, cone):
    """The polyhedral approximation of a developable surface has (almost) no angle defect."""
    for m in (half_cylinder, cone):
        interior = ~m.is_boundary_vertex()
        assert np.rad2deg(np.abs(m.angle_defects()[interior]).sum()) < 10.0
        assert np.rad2deg(np.abs(m.angle_defects()[interior]).max()) < 0.5


def test_areas_and_lengths(planar_disk):
    assert abs(planar_disk.area() - np.pi * 100) / (np.pi * 100) < 0.01       # inscribed polygon
    assert abs(planar_disk.face_areas().sum() - planar_disk.area()) < 1e-9
    assert abs(planar_disk.perimeter() - 2 * np.pi * 10) / (2 * np.pi * 10) < 0.01
    e = planar_disk.edges()
    assert len(e) == (3 * planar_disk.n_faces + len(planar_disk.boundary_edges())) // 2
    assert np.array_equal(e, np.sort(e, axis=1)) and len(np.unique(e, axis=0)) == len(e)
    el = planar_disk.edge_lengths()                                   # per face, the edge opposite each corner
    assert el.shape == (planar_disk.n_faces, 3)
    F, V = planar_disk.F, planar_disk.V
    assert np.allclose(el[:, 0], np.linalg.norm(V[F[:, 1]] - V[F[:, 2]], axis=1)) or \
        np.allclose(el[:, 0], np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1))
    assert np.allclose(planar_disk.face_angles().sum(axis=1), np.pi)
    assert np.allclose(planar_disk.face_normals(), [0, 0, 1]) or np.allclose(planar_disk.face_normals(), [0, 0, -1])


def test_vtk_roundtrip(sphere_cap):
    pd = sphere_cap.to_vtk()
    assert pd.GetNumberOfPoints() == sphere_cap.n_vertices and pd.GetNumberOfCells() == sphere_cap.n_faces
    m = TriMesh.from_vtk(pd, "back")
    assert np.array_equal(m.V, sphere_cap.V) and np.array_equal(m.F, sphere_cap.F) and m.name == "back"


def test_from_vtk_triangulates_quads():
    import vtk
    pts = vtk.vtkPoints()
    for p in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 0, 0), (2, 1, 0)]:
        pts.InsertNextPoint(p)
    quads = vtk.vtkCellArray()
    for q in [(0, 1, 2, 3), (1, 4, 5, 2)]:
        quads.InsertNextCell(4, q)
    pd = vtk.vtkPolyData(); pd.SetPoints(pts); pd.SetPolys(quads)
    m = TriMesh.from_vtk(pd)
    assert m.n_faces == 4 and m.n_vertices == 6 and abs(m.area() - 2.0) < 1e-12


@pytest.mark.parametrize("ext", ["vtp", "stl", "ply", "obj"])
def test_save_load(tmp_path, sphere_cap, ext):
    path = str(tmp_path / f"cap.{ext}")
    sphere_cap.save(path)
    m = TriMesh.load(path)
    assert m.name == "cap" and m.n_faces == sphere_cap.n_faces
    # STL carries no vertex indices, points are merged back by vtkCleanPolyData
    assert m.n_vertices == sphere_cap.n_vertices
    assert abs(m.area() - sphere_cap.area()) < 1e-3 * sphere_cap.area()
    assert abs(m.perimeter() - sphere_cap.perimeter()) < 1e-3 * sphere_cap.perimeter()


def test_remove_unreferenced(sphere_cap):
    V = np.r_[sphere_cap.V, [[100.0, 100.0, 100.0]]]
    m = TriMesh(V, sphere_cap.F.copy())
    assert m.n_vertices == sphere_cap.n_vertices + 1
    r = m.remove_unreferenced()
    assert r.n_vertices == sphere_cap.n_vertices and np.array_equal(r.F, sphere_cap.F)


def test_orient_consistently_with_hint(sphere_cap):
    m = sphere_cap.orient_consistently(outward_hint=np.array([0.0, 0.0, 1.0]))
    assert np.dot(m.face_normals().mean(0), [0, 0, 1]) > 0
    m = sphere_cap.orient_consistently(outward_hint=np.array([0.0, 0.0, -1.0]))
    assert np.dot(m.face_normals().mean(0), [0, 0, 1]) < 0
    assert m.check_orientation()


def test_geodesic_distance_planar(planar_disk):
    """On a flat mesh the exact geodesic is the straight line; Dijkstra on the edge graph overestimates."""
    loop = planar_disk.boundary_loop()
    i, j = int(loop[0]), int(loop[len(loop) // 2])
    straight = np.linalg.norm(planar_disk.V[i] - planar_disk.V[j])
    dijkstra = planar_disk.geodesic_distance(i, j, exact=False)
    assert dijkstra >= straight - 1e-9 and dijkstra < 1.15 * straight
    pytest.importorskip("pygeodesic")
    assert abs(planar_disk.geodesic_distance(i, j) - straight) < 1e-6
    assert planar_disk.geodesic_distance(i, i) == 0.0


def test_geodesic_distance_sphere(sphere_cap):
    """On the sphere of radius R the geodesic between two vertices is the great-circle arc R * angle."""
    pytest.importorskip("pygeodesic")
    R = 10.0
    c = np.array([0.0, 0.0, R])                   # lift_sphere puts the centre at (0, 0, R)
    loop = sphere_cap.boundary_loop()
    i, j = int(loop[0]), int(loop[len(loop) // 2])
    a, b = (sphere_cap.V[i] - c) / R, (sphere_cap.V[j] - c) / R
    arc = R * np.arccos(np.clip(a @ b, -1, 1))
    assert abs(sphere_cap.geodesic_distance(i, j) - arc) / arc < 0.01
    path = sphere_cap.geodesic_path(i, j)
    assert path[0] == i and path[-1] == j and len(path) > 2
    edges = {tuple(sorted(e)) for e in sphere_cap.edges()}
    assert all(tuple(sorted((a, b))) in edges for a, b in zip(path[:-1], path[1:]))
    length = np.linalg.norm(np.diff(sphere_cap.V[path], axis=0), axis=1).sum()
    assert abs(length - sphere_cap.geodesic_distance(i, j, exact=False)) < 1e-5   # VTK single precision
