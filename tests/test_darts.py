"""Dart recommendation, mesh cutting and re-flattening."""
import numpy as np

from slicerheart_patch_flattening.geometric import flatten
from slicerheart_patch_flattening.geometric.darts import recommend_darts, cut_mesh_along_path, flatten_with_darts, \
    smoothed_curvature
from slicerheart_patch_flattening.metrics import distortion_metrics
from conftest import euler_characteristic


def test_smoothed_curvature(sphere_cap, half_cylinder):
    C = smoothed_curvature(sphere_cap, radius_mm=4.0)
    assert C.shape == (sphere_cap.n_vertices,)
    interior = ~sphere_cap.is_boundary_vertex()
    assert (C[interior] > 0).all()                                   # positive curvature everywhere on a cap
    # a 4 mm ball on a sphere of R = 10 encloses about A / R^2 = pi 16 / 100 rad of curvature
    assert 0.5 * 0.16 * np.pi < C[interior].max() < 1.5 * 0.16 * np.pi
    assert np.abs(smoothed_curvature(half_cylinder)).max() < np.deg2rad(2.0)


def test_no_darts_on_developable(half_cylinder, cone):
    assert recommend_darts(half_cylinder) == []
    assert recommend_darts(cone) == []


def test_recommend_darts(sphere_cap, saddle):
    darts = recommend_darts(sphere_cap, max_darts=2)
    assert 1 <= len(darts) <= 2
    d = darts[0]
    assert d["kind"] == "dart" and d["angle_deg"] > 8.0 and d["length_mm"] > 0
    assert d["path"][0] == d["entry"] and d["path"][-1] == d["apex"]
    assert sphere_cap.is_boundary_vertex()[d["entry"]] and not sphere_cap.is_boundary_vertex()[d["apex"]]
    # the path is an edge path
    edges = {tuple(sorted(e)) for e in sphere_cap.edges()}
    assert all(tuple(sorted((a, b))) in edges for a, b in zip(d["path"][:-1], d["path"][1:]))
    # a saddle (K < 0) wants a gusset (insert a wedge)
    g = recommend_darts(saddle, max_darts=1)
    assert g and g[0]["kind"] == "gusset" and g[0]["angle_deg"] < 0
    # the first dart has the largest enclosed curvature
    assert recommend_darts(sphere_cap, max_darts=1)[0]["apex"] == d["apex"]


def test_cut_mesh_along_path(sphere_cap):
    d = recommend_darts(sphere_cap, max_darts=1)[0]
    cut, orig = cut_mesh_along_path(sphere_cap, d["path"])
    assert cut.n_vertices == sphere_cap.n_vertices + len(d["path"]) - 1     # every path vertex but the apex is duplicated
    assert cut.n_faces == sphere_cap.n_faces
    assert abs(cut.area() - sphere_cap.area()) < 1e-9
    assert len(orig) == cut.n_vertices and np.array_equal(cut.V, sphere_cap.V[orig])
    assert euler_characteristic(cut) == 1                                   # still a disk
    loop = cut.boundary_loop()
    assert len(loop) == len(sphere_cap.boundary_loop()) + 2 * (len(d["path"]) - 1)
    assert cut.check_orientation()
    # the slit opens: the perimeter grows by twice the path length
    assert abs(cut.perimeter() - sphere_cap.perimeter() - 2 * d["length_mm"]) < 1e-9


def test_flatten_with_darts(sphere_cap, metrics):
    darts = recommend_darts(sphere_cap, max_darts=1)
    res, cut, applied = flatten_with_darts(sphere_cap, "arap", darts)
    assert len(applied) == 1 and "uv_path" in applied[0]
    assert len(applied[0]["uv_path"]) == len(darts[0]["path"])
    assert len(applied[0]["uv_path"][0]) == 2 and len(applied[0]["uv_path"][-1]) == 1   # entry split, apex not
    M0 = distortion_metrics(sphere_cap, flatten(sphere_cap, "arap").uv)
    M1 = distortion_metrics(cut, res.uv)
    assert M1["flipped_triangles"] == 0
    assert M1["boundary_cumulative_mismatch_mm"] < M0["boundary_cumulative_mismatch_mm"]
    assert M1["stretch_Linf"] < M0["stretch_Linf"] and M1["stretch_min"] > M0["stretch_min"]
    metrics.record("darts/sphere_cap/arap/mismatch_before_mm", M0["boundary_cumulative_mismatch_mm"], atol=0.01)
    metrics.record("darts/sphere_cap/arap/mismatch_after_mm", M1["boundary_cumulative_mismatch_mm"], atol=0.01)
    metrics.record("darts/sphere_cap/arap/stretch_Linf_after", M1["stretch_Linf"], atol=1e-3)


def test_two_darts(sphere_cap):
    darts = recommend_darts(sphere_cap, max_darts=2)
    res, cut, applied = flatten_with_darts(sphere_cap, "arap", darts)
    assert len(applied) == len(darts)
    assert cut.n_faces == sphere_cap.n_faces and euler_characteristic(cut) == 1
    assert distortion_metrics(cut, res.uv)["flipped_triangles"] == 0
