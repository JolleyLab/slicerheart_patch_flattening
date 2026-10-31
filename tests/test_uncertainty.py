"""Uncertainty analysis: method spread and mesh-refinement sensitivity."""
import numpy as np

from slicerheart_patch_flattening.geometric import flatten
from slicerheart_patch_flattening.uncertainty import rigid_align, method_spread, subdivide, refinement_sensitivity


def test_rigid_align():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((50, 2)) * 10
    a = np.radians(33.0)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    moved = ref @ R.T + [3.0, -7.0]
    assert np.abs(rigid_align(moved, ref) - ref).max() < 1e-9
    # alignment fitted on a subset is applied to all rows
    idx = np.arange(0, 50, 2)
    assert np.abs(rigid_align(moved, ref, idx) - ref).max() < 1e-9


def test_method_spread(sphere_cap):
    uv = flatten(sphere_cap, "arap").uv
    a = np.radians(20.0)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    same = {"arap": uv, "copy": uv.copy(), "moved": uv @ R.T + [5.0, 1.0]}
    s = method_spread(sphere_cap, same)
    assert s["boundary_spread_max_mm"] < 1e-9 and s["boundary_spread_rms_mm"] < 1e-9
    uvs = {m: flatten(sphere_cap, m).uv for m in ("arap", "lscm", "cetm")}
    s = method_spread(sphere_cap, uvs)
    assert 0 < s["boundary_spread_rms_mm"] <= s["boundary_spread_max_mm"] < 2.0
    assert set(s["per_method_rms_mm"]) == set(uvs)
    assert len(s["per_vertex_spread_mm"]) == len(s["boundary_loop"]) == len(sphere_cap.boundary_loop())
    assert all(s["per_method_rms_mm"][m] <= s["per_method_max_mm"][m] for m in uvs)


def test_subdivide(sphere_cap):
    fine = subdivide(sphere_cap)
    assert fine.n_faces == 4 * sphere_cap.n_faces
    assert fine.n_vertices == sphere_cap.n_vertices + len(sphere_cap.edges())
    assert np.allclose(fine.V[: sphere_cap.n_vertices], sphere_cap.V)
    assert abs(fine.area() - sphere_cap.area()) < 1e-6 * sphere_cap.area()   # linear subdivision keeps the polyhedron (VTK single precision)
    assert fine.check_orientation()


def test_refinement_sensitivity(half_cylinder, sphere_cap, metrics):
    """Flattening a developable shape does not change with the mesh resolution; on a curved shape
    the outline moves by a fraction of a millimetre (tracked as a regression metric)."""
    r = refinement_sensitivity(half_cylinder, "arap")
    assert r["n_vertices_fine"] > r["n_vertices_coarse"] == half_cylinder.n_vertices
    assert r["refine_boundary_max_mm"] < 0.05
    r = refinement_sensitivity(sphere_cap, "arap")
    assert r["refine_boundary_max_mm"] < 0.5
    metrics.record("uncertainty/sphere_cap/arap/refine_boundary_max_mm", r["refine_boundary_max_mm"], atol=0.005)
    metrics.record("uncertainty/sphere_cap/arap/refine_boundary_rms_mm", r["refine_boundary_rms_mm"], atol=0.002)
