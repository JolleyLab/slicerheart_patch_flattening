"""Tests: mesh utilities, solver correctness on analytic shapes, metrics, darts, API."""
import numpy as np
import pytest

from slicerheart_patch_flattening.library import generators as G
from slicerheart_patch_flattening.mesh import TriMesh
from slicerheart_patch_flattening.geometric import flatten, METHODS
from slicerheart_patch_flattening.metrics import distortion_metrics
from slicerheart_patch_flattening.geometric.darts import recommend_darts, cut_mesh_along_path, flatten_with_darts
from slicerheart_patch_flattening.slicer_api import flatten_polydata

H = 1.0


@pytest.fixture(scope="module")
def half_cylinder():
    m = G.parametric_patch(G.superellipse(10, 8, 2.0, 64), G.lift_cylinder, H, R=7.0).orient_consistently()
    return m


@pytest.fixture(scope="module")
def sphere_cap():
    return G.parametric_patch(G.superellipse(8, 8, 2.0, 64), G.lift_sphere, H, R=10.0).orient_consistently()


def test_boundary_loop_single(half_cylinder):
    loop = half_cylinder.boundary_loop()
    assert len(loop) == len(half_cylinder.boundary_edges())
    assert half_cylinder.check_orientation()


def test_gauss_bonnet(sphere_cap):
    # sum of angle defects (interior + boundary geodesic curvature) = 2 pi * Euler char (disk: 1)
    assert abs(sphere_cap.angle_defects().sum() - 2 * np.pi) < 1e-6
    # cap of radius R: integrated K over the interior vertices' areas = (interior area) / R^2
    interior = ~sphere_cap.is_boundary_vertex()
    K_int = sphere_cap.angle_defects()[interior].sum()
    A_int = sphere_cap.vertex_areas()[interior].sum()
    assert abs(K_int - A_int / 100.0) / K_int < 0.05


@pytest.mark.parametrize("method", ["lscm", "lscm_edge", "asap", "arap", "arap_hybrid", "spring", "membrane_stvk"])
def test_developable_is_isometric(half_cylinder, method):
    res = flatten(half_cylinder, method)
    M = distortion_metrics(half_cylinder, res.uv)
    assert M["flipped_triangles"] == 0
    assert M["edge_rel_err_max_abs_pct"] < 0.3
    assert abs(M["area_total_error_pct"]) < 0.1
    assert M["passes_size_tolerance"]


def test_fixed_boundary_methods_run(half_cylinder):
    for m in ("tutte", "harmonic", "mean_value", "pca_projection"):
        res = flatten(half_cylinder, m)
        assert np.isfinite(res.uv).all()


def test_lscm_matches_igl(sphere_cap):
    igl = pytest.importorskip("igl")
    from slicerheart_patch_flattening.geometric.common import default_pins, pin_coordinates
    from scipy.linalg import orthogonal_procrustes
    pins = default_pins(sphere_cap); puv = pin_coordinates(sphere_cap, pins)
    ref, _ = igl.lscm(sphere_cap.V, sphere_cap.F.astype(np.int64), pins.astype(np.int64), puv)
    mine = flatten(sphere_cap, "lscm", scale="none").uv
    A = mine - mine.mean(0); B = ref - ref.mean(0)
    R, _ = orthogonal_procrustes(B, A)
    assert np.abs(B @ R - A).max() < 1e-6


def test_arap_beats_lscm_on_lengths(sphere_cap):
    Ml = distortion_metrics(sphere_cap, flatten(sphere_cap, "lscm").uv)
    Ma = distortion_metrics(sphere_cap, flatten(sphere_cap, "arap").uv)
    # ARAP minimises isometric distortion, LSCM minimises angle distortion
    assert Ma["isometric_err_mean"] < Ml["isometric_err_mean"]
    assert Ma["stretch_Linf"] < Ml["stretch_Linf"] and Ma["stretch_min"] > Ml["stretch_min"]
    assert Ml["angle_err_deg_p95"] < Ma["angle_err_deg_p95"]


def test_dart_cut_and_improvement(sphere_cap):
    darts = recommend_darts(sphere_cap, max_darts=1)
    assert darts and darts[0]["kind"] == "dart" and darts[0]["angle_deg"] > 0
    cut, orig = cut_mesh_along_path(sphere_cap, darts[0]["path"])
    assert cut.n_vertices == sphere_cap.n_vertices + len(darts[0]["path"]) - 1
    assert cut.n_faces == sphere_cap.n_faces
    assert abs(cut.area() - sphere_cap.area()) < 1e-9
    res, cutm, applied = flatten_with_darts(sphere_cap, "arap", darts)
    M0 = distortion_metrics(sphere_cap, flatten(sphere_cap, "arap").uv)
    M1 = distortion_metrics(cutm, res.uv)
    assert M1["boundary_cumulative_mismatch_mm"] < M0["boundary_cumulative_mismatch_mm"]
    assert M1["flipped_triangles"] == 0


def test_slicer_api_roundtrip(half_cylinder):
    pd = half_cylinder.to_vtk()
    fixed = half_cylinder.V[[0, half_cylinder.n_vertices // 2]]
    out, res, M = flatten_polydata(pd, "lscm", fixed_points_world=fixed)
    assert out.GetNumberOfPoints() == pd.GetNumberOfPoints()
    assert out.GetNumberOfCells() == pd.GetNumberOfCells()
    assert out.GetPointData().GetArray("stretch_s1") is not None
    # pin 0 at origin, pin 1 on +u axis
    assert np.allclose(res.uv[0], 0, atol=1e-9)
    assert abs(res.uv[half_cylinder.n_vertices // 2][1]) < 1e-9


def test_registry_complete():
    assert {"lscm", "arap", "membrane_stvk", "pca_projection"} <= set(METHODS)


@pytest.mark.parametrize("method", ["symmetric_dirichlet", "cetm", "bff_lite", "lscm_perimeter", "membrane_suture"])
def test_extra_methods_isometric_on_developable(half_cylinder, method):
    res = flatten(half_cylinder, method)
    M = distortion_metrics(half_cylinder, res.uv)
    assert M["flipped_triangles"] == 0
    assert M["edge_rel_err_max_abs_pct"] < 0.3


def test_cetm_and_bff_hold_boundary_lengths(sphere_cap):
    for method in ("cetm", "bff_lite"):
        M = distortion_metrics(sphere_cap, flatten(sphere_cap, method).uv)
        assert M["boundary_edge_rel_err_max_abs_pct"] < 0.1
        assert abs(M["perimeter_err_pct"]) < 0.01


def test_cetm_matches_bff(sphere_cap):
    Mc = distortion_metrics(sphere_cap, flatten(sphere_cap, "cetm").uv)
    Mb = distortion_metrics(sphere_cap, flatten(sphere_cap, "bff_lite").uv)
    assert abs(Mc["area_total_error_pct"] - Mb["area_total_error_pct"]) < 0.5


def test_developabilize_reduces_curvature(sphere_cap):
    pytest.importorskip("torch")
    from slicerheart_patch_flattening.geometric.developabilize import developabilize
    r = developabilize(sphere_cap, alpha=1e-5, max_iter=200)
    assert r["total_abs_K_after_deg"] < r["total_abs_K_before_deg"]
    assert r["displacement_mm"][sphere_cap.is_boundary_vertex()].max() < 1e-9
