"""Geometric flattening methods: exactness on developable shapes, known orderings on curved
shapes, cross-check against libigl, whole-library sanity, regression metrics."""
import numpy as np
import pytest

from slicerheart_patch_flattening.geometric import flatten, METHODS
from slicerheart_patch_flattening.geometric.methods_extra import EXTRA_METHODS
from slicerheart_patch_flattening.geometric.common import default_pins, pin_coordinates, rescale, FlatResult
from slicerheart_patch_flattening.metrics import distortion_metrics
from slicerheart_patch_flattening.mesh import TriMesh
from conftest import rigid_distance, DEVELOPABLE_SHAPES, ANALYTIC_SHAPES

# free-boundary methods (expected to be isometric on a developable surface)
FREE_BOUNDARY = ["lscm", "lscm_edge", "lscm_perimeter", "asap", "arap", "arap_hybrid", "spring", "membrane_stvk",
                 "membrane_suture", "symmetric_dirichlet", "cetm", "bff_lite"]
# fixed-boundary methods (the boundary is mapped onto a circle: never isometric, never flipped)
FIXED_BOUNDARY = ["tutte", "harmonic", "mean_value"]
# the methods compared on the whole library on every run
LIBRARY_METHODS = ["lscm", "arap", "cetm"]


def test_registry():
    assert set(FREE_BOUNDARY + FIXED_BOUNDARY + ["pca_projection"]) <= set(METHODS)
    assert {"symmetric_dirichlet", "cetm", "bff_lite", "membrane_suture", "lscm_perimeter", "exp_map"} <= set(EXTRA_METHODS)
    for name, fn in METHODS.items():
        assert callable(fn), name
    with pytest.raises(KeyError):
        flatten(TriMesh(np.zeros((3, 3)), np.array([[0, 1, 2]])), "no_such_method")


def test_result_fields(half_cylinder):
    res = flatten(half_cylinder, "arap")
    assert isinstance(res, FlatResult) and res.method == "arap"
    assert res.uv.shape == (half_cylinder.n_vertices, 2) and np.isfinite(res.uv).all()
    assert res.time_s >= 0 and res.iterations >= 1 and res.converged
    flat = res.as_mesh(half_cylinder)
    assert flat.V.shape == (half_cylinder.n_vertices, 3) and np.all(flat.V[:, 2] == 0)
    assert np.array_equal(flat.F, half_cylinder.F) and flat.meta["flatten_method"] == "arap"


@pytest.mark.parametrize("shape", DEVELOPABLE_SHAPES)
@pytest.mark.parametrize("method", FREE_BOUNDARY)
def test_developable_is_isometric(shape, method, metrics):
    """A developable surface flattens without distortion: every free-boundary method must
    recover its edge lengths, area and template size within the clinical tolerance."""
    m = ANALYTIC_SHAPES[shape]()[0]
    res = flatten(m, method)
    M = distortion_metrics(m, res.uv)
    assert M["flipped_triangles"] == 0
    assert M["edge_rel_err_max_abs_pct"] < 0.3
    assert abs(M["area_total_error_pct"]) < 0.1
    assert abs(M["perimeter_err_pct"]) < 0.1
    assert M["passes_size_tolerance"]
    metrics.record(f"geometric/{shape}/{method}/edge_rel_err_max_abs_pct", M["edge_rel_err_max_abs_pct"], atol=0.01)
    metrics.record(f"geometric/{shape}/{method}/boundary_cumulative_mismatch_mm", M["boundary_cumulative_mismatch_mm"], atol=0.01)


@pytest.mark.parametrize("method", ["lscm", "arap", "cetm", "bff_lite", "symmetric_dirichlet", "membrane_stvk"])
def test_half_cylinder_recovers_exact_template(half_cylinder, half_cylinder_domain, method):
    """The cylinder lift is an isometry of its parameter domain, so the domain is the exact flat
    template and the flattening must coincide with it up to a rigid motion."""
    uv = flatten(half_cylinder, method).uv
    assert rigid_distance(uv, half_cylinder_domain.V) < 0.05       # mm, on a 20 x 16 mm patch


@pytest.mark.parametrize("method", FIXED_BOUNDARY)
def test_fixed_boundary_methods(sphere_cap, method):
    """Boundary on a circle, interior strictly inside it, no flipped triangles (Tutte's theorem for
    a convex boundary with positive weights)."""
    res = flatten(sphere_cap, method)
    uv = res.uv
    loop = sphere_cap.boundary_loop()
    # algebraic (Kasa) circle fit of the boundary: |p|^2 = 2 c . p + k
    P = uv[loop]
    A = np.c_[2 * P, np.ones(len(P))]
    sol, *_ = np.linalg.lstsq(A, (P**2).sum(1), rcond=None)
    c = sol[:2]
    r = np.linalg.norm(P - c, axis=1)
    assert r.std() / r.mean() < 1e-9
    interior = ~sphere_cap.is_boundary_vertex()
    assert (np.linalg.norm(uv[interior] - c, axis=1) < r.mean()).all()
    M = distortion_metrics(sphere_cap, uv)
    assert M["flipped_triangles"] == 0


def test_pca_projection(saddle):
    """Orthogonal projection onto the best plane: never stretches, never flips on a graph surface."""
    M = distortion_metrics(saddle, flatten(saddle, "pca_projection").uv)
    assert M["stretch_Linf"] <= 1.0 + 1e-9 and M["flipped_triangles"] == 0
    assert M["area_total_error_pct"] < 0


def test_lscm_matches_igl(sphere_cap):
    """Our LSCM reproduces libigl's (same pins) up to a rigid motion."""
    igl = pytest.importorskip("igl")
    pins = default_pins(sphere_cap)
    puv = pin_coordinates(sphere_cap, pins)
    ref, _ = igl.lscm(sphere_cap.V, sphere_cap.F.astype(np.int64), pins.astype(np.int64), puv)
    mine = flatten(sphere_cap, "lscm", scale="none").uv
    assert rigid_distance(mine, ref, allow_reflection=False) < 1e-6


def test_lscm_pins(sphere_cap):
    """Explicit pins land exactly where asked (before the area rescale) and change the result."""
    loop = sphere_cap.boundary_loop()
    pins = np.array([loop[0], loop[len(loop) // 3]])
    puv = np.array([[1.0, 2.0], [9.0, 2.0]])
    res = flatten(sphere_cap, "lscm", pins=pins, pin_uv=puv, scale="none")
    assert res.params["pins"] == pins.tolist()
    # canonical_pose recentres and rotates: compare the pin separation, which is invariant
    assert abs(np.linalg.norm(res.uv[pins[0]] - res.uv[pins[1]]) - 8.0) < 1e-9
    other = flatten(sphere_cap, "lscm", scale="none").uv
    assert rigid_distance(res.uv, other) > 1e-3


def test_conformal_vs_isometric_tradeoff(sphere_cap, metrics):
    """LSCM minimises angle distortion, ARAP isometric distortion: each must win its own metric."""
    Ml = distortion_metrics(sphere_cap, flatten(sphere_cap, "lscm").uv)
    Ma = distortion_metrics(sphere_cap, flatten(sphere_cap, "arap").uv)
    assert Ma["isometric_err_mean"] < Ml["isometric_err_mean"]
    assert Ma["principal_stretch_absdev_mean"] < Ml["principal_stretch_absdev_mean"]
    assert Ma["edge_rel_err_mean_abs_pct"] < Ml["edge_rel_err_mean_abs_pct"]
    assert Ml["angle_err_deg_p95"] < Ma["angle_err_deg_p95"]
    assert Ml["conformal_ratio_mean"] < Ma["conformal_ratio_mean"]
    assert Ml["angle_err_deg_mean"] < 1.0                       # conformal: angles essentially preserved
    for name, M in (("lscm", Ml), ("arap", Ma)):
        metrics.record(f"geometric/sphere_cap/{name}/stretch_Linf", M["stretch_Linf"], atol=1e-4)
        metrics.record(f"geometric/sphere_cap/{name}/isometric_err_mean", M["isometric_err_mean"], atol=1e-6)
        metrics.record(f"geometric/sphere_cap/{name}/angle_err_deg_p95", M["angle_err_deg_p95"], atol=1e-3)


def test_area_error_bounded_on_curved_shapes(sphere_cap, saddle, metrics):
    """ARAP on a cap of R = 10 mm of this size: a few percent of local stretch and no flips.
    The exact amount is tracked as a regression metric."""
    for name, m in (("sphere_cap", sphere_cap), ("saddle", saddle)):
        M = distortion_metrics(m, flatten(m, "arap").uv)
        assert M["flipped_triangles"] == 0
        assert abs(M["area_total_error_pct"]) < 1.0                 # ARAP keeps the total area
        assert 0.85 < M["stretch_min"] and M["stretch_Linf"] < 1.15
        metrics.record(f"geometric/{name}/arap/boundary_cumulative_mismatch_mm", M["boundary_cumulative_mismatch_mm"], atol=0.01)
        metrics.record(f"geometric/{name}/arap/edge_rel_err_p95_abs_pct", M["edge_rel_err_p95_abs_pct"], atol=0.01)


def test_cetm_and_bff_hold_boundary_lengths(sphere_cap):
    """Suture-exact methods: every boundary edge keeps its length."""
    for method in ("cetm", "bff_lite"):
        M = distortion_metrics(sphere_cap, flatten(sphere_cap, method).uv)
        assert M["boundary_edge_rel_err_max_abs_pct"] < 0.1, method
        assert abs(M["perimeter_err_pct"]) < 0.01, method
        assert M["flipped_triangles"] == 0, method


def test_cetm_matches_bff(sphere_cap):
    Mc = distortion_metrics(sphere_cap, flatten(sphere_cap, "cetm").uv)
    Mb = distortion_metrics(sphere_cap, flatten(sphere_cap, "bff_lite").uv)
    assert abs(Mc["area_total_error_pct"] - Mb["area_total_error_pct"]) < 0.5


def test_exp_map_runs(sphere_cap):
    res = flatten(sphere_cap, "exp_map")
    assert np.isfinite(res.uv).all() and distortion_metrics(sphere_cap, res.uv)["flipped_triangles"] == 0


@pytest.mark.parametrize("mode", ["area", "perimeter", "edge", "none"])
def test_rescale_modes(sphere_cap, mode):
    uv0 = flatten(sphere_cap, "lscm", scale="none").uv * 1.7
    uv, s = rescale(sphere_cap, uv0, mode)
    assert np.allclose(uv, uv0 * s)
    flat = TriMesh(uv, sphere_cap.F)
    if mode == "area":
        assert abs(flat.area() - sphere_cap.area()) < 1e-9 * sphere_cap.area()
    elif mode == "perimeter":
        assert abs(flat.perimeter() - sphere_cap.perimeter()) < 1e-9 * sphere_cap.perimeter()
    elif mode == "edge":
        e = sphere_cap.edges()
        l3 = np.linalg.norm(sphere_cap.V[e[:, 0]] - sphere_cap.V[e[:, 1]], axis=1)
        l2 = np.linalg.norm(uv[e[:, 0]] - uv[e[:, 1]], axis=1)
        assert abs(np.dot(l3 - l2, l2)) < 1e-9 * np.dot(l2, l2)      # least-squares optimum: residual orthogonal to l2
    else:
        assert s == 1.0
    with pytest.raises(ValueError):
        rescale(sphere_cap, uv0, "nonsense")


@pytest.mark.parametrize("method", LIBRARY_METHODS)
def test_whole_library(library, method, metrics):
    """Every shape of the synthetic library flattens with every main method: finite, no flipped
    triangles, isometric on the developable shapes. The per-family distortion is a regression metric."""
    by_family = {}
    for m in library:
        res = flatten(m, method)
        assert np.isfinite(res.uv).all(), (m.name, method)
        M = distortion_metrics(m, res.uv)
        assert M["flipped_triangles"] == 0, (m.name, method)
        if m.meta["developable"]:
            assert M["edge_rel_err_max_abs_pct"] < 0.5, (m.name, method)
            assert abs(M["area_total_error_pct"]) < 0.2, (m.name, method)
        by_family.setdefault(m.meta["family"], []).append(M)
    for family, Ms in by_family.items():
        metrics.record(f"library/{family}/{method}/stretch_Linf_max", max(M["stretch_Linf"] for M in Ms), atol=1e-3)
        metrics.record(f"library/{family}/{method}/edge_rel_err_p95_abs_pct_median",
                       np.median([M["edge_rel_err_p95_abs_pct"] for M in Ms]), atol=0.01)
        metrics.record(f"library/{family}/{method}/boundary_cumulative_mismatch_mm_median",
                       np.median([M["boundary_cumulative_mismatch_mm"] for M in Ms]), atol=0.01)


def test_developabilize_reduces_curvature(sphere_cap):
    pytest.importorskip("torch")
    from slicerheart_patch_flattening.geometric.developabilize import developabilize
    r = developabilize(sphere_cap, alpha=1e-5, max_iter=200)
    assert r["total_abs_K_after_deg"] < r["total_abs_K_before_deg"]
    assert r["displacement_mm"][sphere_cap.is_boundary_vertex()].max() < 1e-9
