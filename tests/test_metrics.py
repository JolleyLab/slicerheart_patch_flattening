"""Distortion metrics on flattenings with known distortion."""
import numpy as np
import pytest

from slicerheart_patch_flattening.metrics import distortion_metrics, per_face_stretch, per_vertex_stretch, \
    feret_pairs, KEY_METRICS, TOL_ABS_MM, TOL_REL
from slicerheart_patch_flattening.geometric.common import count_flips
from slicerheart_patch_flattening.mesh import TriMesh


def test_identity_is_perfect(planar_disk):
    """A planar mesh mapped onto itself: no distortion of any kind."""
    M = distortion_metrics(planar_disk, planar_disk.V[:, :2])
    assert set(KEY_METRICS) <= set(M)
    assert M["flipped_triangles"] == 0
    for k in ("area_total_error_pct", "edge_rel_err_max_abs_pct", "boundary_edge_rel_err_max_abs_pct",
              "perimeter_err_mm", "boundary_cumulative_mismatch_mm", "isometric_err_max", "angle_err_deg_max",
              "area_ratio_absdev_max", "principal_stretch_absdev_max", "shape_total_abs_K_deg"):
        assert abs(M[k]) < 1e-9, k
    assert abs(M["stretch_Linf"] - 1) < 1e-12 and abs(M["stretch_min"] - 1) < 1e-12
    assert abs(M["conformal_ratio_max"] - 1) < 1e-12
    assert M["passes_size_tolerance"]
    assert abs(M["length_2d_mm"] - M["length_geodesic_3d_mm"]) < 1e-6      # straight line on a plane
    assert abs(M["width_2d_mm"] - M["width_geodesic_3d_mm"]) < 1e-6
    assert abs(M["shape_area_mm2"] - planar_disk.area()) < 1e-9
    assert abs(M["perimeter_3d_mm"] - planar_disk.perimeter()) < 1e-9


@pytest.mark.parametrize("s", [1.02, 0.9])
def test_uniform_scale(planar_disk, s):
    """Scaling the template by s: every stretch is s, area error s^2 - 1, lengths off by s - 1."""
    M = distortion_metrics(planar_disk, planar_disk.V[:, :2] * s)
    assert abs(M["stretch_Linf"] - s) < 1e-12 and abs(M["stretch_min"] - s) < 1e-12
    assert abs(M["area_total_error_pct"] - (s**2 - 1) * 100) < 1e-9
    assert abs(M["edge_rel_err_mean_abs_pct"] - abs(s - 1) * 100) < 1e-9
    assert abs(M["edge_rel_err_max_abs_pct"] - abs(s - 1) * 100) < 1e-9
    assert abs(M["perimeter_err_pct"] - (s - 1) * 100) < 1e-9
    assert abs(M["length_err_pct"] - (s - 1) * 100) < 1e-6
    assert abs(M["angle_err_deg_max"]) < 1e-9                   # a similarity keeps angles
    assert abs(M["conformal_ratio_max"] - 1) < 1e-12
    assert M["flipped_triangles"] == 0
    assert not M["passes_size_tolerance"]                       # 2 % of 20 mm is 0.4 mm > 0.1 mm


def test_size_tolerance_thresholds(planar_disk):
    """The Kizilski et al. tolerance: within 0.1 mm or 0.5 % on both length and width."""
    L = distortion_metrics(planar_disk, planar_disk.V[:, :2])["length_2d_mm"]
    s_ok = 1 + TOL_REL * 0.9                                     # inside the relative tolerance
    assert distortion_metrics(planar_disk, planar_disk.V[:, :2] * s_ok)["passes_size_tolerance"]
    s_bad = 1 + 3 * max(TOL_REL, TOL_ABS_MM / L)
    assert not distortion_metrics(planar_disk, planar_disk.V[:, :2] * s_bad)["passes_size_tolerance"]
    # a looser tolerance passes the same flattening
    assert distortion_metrics(planar_disk, planar_disk.V[:, :2] * s_bad, tol_rel=0.1)["passes_size_tolerance"]


def test_anisotropic_stretch(planar_disk):
    """Stretching u by a: principal stretches a and 1, conformal ratio a, angles change."""
    a = 1.1
    uv = planar_disk.V[:, :2] * np.array([a, 1.0])
    s1, s2, area, J = per_face_stretch(planar_disk, uv)
    assert np.allclose(s1, a) and np.allclose(s2, 1.0)
    assert np.allclose(area, planar_disk.face_areas())
    M = distortion_metrics(planar_disk, uv)
    assert abs(M["stretch_Linf"] - a) < 1e-12 and abs(M["stretch_min"] - 1) < 1e-12
    assert abs(M["conformal_ratio_max"] - a) < 1e-12
    assert abs(M["area_total_error_pct"] - (a - 1) * 100) < 1e-9
    assert M["angle_err_deg_max"] > 1.0
    assert abs(M["isometric_err_max"] - (a - 1) ** 2) < 1e-12


def test_flipped_triangles(planar_disk):
    """Flips are counted against the majority orientation: a globally mirrored template (which a
    cutting template may be) has none; a fold-over counts the triangles it inverts."""
    mirrored = planar_disk.V[:, :2] * np.array([1.0, -1.0])
    assert count_flips(planar_disk, mirrored) == 0
    assert distortion_metrics(planar_disk, mirrored)["flipped_triangles"] == 0
    assert distortion_metrics(planar_disk, mirrored)["edge_rel_err_max_abs_pct"] < 1e-9
    # one interior vertex pulled far across its neighbours folds over its incident triangles
    uv = planar_disk.V[:, :2].copy()
    v = np.flatnonzero(~planar_disk.is_boundary_vertex())[0]
    uv[v] += 5.0
    n_incident = int((planar_disk.F == v).any(axis=1).sum())
    n_flips = count_flips(planar_disk, uv)
    assert 0 < n_flips <= n_incident
    assert distortion_metrics(planar_disk, uv)["flipped_triangles"] == n_flips


def test_per_vertex_stretch(planar_disk):
    pv = per_vertex_stretch(planar_disk, planar_disk.V[:, :2] * 1.3)
    assert set(pv) == {"s1", "s2", "area_ratio"}
    assert np.allclose(pv["s1"], 1.3) and np.allclose(pv["s2"], 1.3) and np.allclose(pv["area_ratio"], 1.69)


def test_feret_pairs(planar_disk):
    loop = planar_disk.boundary_loop()
    uv = planar_disk.V[:, :2]
    (i, j), (k, l) = feret_pairs(uv, loop)
    assert {i, j, k, l} <= set(loop.tolist())
    d = np.linalg.norm(uv[loop][:, None] - uv[loop][None], axis=2)
    assert abs(np.linalg.norm(uv[i] - uv[j]) - d.max()) < 1e-12        # the longest chord
    assert np.linalg.norm(uv[k] - uv[l]) <= d.max() + 1e-12
    # width is measured perpendicular to the length axis
    axis = (uv[j] - uv[i]) / np.linalg.norm(uv[j] - uv[i])
    assert abs(np.dot(uv[l] - uv[k], axis)) < 0.3 * np.linalg.norm(uv[l] - uv[k])


def test_suture_mismatch_is_start_independent(planar_disk):
    """The cumulative suture-line mismatch (range of the running arclength error) does not
    depend on where the boundary loop starts."""
    rng = np.random.default_rng(0)
    uv = planar_disk.V[:, :2] * (1 + 0.02 * rng.standard_normal((planar_disk.n_vertices, 1)))
    M0 = distortion_metrics(planar_disk, uv)["boundary_cumulative_mismatch_mm"]
    assert M0 > 0
    # rotate the vertex numbering of the boundary loop by renumbering all vertices
    perm = rng.permutation(planar_disk.n_vertices)
    inv = np.argsort(perm)
    m2 = TriMesh(planar_disk.V[inv], perm[planar_disk.F])
    M1 = distortion_metrics(m2, uv[inv])["boundary_cumulative_mismatch_mm"]
    assert abs(M1 - M0) < 1e-9
