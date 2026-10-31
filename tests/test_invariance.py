"""A flattening must not depend on the world coordinate system of the input (LPS/RAS, rotation,
translation, reflection) nor on the order of the faces; scaling the input scales the template."""
import numpy as np
import pytest

from slicerheart_patch_flattening.geometric import flatten
from slicerheart_patch_flattening.metrics import distortion_metrics
from slicerheart_patch_flattening.mesh import TriMesh
from conftest import WORLD_TRANSFORMS, transformed, rigid_distance

METHODS = ["lscm", "arap", "cetm", "bff_lite", "symmetric_dirichlet", "membrane_stvk", "spring", "tutte",
           "harmonic", "pca_projection"]
# metrics that are dimensionless (must agree exactly), and metrics in mm (scale with the input)
DIMENSIONLESS = ["stretch_Linf", "stretch_min", "area_total_error_pct", "edge_rel_err_max_abs_pct",
                 "angle_err_deg_p95", "perimeter_err_pct", "flipped_triangles"]
IN_MM = ["boundary_cumulative_mismatch_mm", "perimeter_3d_mm", "length_2d_mm", "width_2d_mm"]


# methods whose iterative solver stops at an absolute tolerance (scipy L-BFGS gtol, energy decrease):
# a scaled input converges to a slightly different point, so they get a looser tolerance under scaling
ITERATIVE = {"arap", "spring", "membrane_stvk", "symmetric_dirichlet"}


@pytest.mark.parametrize("transform", list(WORLD_TRANSFORMS))
@pytest.mark.parametrize("method", METHODS)
def test_world_transform(sphere_cap, method, transform):
    T = WORLD_TRANSFORMS[transform]()
    scale = np.cbrt(abs(np.linalg.det(T[:3, :3])))
    loose = method in ITERATIVE and scale != 1.0
    tol_mm, tol_rel = (5e-4, 1e-3) if loose else (1e-8, 1e-8)
    moved = transformed(sphere_cap, T)
    uv0 = flatten(sphere_cap, method).uv
    uv1 = flatten(moved, method).uv
    assert rigid_distance(uv1 / scale, uv0) < tol_mm, (method, transform)
    M0 = distortion_metrics(sphere_cap, uv0)
    M1 = distortion_metrics(moved, uv1)
    for k in DIMENSIONLESS:
        assert abs(M1[k] - M0[k]) < tol_rel * max(1.0, abs(M0[k])), (method, transform, k)
    for k in IN_MM:
        assert abs(M1[k] - scale * M0[k]) < tol_rel * max(1.0, abs(M0[k])), (method, transform, k)


@pytest.mark.parametrize("method", ["lscm", "arap", "cetm", "membrane_stvk"])
def test_face_order(sphere_cap, method):
    """Permuting the faces (same vertices) must give the same template."""
    rng = np.random.default_rng(3)
    F = sphere_cap.F[rng.permutation(sphere_cap.n_faces)]
    shuffled = TriMesh(sphere_cap.V.copy(), F)
    uv0 = flatten(sphere_cap, method).uv
    uv1 = flatten(shuffled, method).uv
    assert rigid_distance(uv1, uv0) < 1e-6


@pytest.mark.parametrize("method", ["lscm", "arap", "cetm", "membrane_stvk"])
def test_vertex_order(sphere_cap, method):
    """Renumbering the vertices must give the same template (compared vertex by vertex)."""
    rng = np.random.default_rng(5)
    perm = rng.permutation(sphere_cap.n_vertices)      # new index of old vertex i is perm[i]
    inv = np.argsort(perm)
    renumbered = TriMesh(sphere_cap.V[inv], perm[sphere_cap.F])
    uv0 = flatten(sphere_cap, method).uv
    uv1 = flatten(renumbered, method).uv[perm]          # back in the original order
    assert rigid_distance(uv1, uv0) < 1e-6


def test_flipped_orientation_input(sphere_cap):
    """A mesh given with inward-facing normals flattens to the same template."""
    flipped = TriMesh(sphere_cap.V.copy(), sphere_cap.F[:, ::-1].copy())
    for method in ("lscm", "arap"):
        uv0 = flatten(sphere_cap, method).uv
        uv1 = flatten(flipped, method).uv
        assert rigid_distance(uv1, uv0) < 1e-6, method
        assert distortion_metrics(flipped, uv1)["flipped_triangles"] == 0
