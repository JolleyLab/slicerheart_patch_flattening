"""Distortion and fit-for-purpose metrics for a flattened patch template.

The question a surgeon (or a laser cutter) needs answered is not "how conformal
is the map" but "if I cut this 2D outline and sew its edge onto the 3D boundary,
how far off will lengths be, and where". The metrics therefore come in three
groups:

1. Per-triangle metric distortion, from the singular values sigma_1 >= sigma_2 of
   the 3D->2D Jacobian (Sander et al. 2001; Hormann & Greiner 2000):
     area ratio        sigma_1 * sigma_2            (1 = area preserved)
     conformal ratio   sigma_1 / sigma_2            (1 = angles preserved)
     L2 stretch        sqrt((s1^2 + s2^2) / 2)       (Sander's area-weighted RMS)
     isometric error   (s1 - 1)^2 + (s2 - 1)^2       (ARAP-type energy density)
   reported as area-weighted means, 95th percentiles and maxima.

2. Edge and boundary (suture-line) errors in millimetres and percent: all mesh
   edges, boundary edges only, total perimeter, and the running (cumulative)
   arclength mismatch along the boundary, which is the error a surgeon would see
   when matching marks along the suture line.

3. Template-size checks in the spirit of Kizilski et al. 2025: the longest
   boundary-to-boundary distance (length) and the largest perpendicular extent
   (width) of the 2D template are compared with the on-surface geodesic
   distances between the same vertex pairs, against a 0.1 mm / 0.5 % tolerance.

Also reported: fold-overs (inverted triangles), and shape-intrinsic developability
measures (integrated |Gaussian curvature|) that bound what any flattening can do.
"""
from __future__ import annotations

import numpy as np

from .mesh import TriMesh
from .geometric.common import jacobians, rest_edge_matrices, singular_values_2x2, count_flips

TOL_ABS_MM = 0.1     # Kizilski et al. 2025 tolerance on patch width/length change
TOL_REL = 0.005      # 0.5 %


def _wstats(x: np.ndarray, w: np.ndarray) -> dict:
    """Area-weighted mean, p95 and max of a per-face quantity."""
    w = w / w.sum()
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    p95 = x[order][np.searchsorted(cw, 0.95)]
    return dict(mean=float(np.sum(w * x)), p95=float(p95), max=float(x.max()))


def per_face_stretch(mesh: TriMesh, uv: np.ndarray):
    _, Xinv, area = rest_edge_matrices(mesh)
    J = jacobians(mesh, uv, Xinv)
    s1, s2 = singular_values_2x2(J)
    return s1, s2, area, J


def feret_pairs(uv: np.ndarray, loop: np.ndarray):
    """Boundary vertex pair with the largest 2D separation (length axis) and the pair with
    the largest extent perpendicular to it (width axis)."""
    P = uv[loop]
    d = np.linalg.norm(P[:, None] - P[None], axis=2)
    i, j = np.unravel_index(d.argmax(), d.shape)
    axis = (P[j] - P[i]) / d[i, j]
    perp = np.array([-axis[1], axis[0]])
    proj = P @ perp
    k, l = int(proj.argmin()), int(proj.argmax())
    return (int(loop[i]), int(loop[j])), (int(loop[k]), int(loop[l]))


def distortion_metrics(mesh: TriMesh, uv: np.ndarray, tol_abs: float = TOL_ABS_MM,
                       tol_rel: float = TOL_REL) -> dict:
    """All metrics for one flattening. Returns a flat dict (JSON/CSV friendly)."""
    s1, s2, area, J = per_face_stretch(mesh, uv)
    w = area
    out = {}

    # -- 1. per-triangle distortion ------------------------------------------------
    area_ratio = s1 * s2
    out["area_total_error_pct"] = float((area_ratio * area).sum() / area.sum() - 1) * 100
    for k, v in _wstats(np.abs(area_ratio - 1), w).items():
        out[f"area_ratio_absdev_{k}"] = v
    for k, v in _wstats(s1 / np.maximum(s2, 1e-12), w).items():
        out[f"conformal_ratio_{k}"] = v
    out["stretch_L2"] = float(np.sqrt(np.sum(w * (s1**2 + s2**2) / 2) / w.sum()))
    out["stretch_Linf"] = float(s1.max())
    out["stretch_min"] = float(s2.min())
    for k, v in _wstats((s1 - 1) ** 2 + (s2 - 1) ** 2, w).items():
        out[f"isometric_err_{k}"] = v
    for k, v in _wstats(np.maximum(np.abs(s1 - 1), np.abs(s2 - 1)), w).items():
        out[f"principal_stretch_absdev_{k}"] = v
    # angle error per face (max over its 3 angles), degrees
    ang3 = mesh.face_angles()
    ang2 = TriMesh(uv, mesh.F).face_angles()
    aerr = np.rad2deg(np.abs(ang2 - ang3).max(axis=1))
    for k, v in _wstats(aerr, w).items():
        out[f"angle_err_deg_{k}"] = v
    out["flipped_triangles"] = count_flips(mesh, uv)

    # -- 2. edge / suture-line errors -----------------------------------------------
    e = mesh.edges()
    l3 = np.linalg.norm(mesh.V[e[:, 0]] - mesh.V[e[:, 1]], axis=1)
    l2 = np.linalg.norm(uv[e[:, 0]] - uv[e[:, 1]], axis=1)
    rel = (l2 - l3) / l3
    out["edge_rel_err_mean_abs_pct"] = float(np.mean(np.abs(rel)) * 100)
    out["edge_rel_err_p95_abs_pct"] = float(np.percentile(np.abs(rel), 95) * 100)
    out["edge_rel_err_max_abs_pct"] = float(np.abs(rel).max() * 100)
    loop = mesh.boundary_loop()
    B3 = mesh.V[loop]; B2 = uv[loop]
    b3 = np.linalg.norm(np.roll(B3, -1, 0) - B3, axis=1)
    b2 = np.linalg.norm(np.roll(B2, -1, 0) - B2, axis=1)
    brel = (b2 - b3) / b3
    out["perimeter_3d_mm"] = float(b3.sum())
    out["perimeter_err_mm"] = float(b2.sum() - b3.sum())
    out["perimeter_err_pct"] = float((b2.sum() / b3.sum() - 1) * 100)
    out["boundary_edge_rel_err_mean_abs_pct"] = float(np.mean(np.abs(brel)) * 100)
    out["boundary_edge_rel_err_max_abs_pct"] = float(np.abs(brel).max() * 100)
    # running arclength mismatch along the suture line (start-point independent range)
    cum = np.cumsum(b2 - b3)
    out["boundary_cumulative_mismatch_mm"] = float(cum.max() - cum.min())

    # -- 3. template length / width vs geodesic (Kizilski-style check) -----------------
    (i, j), (k, l) = feret_pairs(uv, loop)
    L2d = float(np.linalg.norm(uv[i] - uv[j])); L3d = mesh.geodesic_distance(i, j)
    W2d = float(np.linalg.norm(uv[k] - uv[l])); W3d = mesh.geodesic_distance(k, l)
    out["length_2d_mm"] = L2d; out["length_geodesic_3d_mm"] = L3d
    out["width_2d_mm"] = W2d; out["width_geodesic_3d_mm"] = W3d
    out["length_err_mm"] = L2d - L3d; out["length_err_pct"] = (L2d / L3d - 1) * 100 if L3d else np.nan
    out["width_err_mm"] = W2d - W3d; out["width_err_pct"] = (W2d / W3d - 1) * 100 if W3d else np.nan
    # Note: the reference is the exact polyhedral geodesic (pygeodesic) when available;
    # with the Dijkstra fallback the 3D reference is a few percent too long.
    out["passes_size_tolerance"] = bool(
        (abs(out["length_err_mm"]) <= tol_abs or abs(out["length_err_pct"]) <= tol_rel * 100) and
        (abs(out["width_err_mm"]) <= tol_abs or abs(out["width_err_pct"]) <= tol_rel * 100))

    # -- 4. intrinsic developability of the input shape ------------------------------
    defects = mesh.angle_defects()
    interior = ~mesh.is_boundary_vertex()
    out["shape_total_abs_K_deg"] = float(np.rad2deg(np.abs(defects[interior]).sum()))
    out["shape_net_K_deg"] = float(np.rad2deg(defects[interior].sum()))
    out["shape_area_mm2"] = float(area.sum())
    return out


def per_vertex_stretch(mesh: TriMesh, uv: np.ndarray) -> dict:
    """Area-weighted vertex averages of sigma_1, sigma_2 and area ratio (for maps/plots)."""
    s1, s2, area, _ = per_face_stretch(mesh, uv)
    n = mesh.n_vertices
    acc = {k: np.zeros(n) for k in ("s1", "s2", "area_ratio")}
    wsum = np.zeros(n)
    for c in range(3):
        idx = mesh.F[:, c]
        np.add.at(wsum, idx, area)
        np.add.at(acc["s1"], idx, area * s1)
        np.add.at(acc["s2"], idx, area * s2)
        np.add.at(acc["area_ratio"], idx, area * s1 * s2)
    return {k: v / np.maximum(wsum, 1e-12) for k, v in acc.items()}


KEY_METRICS = [
    "area_total_error_pct", "stretch_L2", "stretch_Linf", "stretch_min",
    "principal_stretch_absdev_p95", "principal_stretch_absdev_max",
    "conformal_ratio_mean", "angle_err_deg_p95",
    "edge_rel_err_p95_abs_pct", "edge_rel_err_max_abs_pct",
    "boundary_edge_rel_err_max_abs_pct", "boundary_cumulative_mismatch_mm", "perimeter_err_pct",
    "length_err_mm", "width_err_mm", "passes_size_tolerance", "flipped_triangles",
]
