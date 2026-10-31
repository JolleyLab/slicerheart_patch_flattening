"""Synthetic 3D patch / baffle shape generators.

All sizes are in millimetres and chosen for infants and young children
(roughly 5-15 kg), the population in Vigil et al. (DORV baffle, 7-8 kg) and the
usual age for atrial switch and PAPVR repair.

Shape families
--------------
analytic developable      half-cylinder, cone segments            -> zero-distortion controls
analytic non-developable  spherical cap, saddle, ellipsoid cap     -> known Gaussian curvature
dorv_tunnel               partial tube swept along a curved, tapering, twisting centerline
baffle_planner (TPS)      flat disk thin-plate-spline warped to a 3D boundary curve plus
                          interior "surface points", exactly as the SlicerHeart Baffle
                          Planner builds a baffle (Vigil et al. 2021)
mustard                   trouser-shaped intra-atrial baffle on a curved sheet
papvr                     ovoid half-tunnel directing anomalous pulmonary veins to the LA

Each generator returns a TriMesh with meta describing family, parameters and a
``developable`` flag.
"""
from __future__ import annotations

import numpy as np
import vtk
from vtk.util import numpy_support as vnp
from shapely.geometry import Polygon, Point

from ..mesh import TriMesh


# --------------------------------------------------------------------------
# 2D domain triangulation
# --------------------------------------------------------------------------
def _delaunay_2d(points2d: np.ndarray) -> np.ndarray:
    pts = vtk.vtkPoints()
    P3 = np.c_[points2d, np.zeros(len(points2d))]
    pts.SetData(vnp.numpy_to_vtk(np.ascontiguousarray(P3), deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    d = vtk.vtkDelaunay2D()
    d.SetInputData(pd)
    d.SetTolerance(1e-9)
    d.Update()
    out = d.GetOutput()
    conn = vnp.vtk_to_numpy(out.GetPolys().GetConnectivityArray())
    offs = vnp.vtk_to_numpy(out.GetPolys().GetOffsetsArray())
    assert np.all(np.diff(offs) == 3), "vtkDelaunay2D produced non-triangular cells"
    return conn.reshape(-1, 3).astype(np.int64)


def triangulate_polygon(boundary: np.ndarray, h: float, jitter: float = 0.15,
                        seed: int = 0) -> TriMesh:
    """Uniform triangulation of the region bounded by a closed 2D polyline.

    boundary : (k, 2) closed polyline (last point != first point)
    h        : target edge length
    Interior points are placed on a hexagonal lattice (jittered) at least 0.6 h
    from the boundary; boundary is resampled at spacing h. Delaunay triangles
    whose centroid is outside the polygon are discarded.
    """
    rng = np.random.default_rng(seed)
    poly = Polygon(boundary)
    if not poly.is_valid:
        poly = poly.buffer(0)
    # resample boundary at spacing ~h
    ring = np.asarray(poly.exterior.coords)[:-1]
    seg = np.linalg.norm(np.roll(ring, -1, axis=0) - ring, axis=1)
    L = seg.sum()
    nb = max(int(round(L / h)), 12)
    s_target = np.linspace(0, L, nb, endpoint=False)
    cum = np.r_[0, np.cumsum(seg)]
    bpts = []
    for s in s_target:
        i = np.searchsorted(cum, s, side="right") - 1
        i = min(i, len(ring) - 1)
        t = (s - cum[i]) / max(seg[i], 1e-12)
        bpts.append(ring[i] + t * (ring[(i + 1) % len(ring)] - ring[i]))
    bpts = np.asarray(bpts)
    # interior hex lattice
    minx, miny, maxx, maxy = poly.bounds
    dy = h * np.sqrt(3) / 2
    xs = np.arange(minx - h, maxx + h, h)
    ys = np.arange(miny - h, maxy + h, dy)
    inner = poly.buffer(-0.6 * h)
    ipts = []
    for j, y in enumerate(ys):
        off = 0.5 * h if j % 2 else 0.0
        for x in xs:
            p = np.array([x + off, y]) + jitter * h * (rng.random(2) - 0.5)
            if inner.contains(Point(p)):
                ipts.append(p)
    ipts = np.asarray(ipts).reshape(-1, 2)
    P = np.vstack([bpts, ipts])
    F = _delaunay_2d(P)
    cen = P[F].mean(axis=1)
    keep = np.array([poly.contains(Point(c)) for c in cen])
    F = F[keep]
    # enforce CCW orientation in 2D (positive signed area)
    a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    sa = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    F[sa < 0] = F[sa < 0][:, ::-1]
    m = TriMesh(P, F).remove_unreferenced()
    m.boundary_loop()  # raises if not a single loop
    return m


def disk_mesh(radius: float, h: float, seed: int = 0) -> TriMesh:
    t = np.linspace(0, 2 * np.pi, max(int(2 * np.pi * radius / h), 16), endpoint=False)
    return triangulate_polygon(np.c_[radius * np.cos(t), radius * np.sin(t)], h, seed=seed)


# --------------------------------------------------------------------------
# 2D outline families (template-space outlines before lifting to 3D)
# --------------------------------------------------------------------------
def superellipse(a: float, b: float, n: float = 2.5, k: int = 128,
                 harmonics: np.ndarray | None = None) -> np.ndarray:
    """Rounded-rectangle / ellipse outline with optional radial harmonic perturbation."""
    t = np.linspace(0, 2 * np.pi, k, endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    x = a * np.sign(ct) * np.abs(ct) ** (2 / n)
    y = b * np.sign(st) * np.abs(st) ** (2 / n)
    r = 1.0
    if harmonics is not None:
        for m, (amp, ph) in enumerate(harmonics, start=2):
            r = r + amp * np.cos(m * t + ph)
    return np.c_[x * r, y * r]


def trouser_outline(length: float, top_width: float, leg_width: float, leg_length: float,
                    crotch_round: float = 4.0, k: int = 160) -> np.ndarray:
    """Mustard-style 'trouser' baffle outline, long axis along y."""
    from shapely.geometry import box
    from shapely.ops import unary_union
    body = box(-top_width / 2, leg_length, top_width / 2, length)
    gap = top_width - 2 * leg_width
    legL = box(-top_width / 2, 0, -top_width / 2 + leg_width, leg_length + 1)
    legR = box(top_width / 2 - leg_width, 0, top_width / 2, leg_length + 1)
    shape = unary_union([body, legL, legR])
    # round everything (buffer out then in)
    shape = shape.buffer(crotch_round, join_style=1).buffer(-2 * crotch_round, join_style=1).buffer(crotch_round, join_style=1)
    ring = np.asarray(shape.exterior.coords)[:-1]
    ring = ring - ring.mean(axis=0)
    return ring


# --------------------------------------------------------------------------
# Lifting maps (u,v) -> 3D
# --------------------------------------------------------------------------
def lift_cylinder(P, R):
    u, v = P[:, 0], P[:, 1]
    return np.c_[R * np.sin(u / R), v, R * (1 - np.cos(u / R))]


def lift_cone(P, R0, slope):
    """Developable cone: u along circumference at base radius R0, v along slant; radius shrinks with v."""
    u, v = P[:, 0], P[:, 1]
    r = R0 - slope * v
    th = u / R0
    return np.c_[r * np.sin(th), v * np.sqrt(1 - slope**2), r * np.cos(th) - R0]


def lift_sphere(P, R):
    """Azimuthal-equidistant inverse: geodesic polar coords on sphere of radius R."""
    rho = np.linalg.norm(P, axis=1)
    phi = np.arctan2(P[:, 1], P[:, 0])
    th = rho / R
    return np.c_[R * np.sin(th) * np.cos(phi), R * np.sin(th) * np.sin(phi), R * (1 - np.cos(th))]


def lift_quadric(P, kx, ky):
    u, v = P[:, 0], P[:, 1]
    return np.c_[u, v, 0.5 * (kx * u**2 + ky * v**2)]


def lift_ellipsoid(P, a, b, c):
    """Graph of the upper ellipsoid cap z = c*sqrt(1 - x^2/a^2 - y^2/b^2)."""
    u, v = P[:, 0], P[:, 1]
    w = np.clip(1 - u**2 / a**2 - v**2 / b**2, 0, None)
    return np.c_[u, v, c * np.sqrt(w)]


# --------------------------------------------------------------------------
# Tunnel / trough sweep (DORV intracardiac baffle, PAPVR half-tunnel)
# --------------------------------------------------------------------------
def _frames_along_arc(s: np.ndarray, length: float, bend_radius: float, bend_axis_tilt: float = 0.0,
                      out_of_plane: float = 0.0):
    """Centerline C(s) that bends in a plane (radius bend_radius) with an optional
    out-of-plane sinusoidal component; returns C, T, N, B (rotation-minimising frame)."""
    if np.isinf(bend_radius):
        C = np.c_[np.zeros_like(s), s, np.zeros_like(s)]
    else:
        phi = s / bend_radius
        C = np.c_[np.zeros_like(s), bend_radius * np.sin(phi), bend_radius * (1 - np.cos(phi))]
    if out_of_plane:
        C[:, 0] += out_of_plane * np.sin(np.pi * s / length)
    if bend_axis_tilt:
        ca, sa = np.cos(bend_axis_tilt), np.sin(bend_axis_tilt)
        Rm = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
        C = C @ Rm.T
    # tangents
    T = np.gradient(C, s, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    # rotation-minimising frame (double reflection)
    N = np.zeros_like(T)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, T[0])) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    N[0] = ref - np.dot(ref, T[0]) * T[0]
    N[0] /= np.linalg.norm(N[0])
    for i in range(1, len(s)):
        v1 = C[i] - C[i - 1]
        c1 = np.dot(v1, v1)
        rL = N[i - 1] - (2 / c1) * np.dot(v1, N[i - 1]) * v1
        tL = T[i - 1] - (2 / c1) * np.dot(v1, T[i - 1]) * v1
        v2 = T[i] - tL
        c2 = np.dot(v2, v2)
        N[i] = rL - (2 / max(c2, 1e-30)) * np.dot(v2, rL) * v2 if c2 > 1e-30 else rL
        N[i] /= np.linalg.norm(N[i])
    B = np.cross(T, N)
    return C, T, N, B


def lift_tunnel(P, length, r0, r1, bend_radius, twist_deg=0.0, out_of_plane=0.0,
                bend_axis_tilt=0.0, theta0_deg=0.0):
    """Map (s, w) with s in [0,length], w = arc position around the tube (mm, 0 = trough
    bottom) onto a tube of radius r(s) (linear taper r0->r1) around a curved centerline."""
    s = np.clip(P[:, 0], 0, length)
    w = P[:, 1]
    ns = 400
    sg = np.linspace(0, length, ns)
    C, T, N, B = _frames_along_arc(sg, length, bend_radius, bend_axis_tilt, out_of_plane)
    idx = np.clip(np.searchsorted(sg, s) - 1, 0, ns - 2)
    t = (s - sg[idx]) / (sg[1] - sg[0])
    Cs = C[idx] * (1 - t)[:, None] + C[idx + 1] * t[:, None]
    Ns = N[idx] * (1 - t)[:, None] + N[idx + 1] * t[:, None]
    Bs = B[idx] * (1 - t)[:, None] + B[idx + 1] * t[:, None]
    Ns /= np.linalg.norm(Ns, axis=1, keepdims=True)
    Bs /= np.linalg.norm(Bs, axis=1, keepdims=True)
    r = r0 + (r1 - r0) * s / length
    th = w / r + np.deg2rad(theta0_deg) + np.deg2rad(twist_deg) * s / length
    return Cs + r[:, None] * (np.cos(th)[:, None] * Ns + np.sin(th)[:, None] * Bs)


# --------------------------------------------------------------------------
# Thin-plate-spline "Baffle Planner" construction (Vigil et al. 2021)
# --------------------------------------------------------------------------
def tps_warp_points(src: np.ndarray, dst: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Warp 3D points P with the VTK thin-plate spline (R basis) defined by landmark pairs."""
    sp = vtk.vtkPoints(); sp.SetData(vnp.numpy_to_vtk(np.ascontiguousarray(src, dtype=np.float64), deep=True))
    tp = vtk.vtkPoints(); tp.SetData(vnp.numpy_to_vtk(np.ascontiguousarray(dst, dtype=np.float64), deep=True))
    tps = vtk.vtkThinPlateSplineTransform()
    tps.SetSourceLandmarks(sp)
    tps.SetTargetLandmarks(tp)
    tps.SetBasisToR()
    tps.Update()
    out = np.empty_like(P)
    for i, p in enumerate(P):
        out[i] = tps.TransformPoint(float(p[0]), float(p[1]), float(p[2]))
    return out


def baffle_planner_patch(boundary3d: np.ndarray, h: float, surface_points: np.ndarray | None = None,
                         disk_radius: float | None = None, seed: int = 0) -> TriMesh:
    """Build a patch exactly like SlicerHeart's Baffle Planner:
    a flat disk is warped by a thin-plate spline so that equally spaced points on its rim
    land on the 3D boundary curve; optional interior landmarks (surface points) pull the
    interior off the minimal-bending-energy surface.

    boundary3d : (k,3) closed 3D polyline (ordered)
    surface_points : (j, 2+3) rows of [u, v, x, y, z]: disk-space source (u,v) and 3D target
    """
    k = len(boundary3d)
    seg = np.linalg.norm(np.roll(boundary3d, -1, axis=0) - boundary3d, axis=1)
    perim = seg.sum()
    if disk_radius is None:
        disk_radius = perim / (2 * np.pi)
    # arclength-parameterised rim landmarks
    cum = np.r_[0, np.cumsum(seg)][:-1] / perim
    ang = 2 * np.pi * cum
    src = np.c_[disk_radius * np.cos(ang), disk_radius * np.sin(ang), np.zeros(k)]
    dst = boundary3d.copy()
    if surface_points is not None and len(surface_points):
        src = np.vstack([src, np.c_[surface_points[:, :2], np.zeros(len(surface_points))]])
        dst = np.vstack([dst, surface_points[:, 2:5]])
    disk = disk_mesh(disk_radius, h, seed=seed)
    V3 = tps_warp_points(src, dst, np.c_[disk.V, np.zeros(disk.n_vertices)])
    m = TriMesh(V3, disk.F.copy())
    return m


# --------------------------------------------------------------------------
# Convenience: parametric patch from 2D outline + lift
# --------------------------------------------------------------------------
def parametric_patch(outline2d: np.ndarray, lift, h: float, seed: int = 0, **kw) -> TriMesh:
    dom = triangulate_polygon(outline2d, h, seed=seed)
    V3 = lift(dom.V, **kw)
    return TriMesh(V3, dom.F.copy())
