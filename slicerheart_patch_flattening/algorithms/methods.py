"""Flattening solvers.

All solvers share the signature ``solver(mesh, **kw) -> FlatResult`` and are
registered in ``METHODS`` so the evaluation harness can loop over them. Each
docstring states what the method preserves, what it sacrifices, and where it
comes from; see REFERENCES.md for the verified bibliography.

Summary of the family (what is minimised):

  pca_projection   nothing; orthographic projection onto the best-fit plane   (naive baseline)
  tutte / harmonic / mean_value
                   Dirichlet-type energy with the boundary pinned to a circle  (Tutte 1963;
                   Eck et al. 1995 / Pinkall & Polthier 1993; Floater 2003)
  lscm             conformal (angle) error, free boundary, two pinned points   (Levy et al. 2002)
                   -> this is what the SlicerHeart Baffle Planner currently uses
  asap / arap      as-similar / as-rigid-as-possible local-global iteration     (Liu et al. 2008)
  arap_hybrid      Liu et al.'s lambda-blend between ASAP and ARAP
  spring           mass-spring edge-length energy, quasi-static                (physical analogue)
  membrane_stvk    St. Venant-Kirchhoff membrane strain energy, quasi-static   (CPU reference for
                   the NVIDIA Warp FEM stage; same energy FEFlatten-style tools minimise)

Every method returns a template in millimetres. The *scaling* step matters for
templates: conformal maps are defined up to scale, so `lscm` rescales to match
surface area (exactly the Baffle Planner post-processing) or, optionally, to the
least-squares edge-length scale.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import minimize

from ..mesh import TriMesh
from .common import (FlatResult, Timer, canonical_pose, count_flips, default_pins, jacobians,
                     laplacian, local_frames, pin_coordinates, rescale, rest_edge_matrices,
                     solve_with_constraints)


# ---------------------------------------------------------------------------
# 0. Naive baseline
# ---------------------------------------------------------------------------
def pca_projection(mesh: TriMesh, **kw) -> FlatResult:
    """Orthographic projection onto the least-squares plane.

    Equivalent to tracing the patch from a photograph taken along the mean
    normal. Foreshortens every tilted region (area always underestimated), so it
    is the floor every real method must beat; it is included because it is what
    happens when a surgeon eyeballs a printed 3D view.
    """
    with Timer() as t:
        c = mesh.V.mean(axis=0)
        _, _, Vt = np.linalg.svd(mesh.V - c, full_matrices=False)
        uv = (mesh.V - c) @ Vt[:2].T
        uv = canonical_pose(uv)
    return FlatResult(uv, "pca_projection", {}, t.elapsed, 0, True,
                      {"flips": count_flips(mesh, uv)})


# ---------------------------------------------------------------------------
# 1. Fixed-boundary (convex-combination) maps
# ---------------------------------------------------------------------------
def _circle_boundary(mesh: TriMesh, match: str = "perimeter"):
    """Map the boundary loop onto a circle with arclength-proportional spacing.

    match='perimeter': circle circumference = 3D boundary length (suture line
    preserved on average); match='area': circle area = 3D surface area.
    """
    loop = mesh.boundary_loop()
    P = mesh.V[loop]
    seg = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
    L = seg.sum()
    if match == "perimeter":
        R = L / (2 * np.pi)
    else:
        R = np.sqrt(mesh.area() / np.pi)
    s = np.r_[0, np.cumsum(seg)[:-1]] / L
    ang = 2 * np.pi * s
    return loop, np.c_[R * np.cos(ang), R * np.sin(ang)]


def fixed_boundary(mesh: TriMesh, weights: str = "cotangent", boundary: str = "circle",
                   match: str = "perimeter", **kw) -> FlatResult:
    """Convex-combination / harmonic parameterisation with a prescribed boundary.

    The boundary is pinned to a circle and interior vertices satisfy L u = 0.
    weights='uniform'   -> Tutte's barycentric embedding (Tutte 1963): provably
                           bijective for a convex boundary, large distortion.
    weights='cotangent' -> discrete harmonic map (Eck et al. 1995), conformal in the
                           limit; can fold if cot weights go negative on bad triangles.
    weights='mean_value'-> Floater's mean value coordinates (Floater 2003): positive
                           weights (no folds) and close to harmonic quality.
    A circular boundary is a poor template boundary for elongated baffles
    (strong stretch near the rim); these are included as classical references
    and as initialisations, not as recommended template generators.
    """
    with Timer() as t:
        loop, buv = _circle_boundary(mesh, match)
        L = laplacian(mesh, weights)
        uv = solve_with_constraints(L, np.zeros((mesh.n_vertices, 2)), loop, buv)
        uv = canonical_pose(uv)
    name = {"uniform": "tutte", "cotangent": "harmonic", "mean_value": "mean_value"}[weights]
    return FlatResult(uv, name, dict(boundary=boundary, match=match), t.elapsed, 1, True,
                      {"flips": count_flips(mesh, uv)})


def tutte(mesh, **kw):      return fixed_boundary(mesh, "uniform", **kw)
def harmonic(mesh, **kw):   return fixed_boundary(mesh, "cotangent", **kw)
def mean_value(mesh, **kw): return fixed_boundary(mesh, "mean_value", **kw)


# ---------------------------------------------------------------------------
# 2. Least squares conformal maps (Levy et al. 2002)
# ---------------------------------------------------------------------------
def _lscm_matrix(mesh: TriMesh) -> sp.csr_matrix:
    """Sparse real matrix M (2m x 2n) with ||M [u; v]||^2 = sum_T A_T |CR residual|^2.

    In the local frame of triangle T with vertices p_k = (x_k, y_k), the discrete
    Cauchy-Riemann residual is r_T = (1/2A) sum_k W_k z_k with complex weights
    W_k = (y_{k+1} - y_{k+2}) + i (x_{k+2} - x_{k+1}) and z_k = u_k + i v_k.
    Rows are scaled by 1/sqrt(4A) so the energy is the area-weighted sum.
    """
    P = local_frames(mesh)
    m, n = mesh.n_faces, mesh.n_vertices
    x, y = P[..., 0], P[..., 1]
    k1 = np.array([1, 2, 0]); k2 = np.array([2, 0, 1])
    a = y[:, k1] - y[:, k2]            # Re W_k
    b = x[:, k2] - x[:, k1]            # Im W_k
    A = 0.5 * np.abs(x[:, 1] * y[:, 2] - x[:, 2] * y[:, 1])
    s = 1.0 / np.sqrt(4 * A)
    rows_re = np.repeat(2 * np.arange(m), 3)
    rows_im = rows_re + 1
    cols_u = mesh.F.ravel()
    cols_v = cols_u + n
    # Re r = sum a_k u_k - b_k v_k ; Im r = sum b_k u_k + a_k v_k
    data = np.concatenate([(a * s[:, None]).ravel(), (-b * s[:, None]).ravel(),
                           (b * s[:, None]).ravel(), (a * s[:, None]).ravel()])
    rows = np.concatenate([rows_re, rows_re, rows_im, rows_im])
    cols = np.concatenate([cols_u, cols_v, cols_u, cols_v])
    return sp.coo_matrix((data, (rows, cols)), shape=(2 * m, 2 * n)).tocsr()


def lscm(mesh: TriMesh, pins: np.ndarray | None = None, pin_uv: np.ndarray | None = None,
         scale: str = "area", **kw) -> FlatResult:
    """Least Squares Conformal Map (Levy, Petitjean, Ray & Maillot 2002).

    Minimises the discrete conformal (Cauchy-Riemann) energy with a free
    boundary; two vertices are pinned to remove the similarity ambiguity. This
    reproduces the SlicerHeart ConformalTextureMapping CLI used by the Baffle
    Planner: pins = the user's two "fixed points" (default: farthest boundary
    pair), followed by a uniform rescale so the template area equals the 3D
    surface area (``scale='area'``). ``scale='edge'`` instead picks the
    least-squares edge-length scale, which is usually a better fit for cutting.

    Preserves angles exactly in the continuous limit; area/length distortion is
    whatever the Gaussian curvature forces and is concentrated where the
    surface is most curved.
    """
    with Timer() as t:
        n = mesh.n_vertices
        if pins is None:
            pins = default_pins(mesh)
        pins = np.asarray(pins, dtype=np.int64)
        if pin_uv is None:
            pin_uv = pin_coordinates(mesh, pins)
        M = _lscm_matrix(mesh)
        fixed = np.r_[pins, pins + n]
        fixed_val = np.r_[pin_uv[:, 0], pin_uv[:, 1]]
        free = np.setdiff1d(np.arange(2 * n), fixed)
        Mf = M[:, free]
        Mp = M[:, fixed]
        rhs = -(Mp @ fixed_val)
        # normal equations (SPD on the free set); scipy's spsolve is adequate at this size
        A = (Mf.T @ Mf).tocsc()
        b = Mf.T @ rhs
        x = np.zeros(2 * n)
        x[fixed] = fixed_val
        x[free] = spla.spsolve(A, b)
        uv = np.c_[x[:n], x[n:]]
        uv, s = rescale(mesh, uv, scale)
        uv = canonical_pose(uv)
    return FlatResult(uv, "lscm", dict(scale=scale, pins=pins.tolist()), t.elapsed, 1, True,
                      {"flips": count_flips(mesh, uv), "post_scale": s})


# ---------------------------------------------------------------------------
# 3. ASAP / ARAP local-global (Liu, Zhang, Gotsman & Gortler 2008)
# ---------------------------------------------------------------------------
def _closest_rotation_2x2(J: np.ndarray) -> np.ndarray:
    """Polar-decomposition rotation of each 2x2 Jacobian (closed form)."""
    a = J[:, 0, 0] + J[:, 1, 1]
    b = J[:, 0, 1] - J[:, 1, 0]
    r = np.hypot(a, b)
    r = np.where(r < 1e-14, 1.0, r)
    c, s = a / r, -b / r      # rotation [[c, -s],[s, c]] closest to J (det > 0)
    R = np.empty_like(J)
    R[:, 0, 0] = c; R[:, 0, 1] = -s
    R[:, 1, 0] = s; R[:, 1, 1] = c
    # if J is reflected (det<0) the closest rotation still has det>0; fine for our use
    return R


def _closest_similarity_2x2(J: np.ndarray, lam: float = 0.0) -> np.ndarray:
    """Local step of Liu et al. (2008): closest L = [[a, b], [-b, a]] to J.

    lam = 0        -> ASAP (any similarity: conformal, angles only)
    lam -> inf     -> ARAP (rotation only, lengths)
    0 < lam < inf  -> hybrid: minimise |J - L|^2 + lam (a^2 + b^2 - 1)^2, solved
                      for the magnitude rho via the cubic 2 lam rho^3 + (1 - 2 lam) rho - rho0 = 0
                      (Newton iterations, vectorised).
    """
    a0 = 0.5 * (J[:, 0, 0] + J[:, 1, 1])
    b0 = 0.5 * (J[:, 0, 1] - J[:, 1, 0])
    rho0 = np.hypot(a0, b0)
    rho0 = np.where(rho0 < 1e-14, 1e-14, rho0)
    if lam == 0.0:
        rho = rho0
    elif np.isinf(lam):
        rho = np.ones_like(rho0)
    else:
        rho = np.ones_like(rho0)
        for _ in range(20):
            f = 2 * lam * rho**3 + (1 - 2 * lam) * rho - rho0
            fp = 6 * lam * rho**2 + (1 - 2 * lam)
            rho = rho - f / fp
    a = a0 * rho / rho0
    b = b0 * rho / rho0
    L = np.empty_like(J)
    L[:, 0, 0] = a; L[:, 0, 1] = b
    L[:, 1, 0] = -b; L[:, 1, 1] = a
    return L


def local_global(mesh: TriMesh, lam: float = np.inf, init: np.ndarray | None = None,
                 max_iter: int = 300, tol: float = 1e-7, name: str | None = None, **kw) -> FlatResult:
    """Local/global parameterisation (Liu et al. 2008).

    Alternates (local) fitting a rotation (ARAP, lam=inf) or similarity (ASAP,
    lam=0) to each triangle's Jacobian with (global) solving the cotangent-
    weighted Poisson system L u = b for the vertex positions that best match
    those per-triangle targets. Initialised from LSCM. ARAP minimises an
    isometric (length) distortion and is the standard "developable-sheet"
    flattening used for pattern making; for a surgeon's template it is the
    natural default because suture-line lengths are what must match.

    Convergence: stops when the relative energy decrease < tol or at max_iter.
    """
    with Timer() as t:
        n = mesh.n_vertices
        X, Xinv, area = rest_edge_matrices(mesh)
        F = mesh.F
        if init is None:
            init = lscm(mesh, scale="edge").uv
        uv = init.copy()
        # cot weights per half-edge (k, k+1) within each face = 0.5 cot(angle at k+2)
        ang = mesh.face_angles()
        cot = 0.5 / np.tan(np.clip(ang, 1e-6, np.pi - 1e-6))
        k = np.array([0, 1, 2]); kn = np.array([1, 2, 0]); ko = np.array([2, 0, 1])
        I = F[:, k].ravel(); Jn = F[:, kn].ravel(); W = cot[:, ko].ravel()
        Lap = laplacian(mesh, "cotangent").tolil()
        # pin one vertex to kill the translation null space
        pin = int(mesh.boundary_loop()[0])
        Lap[pin, :] = 0; Lap[pin, pin] = 1.0
        lu = spla.splu(Lap.tocsc())
        # rest-frame edge vectors (k -> k+1) in local 2D coordinates
        P = local_frames(mesh)
        E = P[:, kn] - P[:, k]                                       # (m,3,2)
        it = 0; converged = False; energy_prev = np.inf; energy = np.inf
        for it in range(1, max_iter + 1):
            J = jacobians(mesh, uv, Xinv)
            Lt = _closest_similarity_2x2(J, lam)
            # local-global energy sum_T A_T |J_T - L_T|_F^2 (Liu et al. 2008, eq. 1/4)
            energy = float(np.sum(area * np.sum((J - Lt) ** 2, axis=(1, 2))))
            # target edge vectors L_T e
            tgt = np.einsum("fij,fkj->fki", Lt, E).reshape(-1, 2)       # (3m,2)
            rhs = np.zeros((n, 2))
            np.add.at(rhs, I, W[:, None] * tgt)
            np.add.at(rhs, Jn, -W[:, None] * tgt)
            rhs[pin] = uv[pin]
            uv = np.c_[lu.solve(rhs[:, 0]), lu.solve(rhs[:, 1])]
            # relative energy decrease is a pose-invariant stopping rule (vertex motion
            # keeps drifting by a slow global rotation long after the shape has converged)
            if energy_prev - energy < tol * max(energy, 1e-12):
                converged = True
                break
            energy_prev = energy
        uv = canonical_pose(uv)
    if name is None:
        name = "arap" if np.isinf(lam) else ("asap" if lam == 0 else f"hybrid_lam{lam:g}")
    return FlatResult(uv, name, dict(lam=float(lam), tol=tol), t.elapsed, it, converged,
                      {"flips": count_flips(mesh, uv), "energy": energy})


def arap(mesh, **kw):          return local_global(mesh, lam=np.inf, name="arap", **kw)
def asap(mesh, **kw):          return local_global(mesh, lam=0.0, name="asap", **kw)
def arap_hybrid(mesh, lam=1.0, **kw): return local_global(mesh, lam=lam, name="arap_hybrid", **kw)


# ---------------------------------------------------------------------------
# 4. Mass-spring relaxation (edge lengths only)
# ---------------------------------------------------------------------------
def spring(mesh: TriMesh, init: np.ndarray | None = None, max_iter: int = 500, tol: float = 1e-8,
           **kw) -> FlatResult:
    """Quasi-static mass-spring flattening: minimise sum_e (|u_i - u_j| - l_e)^2 / l_e.

    The classic "stretchable rubber sheet" picture: every mesh edge is a linear
    spring at its 3D rest length, and the 2D equilibrium is found with L-BFGS
    (analytic gradient). Initialised from ARAP, so it mostly refines edge
    lengths at the cost of angles. No shear stiffness, so it can drift into
    folds on strongly curved patches; the membrane model below is the stiffer,
    better-posed version of the same idea.
    """
    with Timer() as t:
        e = mesh.edges()
        l0 = np.linalg.norm(mesh.V[e[:, 0]] - mesh.V[e[:, 1]], axis=1)
        w = 1.0 / l0
        if init is None:
            init = arap(mesh).uv
        n = mesh.n_vertices

        def fg(x):
            uv = x.reshape(n, 2)
            d = uv[e[:, 0]] - uv[e[:, 1]]
            l = np.linalg.norm(d, axis=1)
            r = l - l0
            E = np.sum(w * r * r)
            g_e = (2 * w * r / np.maximum(l, 1e-12))[:, None] * d
            g = np.zeros((n, 2))
            np.add.at(g, e[:, 0], g_e)
            np.add.at(g, e[:, 1], -g_e)
            return E, g.ravel()

        res = minimize(fg, init.ravel(), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=max_iter, ftol=tol, gtol=1e-8))
        uv = canonical_pose(res.x.reshape(n, 2))
    return FlatResult(uv, "spring", dict(max_iter=max_iter), t.elapsed, int(res.nit), bool(res.success),
                      {"flips": count_flips(mesh, uv), "energy": float(res.fun)})


# ---------------------------------------------------------------------------
# 5. St. Venant-Kirchhoff membrane (CPU reference for the Warp FEM stage)
# ---------------------------------------------------------------------------
def membrane_stvk(mesh: TriMesh, nu: float = 0.3, init: np.ndarray | None = None,
                  max_iter: int = 500, tol: float = 1e-10, **kw) -> FlatResult:
    """Quasi-static linear-membrane (St. Venant-Kirchhoff) flattening.

    Each triangle is a constant-strain membrane element whose rest shape is the
    3D triangle. With deformation gradient F = U X^{-1} and Green strain
    E = (F^T F - I)/2, the plane-stress StVK energy density is
        W = mu |E|_F^2 + (lambda/2) tr(E)^2,  mu = 1/(2(1+nu)), lambda = nu/(1-nu^2)
    (Young's modulus 1; the result is independent of E because there are no
    external loads). The template is the 2D configuration of minimum stored
    energy, found with L-BFGS from the ARAP solution. This is the same problem
    FE flattening tools solve (e.g. the Composites "FEFlatten" step used by
    Kizilski et al. 2025), without pressure, contact or material anisotropy;
    those are the Warp stage. Gradient: dW/dU = A P X^{-T}, P = F S,
    S = 2 mu E + lambda tr(E) I.

    Compared with ARAP it penalises shear and area change through one
    consistent constitutive law and gives strain fields that are directly
    interpretable (principal stretches = what the fabric would have to do).
    """
    with Timer() as t:
        X, Xinv, area = rest_edge_matrices(mesh)
        F = mesh.F
        n = mesh.n_vertices
        mu = 1.0 / (2 * (1 + nu))
        lam = nu / (1 - nu**2)
        XinvT = np.transpose(Xinv, (0, 2, 1))
        if init is None:
            init = arap(mesh).uv

        def fg(x):
            uv = x.reshape(n, 2)
            u0, u1, u2 = uv[F[:, 0]], uv[F[:, 1]], uv[F[:, 2]]
            U = np.stack([u1 - u0, u2 - u0], axis=2)
            Fd = U @ Xinv
            C = np.transpose(Fd, (0, 2, 1)) @ Fd
            Eg = 0.5 * (C - np.eye(2))
            trE = Eg[:, 0, 0] + Eg[:, 1, 1]
            W = mu * np.einsum("fij,fij->f", Eg, Eg) + 0.5 * lam * trE**2
            Etot = np.sum(area * W)
            S = 2 * mu * Eg + lam * trE[:, None, None] * np.eye(2)
            Pk = Fd @ S
            G = area[:, None, None] * (Pk @ XinvT)          # dW/dU, columns = edge gradients
            g = np.zeros((n, 2))
            g1 = G[:, :, 0]; g2 = G[:, :, 1]
            np.add.at(g, F[:, 1], g1)
            np.add.at(g, F[:, 2], g2)
            np.add.at(g, F[:, 0], -(g1 + g2))
            return Etot, g.ravel()

        res = minimize(fg, init.ravel(), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=max_iter, ftol=tol, gtol=1e-9))
        uv = canonical_pose(res.x.reshape(n, 2))
    return FlatResult(uv, "membrane_stvk", dict(nu=nu), t.elapsed, int(res.nit), bool(res.success),
                      {"flips": count_flips(mesh, uv), "energy": float(res.fun)})


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
METHODS = {
    "pca_projection": pca_projection,
    "tutte": tutte,
    "harmonic": harmonic,
    "mean_value": mean_value,
    "lscm": lscm,                       # Baffle Planner equivalent (area-scaled)
    "lscm_edge": lambda m, **kw: lscm(m, scale="edge", **kw),
    "asap": asap,
    "arap": arap,
    "arap_hybrid": arap_hybrid,
    "spring": spring,
    "membrane_stvk": membrane_stvk,
}


def flatten(mesh: TriMesh, method: str = "arap", **kw) -> FlatResult:
    """Flatten `mesh` with the named method (see METHODS). Default: ARAP, the recommended
    template method after the benchmarks in README.md."""
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}; available: {sorted(METHODS)}")
    res = METHODS[method](mesh, **kw)
    if res.method != method:
        res.method = method
    return res
