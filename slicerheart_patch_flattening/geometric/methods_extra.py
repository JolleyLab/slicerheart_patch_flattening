"""Additional flattening solvers explored after the first benchmark.

  symmetric_dirichlet   isometric energy with a fold-over barrier (Smith & Schaefer 2015;
                        the energy used by SLIM, Rabinovich et al. 2017)
  membrane_suture       St. Venant-Kirchhoff membrane + strong length springs on the
                        boundary edges only: the suture line is forced to its 3D length,
                        all unavoidable distortion goes to the interior
  bff_lite              boundary-first flattening with boundary scale factor u = 0
                        (every boundary edge keeps its 3D length), interior conformal
                        (Sawhney & Crane 2017). Simplified: the boundary curve is closed
                        by a least-squares correction and the interior is the harmonic
                        extension of the boundary curve (the paper's conformal extension
                        differs only by the conjugate-harmonic correction).

Each is registered in METHODS by algorithms/__init__.py.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

from ..mesh import TriMesh
from .common import (FlatResult, Timer, canonical_pose, count_flips, laplacian, rest_edge_matrices,
                     solve_with_constraints)
from .methods import arap, lscm


# ---------------------------------------------------------------------------
# 1. Symmetric Dirichlet
# ---------------------------------------------------------------------------
def symmetric_dirichlet(mesh: TriMesh, init: np.ndarray | None = None, max_iter: int = 500,
                        tol: float = 1e-10, **kw) -> FlatResult:
    """Minimise sum_f A_f (|J_f|_F^2 + |J_f^{-1}|_F^2) = sum A (s1^2 + s2^2 + 1/s1^2 + 1/s2^2).

    Symmetric in stretch and compression and infinite for a degenerate or flipped
    triangle, so an L-BFGS line search cannot cross a fold (a barrier the ARAP
    local-global iteration lacks). Minimum 4 per unit area at an isometry.
    Gradient: d/dJ |J|^2 = 2J ;  d/dJ |J^-1|^2 = -2 J^-T J^-1 J^-T.
    """
    with Timer() as t:
        X, Xinv, area = rest_edge_matrices(mesh)
        F = mesh.F; n = mesh.n_vertices
        XinvT = np.transpose(Xinv, (0, 2, 1))
        if init is None:
            init = arap(mesh).uv

        def fg(x):
            uv = x.reshape(n, 2)
            u0, u1, u2 = uv[F[:, 0]], uv[F[:, 1]], uv[F[:, 2]]
            U = np.stack([u1 - u0, u2 - u0], axis=2)
            J = U @ Xinv
            det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
            if np.any(det <= 1e-12):
                return np.inf, np.zeros(2 * n)
            Jinv = np.empty_like(J)
            Jinv[:, 0, 0] = J[:, 1, 1]; Jinv[:, 0, 1] = -J[:, 0, 1]
            Jinv[:, 1, 0] = -J[:, 1, 0]; Jinv[:, 1, 1] = J[:, 0, 0]
            Jinv /= det[:, None, None]
            W = np.einsum("fij,fij->f", J, J) + np.einsum("fij,fij->f", Jinv, Jinv)
            E = np.sum(area * W)
            JinvT = np.transpose(Jinv, (0, 2, 1))
            dW = 2 * J - 2 * (JinvT @ Jinv @ JinvT)
            G = area[:, None, None] * (dW @ XinvT)
            g = np.zeros((n, 2))
            g1 = G[:, :, 0]; g2 = G[:, :, 1]
            np.add.at(g, F[:, 1], g1); np.add.at(g, F[:, 2], g2); np.add.at(g, F[:, 0], -(g1 + g2))
            return E, g.ravel()

        res = minimize(fg, init.ravel(), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=max_iter, ftol=tol, gtol=1e-9))
        uv = canonical_pose(res.x.reshape(n, 2))
    return FlatResult(uv, "symmetric_dirichlet", {}, t.elapsed, int(res.nit), bool(res.success),
                      {"flips": count_flips(mesh, uv), "energy": float(res.fun)})


# ---------------------------------------------------------------------------
# 2. Membrane + suture-line length constraint
# ---------------------------------------------------------------------------
def membrane_suture(mesh: TriMesh, nu: float = 0.3, suture_weight: float = 50.0,
                    init: np.ndarray | None = None, max_iter: int = 800, tol: float = 1e-11, **kw) -> FlatResult:
    """StVK membrane energy (as membrane_stvk) plus w * sum_{boundary edges} (|du| - l)^2 / l.

    The boundary springs are stiff relative to the membrane (Young's modulus 1 on
    ~1 mm^2 triangles vs suture_weight per mm), so the template's rim reproduces
    the 3D suture-line lengths edge by edge; the interior then carries the
    distortion the curvature forces. This mimics what happens physically when a
    patch is sutured to a fixed-length boundary: the fabric in the middle takes
    the strain.
    """
    with Timer() as t:
        X, Xinv, area = rest_edge_matrices(mesh)
        F = mesh.F; n = mesh.n_vertices
        mu = 1.0 / (2 * (1 + nu)); lam = nu / (1 - nu**2)
        XinvT = np.transpose(Xinv, (0, 2, 1))
        be = mesh.boundary_edges()
        l0 = np.linalg.norm(mesh.V[be[:, 0]] - mesh.V[be[:, 1]], axis=1)
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
            E = np.sum(area * W)
            S = 2 * mu * Eg + lam * trE[:, None, None] * np.eye(2)
            G = area[:, None, None] * ((Fd @ S) @ XinvT)
            g = np.zeros((n, 2))
            g1 = G[:, :, 0]; g2 = G[:, :, 1]
            np.add.at(g, F[:, 1], g1); np.add.at(g, F[:, 2], g2); np.add.at(g, F[:, 0], -(g1 + g2))
            # boundary springs
            d = uv[be[:, 0]] - uv[be[:, 1]]
            l = np.linalg.norm(d, axis=1)
            r = l - l0
            E += suture_weight * np.sum(r * r / l0)
            ge = (2 * suture_weight * r / (l0 * np.maximum(l, 1e-12)))[:, None] * d
            np.add.at(g, be[:, 0], ge); np.add.at(g, be[:, 1], -ge)
            return E, g.ravel()

        res = minimize(fg, init.ravel(), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=max_iter, ftol=tol, gtol=1e-9))
        uv = canonical_pose(res.x.reshape(n, 2))
    return FlatResult(uv, "membrane_suture", dict(nu=nu, suture_weight=suture_weight), t.elapsed,
                      int(res.nit), bool(res.success), {"flips": count_flips(mesh, uv), "energy": float(res.fun)})


# ---------------------------------------------------------------------------
# 3. Boundary-first flattening (simplified)
# ---------------------------------------------------------------------------
def bff_lite(mesh: TriMesh, boundary_scale: str = "lengths", **kw) -> FlatResult:
    """Boundary-first flattening with prescribed boundary scale factors (Sawhney & Crane 2017).

    Steps (discrete Yamabe problem, cotangent Laplacian L):
      1. u on the boundary is prescribed: u_B = 0 -> boundary edges keep their 3D length.
      2. Interior scale factor from the Poisson equation  L u = K  (K = angle defects of
         interior vertices; the target flat curvature is 0).
      3. Dirichlet-to-Neumann: the boundary normal derivative h = (L u)_B - K_B gives the
         new boundary geodesic curvature  k~ = k - h  (k = pi - angle sum at boundary vertices).
      4. The flat boundary polygon is integrated from the lengths l~ = e^{(u_i+u_j)/2} l and
         turning angles k~; the (small) closure error is removed by a least-squares
         correction distributed along the loop.
      5. Interior vertices: harmonic extension (L x = 0 with the boundary polygon as
         Dirichlet data). The paper's conformal extension adds a conjugate-harmonic
         correction; the harmonic extension is a close approximation when u is smooth.
    Result: perimeter and every boundary edge length exactly preserved (up to the
    closure correction), conformal-like interior, one sparse factorisation.
    """
    with Timer() as t:
        n = mesh.n_vertices
        loop = mesh.boundary_loop()
        k = len(loop)
        L = laplacian(mesh, "cotangent")
        defects = mesh.angle_defects()              # interior: 2pi - sum, boundary: pi - sum
        isb = mesh.is_boundary_vertex()
        Kint = np.where(isb, 0.0, defects)
        # 1-2. interior scale factor with u_B = 0 (lengths) ; Poisson: L u = K
        uB = np.zeros(k)
        u = solve_with_constraints(L, Kint, loop, uB)
        # 3. new boundary geodesic curvature  k~ = k - h,  h = (L u - K)_B
        h = (L @ u - Kint)[loop]
        kB = defects[loop]
        k_new = kB - h
        # 4. integrate the boundary polygon
        P3 = mesh.V[loop]
        l3 = np.linalg.norm(np.roll(P3, -1, 0) - P3, axis=1)
        if boundary_scale == "lengths":
            l_new = l3 * np.exp(0.5 * (u[loop] + np.roll(u[loop], -1)))
        else:
            raise ValueError(boundary_scale)
        # total turning must be 2*pi; distribute any discrete mismatch evenly
        k_new = k_new + (2 * np.pi - k_new.sum()) / k
        theta = np.cumsum(k_new)                      # direction of edge i after turning at vertex i
        dirs = np.c_[np.cos(theta), np.sin(theta)]
        steps = dirs * l_new[:, None]
        gap = steps.sum(axis=0)                       # closure error
        # least-squares closure: subtract gap distributed proportionally to edge length
        steps = steps - (l_new / l_new.sum())[:, None] * gap[None]
        poly = np.cumsum(steps, axis=0)
        poly = np.vstack([np.zeros((1, 2)), poly[:-1]])
        # 5. harmonic extension of both coordinates
        uv = solve_with_constraints(L, np.zeros((n, 2)), loop, poly)
        uv = canonical_pose(uv)
    return FlatResult(uv, "bff_lite", dict(boundary_scale=boundary_scale), t.elapsed, 1, True,
                      {"flips": count_flips(mesh, uv), "closure_error_mm": float(np.linalg.norm(gap))})


EXTRA_METHODS = {
    "symmetric_dirichlet": symmetric_dirichlet,
    "membrane_suture": membrane_suture,
    "bff_lite": bff_lite,
}


# ---------------------------------------------------------------------------
# 4. Discrete conformal equivalence (CETM / discrete Yamabe) with exact layout
# ---------------------------------------------------------------------------
def _flat_angles(mesh: TriMesh, lnew: np.ndarray):
    """Interior angles of each face from per-face edge lengths lnew (m,3) ordered opposite v0,v1,v2."""
    l0, l1, l2 = lnew[:, 0], lnew[:, 1], lnew[:, 2]
    c0 = (l1**2 + l2**2 - l0**2) / (2 * l1 * l2)
    c1 = (l0**2 + l2**2 - l1**2) / (2 * l0 * l2)
    c2 = (l0**2 + l1**2 - l2**2) / (2 * l0 * l1)
    C = np.stack([c0, c1, c2], 1)
    bad = np.abs(C) > 1.0
    return np.arccos(np.clip(C, -1, 1)), bad.any()


def cetm(mesh: TriMesh, max_iter: int = 50, tol: float = 1e-9, **kw) -> FlatResult:
    """Conformal equivalence of triangle meshes (Springborn, Schroeder & Pinkall 2008).

    Finds per-vertex log scale factors u such that the metric l~_ij = l_ij exp((u_i+u_j)/2)
    is flat at every interior vertex (angle defect 0), with u = 0 prescribed on the
    boundary so every boundary edge keeps its 3D length. Newton's method on the
    discrete Yamabe residual, with Jacobian = half the cotangent Laplacian of the
    current metric. The flat metric is then laid out exactly in the plane face by
    face (breadth-first), which is consistent because the metric has no curvature.

    Result: an exactly flat, discretely conformal template whose boundary edge
    lengths equal the 3D ones. Fails (returns converged=False) if the new metric
    violates a triangle inequality, which happens for extreme curvature with u=0.
    """
    with Timer() as t:
        n = mesh.n_vertices
        F = mesh.F
        L0 = mesh.edge_lengths()                       # (m,3) opposite v0,v1,v2
        isb = mesh.is_boundary_vertex()
        free = np.where(~isb)[0]
        u = np.zeros(n)
        converged = False
        it = 0
        for it in range(1, max_iter + 1):
            # new lengths: edge opposite vertex k joins the other two vertices
            uf = u[F]
            lnew = L0 * np.exp(0.5 * np.stack([uf[:, 1] + uf[:, 2], uf[:, 0] + uf[:, 2], uf[:, 0] + uf[:, 1]], 1))
            ang, bad = _flat_angles(mesh, lnew)
            if bad:
                break
            s = np.zeros(n); np.add.at(s, F.ravel(), ang.ravel())
            K = 2 * np.pi - s                          # interior defects of the new metric
            r = K[free]
            if np.abs(r).max() < tol:
                converged = True
                break
            # Jacobian dK/du = cotangent Laplacian of the new metric (checked by finite differences)
            cot = 1.0 / np.tan(np.clip(ang, 1e-6, np.pi - 1e-6))
            I = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
            Jj = np.concatenate([F[:, 2], F[:, 0], F[:, 1]])
            W = 0.5 * np.concatenate([cot[:, 0], cot[:, 1], cot[:, 2]])
            Wm = sp.coo_matrix((W, (I, Jj)), shape=(n, n)); Wm = (Wm + Wm.T).tocsr()
            Lap = sp.diags(np.asarray(Wm.sum(1)).ravel()) - Wm
            Lff = Lap[free][:, free].tocsc()
            du = sp.linalg.spsolve(Lff, r)
            u[free] -= du
        # layout of the flat metric
        uf = u[F]
        lnew = L0 * np.exp(0.5 * np.stack([uf[:, 1] + uf[:, 2], uf[:, 0] + uf[:, 2], uf[:, 0] + uf[:, 1]], 1))
        uv = _layout_flat_metric(mesh, lnew)
        uv = canonical_pose(uv)
    return FlatResult(uv, "cetm", {}, t.elapsed, it, converged,
                      {"flips": count_flips(mesh, uv), "max_abs_u": float(np.abs(u).max())})


def _layout_flat_metric(mesh: TriMesh, lnew: np.ndarray) -> np.ndarray:
    """Place triangles in the plane from per-face edge lengths, BFS over the dual graph."""
    from collections import deque
    F = mesh.F; n = mesh.n_vertices; m = mesh.n_faces
    # face adjacency through shared edges
    edge_faces = {}
    for fi, f in enumerate(F):
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            edge_faces.setdefault((min(a, b), max(a, b)), []).append(fi)
    uv = np.full((n, 2), np.nan)
    placed_f = np.zeros(m, dtype=bool)
    # first face
    f0 = 0
    a, b, c = F[f0]
    la, lb, lc = lnew[f0]            # la opposite a = |bc|, lb = |ca|, lc = |ab|
    uv[a] = (0.0, 0.0); uv[b] = (lc, 0.0)
    cx = (lb**2 - la**2 + lc**2) / (2 * lc)
    cy = np.sqrt(max(lb**2 - cx**2, 0.0))
    uv[c] = (cx, cy)
    placed_f[f0] = True
    q = deque([f0])
    while q:
        fi = q.popleft()
        f = F[fi]
        for k in range(3):
            a, b = f[k], f[(k + 1) % 3]
            key = (min(a, b), max(a, b))
            for g in edge_faces[key]:
                if placed_f[g]:
                    continue
                gf = F[g]
                kk = [i for i in range(3) if gf[i] not in (a, b)][0]
                cidx = gf[kk]
                # lengths in face g: opposite index j is edge between the other two
                lg = lnew[g]
                ia, ib = list(gf).index(a), list(gf).index(b)
                l_ca = lg[ib]   # edge opposite b = |c a|
                l_cb = lg[ia]   # edge opposite a = |c b|
                l_ab = lg[kk]
                pa, pb = uv[a], uv[b]
                d = pb - pa; dn = np.linalg.norm(d); d = d / dn
                x = (l_ca**2 - l_cb**2 + dn**2) / (2 * dn)
                y = np.sqrt(max(l_ca**2 - x**2, 0.0))
                perp = np.array([-d[1], d[0]])
                # the shared edge a->b appears as b->a in face g (consistent orientation), so c lies to the right of a->b
                pc = pa + x * d - y * perp
                if np.isnan(uv[cidx]).any():
                    uv[cidx] = pc
                placed_f[g] = True
                q.append(g)
    return np.nan_to_num(uv)


# ---------------------------------------------------------------------------
# 5. Discrete exponential map (geodesic polar coordinates)
# ---------------------------------------------------------------------------
def exp_map(mesh: TriMesh, center: int | None = None, n_interior_paths: int = 150, seed: int = 0, **kw) -> FlatResult:
    """Geodesic polar coordinates about a centre vertex (Schmidt, Grimm & Wyvill 2006).

    Every vertex is placed at (r cos th, r sin th) with r its exact geodesic distance
    from the centre (pygeodesic, MMP) and th the departure angle of that geodesic in
    the centre's tangent plane. Radial distances are therefore exact; the error is
    all circumferential, which for a dome means the rim is drawn too long and for a
    saddle too short. It is the formal version of "measure from the middle outwards",
    and a natural template for roughly round patches sutured from a central mark.
    Departure angles are computed exactly for the boundary vertices and a random
    subsample of interior vertices; the rest are interpolated harmonically.
    """
    import pygeodesic.geodesic as geo
    with Timer() as t:
        n = mesh.n_vertices
        rng = np.random.default_rng(seed)
        isb = mesh.is_boundary_vertex()
        if center is None:
            interior = np.where(~isb)[0]
            # interior vertex farthest from the boundary (geodesically central)
            g0 = geo.PyGeodesicAlgorithmExact(mesh.V, mesh.F)
            dB, _ = g0.geodesicDistances(np.where(isb)[0])
            center = int(interior[np.argmax(dB[interior])])
        g = geo.PyGeodesicAlgorithmExact(mesh.V, mesh.F)
        r, _ = g.geodesicDistances(np.array([center]))
        # tangent frame at the centre
        nrm = np.zeros(3)
        for f in mesh.F[np.any(mesh.F == center, axis=1)]:
            a, b, c = mesh.V[f]
            nrm += np.cross(b - a, c - a)
        nrm /= np.linalg.norm(nrm)
        e1 = np.cross(nrm, [1.0, 0, 0]);
        if np.linalg.norm(e1) < 0.3:
            e1 = np.cross(nrm, [0, 1.0, 0])
        e1 /= np.linalg.norm(e1); e2 = np.cross(nrm, e1)
        loop = mesh.boundary_loop()
        sample = np.unique(np.r_[loop, rng.choice(np.where(~isb)[0], min(n_interior_paths, (~isb).sum()), replace=False)])
        sample = sample[sample != center]
        dirs = np.zeros((n, 2)); known = np.zeros(n, dtype=bool)
        for v in sample:
            _, path = g.geodesicDistance(center, int(v))
            if len(path) < 2:
                continue
            # path is ordered from v to center (pygeodesic convention); take the segment at the centre
            d = path[-2] - path[-1] if np.linalg.norm(path[-1] - mesh.V[center]) < 1e-9 else path[1] - path[0]
            d = d - np.dot(d, nrm) * nrm
            if np.linalg.norm(d) < 1e-12:
                continue
            d /= np.linalg.norm(d)
            dirs[v] = (np.dot(d, e1), np.dot(d, e2)); known[v] = True
        # harmonic interpolation of the unit directions for the remaining vertices
        L = laplacian(mesh, "cotangent")
        fixed = np.where(known)[0]
        D = solve_with_constraints(L, np.zeros((n, 2)), fixed, dirs[fixed])
        nd = np.linalg.norm(D, axis=1); nd[nd < 1e-12] = 1.0
        D = D / nd[:, None]
        uv = r[:, None] * D
        uv[center] = 0.0
        uv = canonical_pose(uv)
    return FlatResult(uv, "exp_map", dict(center=int(center)), t.elapsed, 1, True,
                      {"flips": count_flips(mesh, uv), "n_exact_paths": int(len(sample))})


def lscm_perimeter(mesh: TriMesh, **kw) -> FlatResult:
    """LSCM rescaled so the template perimeter equals the 3D suture-line length
    (instead of the sqrt(area) rescale the Baffle Planner uses today)."""
    r = lscm(mesh, scale="perimeter", **kw)
    r.method = "lscm_perimeter"
    return r


EXTRA_METHODS.update({
    "cetm": cetm,
    "exp_map": exp_map,
    "lscm_perimeter": lscm_perimeter,
})
