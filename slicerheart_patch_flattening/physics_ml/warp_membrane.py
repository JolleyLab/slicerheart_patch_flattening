"""Differentiable membrane / thin-shell model in NVIDIA Warp.

Separate track from the geometric solvers: a physics-based, material-model-based
forward problem, differentiable with Warp's tape so the inverse problem (which flat
template, under pressure, gives a target 3D baffle?) can be solved by gradient methods.
Runs on CPU or CUDA (device argument); the CPU membrane in algorithms/methods.py is
the reference implementation for the StVK case.

Two configurations share one constitutive law W(C) (C = F^T F, the right
Cauchy-Green tensor of the 2D->2D or 2D->3D map):

  level 1  FLATTEN : rest = 3D surface (triangle metric in its local frame),
                     unknown = 2D positions. Minimises stored energy only.
  level 2  INFLATE : rest = 2D template (the cut patch), unknown = 3D positions,
                     boundary vertices held on the 3D suture line, transmural
                     pressure p as a follower load (potential -p * swept volume),
                     plus a discrete-shell hinge bending term (Grinspun et al. 2003)
                     that regularises the otherwise zero-bending-stiffness membrane.

Materials (per-area energy, multiplied by thickness t):
  stvk          W = mu |E|^2 + lam/2 tr(E)^2,  E = (C - I)/2,  plane stress
  neo_hookean   W = mu/2 (tr C + 1/det C - 3)   incompressible membrane (lambda_3^2 = 1/det C)
  hgo           neo_hookean + k1/(2 k2) (exp(k2 (I4 - 1)^2) - 1), I4 = a0^T C a0,
                fibres along angle `fiber_angle` in template coordinates, tension only
                (Holzapfel-Gasser-Ogden form; parameters are placeholders until
                pericardium / fabric data are fitted)

Units: mm, N, MPa (N/mm^2). 1 mmHg = 1.333e-4 MPa.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

wp.config.quiet = True
wp.init()

MMHG_TO_MPA = 133.322e-6

MAT_STVK, MAT_NEOHOOKEAN, MAT_HGO = 0, 1, 2


@dataclass
class Material:
    model: str = "stvk"          # stvk | neo_hookean | hgo
    E: float = 10.0              # Young's modulus, MPa (pericardium ~ 10-50 MPa in the toe/linear range)
    nu: float = 0.3
    thickness: float = 0.3       # mm (CardioCel ~0.3 mm; porcine BPA ~1 mm per Kizilski et al. 2025)
    k1: float = 0.0              # HGO fibre stiffness, MPa
    k2: float = 1.0              # HGO fibre exponent
    fiber_angle_deg: float = 0.0 # in template (2D rest) coordinates

    @property
    def model_id(self) -> int:
        return {"stvk": MAT_STVK, "neo_hookean": MAT_NEOHOOKEAN, "hgo": MAT_HGO}[self.model]

    @property
    def mu(self) -> float:
        return self.E / (2 * (1 + self.nu))

    @property
    def lam(self) -> float:            # plane-stress Lame parameter
        return self.E * self.nu / (1 - self.nu**2)

    @property
    def bending_stiffness(self) -> float:
        return self.E * self.thickness**3 / (12 * (1 - self.nu**2))

    def params(self) -> np.ndarray:
        a = np.deg2rad(self.fiber_angle_deg)
        return np.array([self.model_id, self.mu, self.lam, self.thickness, self.k1, self.k2,
                         np.cos(a), np.sin(a)], dtype=np.float64)


# ---------------------------------------------------------------------------
# Warp functions and kernels
# ---------------------------------------------------------------------------
@wp.func
def energy_density(C: wp.mat22d, prm: wp.array(dtype=wp.float64)) -> wp.float64:
    """Membrane strain energy per unit rest area (before thickness) for the chosen model."""
    model = prm[0]
    mu = prm[1]
    lam = prm[2]
    trC = C[0, 0] + C[1, 1]
    detC = C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0]
    W = wp.float64(0.0)
    if model < wp.float64(0.5):
        # St. Venant-Kirchhoff, E = (C - I)/2
        e00 = wp.float64(0.5) * (C[0, 0] - wp.float64(1.0))
        e11 = wp.float64(0.5) * (C[1, 1] - wp.float64(1.0))
        e01 = wp.float64(0.5) * C[0, 1]
        trE = e00 + e11
        W = mu * (e00 * e00 + e11 * e11 + wp.float64(2.0) * e01 * e01) + wp.float64(0.5) * lam * trE * trE
    else:
        # incompressible neo-Hookean membrane
        W = wp.float64(0.5) * mu * (trC + wp.float64(1.0) / detC - wp.float64(3.0))
        if model > wp.float64(1.5):
            k1 = prm[4]
            k2 = prm[5]
            ca = prm[6]
            sa = prm[7]
            I4 = C[0, 0] * ca * ca + wp.float64(2.0) * C[0, 1] * ca * sa + C[1, 1] * sa * sa
            if I4 > wp.float64(1.0):
                q = I4 - wp.float64(1.0)
                W = W + k1 / (wp.float64(2.0) * k2) * (wp.exp(k2 * q * q) - wp.float64(1.0))
    return W


@wp.func
def rest_matrix_2d(t0: wp.vec2d, t1: wp.vec2d, t2: wp.vec2d) -> wp.mat22d:
    """Columns = rest edge vectors (t1 - t0, t2 - t0) of a 2D (template) triangle."""
    e1 = t1 - t0
    e2 = t2 - t0
    return wp.mat22d(e1[0], e2[0], e1[1], e2[1])


@wp.func
def rest_matrix_3d(p0: wp.vec3d, p1: wp.vec3d, p2: wp.vec3d) -> wp.mat22d:
    """Rest edge vectors of a 3D triangle expressed in its own orthonormal frame."""
    e1 = p1 - p0
    e2 = p2 - p0
    l1 = wp.length(e1)
    x = e1 / l1
    n = wp.normalize(wp.cross(e1, e2))
    y = wp.cross(n, x)
    return wp.mat22d(l1, wp.dot(e2, x), wp.float64(0.0), wp.dot(e2, y))


@wp.kernel
def flatten_energy_kernel(rest: wp.array(dtype=wp.vec3d), x: wp.array(dtype=wp.vec2d),
                          faces: wp.array(dtype=wp.int32, ndim=2), prm: wp.array(dtype=wp.float64),
                          energy: wp.array(dtype=wp.float64)):
    """Level 1: 3D rest triangle -> 2D deformed triangle."""
    f = wp.tid()
    i0 = faces[f, 0]; i1 = faces[f, 1]; i2 = faces[f, 2]
    X = rest_matrix_3d(rest[i0], rest[i1], rest[i2])
    A = wp.float64(0.5) * wp.abs(wp.determinant(X))
    U = rest_matrix_2d(x[i0], x[i1], x[i2])
    Fd = U * wp.inverse(X)
    C = wp.transpose(Fd) * Fd
    W = energy_density(C, prm)
    wp.atomic_add(energy, 0, A * prm[3] * W)


@wp.kernel
def inflate_energy_kernel(tmpl: wp.array(dtype=wp.vec2d), x: wp.array(dtype=wp.vec3d),
                          faces: wp.array(dtype=wp.int32, ndim=2), prm: wp.array(dtype=wp.float64),
                          pressure: wp.float64, energy: wp.array(dtype=wp.float64)):
    """Level 2: 2D template triangle -> 3D deformed triangle, plus pressure potential."""
    f = wp.tid()
    i0 = faces[f, 0]; i1 = faces[f, 1]; i2 = faces[f, 2]
    X = rest_matrix_2d(tmpl[i0], tmpl[i1], tmpl[i2])
    A = wp.float64(0.5) * wp.abs(wp.determinant(X))
    Xi = wp.inverse(X)
    p0 = x[i0]; p1 = x[i1]; p2 = x[i2]
    e1 = p1 - p0
    e2 = p2 - p0
    # deformation gradient F (3x2) = [e1 e2] Xi ; C = F^T F (2x2)
    f1 = e1 * Xi[0, 0] + e2 * Xi[1, 0]
    f2 = e1 * Xi[0, 1] + e2 * Xi[1, 1]
    C = wp.mat22d(wp.dot(f1, f1), wp.dot(f1, f2), wp.dot(f2, f1), wp.dot(f2, f2))
    W = energy_density(C, prm)
    # pressure potential: -p * swept volume (cone to origin), gradient = p A n / 3 per vertex
    vol = wp.dot(p0, wp.cross(p1, p2)) / wp.float64(6.0)
    wp.atomic_add(energy, 0, A * prm[3] * W - pressure * vol)


@wp.kernel
def bending_energy_kernel(x: wp.array(dtype=wp.vec3d), hinges: wp.array(dtype=wp.int32, ndim=2),
                          hinge_w: wp.array(dtype=wp.float64), theta0: wp.array(dtype=wp.float64),
                          kb: wp.float64, energy: wp.array(dtype=wp.float64)):
    """Discrete-shell hinge energy kb * w_e * (theta - theta0)^2 over interior edges.
    hinges[e] = (a, b, c, d): edge a-b, opposite vertices c (face 1) and d (face 2)."""
    e = wp.tid()
    a = x[hinges[e, 0]]; b = x[hinges[e, 1]]; c = x[hinges[e, 2]]; d = x[hinges[e, 3]]
    ev = b - a
    n1 = wp.cross(ev, c - a)
    n2 = wp.cross(d - a, ev)
    l1 = wp.length(n1); l2 = wp.length(n2)
    n1 = n1 / l1; n2 = n2 / l2
    eh = wp.normalize(ev)
    s = wp.dot(wp.cross(n1, n2), eh)
    cth = wp.dot(n1, n2)
    th = wp.atan2(s, cth)
    dth = th - theta0[e]
    wp.atomic_add(energy, 0, kb * hinge_w[e] * dth * dth)


# ---------------------------------------------------------------------------
# Python-side model
# ---------------------------------------------------------------------------
def _hinges(F: np.ndarray):
    """Interior-edge hinge stencils (a, b, c, d) and weights |e|^2 / (A1 + A2)."""
    edge_faces = {}
    for fi, f in enumerate(F):
        for k in range(3):
            a, b = int(f[k]), int(f[(k + 1) % 3])
            edge_faces.setdefault((min(a, b), max(a, b)), []).append((fi, int(f[(k + 2) % 3])))
    H = []
    for (a, b), lst in edge_faces.items():
        if len(lst) == 2:
            H.append((a, b, lst[0][1], lst[1][1], lst[0][0], lst[1][0]))
    return np.asarray(H, dtype=np.int64)


class WarpMembrane:
    """Membrane/shell energy on a triangle mesh with Warp autodiff.

    mode='flatten': rest = 3D positions (n,3); x = 2D positions (n,2)
    mode='inflate': rest = 2D template (n,2); x = 3D positions (n,3); pressure in MPa
    """

    def __init__(self, faces: np.ndarray, rest: np.ndarray, material: Material, mode: str,
                 pressure_mpa: float = 0.0, bending: bool = True, device: str = "cpu"):
        self.device = device
        self.mode = mode
        self.material = material
        self.F = np.ascontiguousarray(faces, dtype=np.int32)
        self.m = len(self.F)
        self.n = int(self.F.max()) + 1
        self.faces_wp = wp.array(self.F, dtype=wp.int32, device=device)
        self.prm = wp.array(material.params(), dtype=wp.float64, device=device)
        self.pressure = float(pressure_mpa)
        self.energy = wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True)
        if mode == "flatten":
            self.rest = wp.array(np.ascontiguousarray(rest, dtype=np.float64), dtype=wp.vec3d, device=device, requires_grad=True)
            self.x = wp.zeros(self.n, dtype=wp.vec2d, device=device, requires_grad=True)
            self.dim = 2
        elif mode == "inflate":
            self.rest = wp.array(np.ascontiguousarray(rest, dtype=np.float64), dtype=wp.vec2d, device=device, requires_grad=True)
            self.x = wp.zeros(self.n, dtype=wp.vec3d, device=device, requires_grad=True)
            self.dim = 3
            self.bending = bending
            if bending:
                H = _hinges(self.F)
                rest2 = np.asarray(rest, dtype=np.float64)
                def area(fi):
                    t = rest2[self.F[fi]]
                    return 0.5 * abs((t[1, 0] - t[0, 0]) * (t[2, 1] - t[0, 1]) - (t[1, 1] - t[0, 1]) * (t[2, 0] - t[0, 0]))
                w = np.array([np.sum((rest2[h[1]] - rest2[h[0]])**2) / (area(h[4]) + area(h[5])) for h in H])
                self.hinges = wp.array(np.ascontiguousarray(H[:, :4], dtype=np.int32), dtype=wp.int32, device=device)
                self.hinge_w = wp.array(w, dtype=wp.float64, device=device)
                self.theta0 = wp.zeros(len(H), dtype=wp.float64, device=device)
                self.kb = material.bending_stiffness
        else:
            raise ValueError(mode)

    # -- energy and gradient with respect to the unknown positions x --------------
    def _launch(self):
        if self.mode == "flatten":
            wp.launch(flatten_energy_kernel, dim=self.m,
                      inputs=[self.rest, self.x, self.faces_wp, self.prm], outputs=[self.energy], device=self.device)
        else:
            wp.launch(inflate_energy_kernel, dim=self.m,
                      inputs=[self.rest, self.x, self.faces_wp, self.prm, wp.float64(self.pressure)],
                      outputs=[self.energy], device=self.device)
            if self.bending and len(self.hinge_w) > 0:
                wp.launch(bending_energy_kernel, dim=len(self.hinge_w),
                          inputs=[self.x, self.hinges, self.hinge_w, self.theta0, wp.float64(self.kb)],
                          outputs=[self.energy], device=self.device)

    def energy_grad(self, x_np: np.ndarray, wrt: str = "x"):
        """Total potential energy and its gradient w.r.t. x (default) or w.r.t. the rest configuration."""
        self.x.assign(np.ascontiguousarray(x_np.reshape(self.n, self.dim), dtype=np.float64))
        self.energy.zero_()
        self.x.grad.zero_(); self.rest.grad.zero_(); self.energy.grad.zero_()
        tape = wp.Tape()
        with tape:
            self._launch()
        tape.backward(loss=self.energy)
        E = float(self.energy.numpy()[0])
        g = (self.x.grad if wrt == "x" else self.rest.grad).numpy().copy()
        tape.zero()
        return E, g

    # -- quasi-static equilibrium ------------------------------------------------
    def solve(self, x0: np.ndarray, fixed_idx: np.ndarray | None = None, fixed_pos: np.ndarray | None = None,
              max_iter: int = 2000, tol: float = 1e-12, verbose: bool = False):
        """Minimise the total potential over the free vertices (L-BFGS, scipy) with Dirichlet
        constraints on fixed_idx. Returns (x, info)."""
        from scipy.optimize import minimize
        x = np.array(x0, dtype=np.float64).reshape(self.n, self.dim)
        fixed_idx = np.asarray(fixed_idx if fixed_idx is not None else [], dtype=np.int64)
        if len(fixed_idx):
            x[fixed_idx] = fixed_pos
        free = np.setdiff1d(np.arange(self.n), fixed_idx)

        def fg(z):
            x[free] = z.reshape(-1, self.dim)
            E, g = self.energy_grad(x)
            return E, g[free].ravel()

        res = minimize(fg, x[free].ravel(), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=max_iter, ftol=tol, gtol=1e-10, maxcor=30))
        x[free] = res.x.reshape(-1, self.dim)
        return x, dict(energy=float(res.fun), iterations=int(res.nit), converged=bool(res.success),
                       message=str(res.message), free=free)

    # -- Hessian-vector product by central differences of the gradient (for the adjoint) --
    def hess_vec(self, x: np.ndarray, v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        _, gp = self.energy_grad(x + eps * v)
        _, gm = self.energy_grad(x - eps * v)
        return (gp - gm) / (2 * eps)
