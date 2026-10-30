"""Forward inflation and the inverse template problem (adjoint through equilibrium).

Forward (level 2): given a flat template T (n,2), the suture line (3D positions of
the boundary vertices), transmural pressure p and a material, find the 3D
equilibrium x*(T) of the sewn, pressurised patch:
    x*(T) = argmin_x  Pi(x; T, p)     with x_boundary fixed.

Inverse: find T so that x*(T) matches a target surface S* (the baffle the surgeon
planned in the pressurised state):
    minimise  L(T) = 1/2 |x*(T) - S*|^2     over the template vertex positions.

Gradient by the implicit function theorem (adjoint method). At equilibrium
g(x,T) = dPi/dx = 0, so dx*/dT = -H^{-1} dg/dT with H = d2Pi/dx2, and
    dL/dT = -(dg/dT)^T lambda,       H lambda = (x* - S*)      (free dofs only).
Both products come from Warp autodiff: H v by central differences of the
gradient, and (dg/dT)^T lambda = d/dT [ g . lambda ] by differencing the
gradient *with respect to T* along lambda. No second-order AD is required, which
keeps this on Warp alone; PyTorch would be needed only for a learned surrogate or
for exact Hessians.

Baseline for comparison: the analytical uniform unloading of Kizilski et al. 2025
(scale the pressurised flattening by 1/stretch) is `uniform_unload`.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator, cg, lgmres

from .warp_membrane import WarpMembrane, Material, MMHG_TO_MPA


# ---------------------------------------------------------------------------
# Forward problem
# ---------------------------------------------------------------------------
def inflate(template: np.ndarray, faces: np.ndarray, boundary_idx: np.ndarray, boundary_xyz: np.ndarray,
            pressure_mmhg: float, material: Material, x0: np.ndarray | None = None, bending: bool = True,
            device: str = "cpu", model: WarpMembrane | None = None, **solve_kw):
    """Equilibrium 3D shape of the template sewn onto boundary_xyz under pressure. Returns (x, info, model)."""
    if model is None:
        model = WarpMembrane(faces, template, material, "inflate", pressure_mpa=pressure_mmhg * MMHG_TO_MPA,
                             bending=bending, device=device)
    else:
        model.rest.assign(np.ascontiguousarray(template, dtype=np.float64))
    n = model.n
    if x0 is None:
        # start from the template lifted into the plane of the suture line, then pushed to the boundary
        x0 = np.c_[template, np.zeros(n)]
        c3 = boundary_xyz.mean(0); c2 = template[boundary_idx].mean(0)
        x0 = x0 - np.r_[c2, 0.0] + c3
    x, info = model.solve(x0, boundary_idx, boundary_xyz, **solve_kw)
    return x, info, model


# ---------------------------------------------------------------------------
# Inverse problem
# ---------------------------------------------------------------------------
class InverseTemplate:
    """Template design by the adjoint method. Boundary vertices of the template are free
    (they are what gets cut); the 3D boundary is pinned to the target's suture line."""

    def __init__(self, target_xyz: np.ndarray, faces: np.ndarray, boundary_idx: np.ndarray,
                 pressure_mmhg: float, material: Material, bending: bool = True, device: str = "cpu",
                 cg_tol: float = 1e-6, cg_maxiter: int = 300, fd_eps: float = 1e-6,
                 smooth_weight: float = 0.0, T_ref: np.ndarray | None = None, strain_weight: float = 1.0):
        """The membrane inverse is ill-posed: with the suture line pinned in 3D, the template outline
        changes only the rim *stress*, not the rim position, so many templates give the same
        inflated shape. Two regularisers select among them:
          strain_weight (gamma): gamma * U_membrane(x*, T), the stored strain energy of the inflated
              state (N mm). Among templates that reproduce the target it picks the least-strained
              patch (tension without wrinkles). This is the physically meaningful choice and the
              default.
          smooth_weight (beta): beta/2 |L (T - T_ref)|^2 with L the normalised graph Laplacian of the
              template; penalises rough per-vertex changes, leaves affine changes free. Off by default.
        T_ref defaults to the initial template passed to solve()."""
        self.S = np.asarray(target_xyz, dtype=np.float64)
        self.F = np.asarray(faces)
        self.bidx = np.asarray(boundary_idx, dtype=np.int64)
        self.bxyz = self.S[self.bidx]
        self.n = len(self.S)
        self.free = np.setdiff1d(np.arange(self.n), self.bidx)
        self.p = pressure_mmhg
        self.mat = material
        self.bending = bending
        self.device = device
        self.cg_tol, self.cg_maxiter, self.eps = cg_tol, cg_maxiter, fd_eps
        self.model = None
        self.x_prev = None
        self.history = []
        self.beta = smooth_weight
        self.gamma = strain_weight
        self.T_ref = T_ref
        self.strain_model = None            # pressure-free, bending-free copy for U_membrane
        import scipy.sparse as sp
        e = np.unique(np.sort(np.concatenate([self.F[:, [0, 1]], self.F[:, [1, 2]], self.F[:, [2, 0]]]), axis=1), axis=0)
        W = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(self.n, self.n)); W = (W + W.T).tocsr()
        D = sp.diags(np.asarray(W.sum(1)).ravel())
        self.Lap = (sp.diags(1.0 / np.asarray(W.sum(1)).ravel()) @ (D - W)).tocsr()   # normalised: Laplacian smoothing operator
        self.LtL = (self.Lap.T @ self.Lap).tocsr()

    # -- forward with warm start ------------------------------------------------
    def forward(self, T: np.ndarray):
        x, info, self.model = inflate(T, self.F, self.bidx, self.bxyz, self.p, self.mat, x0=self.x_prev,
                                      bending=self.bending, device=self.device, model=self.model,
                                      max_iter=3000, tol=1e-14)
        self.x_prev = x.copy()
        return x, info

    # -- loss and adjoint gradient -----------------------------------------------
    def loss_grad(self, T_flat: np.ndarray):
        T = T_flat.reshape(self.n, 2)
        x, info = self.forward(T)
        r = (x - self.S)[self.free]                       # residual on free dofs, (nf,3)
        L = 0.5 * float(np.sum(r * r))
        free = self.free; nf = len(free)
        # strain-energy regulariser: U(x*, T); needs dU/dx (enters the adjoint rhs) and dU/dT (direct)
        gU_x = np.zeros((self.n, 3)); gU_T = np.zeros((self.n, 2))
        if self.gamma > 0:
            if self.strain_model is None:
                self.strain_model = WarpMembrane(self.F, T, self.mat, "inflate", pressure_mpa=0.0, bending=False, device=self.device)
            self.strain_model.rest.assign(np.ascontiguousarray(T, dtype=np.float64))
            U, gU_x = self.strain_model.energy_grad(x, wrt="x")
            _, gU_T = self.strain_model.energy_grad(x, wrt="rest")
            L += self.gamma * U
            r = r + self.gamma * gU_x[free]

        # H lambda = r  on the free dofs, H v by central differences of the Warp gradient
        def hv(v_flat):
            v = np.zeros((self.n, 3)); v[free] = v_flat.reshape(nf, 3)
            return self.model.hess_vec(x, v, self.eps)[free].ravel()
        Hop = LinearOperator((3 * nf, 3 * nf), matvec=hv, dtype=np.float64)
        lam, flag = cg(Hop, r.ravel(), rtol=self.cg_tol, maxiter=self.cg_maxiter)
        if flag != 0:
            lam, flag = lgmres(Hop, r.ravel(), x0=lam, rtol=self.cg_tol, maxiter=200)
        lam_full = np.zeros((self.n, 3)); lam_full[free] = lam.reshape(nf, 3)

        # dL/dT = - d/dT [ g(x,T) . lambda ]  =  - ( grad_T Pi(x + eps lam) - grad_T Pi(x - eps lam) ) / (2 eps)
        _, gTp = self.model.energy_grad(x + self.eps * lam_full, wrt="rest")
        _, gTm = self.model.energy_grad(x - self.eps * lam_full, wrt="rest")
        gT = -(gTp - gTm) / (2 * self.eps) + self.gamma * gU_T
        # smoothness prior on the template
        if self.beta > 0 and self.T_ref is not None:
            dT = T - self.T_ref
            L += 0.5 * self.beta * float(np.sum((self.Lap @ dT) ** 2))
            gT = gT + self.beta * (self.LtL @ dT)
        rr = (x - self.S)[self.free]
        self.history.append(dict(loss=L, rms_mm=float(np.sqrt(np.mean(np.sum(rr * rr, 1)))),
                                 max_mm=float(np.sqrt(np.sum(rr * rr, 1)).max()), forward_iters=info["iterations"], cg_flag=int(flag)))
        return L, gT.ravel()

    def solve(self, T0: np.ndarray, max_iter: int = 40, verbose: bool = True):
        if self.T_ref is None:
            self.T_ref = np.array(T0, dtype=np.float64)

        def cb(_):
            if verbose:
                h = self.history[-1]
                print(f"  iter {len(self.history):3d}  loss {h['loss']:.4e}  rms {h['rms_mm']:.4f} mm  max {h['max_mm']:.4f} mm  fwd {h['forward_iters']}  cg {h['cg_flag']}", flush=True)
        res = minimize(self.loss_grad, np.asarray(T0, dtype=np.float64).ravel(), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=max_iter, ftol=1e-14, gtol=1e-10, maxcor=20), callback=cb)
        T = res.x.reshape(self.n, 2)
        x, _ = self.forward(T)
        return T, x, dict(loss=float(res.fun), iterations=int(res.nit), converged=bool(res.success), history=self.history)

    def check_gradient(self, T: np.ndarray, seed: int = 0, eps: float = 1e-4) -> dict:
        """Finite-difference check of the adjoint gradient along a random direction."""
        rng = np.random.default_rng(seed)
        d = rng.standard_normal(T.shape); d /= np.linalg.norm(d)
        if self.T_ref is None:
            self.T_ref = np.array(T, dtype=np.float64)
        _, g = self.loss_grad(T.ravel())
        x_base = self.x_prev.copy()
        # perturbed solves warm-started from the base equilibrium so all three land on the same branch
        self.x_prev = x_base.copy(); Lp, _ = self.loss_grad((T + eps * d).ravel())
        self.x_prev = x_base.copy(); Lm, _ = self.loss_grad((T - eps * d).ravel())
        self.x_prev = x_base.copy()
        fd = (Lp - Lm) / (2 * eps); ad = float(np.dot(g, d.ravel()))
        return dict(fd=fd, adjoint=ad, rel_err=abs(fd - ad) / max(abs(fd), 1e-16))


# ---------------------------------------------------------------------------
# Baseline: uniform analytical unloading (Kizilski et al. 2025, isotropic version)
# ---------------------------------------------------------------------------
def uniform_unload(template_loaded: np.ndarray, pressure_mmhg: float, material: Material, radius_mm: float) -> np.ndarray:
    """Scale a template obtained by flattening the *pressurised* geometry back to the unloaded
    state assuming an isotropic cylindrical-membrane stretch: sigma = p r / t, stretch = 1 + sigma / E."""
    p = pressure_mmhg * MMHG_TO_MPA
    sigma = p * radius_mm / material.thickness
    stretch = 1.0 + sigma / material.E
    return template_loaded / stretch
