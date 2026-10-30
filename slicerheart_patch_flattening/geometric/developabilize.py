"""Nearest developable baffle: move the 3D surface, not the template.

A different answer to "this patch cannot be flattened": change the planned
baffle surface as little as possible so that a flat patch fits it exactly. The
boundary (suture line on the anatomy) stays fixed; interior vertices move to
drive the angle defects (discrete Gaussian curvature, Meyer et al. 2003) to
zero while staying close to the original surface. This is the surgical-planning
variant of developability optimisation (Stein, Grinspun & Crane 2018, who
minimise a different per-vertex developability energy with free boundaries).

    minimise  sum_i defect_i(V)^2  +  alpha * sum_i |V_i - V0_i|^2     over interior V_i

Solved with PyTorch autograd + L-BFGS. Output: the modified surface, the
displacement field (mm) and the residual curvature, so the planner can show the
surgeon "a flat patch can make this baffle if it bulges 1.2 mm less here".
"""
from __future__ import annotations

import numpy as np

from ..mesh import TriMesh


def developabilize(mesh: TriMesh, alpha: float = 1e-3, max_iter: int = 300, fixed: np.ndarray | None = None,
                   target_abs_K_deg: float = 2.0, verbose: bool = False) -> dict:
    import torch
    torch.set_default_dtype(torch.float64)
    V0 = torch.tensor(mesh.V)
    F = torch.tensor(mesh.F)
    n = mesh.n_vertices
    isb = mesh.is_boundary_vertex()
    if fixed is None:
        fixed = np.where(isb)[0]
    free = np.setdiff1d(np.arange(n), fixed)
    free_t = torch.tensor(free)
    interior_t = torch.tensor(np.where(~isb)[0])
    X = V0[free].clone().requires_grad_(True)

    def assemble(Xf):
        V = V0.clone()
        V = V.index_put((free_t,), Xf)
        return V

    def defects(V):
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        def ang(p, q, r):
            u = q - p; w = r - p
            cos = (u * w).sum(1) / (u.norm(dim=1) * w.norm(dim=1)).clamp_min(1e-12)
            return torch.acos(cos.clamp(-1 + 1e-9, 1 - 1e-9))
        A = torch.stack([ang(a, b, c), ang(b, c, a), ang(c, a, b)], 1)
        s = torch.zeros(n).index_add(0, F.reshape(-1), A.reshape(-1))
        return 2 * np.pi - s

    opt = torch.optim.LBFGS([X], max_iter=max_iter, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-10, tolerance_change=1e-12, history_size=50)
    hist = []

    def closure():
        opt.zero_grad()
        V = assemble(X)
        d = defects(V)[interior_t]
        E = (d**2).sum() + alpha * ((X - V0[free])**2).sum()
        E.backward()
        hist.append(float(E.detach()))
        return E

    opt.step(closure)
    V = assemble(X).detach().numpy()
    out = TriMesh(V, mesh.F.copy(), name=mesh.name + "_dev")
    out.meta = dict(mesh.meta); out.meta["developabilized"] = True
    disp = np.linalg.norm(V - mesh.V, axis=1)
    d0 = np.rad2deg(np.abs(mesh.angle_defects()[~isb]).sum())
    d1 = np.rad2deg(np.abs(out.angle_defects()[~isb]).sum())
    return dict(mesh=out, displacement_mm=disp, max_displacement_mm=float(disp.max()),
                rms_displacement_mm=float(np.sqrt((disp**2).mean())),
                total_abs_K_before_deg=float(d0), total_abs_K_after_deg=float(d1),
                reached_target=bool(d1 <= target_abs_K_deg), iterations=len(hist), alpha=alpha)
