"""Shared building blocks for the flattening solvers.

Everything here is pure numpy / scipy.sparse. The solvers in this package all
consume a `TriMesh` (open disk-topology triangle surface) and return a
`FlatResult` whose `uv` array has one 2D row per input vertex, so the result can
be written back onto the same connectivity (this is what the SlicerHeart Baffle
Planner expects for its "flattened model").

Conventions
-----------
* Units are millimetres throughout; a flattened template is meant to be printed
  or laser-cut at 1:1, so the absolute scale of `uv` matters.
* Per-face local frames: each 3D triangle is expressed in its own orthonormal
  2D frame (x along edge v0->v1, y in the triangle plane). The Jacobian of the
  map 3D -> 2D is then the 2x2 matrix J = U X^{-1} where X holds the rest
  (3D, in-plane) edge vectors and U the flattened edge vectors. Its singular
  values (sigma_1 >= sigma_2) are the principal stretches that all distortion
  metrics are built from (Sander et al. 2001; Hormann & Greiner 2000).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..mesh import TriMesh


@dataclass
class FlatResult:
    """Output of a flattening solver."""
    uv: np.ndarray                      # (n, 2) flattened vertex positions (mm)
    method: str
    params: dict = field(default_factory=dict)
    time_s: float = 0.0
    iterations: int = 0
    converged: bool = True
    info: dict = field(default_factory=dict)

    def as_mesh(self, mesh: TriMesh) -> TriMesh:
        """Flattened surface on the original connectivity (z = 0)."""
        m = TriMesh(np.c_[self.uv, np.zeros(len(self.uv))], mesh.F.copy(), name=f"{mesh.name}_{self.method}")
        m.meta = dict(mesh.meta)
        m.meta["flatten_method"] = self.method
        return m


# ---------------------------------------------------------------------------
# Local frames and Jacobians
# ---------------------------------------------------------------------------
def local_frames(mesh: TriMesh) -> np.ndarray:
    """Per-face 2D coordinates of the three vertices in the triangle's own plane.

    Returns P (m, 3, 2) with P[f, 0] = (0, 0), P[f, 1] = (|e1|, 0) and P[f, 2]
    the third vertex. Orientation is right-handed w.r.t. the face normal, so a
    flattening that preserves face orientation has det(J) > 0.
    """
    a, b, c = mesh.face_corners()
    e1 = b - a
    e2 = c - a
    l1 = np.linalg.norm(e1, axis=1)
    x = e1 / l1[:, None]
    n = np.cross(e1, e2)
    nn = np.linalg.norm(n, axis=1)
    n = n / np.maximum(nn, 1e-300)[:, None]
    y = np.cross(n, x)
    P = np.zeros((mesh.n_faces, 3, 2))
    P[:, 1, 0] = l1
    P[:, 2, 0] = np.einsum("ij,ij->i", e2, x)
    P[:, 2, 1] = np.einsum("ij,ij->i", e2, y)
    return P


def rest_edge_matrices(mesh: TriMesh):
    """X (m,2,2) with columns = local-frame edge vectors (v1-v0, v2-v0), plus X^{-1} and face areas."""
    P = local_frames(mesh)
    X = np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]], axis=2)  # (m,2,2), columns are edges
    det = X[:, 0, 0] * X[:, 1, 1] - X[:, 0, 1] * X[:, 1, 0]
    Xinv = np.empty_like(X)
    Xinv[:, 0, 0] = X[:, 1, 1]
    Xinv[:, 0, 1] = -X[:, 0, 1]
    Xinv[:, 1, 0] = -X[:, 1, 0]
    Xinv[:, 1, 1] = X[:, 0, 0]
    Xinv /= det[:, None, None]
    area = 0.5 * np.abs(det)
    return X, Xinv, area


def jacobians(mesh: TriMesh, uv: np.ndarray, Xinv: np.ndarray | None = None) -> np.ndarray:
    """Per-face 2x2 Jacobian of the map from the 3D triangle (local frame) to its 2D image."""
    if Xinv is None:
        _, Xinv, _ = rest_edge_matrices(mesh)
    u0, u1, u2 = uv[mesh.F[:, 0]], uv[mesh.F[:, 1]], uv[mesh.F[:, 2]]
    U = np.stack([u1 - u0, u2 - u0], axis=2)      # (m,2,2) columns = flattened edges
    return U @ Xinv


def singular_values_2x2(J: np.ndarray):
    """Closed-form singular values of a batch of 2x2 matrices (sigma_1 >= sigma_2 >= 0)."""
    a, b, c, d = J[:, 0, 0], J[:, 0, 1], J[:, 1, 0], J[:, 1, 1]
    s1 = np.sqrt((a + d) ** 2 + (b - c) ** 2)
    s2 = np.sqrt((a - d) ** 2 + (b + c) ** 2)
    return 0.5 * (s1 + s2), 0.5 * np.abs(s1 - s2)


# ---------------------------------------------------------------------------
# Laplacians
# ---------------------------------------------------------------------------
def cotangent_weights(mesh: TriMesh) -> tuple[np.ndarray, np.ndarray]:
    """Edges (k,2) and their cotangent weights w_ij = 0.5 (cot alpha + cot beta)."""
    ang = mesh.face_angles()                      # angle at vertex k opposite edge k
    cot = 1.0 / np.tan(np.clip(ang, 1e-6, np.pi - 1e-6))
    F = mesh.F
    # edge opposite vertex 0 is (1,2), etc.
    I = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    Jj = np.concatenate([F[:, 2], F[:, 0], F[:, 1]])
    W = 0.5 * np.concatenate([cot[:, 0], cot[:, 1], cot[:, 2]])
    return np.c_[I, Jj], W


def laplacian(mesh: TriMesh, kind: str = "cotangent") -> sp.csr_matrix:
    """Symmetric graph/cotangent Laplacian L = D - W (positive semi-definite).

    kind: 'uniform' (Tutte 1963), 'cotangent' (Pinkall & Polthier 1993),
          'mean_value' (Floater 2003; asymmetric, returned as is).
    """
    n = mesh.n_vertices
    if kind == "uniform":
        e = mesh.edges()
        W = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
        W = (W + W.T).tocsr()
    elif kind == "cotangent":
        e, w = cotangent_weights(mesh)
        W = sp.coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n, n))
        W = (W + W.T).tocsr()
    elif kind == "mean_value":
        W = mean_value_weights(mesh)
    else:
        raise ValueError(kind)
    D = sp.diags(np.asarray(W.sum(axis=1)).ravel())
    return (D - W).tocsr()


def mean_value_weights(mesh: TriMesh) -> sp.csr_matrix:
    """Mean-value coordinates weights w_ij = (tan(a/2)+tan(b/2))/|e_ij| (Floater 2003)."""
    n = mesh.n_vertices
    F = mesh.F
    ang = mesh.face_angles()
    V = mesh.V
    rows, cols, vals = [], [], []
    for k in range(3):
        i = F[:, k]                 # angle at i
        j1 = F[:, (k + 1) % 3]
        j2 = F[:, (k + 2) % 3]
        t = np.tan(0.5 * ang[:, k])
        for j in (j1, j2):
            L = np.linalg.norm(V[i] - V[j], axis=1)
            rows.append(i); cols.append(j); vals.append(t / L)
    W = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return W.tocsr()


# ---------------------------------------------------------------------------
# Pins, scaling and canonical pose
# ---------------------------------------------------------------------------
def default_pins(mesh: TriMesh) -> np.ndarray:
    """Two boundary vertices that are farthest apart (Euclidean), used as LSCM pins
    (the Baffle Planner asks the user for two 'fixed points'; this is the automatic default)."""
    loop = mesh.boundary_loop()
    P = mesh.V[loop]
    # O(k^2) on the boundary only; k is a few hundred at most
    d = np.linalg.norm(P[:, None] - P[None], axis=2)
    i, j = np.unravel_index(d.argmax(), d.shape)
    return np.array([loop[i], loop[j]], dtype=np.int64)


def pin_coordinates(mesh: TriMesh, pins: np.ndarray) -> np.ndarray:
    """Place the two pins on the u-axis separated by their on-surface geodesic distance,
    which makes the LSCM solution come out close to true scale before post-scaling."""
    d = mesh.geodesic_distance(int(pins[0]), int(pins[1]))
    return np.array([[0.0, 0.0], [d, 0.0]])


def rescale(mesh: TriMesh, uv: np.ndarray, mode: str = "area") -> tuple[np.ndarray, float]:
    """Uniformly rescale a flattening.

    mode 'area'      : match total surface area (what the Baffle Planner does)
    mode 'edge'      : least-squares optimal scale for edge lengths, s = sum(l3d*l2d)/sum(l2d^2)
    mode 'perimeter' : match the boundary (suture-line) length
    mode 'none'      : leave as is
    Returns (uv_scaled, scale).
    """
    if mode == "none":
        return uv, 1.0
    if mode == "area":
        A3 = mesh.area()
        A2 = TriMesh(uv, mesh.F).area()
        s = np.sqrt(A3 / A2)
    elif mode == "perimeter":
        s = mesh.perimeter() / TriMesh(uv, mesh.F).perimeter()
    elif mode == "edge":
        e = mesh.edges()
        l3 = np.linalg.norm(mesh.V[e[:, 0]] - mesh.V[e[:, 1]], axis=1)
        l2 = np.linalg.norm(uv[e[:, 0]] - uv[e[:, 1]], axis=1)
        s = float(np.dot(l3, l2) / np.dot(l2, l2))
    else:
        raise ValueError(mode)
    return uv * s, float(s)


def canonical_pose(uv: np.ndarray, F: np.ndarray | None = None) -> np.ndarray:
    """Center at the centroid and rotate so the principal axis is along u.
    If F is given and the mapping is mirrored (all det < 0), flip v so the template
    is shown from the same side as the 3D surface normal."""
    c = uv.mean(axis=0)
    X = uv - c
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    R = Vt.T
    if np.linalg.det(R) < 0:
        R[:, 1] *= -1
    return X @ R


def count_flips(mesh: TriMesh, uv: np.ndarray) -> int:
    """Number of triangles whose orientation is inverted in the flattening (fold-overs)."""
    a, b, c = uv[mesh.F[:, 0]], uv[mesh.F[:, 1]], uv[mesh.F[:, 2]]
    s = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    # compare against the majority orientation, so a globally mirrored map is not counted
    sign = np.sign(np.median(s))
    return int(np.sum(s * sign <= 0))


def solve_with_constraints(L: sp.spmatrix, rhs: np.ndarray, fixed_idx: np.ndarray, fixed_val: np.ndarray) -> np.ndarray:
    """Solve L x = rhs with hard Dirichlet constraints by elimination (x may be (n,2))."""
    n = L.shape[0]
    free = np.setdiff1d(np.arange(n), fixed_idx)
    L = L.tocsr()
    Lff = L[free][:, free]
    Lfc = L[free][:, fixed_idx]
    b = rhs[free] - Lfc @ fixed_val
    x = np.zeros((n, rhs.shape[1]) if rhs.ndim == 2 else n)
    x[fixed_idx] = fixed_val
    x[free] = spla.spsolve(Lff.tocsc(), b)
    return x


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.elapsed = time.perf_counter() - self.t0
