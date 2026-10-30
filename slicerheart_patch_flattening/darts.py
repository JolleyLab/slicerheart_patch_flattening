"""Darting: relief cuts for patches that are not developable.

Gauss-Bonnet makes the problem precise. If a region of the patch encloses an
integrated Gaussian curvature of Omega (radians), no flattening can be
isometric there; the best a flat sheet can do is a *dart* (a wedge of opening
angle Omega removed from the template for Omega > 0, dome-like regions) or a
*gusset* (a wedge inserted, Omega < 0, saddle-like regions) running from the
boundary to the region. Kizilski et al. 2025 note the same requirement for
"elliptical or saddle-like" patches.

This module
1. finds the interior region(s) of concentrated curvature,
2. proposes a dart: apex vertex, boundary entry vertex, cut path along mesh
   edges (Dijkstra), and opening angle = enclosed curvature,
3. cuts the mesh along that path (duplicating the path vertices so the
   surface stays a disk with one longer boundary), and
4. re-flattens with any method, so the gain can be measured with the same
   metrics as the uncut template.

The cut path is also what gets drawn on the template for the surgeon.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from .mesh import TriMesh
from .algorithms.methods import flatten
from .algorithms.common import FlatResult


# ---------------------------------------------------------------------------
# 1. Curvature concentration
# ---------------------------------------------------------------------------
def smoothed_curvature(mesh: TriMesh, radius_mm: float = 4.0) -> np.ndarray:
    """Integrated angle defect inside a geodesic ball of `radius_mm` around each
    interior vertex (edge-graph Dijkstra distances). Peaks mark dart apices."""
    defects = mesh.angle_defects()
    interior = ~mesh.is_boundary_vertex()
    defects = np.where(interior, defects, 0.0)
    e = mesh.edges()
    w = np.linalg.norm(mesh.V[e[:, 0]] - mesh.V[e[:, 1]], axis=1)
    n = mesh.n_vertices
    G = sp.coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    G = G + G.T
    D = dijkstra(G, directed=False, limit=radius_mm)
    inside = np.isfinite(D)
    return inside @ defects


def recommend_darts(mesh: TriMesh, max_darts: int = 2, radius_mm: float = 4.0,
                    min_angle_deg: float = 8.0) -> list[dict]:
    """Propose up to `max_darts` darts/gussets.

    Returns dicts with apex (vertex id), entry (boundary vertex id), path (vertex ids
    from entry to apex), angle_deg (positive: remove wedge = dart; negative: insert
    wedge = gusset) and kind. Regions whose enclosed curvature is below
    `min_angle_deg` are not worth a cut (a few degrees is absorbed by fabric).
    """
    C = smoothed_curvature(mesh, radius_mm)
    interior = ~mesh.is_boundary_vertex()
    loop = mesh.boundary_loop()
    n = mesh.n_vertices
    e = mesh.edges()
    w = np.linalg.norm(mesh.V[e[:, 0]] - mesh.V[e[:, 1]], axis=1)
    G = sp.coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n, n)).tocsr(); G = G + G.T
    darts = []
    taken = np.zeros(n, dtype=bool)
    for _ in range(max_darts):
        score = np.where(interior & ~taken, np.abs(C), 0.0)
        apex = int(score.argmax())
        ang = float(np.rad2deg(C[apex]))
        if abs(ang) < min_angle_deg:
            break
        # nearest boundary vertex along the surface
        dist, pred = dijkstra(G, directed=False, indices=apex, return_predecessors=True)
        entry = int(loop[np.argmin(dist[loop])])
        path = [entry]
        while path[-1] != apex:
            path.append(int(pred[path[-1]]))
        darts.append(dict(apex=apex, entry=entry, path=np.array(path, dtype=np.int64),
                          angle_deg=ang, kind="dart" if ang > 0 else "gusset",
                          length_mm=float(dist[entry])))
        # suppress the neighbourhood of this apex for the next pick
        taken |= np.isfinite(dijkstra(G, directed=False, indices=apex, limit=2.5 * radius_mm))
    return darts


# ---------------------------------------------------------------------------
# 2. Cutting the mesh along a vertex path
# ---------------------------------------------------------------------------
def cut_mesh_along_path(mesh: TriMesh, path: np.ndarray) -> tuple[TriMesh, np.ndarray]:
    """Slit the mesh along an edge path that starts on the boundary and ends at an
    interior apex. Every path vertex except the apex is duplicated; the faces on
    the left of the directed path keep the original vertex, faces on the right
    get the duplicate. Returns (cut_mesh, old_index_of_each_vertex).
    """
    F = mesh.F.copy()
    V = mesh.V.copy()
    path = [int(p) for p in path]
    path_edges = {(path[i], path[i + 1]) for i in range(len(path) - 1)}
    path_edges |= {(b, a) for a, b in path_edges}
    n = len(V)
    orig = list(range(n))

    # face lists per vertex
    vf = [[] for _ in range(n)]
    for fi, f in enumerate(F):
        for v in f:
            vf[v].append(fi)

    def other_two(f, v):
        i = list(f).index(v)
        return f[(i + 1) % 3], f[(i + 2) % 3]

    for pi, p in enumerate(path[:-1]):
        faces = vf[p]
        # group faces around p into components separated by path edges
        comp = {}
        for f0 in faces:
            if f0 in comp:
                continue
            cid = len(set(comp.values()))
            stack = [f0]
            while stack:
                f = stack.pop()
                if f in comp:
                    continue
                comp[f] = cid
                a, b = other_two(F[f], p)
                for q in (a, b):
                    if (p, q) in path_edges:
                        continue                              # do not cross the cut
                    # neighbour face across edge (p, q)
                    for g in faces:
                        if g != f and g not in comp and q in F[g]:
                            stack.append(g)
        ncomp = len(set(comp.values()))
        if ncomp != 2:
            raise ValueError(f"cut at vertex {p} splits its fan into {ncomp} parts (expected 2)")
        # the face that has directed edge p->next in its CCW order lies to the LEFT of the path
        nxt = path[pi + 1]
        left_face = None
        for f in faces:
            if nxt in F[f]:
                i = list(F[f]).index(p)
                if F[f][(i + 1) % 3] == nxt:
                    left_face = f
        left_cid = comp[left_face]
        # duplicate p for the right-hand component
        new_id = len(V)
        V = np.vstack([V, V[p][None]])
        orig.append(p)
        vf.append([])
        for f, cid in comp.items():
            if cid != left_cid:
                F[f][F[f] == p] = new_id
                vf[new_id].append(f)
        vf[p] = [f for f, cid in comp.items() if cid == left_cid]
    m = TriMesh(V, F, name=mesh.name + "_cut")
    m.meta = dict(mesh.meta)
    m.boundary_loop()      # sanity: still a single loop
    return m, np.asarray(orig, dtype=np.int64)


def flatten_with_darts(mesh: TriMesh, method: str = "arap", darts: list[dict] | None = None,
                       **kw) -> tuple[FlatResult, TriMesh, list[dict]]:
    """Apply the recommended darts (or the given list), re-flatten, return
    (result on the cut mesh, cut mesh, darts). Each dart dict gains 'uv_path'
    with the two 2D sides of the slit for drawing."""
    if darts is None:
        darts = recommend_darts(mesh)
    cut = mesh
    orig = np.arange(mesh.n_vertices)
    applied = []
    for d in darts:
        # map path through earlier cuts (vertex ids of earlier duplicates are >= old n)
        try:
            cut, o = cut_mesh_along_path(cut, d["path"])
        except ValueError:
            continue
        orig = orig[o]
        applied.append(d)
    res = flatten(cut, method, **kw)
    for d in applied:
        ids = [np.where(orig == v)[0] for v in d["path"]]
        d["uv_path"] = [res.uv[i] for i in ids]      # one or two 2D points per path vertex
    return res, cut, applied
