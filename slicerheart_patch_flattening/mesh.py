"""Triangle-mesh container and geometry utilities.

Meshes are open, disk-topology triangle surfaces (a surgical patch). Storage is
plain numpy (V: (n,3) float64, F: (m,3) int64). VTK is used for I/O, thin-plate
spline warping, normals and geodesics; everything else is numpy/scipy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import vtk
from vtk.util import numpy_support as vnp


@dataclass
class TriMesh:
    V: np.ndarray                      # (n, 3) or (n, 2) vertex positions
    F: np.ndarray                      # (m, 3) triangle vertex indices
    name: str = "mesh"
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ basics
    def __post_init__(self):
        self.V = np.ascontiguousarray(self.V, dtype=np.float64)
        self.F = np.ascontiguousarray(self.F, dtype=np.int64)

    @property
    def n_vertices(self) -> int:
        return self.V.shape[0]

    @property
    def n_faces(self) -> int:
        return self.F.shape[0]

    def copy(self) -> "TriMesh":
        return TriMesh(self.V.copy(), self.F.copy(), self.name, dict(self.meta))

    # --------------------------------------------------------------- geometry
    def face_corners(self):
        return self.V[self.F[:, 0]], self.V[self.F[:, 1]], self.V[self.F[:, 2]]

    def face_areas(self) -> np.ndarray:
        a, b, c = self.face_corners()
        if self.V.shape[1] == 2:
            e1, e2 = b - a, c - a
            return 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

    def area(self) -> float:
        return float(self.face_areas().sum())

    def face_normals(self) -> np.ndarray:
        a, b, c = self.face_corners()
        n = np.cross(b - a, c - a)
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)

    def edge_lengths(self) -> np.ndarray:
        """Lengths of the three edges per face, ordered opposite to vertex 0,1,2."""
        a, b, c = self.face_corners()
        return np.stack([np.linalg.norm(c - b, axis=1),
                         np.linalg.norm(a - c, axis=1),
                         np.linalg.norm(b - a, axis=1)], axis=1)

    def face_angles(self) -> np.ndarray:
        """Interior angles per face at vertices 0,1,2 (radians)."""
        L = self.edge_lengths()
        l0, l1, l2 = L[:, 0], L[:, 1], L[:, 2]
        cos0 = (l1**2 + l2**2 - l0**2) / (2 * l1 * l2)
        cos1 = (l0**2 + l2**2 - l1**2) / (2 * l0 * l2)
        cos2 = (l0**2 + l1**2 - l2**2) / (2 * l0 * l1)
        return np.arccos(np.clip(np.stack([cos0, cos1, cos2], 1), -1.0, 1.0))

    def edges(self) -> np.ndarray:
        """Unique undirected edges (k, 2), sorted."""
        e = np.concatenate([self.F[:, [0, 1]], self.F[:, [1, 2]], self.F[:, [2, 0]]])
        e = np.sort(e, axis=1)
        return np.unique(e, axis=0)

    def boundary_edges(self) -> np.ndarray:
        e = np.concatenate([self.F[:, [0, 1]], self.F[:, [1, 2]], self.F[:, [2, 0]]])
        es = np.sort(e, axis=1)
        uniq, counts = np.unique(es, axis=0, return_counts=True)
        return uniq[counts == 1]

    def boundary_loop(self) -> np.ndarray:
        """Ordered vertex indices of the (single) boundary loop.

        Orientation follows the face orientation (counter-clockwise when faces
        are consistently oriented with outward normals).
        """
        e = np.concatenate([self.F[:, [0, 1]], self.F[:, [1, 2]], self.F[:, [2, 0]]])
        es = np.sort(e, axis=1)
        uniq, inv, counts = np.unique(es, axis=0, return_inverse=True, return_counts=True)
        bmask = counts[inv.ravel()] == 1
        be = e[bmask]                      # directed boundary half-edges
        if len(be) == 0:
            raise ValueError("mesh has no boundary")
        nxt = {int(a): int(b) for a, b in be}
        if len(nxt) != len(be):
            raise ValueError("boundary is not a simple loop (non-manifold vertex)")
        start = int(be[0, 0])
        loop = [start]
        cur = nxt[start]
        while cur != start:
            loop.append(cur)
            cur = nxt[cur]
            if len(loop) > len(be):
                raise ValueError("boundary traversal failed")
        if len(loop) != len(be):
            raise ValueError(f"multiple boundary loops ({len(be)} boundary edges, loop of {len(loop)})")
        return np.asarray(loop, dtype=np.int64)

    def is_boundary_vertex(self) -> np.ndarray:
        mask = np.zeros(self.n_vertices, dtype=bool)
        mask[self.boundary_edges().ravel()] = True
        return mask

    def vertex_angle_sums(self) -> np.ndarray:
        ang = self.face_angles()
        s = np.zeros(self.n_vertices)
        np.add.at(s, self.F.ravel(), ang.ravel())
        return s

    def angle_defects(self) -> np.ndarray:
        """Discrete Gaussian curvature (integrated): 2*pi - angle sum for interior
        vertices, pi - angle sum for boundary vertices (geodesic curvature term)."""
        s = self.vertex_angle_sums()
        b = self.is_boundary_vertex()
        return np.where(b, np.pi - s, 2 * np.pi - s)

    def vertex_areas(self) -> np.ndarray:
        """Barycentric (1/3 of incident face areas) vertex areas."""
        fa = self.face_areas()
        va = np.zeros(self.n_vertices)
        for k in range(3):
            np.add.at(va, self.F[:, k], fa / 3.0)
        return va

    def gaussian_curvature(self) -> np.ndarray:
        """Pointwise Gaussian curvature estimate K = defect / vertex area (interior)."""
        d = self.angle_defects()
        a = self.vertex_areas()
        K = d / np.maximum(a, 1e-12)
        K[self.is_boundary_vertex()] = 0.0
        return K

    def perimeter(self) -> float:
        loop = self.boundary_loop()
        P = self.V[loop]
        return float(np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1).sum())

    def check_orientation(self) -> bool:
        """True if every interior edge is traversed in opposite directions by its two faces."""
        e = np.concatenate([self.F[:, [0, 1]], self.F[:, [1, 2]], self.F[:, [2, 0]]])
        uniq, counts = np.unique(e, axis=0, return_counts=True)
        return bool(np.all(counts == 1))

    def adjacency(self):
        import scipy.sparse as sp
        e = self.edges()
        n = self.n_vertices
        A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
        return (A + A.T).tocsr()

    # ---------------------------------------------------------------- VTK I/O
    def to_vtk(self) -> vtk.vtkPolyData:
        V = self.V if self.V.shape[1] == 3 else np.c_[self.V, np.zeros(self.n_vertices)]
        pts = vtk.vtkPoints()
        pts.SetData(vnp.numpy_to_vtk(np.ascontiguousarray(V), deep=True))
        cells = vtk.vtkCellArray()
        offsets = np.arange(0, 3 * self.n_faces + 1, 3, dtype=np.int64)
        conn = np.ascontiguousarray(self.F.ravel(), dtype=np.int64)
        cells.SetData(vnp.numpy_to_vtkIdTypeArray(offsets, deep=True),
                      vnp.numpy_to_vtkIdTypeArray(conn, deep=True))
        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        pd.SetPolys(cells)
        return pd

    @classmethod
    def from_vtk(cls, pd: vtk.vtkPolyData, name: str = "mesh") -> "TriMesh":
        tri = vtk.vtkTriangleFilter()
        tri.SetInputData(pd)
        tri.PassLinesOff()
        tri.PassVertsOff()
        tri.Update()
        pd = tri.GetOutput()
        V = vnp.vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64)
        conn = vnp.vtk_to_numpy(pd.GetPolys().GetConnectivityArray())
        offs = vnp.vtk_to_numpy(pd.GetPolys().GetOffsetsArray())
        assert np.all(np.diff(offs) == 3), "non-triangular cells after vtkTriangleFilter"
        return cls(V, conn.reshape(-1, 3).astype(np.int64), name=name)

    def save(self, path: str):
        pd = self.to_vtk()
        if path.endswith(".stl"):
            w = vtk.vtkSTLWriter()
        elif path.endswith(".ply"):
            w = vtk.vtkPLYWriter()
        elif path.endswith(".obj"):
            w = vtk.vtkOBJWriter()
        else:
            w = vtk.vtkXMLPolyDataWriter()
        w.SetFileName(path)
        w.SetInputData(pd)
        w.Write()

    @classmethod
    def load(cls, path: str) -> "TriMesh":
        if path.endswith(".stl"):
            r = vtk.vtkSTLReader()
        elif path.endswith(".ply"):
            r = vtk.vtkPLYReader()
        elif path.endswith(".obj"):
            r = vtk.vtkOBJReader()
        else:
            r = vtk.vtkXMLPolyDataReader()
        r.SetFileName(path)
        r.Update()
        pd = r.GetOutput()
        if path.endswith(".stl"):
            clean = vtk.vtkCleanPolyData()
            clean.SetInputData(pd)
            clean.Update()
            pd = clean.GetOutput()
        import os
        return cls.from_vtk(pd, name=os.path.splitext(os.path.basename(path))[0])

    # ------------------------------------------------------------- utilities
    def remove_unreferenced(self) -> "TriMesh":
        used = np.unique(self.F)
        remap = -np.ones(self.n_vertices, dtype=np.int64)
        remap[used] = np.arange(len(used))
        return TriMesh(self.V[used], remap[self.F], self.name, dict(self.meta))

    def orient_consistently(self, outward_hint: Optional[np.ndarray] = None) -> "TriMesh":
        """Use vtkPolyDataNormals to make face orientation consistent."""
        n = vtk.vtkPolyDataNormals()
        n.SetInputData(self.to_vtk())
        n.ConsistencyOn()
        n.SplittingOff()
        n.AutoOrientNormalsOff()
        n.Update()
        m = TriMesh.from_vtk(n.GetOutput(), self.name)
        m.meta = dict(self.meta)
        if outward_hint is not None:
            fn = m.face_normals()
            if np.dot(fn.mean(0), outward_hint) < 0:
                m.F = m.F[:, ::-1].copy()
        return m

    def geodesic_distance(self, src: int, dst: int, exact: bool = True) -> float:
        """On-surface distance between two vertices.

        Uses the exact polyhedral geodesic (MMP algorithm, `pygeodesic`) when the
        package is available, otherwise the edge-path Dijkstra distance from VTK
        (which overestimates by a few percent on coarse meshes)."""
        if exact:
            try:
                import pygeodesic.geodesic as geo
                if not hasattr(self, "_geo_alg") or self._geo_alg_n != self.n_vertices:
                    self._geo_alg = geo.PyGeodesicAlgorithmExact(self.V.astype(np.float64), self.F.astype(np.int64))
                    self._geo_alg_n = self.n_vertices
                d, _ = self._geo_alg.geodesicDistance(int(src), int(dst))
                return float(d)
            except ImportError:
                pass
        d = vtk.vtkDijkstraGraphGeodesicPath()
        d.SetInputData(self.to_vtk())
        d.SetStartVertex(int(src))
        d.SetEndVertex(int(dst))
        d.Update()
        P = vnp.vtk_to_numpy(d.GetOutput().GetPoints().GetData())
        if len(P) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())

    def geodesic_path(self, src: int, dst: int) -> np.ndarray:
        """Vertex indices of the Dijkstra shortest edge path from src to dst."""
        d = vtk.vtkDijkstraGraphGeodesicPath()
        d.SetInputData(self.to_vtk())
        d.SetStartVertex(int(src))
        d.SetEndVertex(int(dst))
        d.Update()
        ids = d.GetIdList()
        path = np.array([ids.GetId(i) for i in range(ids.GetNumberOfIds())], dtype=np.int64)
        return path[::-1] if len(path) and path[0] == dst else path     # VTK lists the path from the end vertex
