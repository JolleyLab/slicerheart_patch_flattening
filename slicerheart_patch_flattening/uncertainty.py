"""Uncertainty estimates for a flattened template.

A single distortion number says how much the sheet must stretch; the surgeon
also needs to know how much the *outline itself* is uncertain. Two sources are
quantified here, both in millimetres on the template boundary:

* method uncertainty  - disagreement between algorithms after rigid (Procrustes)
                        alignment. For a developable patch all sound methods
                        coincide; the spread grows with Gaussian curvature and
                        tells us how much of the outline is "choice of energy".
* discretisation      - change of the outline when the mesh is refined once
                        (linear subdivision, re-flatten, compare at the shared
                        original vertices). Verification-type error in the sense
                        of ASME V&V40 calculation verification.

Both are reported as RMS and max boundary displacement, plus a per-boundary-vertex
spread that can be drawn on the template as an uncertainty band.
"""
from __future__ import annotations

import numpy as np
import vtk
from scipy.linalg import orthogonal_procrustes

from .mesh import TriMesh
from .algorithms.methods import flatten


def rigid_align(src: np.ndarray, ref: np.ndarray, idx: np.ndarray | None = None):
    """Rigid (rotation + translation, optional reflection) alignment of src onto ref,
    fitted on the rows `idx` (all rows if None). Returns the transformed full src."""
    if idx is None:
        idx = np.arange(len(ref))
    cs, cr = src[idx].mean(0), ref[idx].mean(0)
    R, _ = orthogonal_procrustes(src[idx] - cs, ref[idx] - cr)
    return (src - cs) @ R + cr


def method_spread(mesh: TriMesh, uvs: dict[str, np.ndarray], reference: str = "arap") -> dict:
    """Boundary disagreement between methods, aligned rigidly on the boundary to `reference`."""
    loop = mesh.boundary_loop()
    ref = uvs[reference]
    aligned = {k: rigid_align(v, ref, loop) for k, v in uvs.items()}
    stack = np.stack([aligned[k][loop] for k in aligned])          # (n_methods, k, 2)
    mean = stack.mean(0)
    dev = np.linalg.norm(stack - mean[None], axis=2)              # (n_methods, k)
    per_vertex_spread = dev.max(0) + dev.max(0) * 0                 # max deviation from consensus
    out = {
        "boundary_spread_rms_mm": float(np.sqrt((dev**2).mean())),
        "boundary_spread_max_mm": float(dev.max()),
        "per_method_rms_mm": {k: float(np.sqrt((dev[i]**2).mean())) for i, k in enumerate(aligned)},
        "per_method_max_mm": {k: float(dev[i].max()) for i, k in enumerate(aligned)},
        "per_vertex_spread_mm": per_vertex_spread,
        "boundary_loop": loop,
    }
    return out


def subdivide(mesh: TriMesh) -> TriMesh:
    """One level of linear (midpoint) subdivision; the original vertices keep their indices."""
    f = vtk.vtkLinearSubdivisionFilter()
    f.SetInputData(mesh.to_vtk())
    f.SetNumberOfSubdivisions(1)
    f.Update()
    m = TriMesh.from_vtk(f.GetOutput(), mesh.name + "_sub")
    assert np.allclose(m.V[: mesh.n_vertices], mesh.V)
    m.meta = dict(mesh.meta)
    return m


def refinement_sensitivity(mesh: TriMesh, method: str, **kw) -> dict:
    """Flatten the mesh and its once-subdivided version; compare the outline at shared vertices."""
    r0 = flatten(mesh, method, **kw)
    ms = subdivide(mesh)
    r1 = flatten(ms, method, **kw)
    n = mesh.n_vertices
    loop = mesh.boundary_loop()
    uv1 = rigid_align(r1.uv[:n], r0.uv, loop)
    d = np.linalg.norm(uv1[loop] - r0.uv[loop], axis=1)
    return {"refine_boundary_rms_mm": float(np.sqrt((d**2).mean())),
            "refine_boundary_max_mm": float(d.max()),
            "n_vertices_coarse": int(n), "n_vertices_fine": int(ms.n_vertices)}
