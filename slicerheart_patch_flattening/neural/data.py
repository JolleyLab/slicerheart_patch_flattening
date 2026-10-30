"""Featurisation of synthetic samples for the flattening surrogate.

Inputs are mesh-independent so one model serves any triangulation:
  boundary   (K, 3)  suture line resampled by arclength, canonical start and direction,
                     in the patch's canonical frame (boundary centroid at the origin, PCA
                     axes of the boundary with the third axis along the mean surface normal)
  surface    (M, 6)  M points sampled uniformly by area on the patch with unit normals, same frame
Target:
  residual   (K, 2)  template outline (ARAP) minus the orthographic projection of the boundary,
                     after rigid alignment of the outline to that projection. Predicting the
                     residual gives the model the right inductive bias: for a developable,
                     nearly flat patch the residual is ~0.
  scalars            cumulative suture-line mismatch (mm), total |K| (deg), dart count

All units mm. The canonical frame is removed again by `uncanonicalize` when drawing.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

K_BOUNDARY = 128
M_SURFACE = 256


def sample_surface(V: np.ndarray, F: np.ndarray, M: int, rng: np.random.Generator):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(b - a, c - a); area = 0.5 * np.linalg.norm(n, axis=1)
    nrm = n / np.maximum(2 * area, 1e-12)[:, None]
    fi = rng.choice(len(F), size=M, p=area / area.sum())
    r1, r2 = rng.random(M), rng.random(M)
    s = np.sqrt(r1)
    P = (1 - s)[:, None] * a[fi] + (s * (1 - r2))[:, None] * b[fi] + (s * r2)[:, None] * c[fi]
    return P, nrm[fi]


def canonical_frame(B: np.ndarray, mean_normal: np.ndarray):
    """Rotation R (3x3, rows = axes) and origin for the boundary polyline B."""
    c = B.mean(0)
    _, _, Vt = np.linalg.svd(B - c, full_matrices=False)
    e1, e2, e3 = Vt
    if np.dot(e3, mean_normal) < 0:
        e3 = -e3
    e2 = np.cross(e3, e1)
    # fix the sign of e1 with the start point (index 0) -> positive x
    if np.dot(B[0] - c, e1) < 0:
        e1 = -e1; e2 = np.cross(e3, e1)
    return np.stack([e1, e2, e3]), c


def rigid_align_2d(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Rotate/translate src (K,2) onto ref (K,2), no scaling, no reflection."""
    cs, cr = src.mean(0), ref.mean(0)
    H = (src - cs).T @ (ref - cr)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, d]) @ Vt
    return (src - cs) @ R + cr


def featurize(rec: dict, rng: np.random.Generator, M: int = M_SURFACE) -> dict:
    V, F = rec["shape_xyz"], rec["faces"]
    B = np.asarray(rec["boundary_resampled"]); T = np.asarray(rec["arap_outline"])
    K = len(B)
    # canonical start: boundary point farthest from the centroid; direction: outline CCW
    c = B.mean(0); k0 = int(np.argmax(np.linalg.norm(B - c, axis=1)))
    B = np.roll(B, -k0, axis=0); T = np.roll(T, -k0, axis=0)
    area2 = 0.5 * np.sum(T[:, 0] * np.roll(T[:, 1], -1) - np.roll(T[:, 0], -1) * T[:, 1])
    if area2 < 0:
        B = np.concatenate([B[:1], B[1:][::-1]]); T = np.concatenate([T[:1], T[1:][::-1]])
    a, b_, cc = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    mean_normal = np.cross(b_ - a, cc - a).sum(0); mean_normal /= np.linalg.norm(mean_normal)
    R, o = canonical_frame(B, mean_normal)
    Bc = (B - o) @ R.T
    P, N = sample_surface(V, F, M, rng)
    Pc = (P - o) @ R.T; Nc = N @ R.T
    proj = Bc[:, :2]
    Ta = rigid_align_2d(T, proj)
    resid = Ta - proj
    met = rec["metrics"]
    return dict(boundary=Bc.astype(np.float32), surface=np.c_[Pc, Nc].astype(np.float32),
                residual=resid.astype(np.float32), proj=proj.astype(np.float32),
                scalars=np.array([met["boundary_cumulative_mismatch_mm"], met["shape_total_abs_K_deg"], len(rec["darts"])], dtype=np.float32),
                family=rec["family"], R=R, origin=o)


def load_raw(dirs: list[str]) -> list[dict]:
    """Records from directories of sample_*.npz (same reader as dataset_io.load_shards)."""
    from .dataset_io import load_shards
    return load_shards(list(dirs))


def build_arrays(recs: list[dict], seed: int = 0, M: int = M_SURFACE) -> dict:
    rng = np.random.default_rng(seed)
    feats = [featurize(r, rng, M) for r in recs]
    return dict(boundary=np.stack([f["boundary"] for f in feats]), surface=np.stack([f["surface"] for f in feats]),
                residual=np.stack([f["residual"] for f in feats]), proj=np.stack([f["proj"] for f in feats]),
                scalars=np.stack([f["scalars"] for f in feats]), family=np.array([f["family"] for f in feats]))
