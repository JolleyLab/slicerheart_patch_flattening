"""Synthetic database generator for learning-based flattening and pressurisation.

Each sample is one random baffle shape from the generator families used in the
30-shape library (tunnels, Baffle-Planner TPS baffles, trouser patches, caps), with
labels from the traditional solvers and, optionally, from the Warp forward model:

  shape_xyz, faces, boundary_loop            geometry (mm)
  boundary_resampled (K,3)                   fixed-size boundary descriptor
  arap_uv, arap_outline (K,2)                geometric template (level 1 label)
  metrics                                    distortion_metrics() of the ARAP template
  darts                                      recommended darts (kind, angle, apex, entry)
  pressure_mmhg, material, inflated_xyz      Warp forward solution of the ARAP template sewn
                                             on the suture line (level 2 label; optional)

Intended uses: (a) a surrogate that maps a boundary curve + a few surface samples to the
template outline and to the expected suture mismatch / dart need, for instant feedback
while the surgeon edits the baffle; (b) a prior/initialiser for the adjoint inverse;
(c) later, labels from the pressurised inverse itself. The record layout leaves fields
for pressure and material so level-2 labels can be added without changing consumers.
"""
from __future__ import annotations

import json
import os

import numpy as np

from ..library import generators as G
from ..mesh import TriMesh
from ..geometric.methods import flatten
from ..metrics import distortion_metrics
from ..geometric.darts import recommend_darts


def random_shape(rng: np.random.Generator, h: float = 1.0, wide: bool = True) -> TriMesh:
    """One random baffle from the library's families with parameters drawn from the pediatric ranges.

    wide=True (dataset v1) adds variability on top of the family parameters so a large set adds
    variety rather than density: patient-size scale 0.7-1.6, mesh resolution h x (0.75-1.3),
    stronger boundary harmonics, tunnel baffles with an added TPS bulge, asymmetric trousers,
    and a random rigid pose (the learner must not rely on the generator's frame)."""
    fam = rng.choice(["tunnel", "tps", "trouser", "cap"], p=[0.45, 0.3, 0.15, 0.1])
    scale = rng.uniform(0.7, 1.6) if wide else 1.0
    if wide:
        h = h * rng.uniform(0.75, 1.3)
    if fam == "tunnel":
        L = rng.uniform(18, 36); r0 = rng.uniform(5, 9); r1 = r0 * rng.uniform(0.7, 1.15)
        Rb = rng.choice([np.inf, rng.uniform(14, 40)]); tw = rng.uniform(-40, 40); oop = rng.uniform(-2, 2)
        ext = rng.uniform(150, 250); wmax = np.deg2rad(ext) * 0.5 * min(r0, r1)
        amp = 0.10 if wide else 0.06
        harm = np.array([[rng.uniform(0, amp), rng.uniform(0, 2 * np.pi)], [rng.uniform(0, 0.6 * amp), rng.uniform(0, 2 * np.pi)],
                         [rng.uniform(0, 0.3 * amp), rng.uniform(0, 2 * np.pi)]])
        outline = G.superellipse(L / 2, wmax, rng.uniform(2.0, 3.0), 96, harmonics=harm) + np.array([L / 2, 0])
        m = G.parametric_patch(outline, G.lift_tunnel, h, length=L, r0=r0, r1=r1, bend_radius=Rb, twist_deg=tw, out_of_plane=oop)
        bulge = 0.0
        if wide and rng.random() < 0.35:
            # add a local bulge/dent through a TPS warp of the tunnel surface (surface-point style)
            bulge = rng.uniform(-3.0, 4.0)
            loop = m.boundary_loop()
            src = m.V[loop][::2]
            interior = np.where(~m.is_boundary_vertex())[0]
            k = int(rng.choice(interior)); nrm = m.face_normals().mean(0); nrm /= np.linalg.norm(nrm)
            src = np.vstack([src, m.V[k:k + 1]]); dst = np.vstack([src[:-1], m.V[k:k + 1] + bulge * nrm])
            m = TriMesh(G.tps_warp_points(src, dst, m.V), m.F)
        params = dict(length=L, r0=r0, r1=r1, bend_radius=Rb, twist_deg=tw, out_of_plane=oop, extent_deg=ext, bulge=bulge)
    elif fam == "tps":
        a = rng.uniform(9, 17); b = a * rng.uniform(0.55, 0.9); host = rng.choice(["sphere", "tube", "saddle"])
        R = rng.uniform(8, 16)
        harm = np.array([[rng.uniform(0, 0.08), rng.uniform(0, 2 * np.pi)], [rng.uniform(0, 0.05), rng.uniform(0, 2 * np.pi)]])
        o2 = G.superellipse(a, b, 2.2, 80, harmonics=harm)
        if host == "sphere":
            b3 = G.lift_sphere(o2, R)
        elif host == "tube":
            b3 = G.lift_tunnel(o2 + np.array([a, 0]), 2 * a, R, R, 2.5 * a)
        else:
            b3 = G.lift_quadric(o2, 1 / R, -1 / R)
        n_sp = int(rng.integers(0, 4)); sp = None
        if n_sp:
            perim = np.linalg.norm(np.roll(b3, -1, 0) - b3, axis=1).sum(); rd = perim / (2 * np.pi)
            base = G.baffle_planner_patch(b3, h, None)
            nrm = base.face_normals().mean(0); nrm /= np.linalg.norm(nrm)
            uv = rng.uniform(-0.45, 0.45, size=(n_sp, 2)) * rd
            disk = G.disk_mesh(rd, h); idx = np.linalg.norm(disk.V[None] - uv[:, None], axis=2).argmin(1)
            tgt = base.V[idx] + rng.uniform(-4, 5) * nrm
            sp = np.c_[uv, tgt]
        m = G.baffle_planner_patch(b3, h, sp)
        params = dict(host=host, R=R, a=a, b=b, n_surface_points=n_sp)
    elif fam == "trouser":
        L = rng.uniform(36, 56); tw_ = rng.uniform(26, 40); lw = tw_ * rng.uniform(0.38, 0.46); ll = rng.uniform(13, 21)
        o = G.trouser_outline(L, tw_, lw, ll); R = rng.uniform(18, 40)
        if wide:
            # asymmetric legs / skew via a smooth affine + harmonic perturbation of the outline
            sk = rng.uniform(-0.15, 0.15); o = o @ np.array([[1.0, sk], [0.0, 1.0]]).T
            t = np.arctan2(o[:, 1], o[:, 0]); o = o * (1 + rng.uniform(0, 0.05) * np.cos(2 * t + rng.uniform(0, 6.3)))[:, None]
        choice = int(rng.integers(0, 3))
        lift, kw = [(G.lift_cylinder, dict(R=R)), (G.lift_sphere, dict(R=R)), (G.lift_quadric, dict(kx=1 / R, ky=-0.5 / R))][choice]
        m = G.parametric_patch(o, lift, h, **kw); params = dict(length=L, R=R, lift=lift.__name__)
    else:
        a = rng.uniform(7, 12); b = a * rng.uniform(0.6, 1.0); R = rng.uniform(9, 18)
        m = G.parametric_patch(G.superellipse(a, b, 2.1, 80), G.lift_sphere, h, R=R); params = dict(a=a, b=b, R=R)
    if wide:
        # patient-size scale and random rigid pose
        q = rng.standard_normal(4); q /= np.linalg.norm(q)
        a, b, c, d = q
        Rm = np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)], [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                       [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])
        m = TriMesh((scale * m.V) @ Rm.T + rng.uniform(-30, 30, 3), m.F)
        params["scale"] = scale; params["h"] = h
    m = m.orient_consistently()
    m.meta = dict(family=str(fam), params={k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v)) for k, v in params.items()})
    return m


def resample_loop(P: np.ndarray, K: int) -> np.ndarray:
    seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1); L = seg.sum(); cum = np.r_[0, np.cumsum(seg)]
    out = []
    for s in np.linspace(0, L, K, endpoint=False):
        i = min(np.searchsorted(cum, s, side="right") - 1, len(P) - 1)
        t = (s - cum[i]) / max(seg[i], 1e-12)
        out.append(P[i] + t * (P[(i + 1) % len(P)] - P[i]))
    return np.asarray(out)


def make_sample(rng: np.random.Generator, K: int = 128, pressure_mmhg: float | None = None, material=None, h: float = 1.0) -> dict:
    m = random_shape(rng, h)
    loop = m.boundary_loop()
    res = flatten(m, "arap")
    M = distortion_metrics(m, res.uv)
    darts = recommend_darts(m)
    from ..metrics import per_vertex_stretch
    pv = per_vertex_stretch(m, res.uv)
    lscm_uv = flatten(m, "lscm").uv
    rec = dict(shape_xyz=m.V.astype(np.float32), faces=m.F.astype(np.int32), boundary_loop=loop.astype(np.int32),
               family=m.meta["family"], params=m.meta["params"],
               boundary_resampled=resample_loop(m.V[loop], K).astype(np.float32),
               arap_uv=res.uv.astype(np.float32), arap_outline=resample_loop(res.uv[loop], K).astype(np.float32),
               lscm_uv=lscm_uv.astype(np.float32),
               vertex_stretch_s1=pv["s1"].astype(np.float32), vertex_stretch_s2=pv["s2"].astype(np.float32),
               vertex_gauss_curvature=m.gaussian_curvature().astype(np.float32),
               vertex_angle_defect=m.angle_defects().astype(np.float32),
               metrics=M, darts=[dict(kind=d["kind"], angle_deg=d["angle_deg"], apex=int(d["apex"]), entry=int(d["entry"])) for d in darts],
               pressure_mmhg=pressure_mmhg, material=None)
    if pressure_mmhg is not None:
        # physics labels: inflated states of the ARAP template sewn on the suture line, for each
        # (pressure, material) pair; pressure_mmhg may be a list, material a list of Material
        from .warp_membrane import Material
        from .inverse import inflate
        pressures = list(np.atleast_1d(pressure_mmhg))
        materials = material if isinstance(material, (list, tuple)) else [material or Material()]
        rec["material"] = [mt.__dict__ for mt in materials]; rec["pressure_mmhg"] = pressures
        rec["inflate_converged"] = []
        for mi, mt in enumerate(materials):
            for pi, p in enumerate(pressures):
                x, info, _ = inflate(res.uv, m.F, loop, m.V[loop], float(p), mt, x0=m.V.copy())
                rec[f"inflated_xyz_m{mi}_p{pi}"] = x.astype(np.float32)
                rec["inflate_converged"].append(bool(info["converged"]))
    return rec


def make_dataset(n: int, out_dir: str, seed: int = 0, K: int = 128, pressure_mmhg=None, h: float = 1.0,
                 material=None, start_index: int = 0, verbose_every: int = 0):
    """Write n samples sample_XXXXX.npz (+ index.json) to out_dir. Deterministic for a given seed."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    index = []
    i = 0
    while i < n:
        if verbose_every and i % verbose_every == 0:
            print(f"[{out_dir}] {i}/{n}", flush=True)
        try:
            rec = make_sample(rng, K, pressure_mmhg, material=material, h=h)
        except Exception as e:          # degenerate random geometry: skip, but say so
            print(f"skipping a sample: {type(e).__name__}: {e}")
            continue
        sid = start_index + i
        rec["id"] = sid; rec["seed"] = seed; rec["dataset_version"] = "v1"
        np.savez_compressed(os.path.join(out_dir, f"sample_{sid:06d}.npz"),
                            **{k: v for k, v in rec.items() if isinstance(v, np.ndarray)},
                            meta=json.dumps({k: v for k, v in rec.items() if not isinstance(v, np.ndarray)}, default=str))
        index.append(dict(id=sid, seed=seed, family=rec["family"], n_vertices=len(rec["shape_xyz"]),
                          suture_mismatch_mm=rec["metrics"]["boundary_cumulative_mismatch_mm"],
                          total_abs_K_deg=rec["metrics"]["shape_total_abs_K_deg"], n_darts=len(rec["darts"])))
        i += 1
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    return index
