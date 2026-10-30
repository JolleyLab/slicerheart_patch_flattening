"""Build the synthetic patch library (28 shapes) and write it to disk with a manifest.

Size references (mm), infants and young children:
  DORV LV->Ao baffle (Vigil et al., 6-10 mo, 7-8 kg): VSD 10-15 mm; tunnel 18-35 mm long,
     15-25 mm wide, trough radius 5-9 mm, angular extent 150-250 deg, taper toward the
     aortic annulus and some twist as the tunnel turns from the VSD toward the aorta.
  Mustard intra-atrial baffle: trouser-shaped pericardial patch ~35-55 mm long,
     25-40 mm across the top, legs 12-18 mm wide; lies on a gently curved sheet.
  Senning pulmonary venous pathway augmentation: ~20x30 mm rounded rectangle, cylindrical.
  PAPVR intra-atrial baffle (RUPV/SVC to LA via ASD): ovoid 15-28 x 10-20 mm, half-tunnel
     radius 5-8 mm, usually curved along the long axis as well.
"""
from __future__ import annotations

import json
import os

import numpy as np

from ..mesh import TriMesh
from . import generators as G

H_DEFAULT = 0.8  # target edge length (mm)


def _mesh(V, F, name, family, developable, **params) -> TriMesh:
    m = TriMesh(V, F, name=name)
    m = m.orient_consistently()
    m.meta = dict(family=family, developable=developable, params=params)
    return m


def _lifted(name, family, developable, outline, lift, h, seed, **kw) -> TriMesh:
    dom = G.triangulate_polygon(outline, h, seed=seed)
    V3 = lift(dom.V, **kw)
    return _mesh(V3, dom.F, name, family, developable, **{k: (float(v) if np.isscalar(v) else v) for k, v in kw.items()})


def build_library(h: float = H_DEFAULT, seed: int = 7) -> list[TriMesh]:
    rng = np.random.default_rng(seed)
    shapes: list[TriMesh] = []

    # ---------------- analytic developable controls (expect ~0 distortion) ----
    shapes.append(_lifted("ctrl_halfcyl_r8_small", "control_developable", True,
                          G.superellipse(12.0, 9.0, 2.0, 96), G.lift_cylinder, h, 1, R=8.0))
    shapes.append(_lifted("ctrl_halfcyl_r6_long", "control_developable", True,
                          G.superellipse(8.5, 16.0, 2.6, 96), G.lift_cylinder, h, 2, R=6.0))
    shapes.append(_lifted("ctrl_cone_r9", "control_developable", True,
                          G.superellipse(11.0, 13.0, 2.3, 96), G.lift_cone, h, 3, R0=9.0, slope=0.25))

    # ---------------- analytic non-developable (known Gaussian curvature) -----
    shapes.append(_lifted("ctrl_spherecap_r12", "control_nondevelopable", False,
                          G.superellipse(9.0, 9.0, 2.0, 96), G.lift_sphere, h, 4, R=12.0))
    shapes.append(_lifted("ctrl_saddle_k0p08", "control_nondevelopable", False,
                          G.superellipse(11.0, 11.0, 2.0, 96), G.lift_quadric, h, 5, kx=0.08, ky=-0.08))
    shapes.append(_lifted("ctrl_ellipsoidcap", "control_nondevelopable", False,
                          G.superellipse(10.0, 7.0, 2.2, 96), G.lift_ellipsoid, h, 6, a=14.0, b=10.0, c=8.0))

    # ---------------- DORV LV-to-aorta tunnel baffles (8) ----------------------
    dorv_params = [
        # length, r0, r1, bend_radius, twist, out_of_plane, extent_deg
        (22, 7.0, 7.0, np.inf, 0, 0.0, 180),      # straight half-tube
        (26, 7.5, 6.0, 30.0, 10, 0.0, 200),       # mild bend, taper
        (30, 8.0, 6.5, 22.0, 25, 0.0, 210),       # moderate bend + twist
        (34, 8.5, 7.0, 18.0, 40, 2.0, 230),       # strong bend, twist, out of plane
        (24, 6.0, 5.0, 20.0, 15, 1.0, 170),       # small infant
        (28, 7.0, 7.5, 25.0, -20, -1.5, 190),     # flaring, opposite twist
        (32, 9.0, 6.0, 16.0, 30, 0.0, 250),       # wide conus-type pathway
        (20, 6.5, 6.5, 14.0, 0, 0.0, 160),        # short, tightly curved
    ]
    for i, (L, r0, r1, Rb, tw, oop, ext) in enumerate(dorv_params):
        wmax = np.deg2rad(ext) * 0.5 * min(r0, r1)
        harm = np.array([[0.04, rng.uniform(0, 2 * np.pi)], [0.03, rng.uniform(0, 2 * np.pi)]])
        outline = G.superellipse(L / 2, wmax, 2.4 + 0.4 * rng.random(), 128, harmonics=harm)
        outline = outline + np.array([L / 2, 0.0])
        shapes.append(_lifted(f"dorv_tunnel_{i+1:02d}", "dorv_tunnel", bool(np.isinf(Rb) and tw == 0 and r0 == r1),
                              outline, G.lift_tunnel, h, 10 + i,
                              length=float(L), r0=float(r0), r1=float(r1), bend_radius=float(Rb),
                              twist_deg=float(tw), out_of_plane=float(oop)))

    # ---------------- Baffle-Planner (TPS) DORV baffles (6) --------------------
    # Boundary curve: closed curve lying on a sphere/tube-like host surface, like a curve
    # snapped to the septum / conus surface, plus 0-3 interior surface points.
    bp_specs = [
        dict(host="sphere", R=14.0, a=12.0, b=9.0, n_surface=0, bulge=0.0),
        dict(host="sphere", R=12.0, a=13.0, b=8.0, n_surface=1, bulge=3.0),
        dict(host="tube", R=8.0, a=15.0, b=9.0, n_surface=2, bulge=2.5),
        dict(host="tube", R=7.0, a=17.0, b=8.0, n_surface=3, bulge=4.0),
        dict(host="saddle", R=15.0, a=14.0, b=10.0, n_surface=1, bulge=-2.0),
        dict(host="sphere", R=10.0, a=10.0, b=8.0, n_surface=2, bulge=5.0),
    ]
    for i, s in enumerate(bp_specs):
        k = 80
        harm = np.array([[0.06, rng.uniform(0, 2 * np.pi)], [0.04, rng.uniform(0, 2 * np.pi)]])
        o2 = G.superellipse(s["a"], s["b"], 2.2, k, harmonics=harm)
        if s["host"] == "sphere":
            b3 = G.lift_sphere(o2, s["R"])
        elif s["host"] == "tube":
            b3 = G.lift_tunnel(o2 + np.array([s["a"], 0]), 2 * s["a"], s["R"], s["R"], 2.5 * s["a"])
        else:
            b3 = G.lift_quadric(o2, 1 / s["R"], -1 / s["R"])
        # interior surface points: pick disk-space locations and push along mean normal
        sp = None
        if s["n_surface"]:
            perim = np.linalg.norm(np.roll(b3, -1, 0) - b3, axis=1).sum()
            rd = perim / (2 * np.pi)
            uv = rng.uniform(-0.45, 0.45, size=(s["n_surface"], 2)) * rd
            # initial TPS from boundary only to get an in-surface starting point
            base = G.baffle_planner_patch(b3, h, None, seed=20 + i)
            nrm = base.face_normals().mean(axis=0); nrm /= np.linalg.norm(nrm)
            # map uv to 3D via nearest disk vertex of base construction
            disk = G.disk_mesh(rd, h, seed=20 + i)
            d = np.linalg.norm(disk.V[None] - uv[:, None], axis=2)
            idx = d.argmin(axis=1)
            tgt = base.V[idx] + s["bulge"] * nrm * (1 + 0.3 * rng.standard_normal(len(idx)))[:, None]
            sp = np.c_[uv, tgt]
        m = G.baffle_planner_patch(b3, h, sp, seed=20 + i)
        shapes.append(_mesh(m.V, m.F, f"bp_tps_dorv_{i+1:02d}", "baffle_planner_tps", False,
                            host=s["host"], host_R=s["R"], a=s["a"], b=s["b"],
                            n_surface_points=s["n_surface"], bulge=s["bulge"]))

    # ---------------- Mustard / Senning intra-atrial baffles (5) ---------------
    mustard = [
        (40, 28, 12, 14, "cyl", 20.0),
        (48, 34, 15, 17, "cyl", 26.0),
        (55, 38, 17, 20, "saddle", 40.0),
        (44, 30, 13, 15, "sphere", 35.0),
    ]
    for i, (L, tw_, lw, ll, host, R) in enumerate(mustard):
        o = G.trouser_outline(L, tw_, lw, ll)
        if host == "cyl":
            lift, kw = G.lift_cylinder, dict(R=R)
        elif host == "sphere":
            lift, kw = G.lift_sphere, dict(R=R)
        else:
            lift, kw = G.lift_quadric, dict(kx=1 / R, ky=-0.5 / R)
        shapes.append(_lifted(f"mustard_trouser_{i+1:02d}", "mustard_senning", host == "cyl",
                              o, lift, h, 30 + i, **kw))
    # Senning pulmonary venous pathway augmentation patch (rounded rectangle on a cylinder)
    shapes.append(_lifted("senning_pv_patch_01", "mustard_senning", True,
                          G.superellipse(10.0, 15.0, 3.5, 112), G.lift_cylinder, h, 35, R=9.0))

    # ---------------- PAPVR intra-atrial baffles (5) ---------------------------
    papvr = [
        (18, 5.5, 5.5, np.inf, 0, 0.0, 180),
        (22, 6.0, 5.0, 24.0, 10, 0.0, 200),
        (26, 7.0, 6.0, 18.0, 20, 1.0, 210),
        (20, 5.0, 6.5, 20.0, -15, 0.0, 190),
        (28, 8.0, 6.0, 15.0, 30, 1.5, 230),
    ]
    for i, (L, r0, r1, Rb, tw, oop, ext) in enumerate(papvr):
        wmax = np.deg2rad(ext) * 0.5 * min(r0, r1)
        harm = np.array([[0.05, rng.uniform(0, 2 * np.pi)]])
        outline = G.superellipse(L / 2, wmax, 2.0, 112, harmonics=harm) + np.array([L / 2, 0.0])
        shapes.append(_lifted(f"papvr_baffle_{i+1:02d}", "papvr", bool(np.isinf(Rb) and tw == 0 and r0 == r1),
                              outline, G.lift_tunnel, h, 40 + i,
                              length=float(L), r0=float(r0), r1=float(r1), bend_radius=float(Rb),
                              twist_deg=float(tw), out_of_plane=float(oop)))
    return shapes


def shape_summary(m: TriMesh) -> dict:
    defects = m.angle_defects()
    interior = ~m.is_boundary_vertex()
    bb = m.V.max(0) - m.V.min(0)
    return dict(
        name=m.name, family=m.meta.get("family"), developable=bool(m.meta.get("developable")),
        n_vertices=int(m.n_vertices), n_faces=int(m.n_faces),
        area_mm2=float(m.area()), perimeter_mm=float(m.perimeter()),
        bbox_mm=[float(x) for x in bb],
        total_abs_gauss_curvature_deg=float(np.rad2deg(np.abs(defects[interior]).sum())),
        net_gauss_curvature_deg=float(np.rad2deg(defects[interior].sum())),
        max_abs_K_per_mm2=float(np.abs(m.gaussian_curvature()).max()),
    )


def write_library(out_dir: str, shapes: list[TriMesh] | None = None, h: float = H_DEFAULT) -> list[dict]:
    os.makedirs(out_dir, exist_ok=True)
    shapes = shapes or build_library(h=h)
    manifest = []
    for m in shapes:
        m.save(os.path.join(out_dir, m.name + ".vtp"))
        m.save(os.path.join(out_dir, m.name + ".stl"))
        rec = shape_summary(m)
        rec["params"] = {k: (v if not isinstance(v, float) or np.isfinite(v) else "inf")
                         for k, v in m.meta.get("params", {}).items()}
        rec["files"] = {"vtp": m.name + ".vtp", "stl": m.name + ".stl"}
        manifest.append(rec)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_library(dir_path: str) -> list[TriMesh]:
    with open(os.path.join(dir_path, "manifest.json")) as f:
        manifest = json.load(f)
    out = []
    for rec in manifest:
        m = TriMesh.load(os.path.join(dir_path, rec["files"]["vtp"]))
        m.name = rec["name"]
        m.meta = dict(family=rec["family"], developable=rec["developable"], params=rec.get("params", {}))
        out.append(m)
    return out
