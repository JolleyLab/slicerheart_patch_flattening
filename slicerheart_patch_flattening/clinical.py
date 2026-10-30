"""Loader for real (clinical) baffle surfaces, placeholder until cases are added.

Expected layout (see clinical_baffles/README.md): one folder per case holding the
thin baffle surface exported from the Baffle Planner (.vtp/.stl/.obj/.ply) and
optional Slicer markups JSON files for the fixed points and boundary curve.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from .mesh import TriMesh


def _read_markups_points(path: str) -> np.ndarray:
    """Control point positions from a Slicer .mrk.json (RAS, as stored)."""
    with open(path) as f:
        doc = json.load(f)
    pts = []
    for mk in doc.get("markups", []):
        for cp in mk.get("controlPoints", []):
            pts.append(cp["position"])
    return np.asarray(pts, dtype=float).reshape(-1, 3)


def load_clinical_cases(root: str = "clinical_baffles") -> list[TriMesh]:
    """Return TriMesh objects for every case folder containing a surface file."""
    cases = []
    for case_dir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(case_dir):
            continue
        surf = None
        for ext in (".vtp", ".stl", ".obj", ".ply"):
            hits = glob.glob(os.path.join(case_dir, "*" + ext))
            if hits:
                surf = hits[0]
                break
        if surf is None:
            continue
        m = TriMesh.load(surf).orient_consistently()
        m.name = os.path.basename(case_dir)
        meta = dict(family="clinical", developable=False, params={}, source=surf)
        fp = glob.glob(os.path.join(case_dir, "*fixed*.mrk.json"))
        if fp:
            meta["fixed_points_world"] = _read_markups_points(fp[0]).tolist()
        notes = os.path.join(case_dir, "notes.md")
        if os.path.exists(notes):
            with open(notes) as f:
                meta["notes"] = f.read()
        m.meta = meta
        cases.append(m)
    return cases
