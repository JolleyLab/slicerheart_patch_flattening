"""
Shared fixtures and helpers.

All tests run on synthetic shapes (analytic surfaces of known curvature and the
synthetic baffle library); no data is downloaded. Tests that need an optional
package (warp, libigl, torch) are skipped when it is not installed.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from slicerheart_patch_flattening.library import generators as G, build_library
from slicerheart_patch_flattening.mesh import TriMesh

# target edge length (mm) of the analytic test shapes: coarse enough for a fast run,
# fine enough for sub-percent isometry checks
H = 1.2


# ---------------------------------------------------------------------------
# Analytic shapes
# ---------------------------------------------------------------------------
#
# The outlines carry a small radial perturbation (harmonics) that breaks their mirror
# symmetry, so that the farthest boundary pair (the automatic LSCM pins, the template
# length axis) is unique: on a symmetric outline several pairs tie and the pick could
# change with rounding, the vertex order or a rotation.

ASYMMETRIC = [(0.04, 0.3), (0.03, 1.1), (0.02, 2.0)]        # (amplitude, phase) of harmonics 2, 3, 4


def outline(a, b, n=2.0, k=64):
    return G.superellipse(a, b, n, k, harmonics=ASYMMETRIC)


def lifted_patch(outline2d, lift, h=H, seed=0, **kw):
    """(mesh, 2D domain) of an outline lifted to a surface; the domain is kept because for an
    isometric lift (cylinder) it is the exact flat template."""
    dom = G.triangulate_polygon(outline2d, h, seed=seed)
    mesh = TriMesh(lift(dom.V, **kw), dom.F.copy()).orient_consistently()
    return mesh, dom


def make_half_cylinder():
    """Developable: the lift (u, v) -> (R sin(u/R), v, R (1 - cos(u/R))) is an isometry of the plane."""
    return lifted_patch(outline(10.0, 8.0, 2.0), G.lift_cylinder, R=7.0)


def make_cone():
    """Developable (cone of slope 0.25); the flat template is a sector, not the parameter domain."""
    return lifted_patch(outline(11.0, 8.0, 2.3), G.lift_cone, R0=9.0, slope=0.25)


def make_sphere_cap():
    """Positive Gaussian curvature 1/R^2 everywhere."""
    return lifted_patch(outline(10.0, 7.0, 2.2), G.lift_sphere, R=10.0)


def make_saddle():
    """Negative Gaussian curvature (hyperbolic paraboloid)."""
    return lifted_patch(outline(11.0, 9.0, 2.0), G.lift_quadric, kx=0.08, ky=-0.08)


ANALYTIC_SHAPES = {
    "half_cylinder": make_half_cylinder,
    "cone": make_cone,
    "sphere_cap": make_sphere_cap,
    "saddle": make_saddle,
}
DEVELOPABLE_SHAPES = ["half_cylinder", "cone"]
CURVED_SHAPES = ["sphere_cap", "saddle"]


@pytest.fixture(scope="session")
def half_cylinder():
    return make_half_cylinder()[0]


@pytest.fixture(scope="session")
def half_cylinder_domain():
    """The exact flat template of `half_cylinder` (its 2D parameter domain)."""
    return make_half_cylinder()[1]


@pytest.fixture(scope="session")
def cone():
    return make_cone()[0]


@pytest.fixture(scope="session")
def sphere_cap():
    return make_sphere_cap()[0]


@pytest.fixture(scope="session")
def saddle():
    return make_saddle()[0]


@pytest.fixture(scope="session", params=list(ANALYTIC_SHAPES))
def analytic_shape(request):
    """Every analytic shape in turn, (name, mesh)."""
    return request.param, ANALYTIC_SHAPES[request.param]()[0]


@pytest.fixture(scope="session")
def planar_disk():
    """A flat disk (z = 0) of radius 10 mm: its own exact template."""
    d = G.disk_mesh(10.0, 1.0)
    return TriMesh(np.c_[d.V, np.zeros(d.n_vertices)], d.F, name="disk")


@pytest.fixture(scope="session")
def library():
    """The synthetic baffle library (30 shapes) at its default resolution."""
    return build_library()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def rotation_matrix(axis, angle_deg):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    a = np.radians(angle_deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def world_transform(reflect=False, scale=1.0):
    """A similarity transform of world space (rotation, optional reflection, scale, translation), 4x4."""
    t = np.eye(4)
    R = rotation_matrix((1.0, -2.0, 0.5), 37.0)
    if reflect:
        R = R @ np.diag([-1.0, 1.0, 1.0])
    t[:3, :3] = scale * R
    t[:3, 3] = (12.5, -40.0, 7.0)
    return t


# Changes of the world coordinate system that must not change a flattening (up to a rigid
# motion of the template; "scale" scales the template by the same factor)
WORLD_TRANSFORMS = {
    "LPS to RAS": lambda: np.diag([-1.0, -1.0, 1.0, 1.0]),
    "rotation and translation": lambda: world_transform(),
    "reflection": lambda: world_transform(reflect=True),
    "scale x2": lambda: world_transform(scale=2.0),
}


def transformed(mesh, matrix):
    """The mesh in another world coordinate system (with consistent face orientation)."""
    V = mesh.V @ matrix[:3, :3].T + matrix[:3, 3]
    m = TriMesh(V, mesh.F.copy(), mesh.name, dict(mesh.meta)).orient_consistently()
    return m


def rigid_distance(uv, ref, allow_reflection=True):
    """Max vertex distance between two templates after the best rigid alignment of uv onto ref."""
    from slicerheart_patch_flattening.uncertainty import rigid_align
    d = np.abs(rigid_align(uv, ref) - ref).max()
    if allow_reflection:
        d = min(d, np.abs(rigid_align(uv * np.array([1.0, -1.0]), ref) - ref).max())
    return float(d)


def euler_characteristic(mesh):
    return mesh.n_vertices - len(mesh.edges()) + mesh.n_faces


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------
#
# Tests record the numbers behind their assertions (max stretch, edge errors,
# suture-line mismatch per shape and method, ...) through the `metrics` fixture.
# Each recorded value is compared with tests/metrics/baseline.json: an error
# metric may not get worse than the baseline by more than the tolerance, so a
# change that stays inside a test's hard limit but degrades a method still
# fails. All values of a run are written to latest.json in the test report
# folder (.test-reports at the repository root, not committed).
#
#     python -m pytest                     # compare with the baseline
#     python -m pytest --update-baseline   # accept the current values
#     python -m pytest --metrics-tolerance 0.02
#
# A metric that is absent from the baseline is recorded without comparison.

METRICS_DIR = Path(__file__).parent / "metrics"
BASELINE_PATH = METRICS_DIR / "baseline.json"
REPORT_DIR = Path(os.environ.get("SLICERHEART_PATCH_FLATTENING_TEST_REPORT_DIR",
                                 Path(__file__).parent.parent / ".test-reports"))
LATEST_PATH = REPORT_DIR / "latest.json"


def pytest_addoption(parser):
    parser.addoption("--update-baseline", action="store_true", default=False,
                     help="write the metrics recorded in this run into tests/metrics/baseline.json")
    parser.addoption("--metrics-tolerance", type=float, default=0.05,
                     help="relative tolerance for regression of recorded metrics (default 0.05)")


class MetricRecorder:
    """record(name, value, direction="lower", tol=None, atol=0.0)

    direction: "lower" (an error: may not rise above baseline (1 + tol) + atol),
               "higher" (a score: may not fall below baseline (1 - tol) - atol),
               "exact" (must stay within baseline +/- (|baseline| tol + atol)),
               "record" (stored for reference, never compared: timings and
               other machine-dependent values).
    """

    def __init__(self, baseline: dict, tolerance: float, update: bool):
        self.baseline, self.tolerance, self.update = baseline, tolerance, update
        self.values = {}

    def record(self, name, value, direction="lower", tol=None, atol=0.0):
        value = float(value)
        self.values[name] = value
        if direction == "record" or self.update or name not in self.baseline:
            return value
        ref = float(self.baseline[name])
        tol = self.tolerance if tol is None else tol
        margin = abs(ref) * tol + atol
        if direction == "lower":
            ok = value <= ref + margin
        elif direction == "higher":
            ok = value >= ref - margin
        elif direction == "exact":
            ok = abs(value - ref) <= margin
        else:
            raise ValueError(direction)
        assert ok, (f"metric {name!r} regressed: {value:.6g} vs baseline {ref:.6g} "
                    f"(direction {direction}, tolerance {tol:g} + {atol:g})")
        return value


@pytest.fixture(scope="session")
def metrics(request):
    baseline = json.loads(BASELINE_PATH.read_text()) if BASELINE_PATH.exists() else {}
    rec = MetricRecorder(baseline, request.config.getoption("--metrics-tolerance"),
                         request.config.getoption("--update-baseline"))
    yield rec
    LATEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEST_PATH.write_text(json.dumps(dict(sorted(rec.values.items())), indent=2) + "\n")
    if rec.update:
        merged = dict(baseline)
        merged.update(rec.values)
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(dict(sorted(merged.items())), indent=2) + "\n")
