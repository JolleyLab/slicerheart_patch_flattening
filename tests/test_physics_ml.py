"""Tests for the Warp membrane track (CPU)."""
import numpy as np
import pytest

wp = pytest.importorskip("warp")

from slicerheart_patch_flattening.library import generators as G
from slicerheart_patch_flattening.algorithms import flatten
from slicerheart_patch_flattening.metrics import distortion_metrics
from slicerheart_patch_flattening.physics_ml.warp_membrane import WarpMembrane, Material
from slicerheart_patch_flattening.physics_ml.inverse import inflate, InverseTemplate


@pytest.fixture(scope="module")
def cap():
    return G.parametric_patch(G.superellipse(7, 6, 2.0, 48), G.lift_sphere, 1.4, R=10.0).orient_consistently()


def test_warp_flatten_matches_numpy_membrane(cap):
    mat = Material("stvk", E=1.0, nu=0.3, thickness=1.0)
    wm = WarpMembrane(cap.F, cap.V, mat, "flatten")
    uv0 = flatten(cap, "arap").uv
    x, info = wm.solve(uv0)
    ref = flatten(cap, "membrane_stvk")
    assert abs(info["energy"] - ref.info["energy"]) / ref.info["energy"] < 1e-4
    M = distortion_metrics(cap, x); Mr = distortion_metrics(cap, ref.uv)
    assert abs(M["stretch_Linf"] - Mr["stretch_Linf"]) < 1e-3


def test_warp_gradient_finite_difference(cap):
    mat = Material("neo_hookean", E=1.0, nu=0.3, thickness=1.0)
    wm = WarpMembrane(cap.F, cap.V, mat, "flatten")
    uv = flatten(cap, "arap").uv
    rng = np.random.default_rng(0); v = rng.standard_normal(uv.shape); eps = 1e-6
    _, g = wm.energy_grad(uv)
    Ep, _ = wm.energy_grad(uv + eps * v); Em, _ = wm.energy_grad(uv - eps * v)
    assert abs((Ep - Em) / (2 * eps) - np.sum(g * v)) / abs(np.sum(g * v)) < 1e-6


def test_inflate_and_adjoint(cap):
    mat = Material("stvk", E=10.0, nu=0.3, thickness=0.3)
    loop = cap.boundary_loop()
    T = flatten(cap, "arap").uv
    x, info, _ = inflate(T, cap.F, loop, cap.V[loop], 20.0, mat, x0=cap.V.copy())
    assert info["converged"] and np.allclose(x[loop], cap.V[loop])
    inv = InverseTemplate(x, cap.F, loop, 20.0, mat, strain_weight=0.5)
    chk = inv.check_gradient(T * 0.98, eps=1e-2)
    assert chk["rel_err"] < 0.05
