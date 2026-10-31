"""The Warp membrane track (CPU): material models against closed forms, gradients against finite
differences, the pressurised membrane against Hencky's solution, forward inflation and the adjoint
inverse. Skipped when warp is not installed."""
import numpy as np
import pytest

wp = pytest.importorskip("warp")

from slicerheart_patch_flattening.geometric import flatten
from slicerheart_patch_flattening.library import generators as G
from slicerheart_patch_flattening.metrics import distortion_metrics
from slicerheart_patch_flattening.mesh import TriMesh
from slicerheart_patch_flattening.physics_ml.warp_membrane import WarpMembrane, Material, MMHG_TO_MPA
from slicerheart_patch_flattening.physics_ml.inverse import inflate, InverseTemplate, uniform_unload

pytestmark = pytest.mark.physics


@pytest.fixture(scope="module")
def cap():
    return G.parametric_patch(G.superellipse(7, 6, 2.0, 48), G.lift_sphere, 1.4, R=10.0).orient_consistently()


def disk(radius=10.0, h=1.0):
    d = G.disk_mesh(radius, h)
    return TriMesh(np.c_[d.V, np.zeros(d.n_vertices)], d.F, name="disk")


# ---------------------------------------------------------------------------
# Material models
# ---------------------------------------------------------------------------

def test_material_parameters():
    m = Material("stvk", E=10.0, nu=0.3, thickness=0.3)
    assert abs(m.mu - 10 / 2.6) < 1e-12
    assert abs(m.lam - 10 * 0.3 / (1 - 0.09)) < 1e-12                       # plane stress
    assert abs(m.bending_stiffness - 10 * 0.027 / (12 * 0.91)) < 1e-12
    assert m.params().shape == (8,) and m.params()[0] == m.model_id
    assert Material("stvk").model_id != Material("neo_hookean").model_id != Material("hgo").model_id


@pytest.mark.parametrize("model", ["stvk", "neo_hookean"])
@pytest.mark.parametrize("stretch", [1.0, 1.05, 1.2])
def test_energy_closed_form_under_biaxial_stretch(model, stretch):
    """A flat sheet stretched uniformly by lambda in both directions. StVK: W = 2 e^2 (mu + lambda_ps)
    with e = (lambda^2 - 1) / 2; incompressible neo-Hookean: W = mu/2 (2 lambda^2 + lambda^-4 - 3).
    Total energy = thickness * area * W, and the interior gradient vanishes (uniform stress)."""
    d = disk()
    mat = Material(model, E=10.0, nu=0.3, thickness=0.3)
    wm = WarpMembrane(d.F, d.V, mat, "flatten")
    E, g = wm.energy_grad(d.V[:, :2] * stretch)
    e = (stretch**2 - 1) / 2
    W = 2 * e * e * (mat.mu + mat.lam) if model == "stvk" else 0.5 * mat.mu * (2 * stretch**2 + stretch**-4 - 3)
    expected = d.area() * mat.thickness * W
    assert abs(E - expected) < 1e-9 * max(expected, 1e-3)
    assert np.abs(g[~d.is_boundary_vertex()]).max() < 1e-9


def test_hgo_fibres_only_in_tension():
    """The HGO fibre term (fibres along u of the template, so in inflate mode where the rest
    configuration is the template) is active only when the fibres are stretched (I4 > 1), and
    its closed form is k1 / (2 k2) (exp(k2 (I4 - 1)^2) - 1) per unit rest area."""
    d = disk()
    T = d.V[:, :2]
    iso = Material("neo_hookean", E=10.0, nu=0.3, thickness=0.3)
    hgo = Material("hgo", E=10.0, nu=0.3, thickness=0.3, k1=5.0, k2=1.0, fiber_angle_deg=0.0)
    E_iso = lambda x: WarpMembrane(d.F, T, iso, "inflate", bending=False).energy_grad(x)[0]
    E_hgo = lambda x: WarpMembrane(d.F, T, hgo, "inflate", bending=False).energy_grad(x)[0]
    along = d.V * np.array([1.1, 1.0, 1.0])          # fibres stretched: I4 = 1.21
    across = d.V * np.array([1.0, 1.1, 1.0])         # fibres unloaded: I4 = 1
    fibre = 5.0 / 2.0 * (np.exp(1.0 * 0.21**2) - 1) * 0.3 * d.area()
    assert abs(E_hgo(along) - E_iso(along) - fibre) < 1e-9 * fibre
    assert abs(E_hgo(across) - E_iso(across)) < 1e-12 * E_iso(across)
    # the same with the fibres at 90 degrees: the roles swap
    hgo90 = Material("hgo", E=10.0, nu=0.3, thickness=0.3, k1=5.0, k2=1.0, fiber_angle_deg=90.0)
    E_90 = lambda x: WarpMembrane(d.F, T, hgo90, "inflate", bending=False).energy_grad(x)[0]
    assert abs(E_90(across) - E_iso(across) - fibre) < 1e-9 * fibre
    assert abs(E_90(along) - E_iso(along)) < 1e-9 * E_iso(along)


# ---------------------------------------------------------------------------
# Flattening (level 1)
# ---------------------------------------------------------------------------

def test_warp_flatten_matches_numpy_membrane(cap):
    """The Warp StVK membrane flattening reproduces the scipy/numpy membrane_stvk method."""
    mat = Material("stvk", E=1.0, nu=0.3, thickness=1.0)
    wm = WarpMembrane(cap.F, cap.V, mat, "flatten")
    uv0 = flatten(cap, "arap").uv
    x, info = wm.solve(uv0)
    ref = flatten(cap, "membrane_stvk")
    assert abs(info["energy"] - ref.info["energy"]) / ref.info["energy"] < 1e-4
    M = distortion_metrics(cap, x)
    Mr = distortion_metrics(cap, ref.uv)
    assert abs(M["stretch_Linf"] - Mr["stretch_Linf"]) < 1e-3
    assert M["flipped_triangles"] == 0


@pytest.mark.parametrize("model", ["stvk", "neo_hookean", "hgo"])
def test_warp_gradient_finite_difference(cap, model):
    mat = Material(model, E=1.0, nu=0.3, thickness=1.0, k1=0.5 if model == "hgo" else 0.0, fiber_angle_deg=30.0)
    wm = WarpMembrane(cap.F, cap.V, mat, "flatten")
    uv = flatten(cap, "arap").uv * 1.02
    rng = np.random.default_rng(0)
    v = rng.standard_normal(uv.shape)
    eps = 1e-6
    _, g = wm.energy_grad(uv)
    Ep, _ = wm.energy_grad(uv + eps * v)
    Em, _ = wm.energy_grad(uv - eps * v)
    assert abs((Ep - Em) / (2 * eps) - np.sum(g * v)) / abs(np.sum(g * v)) < 1e-6


def test_flatten_energy_small_for_developable(half_cylinder, half_cylinder_domain):
    """The exact template of a developable surface carries (almost) no strain energy: only the
    chord error of the 1.2 mm triangles on R = 7 mm remains, equivalent to a uniform strain well
    below 0.2 % (the energy is quadratic in the strain)."""
    mat = Material("stvk", E=10.0, nu=0.3, thickness=0.3)
    wm = WarpMembrane(half_cylinder.F, half_cylinder.V, mat, "flatten")
    E, g = wm.energy_grad(half_cylinder_domain.V)
    E_1pct, _ = wm.energy_grad(half_cylinder_domain.V * 1.01)
    equivalent_strain = 0.01 * np.sqrt(E / E_1pct)
    assert equivalent_strain < 0.002
    # and the solver does not move away from it by more than the chord error
    x, info = wm.solve(half_cylinder_domain.V)
    assert info["energy"] <= E + 1e-12 and np.abs(x - half_cylinder_domain.V).max() < 0.05


# ---------------------------------------------------------------------------
# Inflation (level 2)
# ---------------------------------------------------------------------------

def test_inflate_zero_pressure_is_the_template():
    """Without pressure, a template sewn onto its own (planar) outline stays flat and unstrained."""
    d = disk()
    loop = d.boundary_loop()
    mat = Material("stvk", E=10.0, nu=0.3, thickness=0.3)
    x, info, model = inflate(d.V[:, :2], d.F, loop, d.V[loop], 0.0, mat)
    assert info["converged"]
    assert np.abs(x - d.V).max() < 1e-6 and info["energy"] < 1e-9


def test_bending_energy_zero_when_flat():
    d = disk()
    mat = Material("stvk", E=10.0, nu=0.3, thickness=0.3)
    with_b = WarpMembrane(d.F, d.V[:, :2], mat, "inflate", bending=True)
    without = WarpMembrane(d.F, d.V[:, :2], mat, "inflate", bending=False)
    x = d.V * np.array([1.05, 1.05, 1.0])
    Eb, gb = with_b.energy_grad(x)
    E0, g0 = without.energy_grad(x)
    assert abs(Eb - E0) < 1e-12 * E0 and np.abs(gb - g0).max() < 1e-9


@pytest.mark.parametrize("pressure_mmhg", [20.0, 60.0])
def test_hencky_pressurised_circular_membrane(pressure_mmhg, metrics):
    """Centre deflection of a clamped circular membrane under uniform pressure (Hencky 1915):
    w0 = c(nu) a (q a / (E h))^(1/3), c(0.3) = 0.662 (Fichter, NASA TP-3658, 1997). The deflection
    must also scale with the cube root of the pressure."""
    a, E, h = 10.0, 10.0, 0.3
    d = disk(a, 1.0)
    loop = d.boundary_loop()
    mat = Material("stvk", E=E, nu=0.3, thickness=h)
    x, info, _ = inflate(d.V[:, :2], d.F, loop, d.V[loop], pressure_mmhg, mat, bending=False, max_iter=5000)
    assert info["converged"] and np.allclose(x[loop], d.V[loop])
    q = pressure_mmhg * MMHG_TO_MPA
    w0 = np.abs(x[:, 2]).max()
    hencky = 0.662 * a * (q * a / (E * h)) ** (1 / 3)
    assert abs(w0 - hencky) / hencky < 0.03
    assert np.abs(x[:, 2]).argmax() == np.linalg.norm(d.V[:, :2], axis=1).argmin()   # at the centre
    metrics.record(f"physics/hencky/p{pressure_mmhg:g}/relative_error", abs(w0 - hencky) / hencky, atol=0.002)


def test_bending_stiffens_the_membrane():
    d = disk(10.0, 1.0)
    loop = d.boundary_loop()
    mat = Material("stvk", E=10.0, nu=0.3, thickness=1.0)      # thick: bending matters
    x0, _, _ = inflate(d.V[:, :2], d.F, loop, d.V[loop], 60.0, mat, bending=False, max_iter=5000)
    x1, _, _ = inflate(d.V[:, :2], d.F, loop, d.V[loop], 60.0, mat, bending=True, max_iter=5000)
    assert 0 < np.abs(x1[:, 2]).max() < np.abs(x0[:, 2]).max()


def test_inflate_and_adjoint(cap):
    """Forward inflation onto the cap's suture line converges; the adjoint gradient of the
    inverse (template design) problem matches finite differences."""
    mat = Material("stvk", E=10.0, nu=0.3, thickness=0.3)
    loop = cap.boundary_loop()
    T = flatten(cap, "arap").uv
    x, info, _ = inflate(T, cap.F, loop, cap.V[loop], 20.0, mat, x0=cap.V.copy())
    assert info["converged"] and np.allclose(x[loop], cap.V[loop])
    inv = InverseTemplate(x, cap.F, loop, 20.0, mat, strain_weight=0.5)
    chk = inv.check_gradient(T * 0.98, eps=1e-2)
    assert chk["rel_err"] < 0.05


def test_uniform_unload():
    """The analytic pre-shrink of a template flattened from a pressurised geometry: a uniformly
    smaller template, by the hoop strain p r / (E h) of a thin-walled cylinder of radius r."""
    T = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    mat = Material("stvk", E=10.0, nu=0.3, thickness=0.3)
    T0 = uniform_unload(T, 20.0, mat, radius_mm=10.0)
    stretch = 1 + 20.0 * MMHG_TO_MPA * 10.0 / (0.3 * 10.0)
    assert np.allclose(T0 * stretch, T, atol=1e-12)
    assert 0.98 < 1 / stretch < 1.0
