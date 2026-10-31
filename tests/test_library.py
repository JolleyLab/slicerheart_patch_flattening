"""Synthetic baffle library: generators, shape validity, reproducibility, manifest I/O."""
import numpy as np

from slicerheart_patch_flattening.library import generators as G, build_library, write_library, load_library
from slicerheart_patch_flattening.library.build import shape_summary
from conftest import euler_characteristic

FAMILIES = {"control_developable", "control_nondevelopable", "dorv_tunnel", "baffle_planner_tps",
            "mustard_senning", "papvr"}


def test_library_size_and_families(library):
    assert len(library) == 30
    names = [m.name for m in library]
    assert len(set(names)) == len(names)
    assert {m.meta["family"] for m in library} == FAMILIES
    assert sum(m.meta["developable"] for m in library) >= 8        # controls plus one per clinical family


def test_every_shape_is_a_valid_disk(library):
    for m in library:
        assert np.isfinite(m.V).all(), m.name
        assert euler_characteristic(m) == 1, m.name
        loop = m.boundary_loop()
        assert len(loop) == len(m.boundary_edges()) == len(np.unique(loop)), m.name
        assert m.check_orientation(), m.name
        assert (m.face_areas() > 1e-8).all(), m.name
        assert abs(m.angle_defects().sum() - 2 * np.pi) < 1e-8, m.name   # Gauss-Bonnet, chi = 1


def test_sizes_are_pediatric(library):
    """Patch sizes in the range of the clinical baffles the library imitates (a few cm across)."""
    for m in library:
        bb = m.V.max(0) - m.V.min(0)
        assert 100 < m.area() < 3000, (m.name, m.area())
        assert 10 < bb.max() < 100, (m.name, bb)


def test_developable_flag_matches_curvature(library):
    """Shapes flagged developable carry little integrated |K|; the others carry a lot."""
    for m in library:
        interior = ~m.is_boundary_vertex()
        total_abs_K = np.rad2deg(np.abs(m.angle_defects()[interior]).sum())
        if m.meta["developable"]:
            assert total_abs_K < 20.0, (m.name, total_abs_K)
        else:
            assert total_abs_K > 20.0, (m.name, total_abs_K)


def test_library_is_reproducible(library):
    again = build_library()
    assert [m.name for m in again] == [m.name for m in library]
    for a, b in zip(again, library):
        assert np.array_equal(a.V, b.V) and np.array_equal(a.F, b.F), a.name
    other = build_library(seed=8)
    assert any(a.n_vertices != b.n_vertices or not np.allclose(a.V, b.V) for a, b in zip(other, library))


def test_resolution_parameter():
    coarse = build_library(h=1.6)
    fine = build_library(h=1.0)
    for c, f in zip(coarse, fine):
        assert c.n_vertices < f.n_vertices, c.name
        assert abs(c.area() - f.area()) < 0.03 * f.area(), c.name


def test_write_and_load_library(tmp_path, library):
    shapes = library[:3] + library[6:7]
    manifest = write_library(str(tmp_path), shapes)
    assert len(manifest) == 4
    assert {"name", "family", "developable", "n_vertices", "area_mm2", "total_abs_gauss_curvature_deg",
            "files", "params"} <= set(manifest[0])
    loaded = load_library(str(tmp_path))
    assert [m.name for m in loaded] == [m.name for m in shapes]
    for a, b in zip(loaded, shapes):
        assert a.n_vertices == b.n_vertices and np.allclose(a.V, b.V, atol=1e-6)
        assert a.meta.get("family") == b.meta["family"]


def test_shape_summary(sphere_cap):
    sphere_cap.meta.update(family="test", developable=False)
    s = shape_summary(sphere_cap)
    assert s["n_vertices"] == sphere_cap.n_vertices and s["area_mm2"] == sphere_cap.area()
    assert s["total_abs_gauss_curvature_deg"] >= abs(s["net_gauss_curvature_deg"]) > 0


def test_generators_outlines():
    s = G.superellipse(10.0, 6.0, 2.0, 64)
    assert s.shape == (64, 2) and abs(s[:, 0].max() - 10) < 1e-9 and abs(s[:, 1].max() - 6) < 1e-9
    t = G.trouser_outline(30.0, 20.0, 8.0, 12.0)
    assert t.shape[1] == 2 and len(t) > 8
    d = G.disk_mesh(5.0, 0.8)
    assert d.V.shape[1] == 2 and abs(d.area() - np.pi * 25) < 0.02 * np.pi * 25


def test_lift_cylinder_is_isometric():
    """The cylinder lift preserves every edge length of the parameter domain."""
    dom = G.triangulate_polygon(G.superellipse(10.0, 8.0, 2.0, 48), 1.5)
    V3 = G.lift_cylinder(dom.V, R=7.0)
    e = dom.edges()
    l2 = np.linalg.norm(dom.V[e[:, 0]] - dom.V[e[:, 1]], axis=1)
    l3 = np.linalg.norm(V3[e[:, 0]] - V3[e[:, 1]], axis=1)
    assert np.abs(l3 / l2 - 1).max() < 0.005       # chords of arcs of < 1.5 mm on R = 7


def test_lift_sphere_radius():
    P = G.superellipse(8.0, 8.0, 2.0, 32)
    V = G.lift_sphere(P, R=10.0)
    assert np.allclose(np.linalg.norm(V - [0, 0, 10.0], axis=1), 10.0)


def test_baffle_planner_tps_generator():
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    b3 = np.c_[10 * np.cos(t), 8 * np.sin(t), 2 * np.sin(2 * t)]
    m = G.baffle_planner_patch(b3, 1.2)
    assert m.n_vertices > 50 and len(m.boundary_loop()) > 30
    assert euler_characteristic(m) == 1
    # the surface passes through the given boundary curve
    d = np.linalg.norm(m.V[m.boundary_loop()][:, None] - b3[None], axis=2).min(1)
    assert d.max() < 1.0
