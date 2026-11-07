"""Electrostatic and thermal solvers (photonfdtd.statics) against analytic cases."""
import numpy as np
import pytest

import photonfdtd as pf
from photonfdtd import statics

D = 0.02e-6


def _line_mesh(n=100):
    z = (np.arange(n) - (n - 1) / 2) * D
    return statics.Mesh((np.array([0.0]), np.array([0.0]), z), (0.0, 0.0, D)), z


def _plates(z):
    top = pf.Box((0, 0, z[-1]), (1, 1, 1.5 * D), pf.Medium())
    bot = pf.Box((0, 0, z[0]), (1, 1, 1.5 * D), pf.Medium())
    return top, bot


def test_parallel_plate_uniform_field():
    mesh, z = _line_mesh()
    top, bot = _plates(z)
    r = statics.solve_potential(mesh, np.ones(mesh.shape), [top, bot], [1.0, 0.0])
    # The electrodes' potential sits on their faces, half a cell in from
    # the end cells' centres.
    gap = z[-1] - z[0] - D
    Ez = r.E[2][0, 0, 2:-2]
    assert np.allclose(Ez, -1.0 / gap, rtol=1e-10)
    inner = slice(1, -1)
    assert np.allclose(r.phi[0, 0, inner], (z[inner] - z[0] - D / 2) / gap, atol=1e-12)


def test_two_layer_dielectric_series_field():
    """D is continuous: eps1 E1 = eps2 E2, and E1 L1 + E2 L2 = V."""
    mesh, z = _line_mesh()
    top, bot = _plates(z)
    eps = np.ones(mesh.shape)
    eps[..., 50:] = 4.0
    r = statics.solve_potential(mesh, eps, [top, bot], [1.0, 0.0])
    L1 = z[49] - z[0]          # electrode face to the interface
    L2 = z[-1] - z[50]
    E2 = -1.0 / (4 * L1 + L2)
    assert r.E[2][0, 0, 25] == pytest.approx(4 * E2, rel=1e-9)
    assert r.E[2][0, 0, 75] == pytest.approx(E2, rel=1e-9)


def test_superposition_matches_direct_solve():
    y = (np.arange(40) - 19.5) * D
    z = (np.arange(30) - 14.5) * D
    mesh = statics.Mesh((np.array([0.0]), y, z), (0.0, D, D))
    eps = np.ones(mesh.shape)
    eps[:, :, :12] = 3.9
    a = pf.Box((-0.3e-6, 0.1e-6), (0.1e-6, 0.1e-6), pf.Medium())
    b = pf.Box((0.3e-6, 0.1e-6), (0.1e-6, 0.1e-6), pf.Medium())
    c = pf.Box((0.0, -0.2e-6), (0.6e-6, 0.04e-6), pf.Medium())
    V = [2.0, -1.5, 0.25]
    r = statics.solve_potential(mesh, eps, [a, b, c], V)
    phi_sum = sum(v * r.basis_phi[i] for i, v in enumerate(V))
    assert np.allclose(r.phi, phi_sum, atol=1e-12)
    # Electrode cells sit at their voltage.
    for g, v in zip((a, b, c), V):
        m = statics.region_mask(g, mesh)
        assert np.allclose(r.phi[m], v)
    Ex, Ey, Ez = r.field(V)
    assert np.allclose(Ey, r.E[1])


def test_anisotropic_dc_permittivity_uses_the_axis_component():
    mesh, z = _line_mesh()
    top, bot = _plates(z)
    eps = np.ones(mesh.shape)
    lo = eps.copy(); hi = eps.copy()
    hi[..., 50:] = 4.0                      # only eps_zz is layered
    r = statics.solve_potential(mesh, (lo, lo, hi), [top, bot], [1.0, 0.0])
    r2 = statics.solve_potential(mesh, hi, [top, bot], [1.0, 0.0])
    assert np.allclose(r.phi, r2.phi)


def test_ground_boundary():
    y = (np.arange(30) - 14.5) * D
    z = (np.arange(30) - 14.5) * D
    mesh = statics.Mesh((np.array([0.0]), y, z), (0.0, D, D))
    e = pf.Box((0.0, 0.0), (0.1e-6, 0.1e-6), pf.Medium())
    r = statics.solve_potential(mesh, np.ones(mesh.shape), [e], [1.0], boundary="ground")
    assert r.phi[0, 0, 0] < 0.2 and r.phi.max() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        statics.solve_potential(mesh, np.ones(mesh.shape), [e], [1.0], boundary="open")


def test_thermal_slab_uniform_heating_is_parabolic():
    mesh, z = _line_mesh()
    k = np.full(mesh.shape, 1.5)
    heater = pf.Box((0, 0, 0), (1, 1, 10), pf.Medium())
    t = statics.solve_thermal(mesh, k, [heater], [1.0], sinks=("zmin", "zmax"))
    L = z.size * D
    q = 1.0 / L
    T_an = q / (2 * 1.5) * ((L / 2) ** 2 - z ** 2)
    assert np.allclose(t.dT[0, 0], T_an, rtol=0, atol=2e-4 * T_an.max())
    assert np.allclose(t.temperature([2.0]) - t.T0, 2 * t.dT)


def test_thermal_line_conduction_and_convection():
    """1-D conduction from a heater at one end through a slab to a convective
    face: dT(heater) = P (L/k + 1/h) for a unit area."""
    mesh, z = _line_mesh()
    k = np.full(mesh.shape, 2.0)
    heater = pf.Box((0, 0, z[0]), (1, 1, 0.5 * D), pf.Medium())   # the first cell
    h = 5e5
    t = statics.solve_thermal(mesh, k, [heater], [1.0], sinks=(), convection={"zmax": h})
    L = z[-1] - z[0] + D / 2                 # heater centre to the face
    assert t.dT[0, 0, 0] == pytest.approx(L / 2.0 + 1.0 / h, rel=1e-3)


def test_thermal_needs_a_sink():
    mesh, z = _line_mesh()
    heater = pf.Box((0, 0, 0), (1, 1, 10), pf.Medium())
    with pytest.raises(ValueError, match="steady state"):
        statics.solve_thermal(mesh, np.ones(mesh.shape), [heater], sinks=())


def test_mesh_of_mode_solver_and_simulation():
    ms = pf.ModeSolver(size=(1e-6, 1e-6), cell_size=0.1e-6, structures=[],
                       wavelength=1.55e-6)
    m = statics.mesh_of(ms)
    assert m.shape == (1, 10, 10)
    grid = pf.Grid(size=(1e-6, 1e-6), cell_size=0.1e-6)
    assert statics.mesh_of(pf.Simulation(grid)).shape == grid.shape


def test_thermal_sink_on_a_collapsed_axis_is_refused():
    y = (np.arange(20) - 9.5) * D
    mesh = statics.Mesh((y, y, np.array([0.0])), (D, D, 0.0))     # a chip plane
    heater = pf.Box((0.0, 0.0), (0.1e-6, 0.1e-6), pf.Medium())
    with pytest.raises(ValueError, match="collapsed"):
        statics.solve_thermal(mesh, np.ones(mesh.shape), [heater], sinks=("zmin",))
