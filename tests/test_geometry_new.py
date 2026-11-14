"""Sphere and Cylinder primitives."""
import math

import numpy as np
import pytest

import photonfdtd as pf


def test_sphere_volume_and_ellipsoid():
    dx = 0.02e-6
    g = pf.Grid(size=(1.2e-6, 1.2e-6, 1.2e-6), cell_size=dx)
    r = 0.4e-6
    m = pf.Sphere(center=(0, 0, 0), radius=r, medium=pf.Medium(2.0)).region_mask(g)
    vol = m.sum() * dx ** 3
    assert abs(vol / (4 / 3 * math.pi * r ** 3) - 1) < 0.02
    e = pf.Sphere(center=(0.1e-6, 0, 0), radius=(0.4e-6, 0.2e-6, 0.1e-6),
                  medium=pf.Medium(2.0)).region_mask(g)
    assert abs(e.sum() * dx ** 3 / (4 / 3 * math.pi * 0.4e-6 * 0.2e-6 * 0.1e-6) - 1) < 0.05
    with pytest.raises(ValueError):
        pf.Sphere(center=(0, 0, 0), radius=0.0, medium=pf.Medium(2.0)).region_mask(g)


def test_cylinder_axes_2d_and_tube():
    dx = 0.02e-6
    g2 = pf.Grid(size=(2e-6, 2e-6), cell_size=dx)
    disk = pf.Cylinder(center=(0, 0, 0), radius=0.5e-6, length=math.inf,
                       medium=pf.Medium(2.0)).region_mask(g2)
    assert abs(disk.sum() * dx ** 2 / (math.pi * 0.25e-12) - 1) < 0.02
    strip = pf.Cylinder(center=(0, 0, 0), radius=0.3e-6, length=1e-6, axis="x",
                        medium=pf.Medium(2.0)).region_mask(g2)
    # An x-cylinder cut by z = 0 is a 1 um x 0.6 um strip.
    assert abs(strip.sum() * dx ** 2 / (1e-6 * 0.6e-6) - 1) < 0.05
    # A hollow tube: cladding then bore.
    g3 = pf.Grid(size=(2e-6, 2e-6, 1e-6), cell_size=0.05e-6)
    sim = pf.Simulation(g3, structures=[
        pf.Cylinder(center=(0, 0, 0), radius=0.8e-6, length=math.inf,
                    medium=pf.Medium.from_index(1.5)),
        pf.Cylinder(center=(0, 0, 0), radius=0.4e-6, length=math.inf,
                    medium=pf.Medium(1.0))], run_time=1e-15)
    i0, j0, _ = g3.index_at((0, 0, 0))
    i1, _, _ = g3.index_at((0.6e-6, 0, 0))
    assert sim.eps_r[i0, j0, 0] == 1.0 and np.isclose(sim.eps_r[i1, j0, 0], 2.25)


def test_sphere_with_subpixel_smoothing():
    g = pf.Grid(size=(1e-6, 1e-6), cell_size=0.05e-6)
    s = pf.Sphere(center=(0.013e-6, 0, 0), radius=0.3e-6, medium=pf.Medium(4.0))
    sim = pf.Simulation(g, structures=[s], subpixel=True, run_time=1e-15)
    ex, ey, ez = sim.eps_components()
    # Interface cells get intermediate values; the fill (mean of Ez's
    # arithmetic average) tracks the disk area.
    assert ((ez > 1.01) & (ez < 3.99)).any()
    fill = (ez.mean() - 1.0) / 3.0
    assert abs(fill / (math.pi * 0.09e-12 / 1e-12) - 1) < 0.02


def test_mode_solver_refuses_magnetic():
    with pytest.raises(NotImplementedError, match="mu_r"):
        pf.ModeSolver(size=(2e-6, 2e-6), cell_size=0.1e-6, structures=[
            pf.Box(center=(0, 0), size=(0.5e-6, 0.5e-6),
                   medium=pf.Medium(permittivity=4.0, permeability=2.0))],
            wavelength=1.55e-6)
