"""Smoke tests: minimum-viable checks on FDTD and mode solver."""
import numpy as np
import pytest
import photonfdtd as pf


def test_grid_basic():
    g = pf.Grid(size=(2e-6, 2e-6), cell_size=50e-9, pml_layers=(8, 8, 0))
    assert g.ndim == 2
    assert g.shape[0] == 40 and g.shape[1] == 40 and g.shape[2] == 1


def test_mode_solver_slab_index_in_bounds():
    """Effective index of a guided mode should sit between cladding and core."""
    n_core, n_clad = 2.0, 1.0
    core = pf.Box(center=(0.0, 0.0), size=(0.5e-6, 0.3e-6),
                  medium=pf.Medium.from_index(n_core))
    ms = pf.ModeSolver(size=(4e-6, 3e-6), cell_size=20e-9,
                       structures=[core], wavelength=1.55e-6,
                       background_eps=n_clad ** 2, num_modes=2)
    res = ms.solve()
    assert res.n_eff[0] < n_core + 1e-3
    assert res.n_eff[0] > n_clad
    assert res.n_eff[0] >= res.n_eff[1]


def test_fdtd_2d_dipole_runs():
    """2D dipole simulation runs without exploding."""
    lam0 = 1.0e-6
    freq0 = pf.C_0 / lam0
    grid = pf.Grid(size=(2e-6, 2e-6), cell_size=lam0 / 15, pml_layers=(8, 8, 0))
    src = pf.PointDipole(
        position=(0.0, 0.0), component="Ez",
        waveform=pf.GaussianPulse(freq0=freq0, fwhm=8e-15),
    )
    mon = pf.FieldMonitor(name="snap", components=("Ez",), interval=20)
    sim = pf.Simulation(grid, sources=[src], monitors=[mon], run_time=40e-15)
    res = sim.run()
    assert "snap" in res.fields
    arr = res.fields["snap"]["Ez"]
    assert arr.ndim == 4  # (n_frames, nx, ny, nz)
    assert np.isfinite(arr).all()
    # field should be non-trivial in the interior
    assert np.abs(arr).max() > 0.0


def test_fdtd_3d_dipole_runs():
    """Tiny 3D simulation should also run cleanly."""
    lam0 = 1.0e-6
    freq0 = pf.C_0 / lam0
    grid = pf.Grid(size=(1.5e-6, 1.5e-6, 1.5e-6),
                   cell_size=lam0 / 10, pml_layers=(6, 6, 6))
    src = pf.PointDipole(position=(0.0, 0.0, 0.0), component="Ez",
                         waveform=pf.GaussianPulse(freq0=freq0, fwhm=6e-15))
    sim = pf.Simulation(grid, sources=[src], run_time=20e-15)
    res = sim.run()
    assert res.times.size > 0
