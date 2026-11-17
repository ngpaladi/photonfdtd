"""EnergyMonitor and closed-surface (bounded) flux."""
import numpy as np
import pytest

import photonfdtd as pf

import jax
jax.config.update("jax_enable_x64", True)

LAM = 1e-6
F0 = pf.C_0 / LAM
DX = LAM / 20


def _dipole_sim(monitors, sym=(0, 0, 0), run_time=60e-15):
    g = pf.Grid(size=(8e-6, 8e-6), cell_size=DX, pml_layers=(14, 14, 0),
                symmetry=sym)
    src = pf.PointDipole(position=(0.0, 0.0) if not any(sym) else (DX / 2, DX / 2),
                         component="Ez",
                         waveform=pf.GaussianPulse(freq0=F0, fwhm=6e-15))
    return pf.Simulation(g, sources=[src], monitors=monitors, run_time=run_time)


def test_energy_balance_box_flux_plus_stored_energy():
    """Energy conservation in a box around a dipole: what the closed surface
    lets out, plus what is still inside, equals what the source put in
    (the inside energy at its peak, once the pulse has been injected)."""
    box = pf.closed_surface("b", center=(0, 0), size=(3e-6, 3e-6))
    assert [m.name for m in box] == ["b-x", "b+x", "b-y", "b+y"] or len(box) == 6
    U = pf.EnergyMonitor(name="u", interval=5,
                         bounds=[(-1.5e-6, 1.5e-6), (-1.5e-6, 1.5e-6), None])
    sim = _dipole_sim(box + [U])
    res = sim.run()
    out = pf.closed_surface_flux(res, "b")
    u = res.energy["u"]
    assert out > 0
    # After the pulse has left, the box is (nearly) empty and the outflow
    # equals the energy the source injected; the peak stored energy is a
    # lower bound on that, and the remaining energy is small.
    assert u[-1] < 1e-3 * u.max()
    assert out >= 0.9 * u.max()


def test_bounded_flux_matches_cropped_photon_spectrum_closure():
    """Four bounded faces in 2-D close: a source outside the box gives zero
    net flux through it."""
    box = pf.closed_surface("b", center=(2e-6, 0), size=(1.5e-6, 1.5e-6))
    sim = _dipole_sim(box)
    res = sim.run()
    one_face = abs(res.flux["b-x"])
    assert abs(pf.closed_surface_flux(res, "b")) < 2e-3 * one_face


def test_energy_on_symmetric_grid_is_whole_domain():
    """The EnergyMonitor of a quarter-domain run equals the full run's."""
    U = pf.EnergyMonitor(name="u", interval=20)
    full = _dipole_sim([U], run_time=10e-15)
    # Full grid with matching node layout: put the dipole at +dx/2 too.
    full = pf.Simulation(pf.Grid(size=(8e-6 + 0 * DX, 8e-6), cell_size=DX,
                                 pml_layers=(14, 14, 0)),
                         sources=[pf.PointDipole(position=(s * DX / 2, t * DX / 2),
                                                 component="Ez",
                                                 waveform=full.sources[0].waveform)
                                  for s in (1, -1) for t in (1, -1)],
                         monitors=[U], run_time=10e-15)
    half = _dipole_sim([U], sym=(1, 1, 0), run_time=10e-15)
    assert half.grid.full_shape == full.grid.shape
    uf, uh = full.run().energy["u"], half.run().energy["u"]
    assert np.allclose(uh, uf, rtol=1e-9, atol=1e-12 * uf.max())


def test_tiled_refuses_energy_monitor():
    sim = _dipole_sim([pf.EnergyMonitor(name="u")], run_time=2e-15)
    with pytest.raises(NotImplementedError, match="EnergyMonitor"):
        sim.run(disk_tiled=True)
