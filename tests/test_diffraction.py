"""Diffraction orders of a periodic cell from a plane DFT monitor."""
import math

import numpy as np
import pytest

import photonfdtd as pf

import jax
jax.config.update("jax_enable_x64", True)

LAM = 1e-6
F0 = pf.C_0 / LAM
DX = LAM / 30
TANG = ("Ey", "Ez", "Hy", "Hz")


def _run(period, structures=(), angle=0.0, pol="Ez"):
    pw = pf.PlaneWave(center=(-2.5e-6, 0, 0), direction="+x", polarization=pol,
                      waveform=pf.GaussianPulse(freq0=F0, fwhm=6e-15),
                      angle=angle, tilt_axis="y" if angle else None)
    g = pf.Grid(size=(12e-6, period, 0), cell_size=(DX, DX, 0),
                pml_layers=(30, 0, 0), periodic=(False, True, False),
                bloch=pw.bloch_vector())
    mons = [pf.DFTMonitor(name=nm, components=TANG, freqs=[F0], plane_axis="x",
                          plane_position=x) for nm, x in (("t", 3.5e-6), ("r", -3.5e-6))]
    sim = pf.Simulation(g, structures=list(structures), sources=[pw],
                        monitors=mons, run_time=150e-15)
    return sim, sim.run()


def test_oblique_wave_is_one_order_at_its_angle():
    sim, res = _run(1.0e-6, angle=30.0)
    d = pf.diffraction_orders(res, sim, "t")
    i0 = d.order(0, 0)
    assert d.fraction[i0] > 0.999
    assert abs(d.sin_theta[i0, 0] - 0.5) < 1e-9          # t_axes = (y, z)
    # Period = lambda at 30 degrees: order -1 propagates (sin = -0.5), +1 not.
    assert d.propagating[d.order(-1)] and not d.propagating[d.order(1)]
    assert abs(d.sin_theta[d.order(-1), 0] + 0.5) < 1e-9


def test_grating_orders_obey_grating_equation_and_parseval():
    """A rectangular phase grating, period 2.5 lambda: orders up to |m| = 2
    propagate at sin(theta) = m lambda / period, the evanescent ones carry
    nothing to a plane far away, and the order powers sum to the flux."""
    period = 2.5e-6
    teeth = pf.Box(center=(0, 0.0, 0), size=(0.6e-6, period / 2, 1),
                   medium=pf.Medium.from_index(1.8))
    sim, res = _run(period, [teeth])
    d = pf.diffraction_orders(res, sim, "t")
    for mo in (-2, -1, 1, 2):
        i = d.order(mo)
        assert d.propagating[i]
        assert abs(d.sin_theta[i, 0] - mo * LAM / period) < 1e-9
    assert not d.propagating[d.order(3)]
    assert np.abs(d.fraction[~d.propagating]).sum() < 1e-4
    # Several orders carry power; the grating is symmetric, so +-m match.
    assert d.fraction[d.order(1)] > 0.02
    assert abs(d.fraction[d.order(1)] / d.fraction[d.order(-1)] - 1) < 1e-3
    # Parseval: the per-order flux sums to the flux summed over the plane.
    E = {c: np.asarray(res.dft["t"][c])[0].ravel() for c in TANG}
    direct = (E["Ey"] * E["Hz"].conj() - E["Ez"] * E["Hy"].conj()).real.sum()
    assert abs(d.plane_sum.sum() / direct - 1) < 1e-10
    ref = pf.photon_spectrum(res, sim, "t").energy_density[0]
    assert np.isclose(d.power.sum(), ref)


def test_diffraction_needs_periodic_plane():
    g = pf.Grid(size=(4e-6, 2e-6), cell_size=DX * 2, pml_layers=(8, 8, 0))
    mon = pf.DFTMonitor(name="t", components=TANG, freqs=[F0], plane_axis="x",
                        plane_position=1e-6)
    sim = pf.Simulation(g, monitors=[mon], run_time=1e-15)
    res = sim.run()
    with pytest.raises(ValueError, match="periodic"):
        pf.diffraction_orders(res, sim, "t")
