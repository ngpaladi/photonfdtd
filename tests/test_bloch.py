"""Bloch-periodic boundaries and oblique plane waves (complex fields)."""
import math

import numpy as np
import pytest

import photonfdtd as pf

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

LAM = 1e-6
F0 = pf.C_0 / LAM
DX = LAM / 40
PERIOD = 0.5e-6


def _wave():
    return pf.GaussianPulse(freq0=F0, fwhm=6e-15)


def _pw(pol, angle, x=-3e-6, direction="+x"):
    return pf.PlaneWave(center=(x, 0, 0), direction=direction, polarization=pol,
                        waveform=_wave(), angle=angle, tilt_axis="y")


def _grid(pw):
    return pf.Grid(size=(14e-6, PERIOD, 0), cell_size=(DX, DX, 0),
                   pml_layers=(30, 0, 0), periodic=(False, True, False),
                   bloch=pw.bloch_vector())


TANG = ("Ey", "Ez", "Hy", "Hz")


def _run(pol, angle, structures=(), planes=(("t", 3e-6), ("b", -4.5e-6))):
    pw = _pw(pol, angle)
    g = _grid(pw)
    mons = [pf.DFTMonitor(name=nm, components=TANG, freqs=[F0], plane_axis="x",
                          plane_position=x) for nm, x in planes]
    sim = pf.Simulation(g, structures=list(structures), sources=[pw],
                        monitors=mons, run_time=160e-15)
    return sim, sim.run()


def _sdf(sim, res, name):
    return float(pf.photon_spectrum(res, sim, name).energy_density[0])


def test_bloch_grid_validation():
    with pytest.raises(ValueError, match="periodic"):
        pf.Grid(size=(4e-6, 1e-6), cell_size=0.1e-6, bloch=(0, 1e6, 0))
    g = pf.Grid(size=(4e-6, 1e-6), cell_size=0.1e-6, periodic=(False, True, False),
                bloch=(0, 1e6, 0))
    assert g.is_complex
    assert np.isclose(g.bloch_phase(1), np.exp(1j * 1e6 * 1e-6))
    pw = _pw("Ez", 30.0)
    with pytest.raises(ValueError, match="Bloch"):
        pf.Simulation(pf.Grid(size=(4e-6, PERIOD), cell_size=DX,
                              pml_layers=(10, 0, 0), periodic=(False, True, False)),
                      sources=[pw], run_time=1e-15)


@pytest.mark.parametrize("pol", ["Ez", "Ey"])
def test_oblique_wave_is_one_way_with_bloch_phase(pol):
    """At 30 degrees the sheet launches forwards only, the field carries
    exp(i k_t y) across the cell, and its tangential E is the requested
    amplitude (s) or amplitude*cos(angle) (p), as at normal incidence."""
    sim, res = _run(pol, 30.0)
    _, ref = _run(pol, 0.0)
    E = np.asarray(res.dft["t"][pol])[0].ravel()
    E0 = np.asarray(ref.dft["t"][pol])[0].ravel()
    k = _pw(pol, 30.0).bloch_vector()[1]
    y = sim.grid.coords[1] + (0.5 * DX if pol == "Ey" else 0.0)
    # The Bloch phase across the cell, sample to sample.
    ph = E / np.exp(1j * k * y)
    assert np.abs(ph / ph[0] - 1).max() < 1e-6
    expect = math.cos(math.radians(30)) if pol == "Ey" else 1.0
    assert abs(np.abs(E).mean() / np.abs(E0).mean() / expect - 1) < 0.01
    back = np.abs(np.asarray(res.dft["b"][pol])[0]).max()
    assert back / np.abs(E).max() < 5e-3


@pytest.mark.parametrize("pol", ["Ez", "Ey"])
def test_oblique_fresnel_transmission(pol):
    """30 degrees onto an n = 2 half-space: the transmitted power at f0
    matches Fresnel's T_s (Ez) and T_p (Ey) to 1%."""
    n2, th = 2.0, math.radians(30)
    half = pf.Box(center=(4e-6, 0, 0), size=(8e-6, 1, 1),
                  medium=pf.Medium.from_index(n2))
    sim0, inc = _run(pol, 30.0)
    sim1, tr = _run(pol, 30.0, structures=[half])
    T = _sdf(sim1, tr, "t") / _sdf(sim0, inc, "t")
    ci = math.cos(th)
    ct = math.sqrt(1 - (math.sin(th) / n2) ** 2)
    if pol == "Ez":
        r = (ci - n2 * ct) / (ci + n2 * ct)
    else:
        r = (n2 * ci - ct) / (n2 * ci + ct)
    assert abs(T / (1 - r ** 2) - 1) < 0.01, (T, 1 - r ** 2)


def test_complex_gradient_matches_finite_difference():
    """The checkpointed adjoint differentiates through the complex fields."""
    dx = LAM / 14
    pw = pf.PlaneWave(center=(-0.8e-6, 0, 0), direction="+x", polarization="Ez",
                      waveform=_wave(), angle=20.0, tilt_axis="y")
    g = pf.Grid(size=(3e-6, 1e-6, 0), cell_size=(dx, dx, 0), pml_layers=(8, 0, 0),
                periodic=(False, True, False), bloch=pw.bloch_vector())
    box = pf.Box(center=(0.2e-6, 0.1e-6, 0), size=(0.6e-6, 0.4e-6, 1),
                 medium=pf.Medium.from_index(2.0))
    mon = pf.DFTMonitor(name="d", components=("Ez",), freqs=[F0],
                        plane_axis="x", plane_position=0.8e-6)
    sim = pf.Simulation(g, structures=[box], sources=[pw], monitors=[mon],
                        run_time=30e-15)
    loss = lambda out: jnp.sum(jnp.abs(out["dft"]["d"]["Ez"]) ** 2)
    v, grad = pf.jax_value_and_grad_eps(sim, loss)
    i, j, _ = sim.grid.index_at((0.2e-6, 0.1e-6))
    h = 1e-4
    vals = []
    eps0 = np.array(sim.eps_r, copy=True)
    for s in (+1, -1):
        eps = eps0.copy()
        eps[i, j, 0] += s * h
        sim.eps_r = eps
        vals.append(pf.jax_value_and_grad_eps(sim, loss)[0])
    fd = (vals[0] - vals[1]) / (2 * h)
    assert abs(grad[i, j, 0] / fd - 1) < 1e-5


def test_complex_refusals():
    pw = _pw("Ez", 30.0)
    g = pf.Grid(size=(4e-6, PERIOD), cell_size=DX, pml_layers=(10, 0, 0),
                periodic=(False, True, False), bloch=pw.bloch_vector())
    flux = pf.FluxMonitor(name="f", plane_axis="x", plane_position=1e-6)
    with pytest.raises(NotImplementedError, match="FluxMonitor"):
        pf.Simulation(g, sources=[pw], monitors=[flux], run_time=1e-15).run()
    mon = pf.DFTMonitor(name="d", components=("Ez",), freqs=[F0])
    sim = pf.Simulation(g, sources=[pw], monitors=[mon], run_time=1e-15)
    loss = lambda out: jnp.sum(jnp.abs(out["dft"]["d"]["Ez"]) ** 2)
    with pytest.raises(NotImplementedError, match="complex"):
        pf.jax_value_and_grad_eps_reversible_pml(sim, loss)
