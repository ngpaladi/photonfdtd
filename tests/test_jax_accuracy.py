"""JAX backend parity + differentiability for the accuracy features.

Subpixel smoothing and dispersive (ADE) media are ported to the JAX device
path. These tests confirm the JAX results match the NumPy reference to
floating-point reordering, and that gradients flow through the dispersive time
evolution (the point of running them under JAX).

Skipped entirely when jax is not installed.
"""
import numpy as np
import pytest

import photonfdtd as pf
from photonfdtd.materials import DispersiveMedium

jax = pytest.importorskip("jax")
import jax.numpy as jnp   # noqa: E402


def test_subpixel_jax_matches_numpy():
    """Subpixel-smoothed run: JAX device path == NumPy reference."""
    def build(use_jax):
        lam0 = 1e-6
        f0 = pf.C_0 / lam0
        dx = lam0 / 20
        grid = pf.Grid(size=(4e-6, 4e-6), cell_size=dx, pml_layers=(10, 10, 0))
        box = pf.Box(center=(0.3e-6, 0.13e-6), size=(1.55e-6, 0.77e-6),
                     medium=pf.Medium.from_index(2.5))          # off-grid edges
        src = pf.PointDipole(position=(-1e-6, 0.0), component="Ez",
                             waveform=pf.GaussianPulse(freq0=f0, fwhm=8e-15))
        mon = pf.DFTMonitor(name="d", components=("Ez",), freqs=[f0])
        sim = pf.Simulation(grid, structures=[box], sources=[src], monitors=[mon],
                            run_time=100e-15, subpixel=True, use_jax=use_jax)
        return np.asarray(sim.run().dft["d"]["Ez"])

    a, b = build(False), build(True)
    assert np.abs(a - b).max() / np.abs(a).max() < 1e-10


def test_dispersion_jax_matches_numpy():
    """Dispersive (Lorentz ADE) run: JAX device path == NumPy reference."""
    def build(use_jax):
        med = DispersiveMedium.lorentz(2.25, [(1.2, 500e12, 0.0)])
        dx = 50e-9
        L = 30e-6
        grid = pf.Grid(size=(L,), cell_size=dx, pml_layers=(20,))
        box = pf.Box(center=(0.0,), size=(2 * L,), medium=med)
        src = pf.PointDipole(position=(-8e-6,), component="Ez",
                             waveform=pf.GaussianPulse(freq0=200e12, fwhm=8e-15))
        mon = pf.DFTMonitor(name="d", components=("Ez",), freqs=[200e12, 250e12])
        sim = pf.Simulation(grid, structures=[box], sources=[src], monitors=[mon],
                            run_time=200e-15, use_jax=use_jax)
        return np.asarray(sim.run().dft["d"]["Ez"])

    a, b = build(False), build(True)
    assert np.abs(a - b).max() / np.abs(a).max() < 1e-10


def test_dispersion_is_differentiable():
    """jax.grad flows through the ADE polarization recursion. Differentiate a
    DFT-intensity loss w.r.t. a pole-strength scale and check finite differences.
    This is what makes dispersive inverse design possible."""
    from photonfdtd import jaxbackend as jb
    jax.config.update("jax_enable_x64", True)

    med = DispersiveMedium.lorentz(2.25, [(1.2, 500e12, 0.0)])
    dx = 60e-9
    L = 16e-6
    grid = pf.Grid(size=(L,), cell_size=dx, pml_layers=(15,))
    box = pf.Box(center=(0.0,), size=(2 * L,), medium=med)
    src = pf.PointDipole(position=(-4e-6,), component="Ez",
                         waveform=pf.GaussianPulse(freq0=250e12, fwhm=8e-15))
    mon = pf.DFTMonitor(name="d", components=("Ez",), freqs=[250e12])
    sim = pf.Simulation(grid, structures=[box], sources=[src], monitors=[mon],
                        run_time=120e-15, use_jax=True)

    static = jb._build_static(sim)
    plans = jb._monitor_plan(sim)
    pplans = jb._particle_plan(sim, static)
    ade = jb._build_ade_jax(sim, static)
    ce = (sim.dt / (np.asarray(sim.eps_r, np.float64) * pf.EPS_0))
    ce_e = {c: jnp.asarray(ce) for c in ("Ex", "Ey", "Ez")}
    Cbase = jnp.asarray(ade["C"])

    def loss(alpha):
        ade_a = dict(ade)
        ade_a["C"] = alpha * Cbase                     # scale pole strength
        sf, sd, sfx = jb._device_simulate(sim, static, plans, ce_e, pplans,
                                          ade=ade_a, ce_particle=jnp.asarray(ce))
        return jnp.sum(jnp.abs(sd["d"]["Ez"]) ** 2).real

    g = float(jax.grad(loss)(1.0))
    h = 1e-4
    fd = float((loss(1.0 + h) - loss(1.0 - h)) / (2 * h))
    assert g != 0.0
    assert abs(g - fd) / abs(fd) < 1e-5


def test_gradient_checkpointing_matches_plain_grad():
    """Nested gradient checkpointing (O(sqrt(n_steps)) adjoint memory) must give
    a bit-for-bit identical gradient to the plain O(n_steps) reverse pass -
    rematerialization is exact, it only trades compute for memory."""
    lam0 = 1e-6
    f0 = pf.C_0 / lam0
    dx = lam0 / 16
    grid = pf.Grid(size=(3e-6, 3e-6), cell_size=dx, pml_layers=(8, 8, 0))
    box = pf.Box(center=(0, 0), size=(1e-6, 1e-6),
                 medium=pf.Medium.from_index(2.0))
    src = pf.PointDipole(position=(-1e-6, 0), component="Ez",
                         waveform=pf.GaussianPulse(freq0=f0, fwhm=6e-15))
    mon = pf.DFTMonitor(name="d", components=("Ez",), freqs=[f0])
    sim = pf.Simulation(grid, structures=[box], sources=[src], monitors=[mon],
                        run_time=120e-15, use_jax=True)
    assert sim.n_steps > 100                       # enough steps for segmentation
    loss = lambda out: out["dft"]["d"]["Ez"].real.sum()

    v0, g0 = pf.jax_value_and_grad_eps(sim, loss, remat="none")
    # 2-level (sqrt(N)) and 3-level (N**(1/3)) checkpointing must both reproduce
    # the plain reverse pass exactly - rematerialization is exact.
    for levels in (2, 3):
        v1, g1 = pf.jax_value_and_grad_eps(sim, loss, remat="nested",
                                           checkpoint_levels=levels)
        assert np.isclose(v0, v1)
        assert np.allclose(g0, g1, rtol=1e-10, atol=1e-12 * np.abs(g0).max())


def test_grad_eps_guards_on_new_features():
    """value_and_grad_eps differentiates scalar eps_r only; it must refuse the
    remapped-coefficient cases rather than return a silently-wrong gradient."""
    grid = pf.Grid(size=(3e-6,), cell_size=1e-7, pml_layers=(8,))
    box = pf.Box(center=(0.0,), size=(1e-6,),
                 medium=DispersiveMedium.drude(1.0, 100e12, 5e12))
    src = pf.PointDipole(position=(-0.8e-6,), component="Ez",
                         waveform=pf.GaussianPulse(freq0=200e12, fwhm=8e-15))
    mon = pf.DFTMonitor(name="d", components=("Ez",), freqs=[200e12])
    sim = pf.Simulation(grid, structures=[box], sources=[src], monitors=[mon],
                        run_time=20e-15, use_jax=True)
    with pytest.raises(NotImplementedError):
        pf.jax_value_and_grad_eps(sim, lambda out: out["dft"]["d"]["Ez"].real.sum())
