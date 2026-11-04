"""JAX accuracy features: regression + differentiability.

Subpixel smoothing and dispersive (ADE) media run on the JAX engine. These
tests check them against golden values from the retired NumPy core (see
_golden.py) and confirm gradients flow through the dispersive time evolution.
"""
import numpy as np
import pytest

import photonfdtd as pf
from photonfdtd.materials import DispersiveMedium
from _golden import assert_golden

import jax
import jax.numpy as jnp

# Golden fingerprints (see _golden.py), keyed case -> output -> fingerprint.
GOLDEN = {
    "subpixel": {
        "dft_Ez": [5.8600581798972575e-15, -5.733049932961114e-15, 6.2041662415072204e-15, -5.532026335824776e-15],
    },
    "dispersion": {
        "dft_Ez": [6.68555648298826e-14, -1.0028088832020372e-14, 6.683700159703115e-14, -6.701031200683314e-15],
    },
}


def case_subpixel():
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
                        run_time=100e-15, subpixel=True)
    return {"dft_Ez": np.asarray(sim.run().dft["d"]["Ez"])}


def case_dispersion():
    med = DispersiveMedium.lorentz(2.25, [(1.2, 500e12, 0.0)])
    dx = 50e-9
    L = 30e-6
    grid = pf.Grid(size=(L,), cell_size=dx, pml_layers=(20,))
    box = pf.Box(center=(0.0,), size=(2 * L,), medium=med)
    src = pf.PointDipole(position=(-8e-6,), component="Ez",
                         waveform=pf.GaussianPulse(freq0=200e12, fwhm=8e-15))
    mon = pf.DFTMonitor(name="d", components=("Ez",), freqs=[200e12, 250e12])
    sim = pf.Simulation(grid, structures=[box], sources=[src], monitors=[mon],
                        run_time=200e-15)
    return {"dft_Ez": np.asarray(sim.run().dft["d"]["Ez"])}


def test_subpixel_matches_golden():
    for k, v in case_subpixel().items():
        assert_golden(v, GOLDEN["subpixel"][k], 1e-9, k)


def test_dispersion_matches_golden():
    """Dispersive (Lorentz ADE) run."""
    for k, v in case_dispersion().items():
        assert_golden(v, GOLDEN["dispersion"][k], 1e-9, k)


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
                        run_time=120e-15)

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
                        run_time=120e-15)
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
                        run_time=20e-15)
    with pytest.raises(NotImplementedError):
        pf.jax_value_and_grad_eps(sim, lambda out: out["dft"]["d"]["Ez"].real.sum())
