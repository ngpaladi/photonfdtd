"""Reversible-in-time adjoint: exact invertibility, gradient parity, O(1) memory.

Skipped when jax is not installed.
"""
import numpy as np
import pytest

import photonfdtd as pf

jax = pytest.importorskip("jax")
import jax.numpy as jnp   # noqa: E402
jax.config.update("jax_enable_x64", True)


def _nopml_sim(run_time=60e-15):
    lam = 1e-6
    f0 = pf.C_0 / lam
    dx = lam / 14
    grid = pf.Grid(size=(2e-6, 2e-6, 1.2e-6), cell_size=dx, pml_layers=(0, 0, 0))
    box = pf.Box(center=(0, 0, 0), size=(0.8e-6, 0.8e-6, 0.4e-6),
                 medium=pf.Medium.from_index(2.5))
    src = pf.PointDipole(position=(-0.5e-6, 0, 0), component="Ez",
                         waveform=pf.GaussianPulse(freq0=f0, fwhm=6e-15))
    mon = pf.DFTMonitor(name="d", components=("Ez",), freqs=[f0])
    return pf.Simulation(grid, structures=[box], sources=[src], monitors=[mon],
                         run_time=run_time, use_jax=True)


def test_forward_reverse_is_identity():
    """reverse_step exactly inverts forward_step: forward N then reverse N
    returns the state (and monitor accumulators) to zero at machine precision."""
    from photonfdtd import reversible as rev
    from photonfdtd.jaxbackend import _build_static, _monitor_plan
    sim = _nopml_sim()
    static = _build_static(sim)
    plans = _monitor_plan(sim)
    ce = sim.dt / (np.asarray(sim.eps_r) * pf.EPS_0)
    ce_e = {c: jnp.asarray(ce) for c in ("Ex", "Ey", "Ez")}
    init, fwd, rev_step, _ = rev._make_reversible(sim, static, plans, ce_e)
    N = sim.n_steps
    c = init()
    for n in range(N):
        c = fwd(c, ce_e, n)
    peak = max(float(jnp.abs(c[0][k]).max()) for k in c[0])
    for n in range(N - 1, -1, -1):
        c = rev_step(c, ce_e, n)
    resid = max(float(jnp.abs(c[0][k]).max()) for k in c[0])
    assert resid / peak < 1e-11
    assert float(jnp.abs(c[1]["d"]["Ez"]).max()) < 1e-25   # DFT accum back to 0


def test_reversible_gradient_matches_plain():
    """The reversible adjoint gradient equals the plain reverse-pass gradient
    to machine precision (it is exact rematerialization, just memory-cheaper)."""
    sim = _nopml_sim()
    loss = lambda out: jnp.sum(jnp.abs(out["dft"]["d"]["Ez"]) ** 2)
    v_ref, g_ref = pf.jax_value_and_grad_eps(sim, loss, remat="none")
    v_rev, g_rev = pf.jax_value_and_grad_eps_reversible(sim, loss)
    assert np.isclose(v_ref, v_rev)
    assert np.abs(g_ref - g_rev).max() / np.abs(g_ref).max() < 1e-10


def test_reversible_gating():
    """The reversible adjoint refuses cases it cannot handle exactly (PML)."""
    lam = 1e-6
    f0 = pf.C_0 / lam
    grid = pf.Grid(size=(2e-6, 2e-6, 1.2e-6), cell_size=lam / 14,
                   pml_layers=(6, 6, 6))                       # PML present
    src = pf.PointDipole(position=(0, 0, 0), component="Ez",
                         waveform=pf.GaussianPulse(freq0=f0, fwhm=6e-15))
    mon = pf.DFTMonitor(name="d", components=("Ez",), freqs=[f0])
    sim = pf.Simulation(grid, sources=[src], monitors=[mon],
                        run_time=20e-15, use_jax=True)
    from photonfdtd.reversible import reversible_available
    ok, _ = reversible_available(sim)
    assert not ok
    with pytest.raises(NotImplementedError):
        pf.jax_value_and_grad_eps_reversible(
            sim, lambda out: out["dft"]["d"]["Ez"].real.sum())
