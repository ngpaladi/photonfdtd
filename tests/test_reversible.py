"""Reversible-in-time adjoint: exact invertibility, gradient parity, O(1) memory.

Skipped when jax is not installed.
"""
import numpy as np
import pytest

import photonfdtd as pf

import jax
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
                         run_time=run_time)


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


def test_reversible_pml_gradient_matches_plain():
    """The PML-shell reversible adjoint (forward-then-reverse reconstructs the
    interior; the PML shell is replayed from a per-step tape) gives the exact
    gradient WITH CPML - validated against the plain reverse pass. (This path is
    correct but only saves memory for a thin PML shell; see module docstring.)"""
    from photonfdtd import reversible_pml as rp
    lam = 1e-6
    f0 = pf.C_0 / lam
    dx = lam / 12
    grid = pf.Grid(size=(2e-6, 2e-6, 1.2e-6), cell_size=dx, pml_layers=(6, 6, 6))
    box = pf.Box(center=(0, 0, 0), size=(0.7e-6, 0.7e-6, 0.4e-6),
                 medium=pf.Medium.from_index(2.5))
    src = pf.PointDipole(position=(-0.4e-6, 0, 0), component="Ez",
                         waveform=pf.GaussianPulse(freq0=f0, fwhm=6e-15))
    mon = pf.DFTMonitor(name="d", components=("Ez",), freqs=[f0])
    sim = pf.Simulation(grid, structures=[box], sources=[src], monitors=[mon],
                        run_time=45e-15)

    # forward-then-reverse returns to the initial state at machine precision.
    from photonfdtd.jaxbackend import _build_static, _monitor_plan
    static, plans = _build_static(sim), _monitor_plan(sim)
    ce = sim.dt / (np.asarray(sim.eps_r) * pf.EPS_0)
    ce_e = {c: jnp.asarray(ce) for c in ("Ex", "Ey", "Ez")}
    init, fwd, rev_step, shell_of = rp.make_pml_reversible(sim, static, plans, ce_e)
    N = sim.n_steps
    c = init()
    tape = []
    for n in range(N):
        tape.append(shell_of(c))
        c = fwd(c, ce_e, n)
    peak = max(float(jnp.abs(c[0][k]).max()) for k in c[0])
    for n in range(N - 1, -1, -1):
        c = rev_step(c, tape[n], ce_e, n)
    assert max(float(jnp.abs(c[0][k]).max()) for k in c[0]) / peak < 1e-11

    loss = lambda out: jnp.sum(jnp.abs(out["dft"]["d"]["Ez"]) ** 2)
    v_ref, g_ref = pf.jax_value_and_grad_eps(sim, loss, remat="none")
    v_rev, g_rev = pf.jax_value_and_grad_eps_reversible_pml(sim, loss)
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
                        run_time=20e-15)
    from photonfdtd.reversible import reversible_available
    ok, _ = reversible_available(sim)
    assert not ok
    with pytest.raises(NotImplementedError):
        pf.jax_value_and_grad_eps_reversible(
            sim, lambda out: out["dft"]["d"]["Ez"].real.sum())


def _dispersive_sim(pml, gamma=0.0):
    """A lossless (Sellmeier-like) or lossy Lorentz box, with a plain box
    stamped over part of it."""
    lam = 1e-6
    f0 = pf.C_0 / lam
    grid = pf.Grid(size=(2e-6, 2e-6, 1.2e-6), cell_size=lam / 14,
                   pml_layers=(pml,) * 3)
    med = pf.DispersiveMedium.lorentz(2.0, [(1.5, 1.5 * f0, gamma),
                                            (0.5, 0.8 * f0, 0.0)])
    boxes = [pf.Box(center=(0, 0, 0), size=(0.8e-6, 0.8e-6, 0.4e-6), medium=med),
             pf.Box(center=(0.2e-6, 0, 0), size=(0.3e-6, 0.3e-6, 0.4e-6),
                    medium=pf.Medium.from_index(1.6))]
    src = pf.PointDipole(position=(-0.5e-6, 0, 0), component="Ez",
                         waveform=pf.GaussianPulse(freq0=f0, fwhm=6e-15))
    mons = [pf.DFTMonitor(name="d", components=("Ez",), freqs=[f0]),
            pf.FluxMonitor(name="f", plane_axis="x", plane_position=0.5e-6)]
    return pf.Simulation(grid, structures=boxes, sources=[src], monitors=mons,
                         run_time=40e-15)


def _disp_loss(out):
    return jnp.sum(jnp.abs(out["dft"]["d"]["Ez"]) ** 2) + 1e-20 * out["flux"]["f"]


@pytest.mark.parametrize("pml", [0, 5])
def test_reversible_lossless_dispersion(pml):
    """Lossless poles are inverted exactly, so both reversible adjoints give
    the plain reverse-pass gradient to machine precision."""
    sim = _dispersive_sim(pml)
    v_ref, g_ref = pf.jax_value_and_grad_eps(sim, _disp_loss, remat="none")
    fns = [pf.jax_value_and_grad_eps_reversible_pml]
    if pml == 0:
        fns.append(pf.jax_value_and_grad_eps_reversible)
    for fn in fns:
        v, g = fn(sim, _disp_loss)
        assert abs(v - v_ref) <= 1e-13 * abs(v_ref)
        assert np.abs(g - g_ref).max() <= 1e-12 * np.abs(g_ref).max()


def test_reversible_lossless_dispersion_round_trip():
    """forward N steps then reverse N steps returns fields and polarization
    to zero."""
    from photonfdtd import reversible as rev
    from photonfdtd.jaxbackend import _build_static, _monitor_plan, _build_ade_jax
    sim = _dispersive_sim(0)
    static = _build_static(sim)
    ade = _build_ade_jax(sim, static)
    ce = sim.dt / (np.asarray(sim.eps_r) * pf.EPS_0)
    ce_e = {c: jnp.asarray(ce) for c in ("Ex", "Ey", "Ez")}
    init, fwd, rev_step, _ = rev._make_reversible(sim, static, _monitor_plan(sim),
                                                  ce_e, ade)
    fwd, rev_step = jax.jit(fwd), jax.jit(rev_step)
    c = init()
    for n in range(sim.n_steps):
        c = fwd(c, ce_e, n)
    peak_E = max(float(jnp.abs(c[0][k]).max()) for k in c[0])
    peak_P = float(jnp.abs(c[3][0]).max())
    assert peak_P > 0
    for n in range(sim.n_steps - 1, -1, -1):
        c = rev_step(c, ce_e, n)
    assert max(float(jnp.abs(c[0][k]).max()) for k in c[0]) <= 1e-11 * peak_E
    assert max(float(jnp.abs(p).max()) for p in c[3]) <= 1e-11 * peak_P


def test_reversible_rejects_lossy_poles():
    from photonfdtd.reversible import reversible_available
    sim = _dispersive_sim(0, gamma=1e13)
    ok, why = reversible_available(sim)
    assert not ok and "lossless" in why
    for fn in (pf.jax_value_and_grad_eps_reversible,
               pf.jax_value_and_grad_eps_reversible_pml):
        with pytest.raises(NotImplementedError, match="lossless"):
            fn(sim, _disp_loss)
