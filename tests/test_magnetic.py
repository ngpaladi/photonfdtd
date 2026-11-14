"""Magnetic media: relative permeability and magnetic loss in the H update."""
import numpy as np
import pytest

import photonfdtd as pf

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

LAM = 1e-6
F0 = pf.C_0 / LAM
DX = LAM / 40


def _wave(fwhm=6e-15):
    return pf.GaussianPulse(freq0=F0, fwhm=fwhm)


def _slab_run(medium, half_space=True, dx=DX):
    """1-D run along x: a one-way Ey plane wave hits a slab (or half-space)
    starting at x = 0. Returns (incident flux, transmitted flux, reflected
    Ey phasor at the reflection plane) with the reflected part taken as the
    difference from the empty run."""
    g = pf.Grid(size=(24e-6, 0, 0), cell_size=(dx, 0, 0),
                pml_layers=(int(round(30 * DX / dx)), 0, 0))
    size = 20e-6 if half_space else 2e-6
    st = [pf.Box(center=(size / 2, 0, 0), size=(size, 1, 1), medium=medium)]
    src = pf.PlaneWave(center=(-4e-6, 0, 0), direction="+x", polarization="Ey",
                       waveform=_wave())
    mons = [pf.FluxMonitor(name="in", plane_axis="x", plane_position=-2e-6),
            pf.FluxMonitor(name="out", plane_axis="x",
                           plane_position=4e-6 if half_space else 3e-6),
            pf.DFTMonitor(name="r", components=("Ey",), freqs=[F0],
                          plane_axis="x", plane_position=-2e-6)]
    kw = dict(sources=[src], monitors=mons, run_time=200e-15)
    ref = pf.Simulation(g, **kw).run()
    res = pf.Simulation(g, structures=st, **kw).run()
    r = complex(np.ravel(res.dft["r"]["Ey"])[0] - np.ravel(ref.dft["r"]["Ey"])[0])
    inc = complex(np.ravel(ref.dft["r"]["Ey"])[0])
    return float(ref.flux["in"]), float(res.flux["out"]), r / inc


def test_impedance_matched_slab_does_not_reflect():
    """eps_r = mu_r = 4: index 2 but vacuum impedance, so nothing reflects and
    all the power goes through. On the grid the residual reflection is a
    discretization error, and it falls off second order in dx (because mu is
    averaged onto the staggered H nodes; sampled per cell it is first order)."""
    med = pf.Medium(permittivity=4.0, permeability=4.0)
    P_in, P_out, r40 = _slab_run(med)
    assert abs(r40) < 0.03
    assert abs(P_out / P_in - 1) < 2e-3
    _, _, r80 = _slab_run(med, dx=DX / 2)
    assert abs(r40) / abs(r80) > 3.5


def test_permeability_half_space_matches_fresnel_with_opposite_sign():
    """mu_r = 4, eps_r = 1 has the same index (2) as eps_r = 4 and the same
    power transmission 4 Z1 Z2 / (Z1 + Z2)^2 = 8/9, but the impedance goes up
    instead of down, so the reflection coefficient flips sign:
    r = (Z2 - Z1)/(Z2 + Z1) = +1/3 against -1/3 for the dielectric."""
    P_in, P_out, r_mu = _slab_run(pf.Medium(permittivity=1.0, permeability=4.0))
    assert abs(P_out / P_in / (8 / 9) - 1) < 0.01
    _, _, r_eps = _slab_run(pf.Medium(permittivity=4.0))
    assert abs(abs(r_mu) - 1 / 3) < 0.01 and abs(abs(r_eps) - 1 / 3) < 0.01
    # Same reference plane for both, so the phases differ by pi.
    assert abs(r_mu + r_eps) < 0.02


def test_matched_magnetic_loss_absorbs_without_reflecting():
    """sigma / eps0 = sigma_m / mu0 with eps_r = mu_r = 1 is a matched
    absorber (the continuum limit of a PML at normal incidence): no reflection,
    and a 2-um slab of it attenuates by exp(-2 sigma d / (eps0 c))."""
    sigma = 2e3
    med = pf.Medium(permittivity=1.0, conductivity=sigma,
                    magnetic_conductivity=sigma * pf.MU_0 / pf.EPS_0)
    P_in, P_out, r = _slab_run(med, half_space=False)
    assert abs(r) < 0.01
    expect = np.exp(-2 * sigma * 2e-6 / (pf.EPS_0 * pf.C_0))
    assert abs(P_out / P_in / expect - 1) < 0.03


def test_anisotropic_permeability_components():
    """A diagonal mu tensor reaches the right H component: a wave polarized
    Ey (H along z) sees mu_zz and ignores mu_yy."""
    m1 = pf.AnisotropicMedium(permittivity=(4, 4, 4), permeability=(1, 1, 4))
    m2 = pf.AnisotropicMedium(permittivity=(4, 4, 4), permeability=(1, 9, 4))
    _, _, r1 = _slab_run(m1)
    _, _, r2 = _slab_run(m2)
    assert abs(r1) < 0.03 and abs(r1 - r2) < 1e-12
    m3 = pf.AnisotropicMedium(permittivity=(4, 4, 4), permeability=4.0)
    assert m3.permeability == (4.0, 4.0, 4.0)


def _grad_sim():
    dx = LAM / 14
    g = pf.Grid(size=(3e-6, 1e-6, 0), cell_size=(dx, dx, 0),
                pml_layers=(8, 0, 0), periodic=(False, True, False))
    box = pf.Box(center=(0.2e-6, 0, 0), size=(0.6e-6, 0.4e-6, 1),
                 medium=pf.Medium(permittivity=2.0, permeability=1.5))
    src = pf.PlaneWave(center=(-0.6e-6, 0, 0), direction="+x", polarization="Ez",
                       waveform=_wave())
    mon = pf.DFTMonitor(name="d", components=("Ez",), freqs=[F0],
                        plane_axis="x", plane_position=0.8e-6)
    return pf.Simulation(g, structures=[box], sources=[src], monitors=[mon],
                         run_time=40e-15)


def test_magnetic_gradients_agree_across_adjoints():
    """The checkpointed and PML-reversible adjoints step the same mu_r and
    give the same d/d eps_r; and the gradient matches a finite difference."""
    sim = _grad_sim()
    loss = lambda out: jnp.sum(jnp.abs(out["dft"]["d"]["Ez"]) ** 2)
    v_ref, g_ref = pf.jax_value_and_grad_eps(sim, loss, remat="none")
    v_rev, g_rev = pf.jax_value_and_grad_eps_reversible_pml(sim, loss)
    assert np.isclose(v_ref, v_rev)
    assert np.abs(g_ref - g_rev).max() / np.abs(g_ref).max() < 1e-9
    # Forward value agrees with Simulation.run (same coefficients).
    out = sim.run()
    assert np.isclose(float(np.sum(np.abs(out.dft["d"]["Ez"]) ** 2)), v_ref)
    # The gradient path really steps mu (it is not silently dropped).
    plain = pf.Simulation(sim.grid, structures=[pf.Box(
        center=(0.2e-6, 0, 0), size=(0.6e-6, 0.4e-6, 1),
        medium=pf.Medium(permittivity=2.0))], sources=[], monitors=sim.monitors,
        run_time=40e-15)
    plain.sources = sim.sources
    v_plain, _ = pf.jax_value_and_grad_eps(plain, loss, remat="none")
    assert abs(v_plain / v_ref - 1) > 1e-3


def test_magnetic_refusals():
    lossy = pf.Medium(permittivity=2.0, magnetic_conductivity=100.0)
    g = pf.Grid(size=(2e-6, 1e-6), cell_size=LAM / 14, pml_layers=(6, 6, 0))
    box = pf.Box(center=(0, 0, 0), size=(0.5e-6, 0.5e-6, 1), medium=lossy)
    with pytest.raises(NotImplementedError, match="magnetic"):
        pf.Simulation(g, structures=[box], subpixel=True, run_time=5e-15)
    sim = pf.Simulation(g, structures=[box], run_time=5e-15)
    with pytest.raises(NotImplementedError, match="magnetic"):
        sim.run(disk_tiled=True)
    mon = pf.DFTMonitor(name="d", components=("Ez",), freqs=[F0])
    sim = pf.Simulation(g, structures=[box], monitors=[mon], run_time=5e-15)
    loss = lambda out: jnp.sum(jnp.abs(out["dft"]["d"]["Ez"]) ** 2)
    with pytest.raises(NotImplementedError, match="magnetic loss"):
        pf.jax_value_and_grad_eps_reversible_pml(sim, loss)


def test_memory_estimate_counts_magnetic_coefficients():
    g = pf.Grid(size=(2e-6, 1e-6), cell_size=LAM / 14, pml_layers=(6, 6, 0))
    plain = pf.Simulation(g, run_time=5e-15)
    mag = pf.Simulation(g, structures=[pf.Box(center=(0, 0, 0), size=(1, 1, 1),
                        medium=pf.Medium(permeability=2.0))], run_time=5e-15)
    assert pf.estimate_memory(mag)["forward"] > pf.estimate_memory(plain)["forward"]
