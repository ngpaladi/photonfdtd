"""Tests for the 2.5-D effective-index method, eigenmode expansion, and the
unconditionally-stable ADI-FDTD stepper (NumPy + Rust)."""
import math
import warnings

import numpy as np
import pytest
from scipy.optimize import brentq

import photonfdtd as pf
from photonfdtd.rustbackend import rust_available


# ======================================================================= #
# 2.5-D effective-index method
# ======================================================================= #
def _analytic_slab_te(n1, n2, d, k0, num=4):
    """Effective indices of a symmetric slab from the TE dispersion relation."""
    roots = []
    for parity in ("even", "odd"):
        def f(neff):
            ka = k0 * np.sqrt(max(n1 ** 2 - neff ** 2, 0.0))
            ga = k0 * np.sqrt(max(neff ** 2 - n2 ** 2, 0.0))
            return (ka * np.tan(ka * d / 2) - ga if parity == "even"
                    else ka * (-1.0 / np.tan(ka * d / 2)) - ga)
        xs = np.linspace(n2 + 1e-6, n1 - 1e-6, 4000)
        fs = np.array([f(x) for x in xs])
        for i in range(len(xs) - 1):
            if (np.isfinite(fs[i]) and np.isfinite(fs[i + 1])
                    and fs[i] * fs[i + 1] < 0 and abs(fs[i]) < 1e6
                    and abs(fs[i + 1]) < 1e6):
                try:
                    roots.append(brentq(f, xs[i], xs[i + 1]))
                except ValueError:
                    pass
    return np.array(sorted(set(np.round(roots, 6)), reverse=True))


def test_slab_te_matches_analytic():
    lam, n1, n2, d = 1.55e-6, 3.48, 1.44, 0.22e-6
    k0 = 2 * np.pi / lam
    ana = _analytic_slab_te(n1, n2, d, k0)
    ana = ana[(ana > n2) & (ana < n1)]
    fd = pf.slab_modes([pf.Layer(d, n1)], lam, polarization="TE",
                       num_modes=1, substrate_index=n2, cladding_index=n2)
    assert abs(fd.n_eff[0] - ana[0]) < 1e-3


def test_slab_converges_second_order():
    lam, n1, n2, d = 1.55e-6, 3.48, 1.44, 0.22e-6
    ana = _analytic_slab_te(n1, n2, d, 2 * np.pi / lam)[0]
    errs = []
    for dz in (5e-9, 2.5e-9):
        e = pf.slab_modes([pf.Layer(d, n1)], lam, polarization="TE",
                          num_modes=1, substrate_index=n2, cladding_index=n2,
                          dz=dz).n_eff[0]
        errs.append(abs(e - ana))
    # halving dz should cut the error by ~4x (second order)
    assert errs[1] < errs[0] / 3.0


def test_eim_soi_reduction():
    lam = 1.55e-6
    eim = pf.EffectiveIndex2D.from_stacks(
        core_layers=[pf.Layer(0.22e-6, 3.48)],
        etched_layers=[pf.Layer(0.22e-6, 1.44)],
        wavelength=lam, polarization="TE",
        substrate_index=1.44, cladding_index=1.0)
    # 220 nm SOI TE0 slab index is ~2.83; background < core.
    assert 2.6 < eim.n_core < 3.0
    assert eim.n_background < eim.n_core
    mc, mb = eim.effective_index_media()
    assert mc.eps_r > mb.eps_r


# ======================================================================= #
# Eigenmode expansion
# ======================================================================= #
def _wg_eps(y, width, n_core=2.8, n_clad=1.44):
    e = np.full(y.size, n_clad ** 2)
    e[np.abs(y) <= width / 2] = n_core ** 2
    return e


def test_eme_uniform_guide_is_transparent():
    lam, dy, ny = 1.55e-6, 20e-9, 200
    y = (np.arange(ny) - ny / 2) * dy
    eps = _wg_eps(y, 0.5e-6)
    for N in (1, 4):
        r = pf.eme_2d([pf.Section(eps, 2e-6), pf.Section(eps, 2e-6)],
                      dy, lam, num_modes=N)
        assert abs(abs(r.transmission(0, 0)) - 1.0) < 1e-6
        assert abs(r.reflection(0, 0)) < 1e-9


def test_eme_uniform_phase():
    lam, dy, ny = 1.55e-6, 20e-9, 200
    y = (np.arange(ny) - ny / 2) * dy
    eps = _wg_eps(y, 0.5e-6)
    L = 4e-6
    r = pf.eme_2d([pf.Section(eps, L)], dy, lam, num_modes=1)
    beta0 = r.n_eff_in[0] * 2 * np.pi / lam
    expected = np.exp(1j * beta0 * L)
    got = r.transmission(0, 0)
    assert abs(got - expected) < 1e-6


def test_eme_step_reflection_matches_fresnel():
    """The fundamental reflection of a waveguide-width step equals the Fresnel
    reflection of the two modes' effective indices."""
    lam, dy, ny = 1.55e-6, 20e-9, 200
    y = (np.arange(ny) - ny / 2) * dy
    r = pf.eme_2d([pf.Section(_wg_eps(y, 0.5e-6), 3e-6),
                   pf.Section(_wg_eps(y, 0.3e-6), 3e-6)],
                  dy, lam, num_modes=8)
    n1, n2 = r.n_eff_in[0], r.n_eff_out[0]
    fresnel = ((n1 - n2) / (n1 + n2)) ** 2
    R = abs(r.reflection(0, 0)) ** 2
    assert abs(R - fresnel) / fresnel < 0.15


# ======================================================================= #
# ADI-FDTD (CFL-free), NumPy + Rust
# ======================================================================= #
def _gauss(f0, fwhm):
    sig = fwhm / (2 * math.sqrt(2 * math.log(2)))
    t0 = 4 * sig
    return lambda t: np.exp(-((t - t0) / sig) ** 2) * np.sin(2 * math.pi * f0 * (t - t0))


def test_adi_stable_far_beyond_cfl():
    """ADI stays finite and bounded at 8x the explicit CFL step, where an
    explicit Yee update diverges."""
    nx, ny, dx = 60, 50, 40e-9
    eps = np.ones((nx, ny))
    src = pf.ADISource(20, 25, _gauss(pf.C_0 / 1e-6, 8e-15))
    sim = pf.ADISimulation2D(eps, dx, courant_factor=8.0, boundary="pec",
                             sources=[src])
    assert sim.courant_factor > 7.9
    r = sim.run(400, probes=[(30, 25)])
    p = r.probes["30,25"]
    assert np.all(np.isfinite(p))
    assert np.abs(p).max() < 1e-3          # bounded, not blowing up


def test_adi_cavity_resonance_accurate():
    """PEC vacuum cavity: the dominant resonance matches an analytic mode."""
    nx, ny, dx = 80, 60, 40e-9
    Lx, Ly = nx * dx, ny * dx
    eps = np.ones((nx, ny))
    src = pf.ADISource(33, 25, _gauss(pf.C_0 / 1e-6, 8e-15))
    sim = pf.ADISimulation2D(eps, dx, courant_factor=2.0, boundary="pec",
                             sources=[src])
    steps = int(2500e-15 / sim.dt)
    p = sim.run(steps, probes=[(33, 25)]).probes["33,25"]
    sp = np.abs(np.fft.rfft(p * np.hanning(len(p))))
    fr = np.fft.rfftfreq(len(p), sim.dt)
    band = (fr > 1e14) & (fr < 5e14)
    fpk = fr[band][np.argmax(sp[band])]
    modes = sorted(pf.C_0 / 2 * np.sqrt((m / Lx) ** 2 + (n / Ly) ** 2)
                   for m in range(1, 6) for n in range(1, 6))
    near = min(modes, key=lambda f: abs(f - fpk))
    assert abs(fpk - near) / near < 0.02


def test_adi_absorber_dissipates():
    """The conductivity absorber removes far more energy than a PEC wall."""
    nx, ny, dx = 100, 100, 40e-9
    eps = np.ones((nx, ny))
    src = pf.ADISource(50, 50, _gauss(pf.C_0 / 1e-6, 6e-15))

    def final_l2(boundary):
        sim = pf.ADISimulation2D(eps, dx, courant_factor=2.0,
                                 boundary=boundary, sources=[src], pml_cells=16)
        r = sim.run(700, snapshot_interval=699)
        return np.sqrt((r.snapshots["Ez"][-1] ** 2).sum())

    assert final_l2("absorber") < 1e-3 * final_l2("pec")


@pytest.mark.skipif(not rust_available(),
                    reason="Rust extension not built")
@pytest.mark.parametrize("boundary", ["pec", "absorber"])
def test_adi_rust_matches_numpy(boundary):
    nx, ny, dx = 80, 60, 40e-9
    eps = np.ones((nx, ny)); eps[30:50, 25:35] = 4.0
    src = pf.ADISource(20, 30, _gauss(pf.C_0 / 1e-6, 8e-15))

    def run(backend):
        sim = pf.ADISimulation2D(eps, dx, courant_factor=2.0, boundary=boundary,
                                 sources=[src], backend=backend)
        return sim.run(400, snapshot_interval=100, probes=[(40, 30), (55, 40)])

    a, b = run("numpy"), run("rust")
    sa = a.snapshots["Ez"][-1]
    np.testing.assert_allclose(b.snapshots["Ez"][-1], sa, rtol=0.0,
                               atol=1e-10 * np.abs(sa).max())
    for key in a.probes:
        pa = a.probes[key]
        np.testing.assert_allclose(b.probes[key], pa, rtol=0.0,
                                   atol=1e-10 * np.abs(pa).max())


@pytest.mark.skipif(not rust_available(),
                    reason="Rust extension not built")
def test_adi_rust_float32():
    nx, ny, dx = 60, 50, 40e-9
    eps = np.ones((nx, ny))
    src = pf.ADISource(20, 25, _gauss(pf.C_0 / 1e-6, 8e-15))

    def run(backend, prec):
        sim = pf.ADISimulation2D(eps, dx, courant_factor=2.0, boundary="pec",
                                 sources=[src], backend=backend, precision=prec)
        return sim.run(300, snapshot_interval=299)

    a = run("numpy", "float64").snapshots["Ez"][-1]
    b = run("rust", "float32").snapshots["Ez"][-1]
    assert np.abs(a - b).max() < 1e-4 * np.abs(a).max()
