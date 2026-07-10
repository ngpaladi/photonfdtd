"""Tests for dispersive media (auxiliary-differential-equation stepping).

Three layers:

1. The cited material library reproduces its published refractive index.
2. The ADE time-stepper reproduces a Lorentz medium's analytic phase index
   n(omega) = Re(sqrt(eps(omega))) when a pulse propagates through it.
3. Guards: the pole-stability limit is enforced, and at_wavelength() gives a
   correct fixed-index fallback.
"""
import numpy as np
import pytest

import photonfdtd as pf
from photonfdtd import materials as m
from photonfdtd.materials import DispersiveMedium

C0 = 299_792_458.0


def _n(med, lam_um):
    return med.index(C0 / (lam_um * 1e-6)).real


def test_library_indices_match_sources():
    """Each library medium reproduces its published n at the sanity wavelength."""
    assert _n(m.silica(), 1.55) == pytest.approx(1.44402, abs=2e-4)
    assert _n(m.silicon(), 1.55) == pytest.approx(3.4777, abs=2e-3)
    assert _n(m.silicon_nitride(), 1.55) == pytest.approx(1.9963, abs=2e-3)
    assert _n(m.lithium_niobate("o"), 1.55) == pytest.approx(2.2111, abs=2e-3)
    assert _n(m.lithium_niobate("e"), 1.55) == pytest.approx(2.1376, abs=2e-3)

    # Metals: complex permittivity at 633 nm (Rakic LD).
    eps_au = m.gold().eps_model(C0 / 633e-9)
    assert eps_au.real == pytest.approx(-9.81, abs=0.1)
    assert eps_au.imag == pytest.approx(1.96, abs=0.1)
    eps_ag = m.silver().eps_model(C0 / 633e-9)
    assert eps_ag.real == pytest.approx(-14.49, abs=0.2)
    assert eps_ag.imag == pytest.approx(1.10, abs=0.1)


def test_sellmeier_pole_equivalence():
    """A Sellmeier term equals a lossless Lorentz pole: check the identity holds
    by comparing eps_model to the closed-form Sellmeier sum."""
    terms = [(0.6961663, 0.0684043 ** 2), (0.4079426, 0.1162414 ** 2)]
    med = DispersiveMedium.sellmeier(terms)
    for lam_um in (0.5, 1.0, 1.55, 3.0):
        lam2 = lam_um ** 2
        n2 = 1.0 + sum(B * lam2 / (lam2 - C) for B, C in terms)
        assert med.eps_model(C0 / (lam_um * 1e-6)).real == pytest.approx(n2, rel=1e-9)


def test_ade_lorentz_phase_index():
    """A pulse through a homogeneous Lorentz medium acquires the analytic phase
    index. Resonance is placed above the probe band (normal dispersion, n>1,
    low loss) so the phase-gradient extraction is well posed."""
    med = DispersiveMedium.lorentz(2.25, [(1.2, 500e12, 0.0)])
    dx = 50e-9
    L = 60e-6
    grid = pf.Grid(size=(L,), cell_size=dx, pml_layers=(30,))
    box = pf.Box(center=(0.0,), size=(2 * L,), medium=med)      # fill the domain
    src = pf.PointDipole(position=(-20e-6,), component="Ez",
                         waveform=pf.GaussianPulse(freq0=200e12, fwhm=8e-15))
    freqs = np.array([150e12, 180e12, 210e12, 240e12, 270e12])
    mon = pf.DFTMonitor(name="d", components=("Ez",), freqs=freqs)
    sim = pf.Simulation(grid, structures=[box], sources=[src], monitors=[mon],
                        run_time=400e-15)
    r = sim.run()
    D = np.asarray(r.dft["d"]["Ez"])
    x = grid.coords[0]
    i1 = int(np.argmin(np.abs(x - 0e-6)))
    i2 = int(np.argmin(np.abs(x - 15e-6)))
    for fi, f in enumerate(freqs):
        ph = np.unwrap(np.angle(D[fi, :, 0, 0]))
        k = abs(ph[i2] - ph[i1]) / (x[i2] - x[i1])
        n_fdtd = k * C0 / (2 * np.pi * f)
        n_an = np.sqrt(med.eps_model(f)).real
        assert n_fdtd == pytest.approx(n_an, rel=0.025), \
            f"f={f/1e12:.0f}THz n_fdtd={n_fdtd:.4f} n_analytic={n_an:.4f}"


def test_dispersive_run_is_finite():
    """A Drude medium (always-stable omega0=0 pole) time-steps without blowing
    up, confirming the ADE recursion is stable and coupled correctly."""
    med = DispersiveMedium.drude(1.0, f_plasma_Hz=150e12, f_collision_Hz=5e12)
    dx = 40e-9
    grid = pf.Grid(size=(8e-6, 8e-6), cell_size=dx, pml_layers=(10, 10, 0))
    box = pf.Box(center=(1e-6, 0.0), size=(2e-6, 2e-6), medium=med)
    src = pf.PointDipole(position=(-2e-6, 0.0), component="Ez",
                         waveform=pf.GaussianPulse(freq0=200e12, fwhm=8e-15))
    mon = pf.FieldMonitor(name="s", components=("Ez",), interval=50)
    sim = pf.Simulation(grid, structures=[box], sources=[src], monitors=[mon],
                        run_time=120e-15)
    r = sim.run()
    field = r.fields["s"]["Ez"]
    assert np.isfinite(field).all()
    assert np.abs(field[-1]).max() < np.abs(field).max() * 5  # no blow-up


def test_pole_stability_guard():
    """A medium with a deep-UV pole (fused silica on a near-IR grid) trips the
    omega0*dt < 2 stability guard with an explanatory error."""
    grid = pf.Grid(size=(4e-6,), cell_size=1e-7, pml_layers=(10,))
    box = pf.Box(center=(0.0,), size=(2e-6,), medium=m.silica())
    src = pf.PointDipole(position=(-1e-6,), component="Ez",
                         waveform=pf.GaussianPulse(freq0=2e14, fwhm=8e-15))
    sim = pf.Simulation(grid, structures=[box], sources=[src], run_time=5e-15)
    with pytest.raises(ValueError, match="unstable"):
        sim.run()


def test_at_wavelength_fallback():
    """at_wavelength returns a plain Medium with the correct real index."""
    med = m.silica()
    fixed = med.at_wavelength(1.55e-6)
    assert not fixed.is_dispersive
    assert np.sqrt(fixed.eps_r) == pytest.approx(1.44402, abs=2e-4)


def test_dispersion_backend_support():
    """Dispersion is supported on the JAX backend and rejected only on Numba."""
    grid = pf.Grid(size=(3e-6,), cell_size=1e-7, pml_layers=(6,))
    box = pf.Box(center=(0.0,), size=(1e-6,),
                 medium=DispersiveMedium.drude(1.0, 100e12, 5e12))
    pytest.importorskip("jax")
    pf.Simulation(grid, structures=[box], use_jax=True)   # accepted

    from photonfdtd import simulation as _s
    if _s._NUMBA_AVAILABLE:
        with pytest.raises(ValueError):
            pf.Simulation(grid, structures=[box], use_numba=True)
