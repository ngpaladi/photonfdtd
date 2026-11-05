"""Photon-count estimates: DFT-plane spectra against analytic Cherenkov yields."""
import math

import numpy as np
import pytest

import photonfdtd as pf
from photonfdtd import photons as ph


def rel(a, b):
    """Relative difference; no absolute floor (these energies are ~1e-26 J,
    where pytest.approx's default abs=1e-12 would accept anything)."""
    return abs(a - b) / abs(b)


def test_frank_tamm_textbook_yield():
    """~490 sin^2(theta_c) photons per cm in 400-700 nm for a unit charge."""
    n, beta = 1.33, 0.99
    sin2 = 1.0 - 1.0 / (n * beta) ** 2
    N = pf.frank_tamm_photons(n, beta, 0.01, (400e-9, 700e-9))
    exact = 2 * math.pi * ph.ALPHA * 0.01 * sin2 * (1 / 400e-9 - 1 / 700e-9)
    assert rel(N, exact) < 1e-14
    assert rel(N / sin2, 491.2566) < 1e-6
    # dispersive callable with constant index agrees; charge scales as z^2
    Nd = pf.frank_tamm_photons(lambda f: np.full_like(f, n), beta, 0.01,
                               (400e-9, 700e-9))
    assert rel(Nd, N) < 1e-15
    N2 = pf.frank_tamm_photons(n, beta, 0.01, (400e-9, 700e-9), charge=2 * pf.Q_E)
    assert rel(N2, 4 * N) < 1e-15


def test_frank_tamm_below_threshold_is_dark():
    f = pf.C_0 / np.linspace(400e-9, 700e-9, 5)
    s = pf.frank_tamm_spectrum(f, 1.2, 0.8, length=1.0)
    assert np.all(s.energy_density == 0.0)
    assert s.total_photons() == 0.0


def test_spectrum_bookkeeping():
    f = np.linspace(1e14, 3e14, 201)
    a = 2e-30 / 1e14
    # dU/df = a f: dN/df = a / h is flat, so the trapezoid rule is exact.
    s = pf.PhotonSpectrum(f, a * f)
    assert rel(s.total_photons(), a / pf.PLANCK_H * 2e14) < 1e-14
    assert rel(s.total_energy(1.5e14, 2.5e14), a * (2.5e14 ** 2 - 1.5e14 ** 2) / 2) < 1e-14
    lam = (pf.C_0 / 2.5e14, pf.C_0 / 1.5e14)
    assert rel(s.total_photons_wavelength(*lam), s.total_photons(1.5e14, 2.5e14)) < 1e-14
    assert rel(s.rescale_charge(2e-9, pf.Q_E).total_photons(),
               s.total_photons() * (pf.Q_E / 2e-9) ** 2) < 1e-14
    assert rel((s + s).total_photons(), 2 * s.total_photons()) < 1e-15
    with pytest.raises(ValueError):
        s.total_photons(0.5e14, 2e14)
    assert rel(pf.photon_count(pf.PLANCK_H * 2e14 * 7, 2e14), 7.0) < 1e-15


def _cherenkov_2d(components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")):
    n, beta, q, dx = 2.0, 0.9, 1e-9, 40e-9
    Lx, Ly, Y = 20e-6, 6e-6, 1.5e-6
    freqs = pf.C_0 / np.linspace(1.5e-6, 4e-6, 6)
    grid = pf.Grid(size=(Lx, Ly), cell_size=dx, pml_layers=(12, 12, 0))
    med = pf.Box(center=(0, 0), size=(3 * Lx, 3 * Ly),
                 medium=pf.Medium.from_index(n))
    p = pf.ChargedParticle(charge=q, velocity=(beta * pf.C_0, 0.0),
                           start=(-Lx / 2, 0.0))
    mons = [pf.DFTMonitor(name=nm, components=components, freqs=freqs,
                          plane_axis="y", plane_position=s * Y)
            for nm, s in (("up", 1), ("dn", -1))]
    sim = pf.Simulation(grid, structures=[med], sources=[p], monitors=mons,
                        run_time=Lx / (beta * pf.C_0) + 2 * Ly * n / pf.C_0)
    return sim, sim.run(), dict(n=n, beta=beta, q=q, dx=dx, freqs=freqs, p=p)


def test_photon_spectrum_matches_2d_cherenkov():
    """Photons through a plane beside a line charge match the analytic 2-D
    Cherenkov spectrum (cloud form factor included) on both sides. The
    remaining difference (0.82% worst, 0.43% in total at 40 nm cells) is the
    grid's discretization error; the two sides agree to ~1e-5."""
    sim, r, c = _cherenkov_2d()
    W = 3e-6
    bounds = [(-W, W), None, None]
    xs = np.asarray(sim.grid.coords[0])
    L = ((xs >= -W) & (xs <= W)).sum() * c["dx"]
    ref = pf.frank_tamm_spectrum(c["freqs"], c["n"], c["beta"], length=L,
                                 charge=c["q"], ndim=2, sides=1,
                                 cloud_sigma=c["p"].cloud_cells * c["dx"])
    sides = [pf.photon_spectrum(r, sim, name, direction=sgn, bounds=bounds)
             for name, sgn in (("up", +1), ("dn", -1))]
    for s in sides:
        assert np.max(np.abs(s.energy_density / ref.energy_density - 1)) < 9e-3
        assert rel(s.total_photons(), ref.total_photons()) < 5e-3
    assert np.max(np.abs(sides[0].energy_density / sides[1].energy_density - 1)) < 1e-4
    # the wrong direction gives exactly the negated count
    s = pf.photon_spectrum(r, sim, "up", direction=-1, bounds=bounds)
    assert s.total_photons() == -sides[0].total_photons()


def test_photon_spectrum_validates_monitor():
    grid = pf.Grid(size=(1e-6, 1e-6), cell_size=100e-9, pml_layers=(3, 3, 0))
    src = pf.PointDipole(position=(0, 0), component="Ez",
                         waveform=pf.GaussianPulse(freq0=3e14, fwhm=6e-15))
    mons = [pf.DFTMonitor(name="partial", components=("Ez", "Hx"), freqs=[3e14],
                          plane_axis="y", plane_position=0.2e-6),
            pf.DFTMonitor(name="vol", components=("Ez",), freqs=[3e14])]
    sim = pf.Simulation(grid, sources=[src], monitors=mons, run_time=5e-15)
    r = sim.run()
    with pytest.raises(ValueError, match="missing"):
        pf.photon_spectrum(r, sim, "partial")
    with pytest.raises(ValueError, match="plane_axis"):
        pf.photon_spectrum(r, sim, "vol")
    with pytest.raises(ValueError, match="not a DFTMonitor"):
        pf.photon_spectrum(r, sim, "nope")


def test_photon_spectrum_symmetric_and_matches_flux_monitor():
    """A symmetric dipole: the plane spectra on opposite sides agree (to the
    grid's O(dx^2) mirror asymmetry, 4.3e-5), and the band integral of the
    spectrum equals the FluxMonitor energy to round-off."""
    f0, dx = pf.C_0 / 1e-6, 50e-9
    comps = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    grid = pf.Grid(size=(3e-6, 3e-6), cell_size=dx, pml_layers=(8, 8, 0))
    src = pf.PointDipole(position=(0, 0), component="Ez",
                         waveform=pf.GaussianPulse(freq0=f0, fwhm=5e-15))
    freqs = np.linspace(0.0, 4.0, 400) * f0
    mons = [pf.DFTMonitor(name=f"d{s:+d}", components=comps, freqs=freqs,
                          plane_axis="x", plane_position=s * 0.5e-6)
            for s in (1, -1)]
    mons.append(pf.FluxMonitor(name="flux", plane_axis="x", plane_position=0.5e-6))
    sim = pf.Simulation(grid, sources=[src], monitors=mons, run_time=60e-15)
    r = sim.run()
    up = pf.photon_spectrum(r, sim, "d+1", +1)
    dn = pf.photon_spectrum(r, sim, "d-1", -1)
    peak = up.energy_density.max()
    assert np.abs(up.energy_density - dn.energy_density).max() < 6e-5 * peak
    assert rel(up.total_energy(), r.flux["flux"]) < 1e-12
