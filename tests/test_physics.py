"""Quantitative physics tests.

These verify that the FDTD core gives the right answer for problems whose
solutions are known analytically. Tolerances are loose; the goal is to catch
gross bugs (sign errors, wrong dt, etc.), not to validate accuracy to many
digits.
"""
import math
import numpy as np
import pytest
import photonfdtd as pf


# -------------------------------------------------------------------------- #
def test_1d_plane_wave_speed():
    """A pulse launched in 1D vacuum should propagate at c.

    We measure the speed by cross-correlating two snapshots separated by a
    known time interval and reading off the spatial shift of the peak
    correlation. This is robust to the carrier oscillation that confuses
    naive peak tracking.
    """
    lam0 = 1.0e-6
    freq0 = pf.C_0 / lam0
    dx = lam0 / 30
    L = 6e-6
    grid = pf.Grid(size=(L,), cell_size=dx, pml_layers=(12,))
    src = pf.PointDipole(
        position=(-2.0e-6,), component="Ey",
        waveform=pf.GaussianPulse(freq0=freq0, fwhm=4e-15),
    )
    mon = pf.FieldMonitor(name="snap", components=("Ey",), interval=5)
    sim = pf.Simulation(grid, sources=[src], monitors=[mon], run_time=22e-15)
    res = sim.run()

    Ey = res.fields["snap"]["Ey"][..., 0, 0]
    times = res.monitor_times["snap"]
    x = grid.coords[0]

    # pick two frames after the pulse has launched, both in the interior
    i1 = int(len(times) * 0.55)
    i2 = int(len(times) * 0.80)
    dt_frames = times[i2] - times[i1]
    f1 = Ey[i1]; f2 = Ey[i2]

    # Cross-correlate; the peak lag (in cells) times dx is the spatial shift.
    corr = np.correlate(f2, f1, mode="full")
    lag = np.argmax(corr) - (len(f1) - 1)
    shift = lag * dx
    speed = shift / dt_frames
    err = abs(speed - pf.C_0) / pf.C_0
    assert err < 0.03, f"speed = {speed:.3e}, c = {pf.C_0:.3e}, err = {err:.3%}"


def test_cpml_absorbs_outgoing_wave_1d():
    """After the pulse should have exited via the PML, energy in the interior
    must drop by orders of magnitude."""
    lam0 = 1.0e-6
    freq0 = pf.C_0 / lam0
    dx = lam0 / 25
    L = 4e-6
    grid = pf.Grid(size=(L,), cell_size=dx, pml_layers=(12,))
    src = pf.PointDipole(
        position=(0.0,), component="Ey",
        waveform=pf.GaussianPulse(freq0=freq0, fwhm=5e-15),
    )
    mon = pf.FieldMonitor(name="snap", components=("Ey",), interval=10)
    sim = pf.Simulation(grid, sources=[src], monitors=[mon], run_time=120e-15)
    res = sim.run()

    Ey_arr = res.fields["snap"]["Ey"][..., 0, 0]
    # Mid-domain energy at the peak vs at the end
    nx = grid.shape[0]
    interior = slice(nx // 4, 3 * nx // 4)
    energy = np.sum(Ey_arr[:, interior] ** 2, axis=1)
    peak = energy.max()
    final = energy[-1]
    # Expect at least 20 dB attenuation
    assert peak > 0
    assert final / peak < 1e-2, f"PML did not absorb: peak={peak:.3e}, final={final:.3e}"


def test_mode_solver_slab_against_eigenvalue_eq():
    """Compare scalar mode-solver n_eff to the analytic 1D-slab dispersion,
    with the analytic correction for the finite y-extent of the 2D domain.

    The TE0 mode of a 1D symmetric slab waveguide (core index n1, cladding n2,
    half-width a) satisfies  a*k1 * tan(a*k1) = a*k2  with
        k1 = k0 sqrt(n1^2 - n_eff^2), k2 = k0 sqrt(n_eff^2 - n_clad^2).
    Our 2D solver discretises on a finite (Ly, Lz) domain with Dirichlet
    boundaries, so the actual eigenvalue is

        beta^2  =  beta_1D^2  -  (pi / Ly)^2.

    We compare against that prediction.
    """
    from scipy.optimize import brentq

    lam0 = 1.55e-6
    k0 = 2 * math.pi / lam0
    n_core, n_clad = 1.46, 1.45
    a = 1.0e-6
    Ly, Lz = 12e-6, 6e-6
    cell = lam0 / 60

    def f(n_eff):
        k1 = k0 * math.sqrt(n_core ** 2 - n_eff ** 2)
        k2 = k0 * math.sqrt(n_eff ** 2 - n_clad ** 2)
        return a * k1 * math.tan(a * k1) - a * k2

    eps = 1e-9
    n_eff_1d = brentq(f, n_clad + eps, n_core - eps)
    beta_1d = k0 * n_eff_1d
    beta_2d_expected = math.sqrt(beta_1d ** 2 - (math.pi / Ly) ** 2)
    n_eff_expected = beta_2d_expected / k0

    core = pf.Box(
        center=(0.0, 0.0), size=(Ly * 2, 2 * a),     # span the full y range
        medium=pf.Medium.from_index(n_core),
    )
    ms = pf.ModeSolver(
        size=(Ly, Lz),
        cell_size=cell,
        structures=[core],
        wavelength=lam0,
        background_eps=n_clad ** 2,
        num_modes=1,
    )
    res = ms.solve()
    n_eff_num = float(res.n_eff[0])
    # 2nd-order central FD biases n_eff low by ~(k0 dx)^2 / 12. At lambda/60
    # that's a few parts in 10^3; tolerance of 5e-3 captures the gross-bug
    # detection we want without forcing an expensive higher-order solve.
    err = abs(n_eff_num - n_eff_expected)
    assert err < 5e-3, (
        f"expected (bounded) n_eff = {n_eff_expected:.6f}, "
        f"numerical = {n_eff_num:.6f},  error = {err:.6f}"
    )
