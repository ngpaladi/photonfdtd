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


# -------------------------------------------------------------------------- #
# Cherenkov radiation from a moving charged particle.
# -------------------------------------------------------------------------- #
def _run_cherenkov(beta, n=2.0, t_snap=22e-15):
    """Fire a charge at speed ``beta*c`` through an index-``n`` dielectric and
    return (Hz snapshot, x coords, y coords, particle-x at the snapshot, source).
    """
    Lx, Ly, dx = 8e-6, 6e-6, 50e-9
    grid = pf.Grid(size=(Lx, Ly), cell_size=dx, pml_layers=(12, 12, 0))
    medium = pf.Box(center=(0.0, 0.0), size=(3 * Lx, 3 * Ly),
                    medium=pf.Medium.from_index(n))
    particle = pf.ChargedParticle(
        charge=2e-9, velocity=(beta * pf.C_0, 0.0), start=(-3e-6, 0.0),
    )
    mon = pf.FieldMonitor(name="snap", components=("Hz",), times=[t_snap])
    sim = pf.Simulation(grid, structures=[medium], sources=[particle],
                        monitors=[mon], run_time=24e-15)
    res = sim.run()
    Hz = res.fields["snap"]["Hz"][0, :, :, 0]
    x_p = particle.position_at(res.monitor_times["snap"][0])[0]
    return Hz, grid.coords[0], grid.coords[1], x_p, particle


def test_cherenkov_radiates_only_above_threshold():
    """A charge radiates a far-off-axis cone only when it outruns the local
    phase velocity (n*beta > 1). Below threshold the field stays bound to the
    trajectory, so lateral energy is far smaller.
    """
    n = 2.0
    Hz_super, _, y, _, _ = _run_cherenkov(beta=0.9, n=n)   # n*beta = 1.8 > 1
    Hz_sub, _, _, _, _ = _run_cherenkov(beta=0.4, n=n)     # n*beta = 0.8 < 1
    assert np.isfinite(Hz_super).all() and np.isfinite(Hz_sub).all()

    lateral = np.abs(y) > 1.2e-6        # energy well away from the trajectory axis
    e_super = float(np.sum(Hz_super[:, lateral] ** 2))
    e_sub = float(np.sum(Hz_sub[:, lateral] ** 2))
    assert e_super > 5.0 * e_sub, (
        f"expected super-luminal lateral energy >> sub-luminal, "
        f"got super={e_super:.3e}, sub={e_sub:.3e} (ratio {e_super / e_sub:.1f})"
    )


def test_cherenkov_cone_angle_matches_theory():
    """The shock-cone half-angle measured from the field equals the analytic
    Mach angle mu = arcsin(1/(n*beta)).
    """
    n, beta = 2.0, 0.9
    Hz, x, y, x_p, particle = _run_cherenkov(beta=beta, n=n)
    mu_true = particle.mach_angle(n)

    # Trace the cone front: in the upper half, behind the particle, follow the
    # y of peak |Hz| in each x-column, then fit |y| = tan(mu) * (x_p - x).
    j_up = y > 0.2e-6
    y_up = y[j_up]
    absHz = np.abs(Hz[:, j_up])
    dist, ypk = [], []
    for ix, xx in enumerate(x):
        d = x_p - xx
        if d < 0.8e-6 or d > 4.5e-6:     # skip near field and the far turn-on transient
            continue
        col = absHz[ix]
        if col.max() <= 0.0:
            continue
        dist.append(d)
        ypk.append(y_up[int(np.argmax(col))])
    dist = np.asarray(dist)
    ypk = np.asarray(ypk)
    assert dist.size > 20, "not enough cone-front samples to fit an angle"

    slope = float(np.sum(dist * ypk) / np.sum(dist * dist))   # LSQ through origin
    mu_meas = math.atan(slope)
    rel_err = abs(mu_meas - mu_true) / mu_true
    assert rel_err < 0.15, (
        f"measured Mach angle {math.degrees(mu_meas):.1f} deg vs "
        f"theory {math.degrees(mu_true):.1f} deg (rel err {rel_err:.1%})"
    )
