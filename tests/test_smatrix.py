"""Mode-decomposition S-parameters and the memory-light port-plane DFT.

Covers:
1. The overlap inner product separates forward/backward modes and is
   normalization-agnostic (a mode fed into itself gives alpha=1).
2. A plane-restricted DFTMonitor records exactly the full-domain DFT sliced on
   that plane - on both the NumPy and JAX backends (the port-plane monitor is
   what keeps a PIC S-parameter run memory-light).
3. End-to-end: a mode launched down a straight lossless waveguide has |S21|~=1.
"""
import numpy as np
import pytest

import photonfdtd as pf


def _wg_mode(dy=40e-9, dz=40e-9, Ly=2.0e-6, Lz=1.4e-6):
    core = pf.Box(center=(0, 0), size=(0.45e-6, 0.22e-6),
                  medium=pf.Medium.from_index(3.48))
    ms = pf.ModeSolver(size=(Ly, Lz), cell_size=(dy, dz), structures=[core],
                       wavelength=1.55e-6, background_eps=1.44 ** 2, num_modes=2)
    return ms, ms.solve()


def test_overlap_separates_forward_backward():
    ms, mr = _wg_mode()
    dA = ms.dy * ms.dz
    from photonfdtd.smatrix import mode_port_fields
    m0 = mode_port_fields(mr, 0)

    # A mode fed in as the "measured" field: forward -> (1, 0), backward -> (0, 1).
    sim_fwd = {k: m0[k][None, ...] for k in m0}
    ap, am = pf.mode_amplitudes(sim_fwd, m0, dA)
    assert abs(ap[0]) == pytest.approx(1.0, abs=1e-6)
    assert abs(am[0]) < 1e-6

    sim_bwd = {"Ey": m0["Ey"][None], "Ez": m0["Ez"][None],
               "Hy": -m0["Hy"][None], "Hz": -m0["Hz"][None]}
    ap, am = pf.mode_amplitudes(sim_bwd, m0, dA)
    assert abs(ap[0]) < 1e-6
    assert abs(am[0]) == pytest.approx(1.0, abs=1e-6)

    # A superposition 0.7*forward + 0.3j*backward decodes to its coefficients.
    sim_mix = {"Ey": (0.7 * m0["Ey"] + 0.3j * m0["Ey"])[None],
               "Ez": (0.7 * m0["Ez"] + 0.3j * m0["Ez"])[None],
               "Hy": (0.7 * m0["Hy"] - 0.3j * m0["Hy"])[None],
               "Hz": (0.7 * m0["Hz"] - 0.3j * m0["Hz"])[None]}
    ap, am = pf.mode_amplitudes(sim_mix, m0, dA)
    assert ap[0] == pytest.approx(0.7, abs=1e-4)
    assert am[0] == pytest.approx(0.3j, abs=1e-4)

    # Different modes are ~orthogonal under this inner product.
    m1 = mode_port_fields(mr, 1)
    sim1 = {k: m1[k][None] for k in m1}
    ap, _ = pf.mode_amplitudes(sim1, m0, dA)
    assert abs(ap[0]) < 0.1


def _plane_parity(use_jax):
    lam = 1.0e-6
    f0 = pf.C_0 / lam
    dx = lam / 16
    grid = pf.Grid(size=(3e-6, 3e-6, 2e-6), cell_size=dx, pml_layers=(6, 6, 6))
    src = pf.PointDipole(position=(-0.5e-6, 0, 0), component="Ez",
                         waveform=pf.GaussianPulse(freq0=f0, fwhm=6e-15))
    x0 = 0.6e-6
    full = pf.DFTMonitor(name="full", components=("Ey", "Hz"), freqs=[f0])
    plane = pf.DFTMonitor(name="plane", components=("Ey", "Hz"), freqs=[f0],
                          plane_axis="x", plane_position=x0)
    sim = pf.Simulation(grid, sources=[src], monitors=[full, plane],
                        run_time=40e-15, use_jax=use_jax)
    res = sim.run()
    xi = int(np.argmin(np.abs(grid.coords[0] - x0)))
    for c in ("Ey", "Hz"):
        f = np.asarray(res.dft["full"][c])[:, xi:xi + 1]
        p = np.asarray(res.dft["plane"][c])
        assert p.shape == f.shape
        assert np.allclose(p, f, rtol=1e-6, atol=1e-12 * np.abs(f).max())


def test_plane_dft_matches_full_numpy():
    _plane_parity(use_jax=False)


def test_plane_dft_matches_full_jax():
    pytest.importorskip("jax")
    _plane_parity(use_jax=True)


@pytest.mark.slow
def test_waveguide_s21_energy_conserving():
    """A mode launched down a straight lossless waveguide transmits |S21|~=1
    between two downstream port planes, on the JAX backend."""
    pytest.importorskip("jax")
    lam = 1.55e-6
    f0 = pf.C_0 / lam
    dy = dz = 40e-9
    Ly, Lz = 2.0e-6, 1.4e-6
    ms, mr = _wg_mode(dy, dz, Ly, Lz)
    Ey = np.real(mr.Ey[0])
    core = pf.Box(center=(0, 0, 0), size=(10e-6, 0.45e-6, 0.22e-6),
                  medium=pf.Medium.from_index(3.48))
    grid = pf.Grid(size=(6e-6, Ly, Lz), cell_size=(dy, dy, dz),
                   pml_layers=(10, 10, 10))
    src = pf.ModeSource(center=(-2e-6, 0, 0), size=(0, Ly, Lz), component="Ey",
                        waveform=pf.GaussianPulse(freq0=f0, fwhm=10e-15),
                        profile=Ey, profile_coords=(ms.y, ms.z))
    p1 = pf.DFTMonitor(name="p1", components=("Ey", "Ez", "Hy", "Hz"),
                       freqs=[f0], plane_axis="x", plane_position=0.0)
    p2 = pf.DFTMonitor(name="p2", components=("Ey", "Ez", "Hy", "Hz"),
                       freqs=[f0], plane_axis="x", plane_position=2.0e-6)
    sim = pf.Simulation(grid, structures=[core], sources=[src],
                        monitors=[p1, p2], run_time=200e-15, use_jax=True)
    res = sim.run()
    dA = dy * dz
    a1p, _ = pf.s_parameters(res, "p1", mr, 0, dA)
    a2p, _ = pf.s_parameters(res, "p2", mr, 0, dA)
    assert abs(a2p[0] / a1p[0]) == pytest.approx(1.0, abs=0.05)
