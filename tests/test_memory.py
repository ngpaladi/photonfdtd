"""Memory-efficiency features for large volumes:

- DFTMonitor: running frequency-domain accumulation (time-axis compression)
- a caller-supplied eps_r grid survives run()
"""
import numpy as np
import pytest

import photonfdtd as pf


def _dipole_sim(monitors, run_time=60e-15, precision="float64"):
    freq0 = pf.C_0 / 1.0e-6
    grid = pf.Grid(size=(3e-6, 3e-6), cell_size=50e-9, pml_layers=(10, 10, 0))
    src = pf.PointDipole(position=(0.0, 0.0), component="Ez",
                         waveform=pf.GaussianPulse(freq0=freq0, fwhm=6e-15))
    return pf.Simulation(grid, sources=[src], monitors=monitors,
                         run_time=run_time, precision=precision)


# --------------------------------------------------------------------------- #
# DFTMonitor
# --------------------------------------------------------------------------- #
def test_dft_monitor_matches_manual_dft():
    """The running DFT equals a post-hoc DFT of the recorded time series to
    machine precision, using each component's own Yee half-step time."""
    freq0 = pf.C_0 / 1.0e-6
    freqs = [0.8 * freq0, freq0, 1.2 * freq0]
    td = pf.FieldMonitor(name="td", components=("Ez", "Hx"), interval=1)
    fd = pf.DFTMonitor(name="fd", components=("Ez", "Hx"), freqs=freqs, interval=1)
    sim = _dipole_sim([td, fd])
    res = sim.run()
    dt = sim.dt

    assert list(res.dft_freqs["fd"]) == pytest.approx(freqs)
    steps = np.round(res.monitor_times["td"] / dt).astype(int)
    for comp in ("Ez", "Hx"):
        series = res.fields["td"][comp]
        got = res.dft["fd"][comp]
        assert got.shape == (len(freqs),) + series.shape[1:]
        assert np.iscomplexobj(got)
        tvec = (steps + (1.0 if comp[0] == "E" else 0.5)) * dt
        manual = np.zeros_like(got)
        for fi, f in enumerate(freqs):
            phase = np.exp(1j * 2 * np.pi * f * tvec)
            manual[fi] = (series * phase[:, None, None, None]).sum(axis=0) * dt
        rel = np.abs(manual - got).max() / np.abs(got).max()
        assert rel < 1e-10, f"{comp}: DFT mismatch rel={rel:.2e}"


def test_dft_monitor_far_smaller_than_time_series():
    """DFT storage scales with n_freq, not n_steps: a big time-axis reduction."""
    freqs = [pf.C_0 / 1.0e-6]
    td = pf.FieldMonitor(name="td", components=("Ez",), interval=1)
    fd = pf.DFTMonitor(name="fd", components=("Ez",), freqs=freqs, interval=1)
    res = _dipole_sim([td, fd]).run()
    td_bytes = res.fields["td"]["Ez"].nbytes
    fd_bytes = res.dft["fd"]["Ez"].nbytes
    n_rec = res.fields["td"]["Ez"].shape[0]
    # complex128 (16 B) vs float64 (8 B) per cell => ratio ~ n_rec / 2.
    assert td_bytes / fd_bytes > n_rec / 4 > 10


def test_dft_monitor_requires_freqs():
    with pytest.raises(ValueError):
        pf.DFTMonitor(name="x", components=("Ez",), freqs=[])


# --------------------------------------------------------------------------- #
# Custom eps_r
# --------------------------------------------------------------------------- #
def test_custom_eps_r_preserved_across_run():
    grid = pf.Grid(size=(2e-6, 2e-6), cell_size=100e-9, pml_layers=(4, 4, 0))
    sim = pf.Simulation(grid, run_time=5e-15)
    custom = np.full(grid.shape, 2.25, dtype=np.float64)
    sim.eps_r = custom
    sim.run()
    assert sim.eps_r is custom                       # custom grid is kept
