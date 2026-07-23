"""Rust-backend parity tests.

The compiled Rust stepping core (``backend="rust"``, built from
``rust/src/lib.rs``) must reproduce the default NumPy backend to
double-precision round-off. Like the Numba/CuPy/JAX parity tests, these run
only when the extension is built and skip cleanly otherwise.
"""
import numpy as np
import pytest

import photonfdtd as pf
from photonfdtd.rustbackend import rust_available

pytestmark = pytest.mark.skipif(
    not rust_available(), reason="Rust extension (_photonfdtd_rs) not built"
)


def _run_2d(backend):
    freq0 = pf.C_0 / 1.0e-6
    grid = pf.Grid(size=(4e-6, 3e-6), cell_size=50e-9, pml_layers=(10, 10, 0))
    box = pf.Box(center=(0.6e-6, 0.0), size=(1.0e-6, 1.0e-6),
                 medium=pf.Medium.from_index(2.0))
    src = pf.PointDipole(position=(-0.8e-6, 0.0, 0.0), component="Ez",
                         waveform=pf.GaussianPulse(freq0=freq0, fwhm=6e-15))
    mon = pf.FieldMonitor(name="s", components=("Ez", "Hx", "Hy"), interval=25)
    dft = pf.DFTMonitor(name="d", components=("Ez",), freqs=[freq0], interval=1)
    sim = pf.Simulation(grid, structures=[box], sources=[src],
                        monitors=[mon, dft], run_time=40e-15, backend=backend)
    return sim.run()


def test_rust_matches_numpy_2d():
    r_np = _run_2d("numpy")
    r_rs = _run_2d("rust")
    for c in ("Ez", "Hx", "Hy"):
        a, b = r_np.fields["s"][c], r_rs.fields["s"][c]
        assert a.shape == b.shape
        np.testing.assert_allclose(b, a, rtol=0.0,
                                   atol=1e-12 * np.abs(a).max())
    a, b = r_np.dft["d"]["Ez"], r_rs.dft["d"]["Ez"]
    np.testing.assert_allclose(b, a, rtol=0.0, atol=1e-12 * np.abs(a).max())
    np.testing.assert_array_equal(r_np.monitor_times["s"],
                                  r_rs.monitor_times["s"])


def test_rust_matches_numpy_3d():
    def run(backend):
        freq0 = pf.C_0 / 1.0e-6
        grid = pf.Grid(size=(2.5e-6, 2e-6, 2e-6), cell_size=80e-9,
                       pml_layers=(8, 8, 8))
        box = pf.Box(center=(0.4e-6, 0, 0), size=(0.8e-6, 0.8e-6, 0.8e-6),
                     medium=pf.Medium.from_index(1.8))
        src = pf.PointDipole(position=(-0.5e-6, 0, 0), component="Ey",
                             waveform=pf.GaussianPulse(freq0=freq0, fwhm=5e-15))
        mon = pf.FieldMonitor(name="s", components=("Ey", "Hz"), times=[14e-15])
        sim = pf.Simulation(grid, structures=[box], sources=[src],
                            monitors=[mon], run_time=15e-15, backend=backend)
        return sim.run().fields["s"]

    a, b = run("numpy"), run("rust")
    for c in a:
        np.testing.assert_allclose(b[c], a[c], rtol=0.0,
                                   atol=1e-12 * np.abs(a[c]).max())


def test_rust_matches_numpy_1d():
    def run(backend):
        freq0 = pf.C_0 / 1.0e-6
        grid = pf.Grid(size=(8e-6,), cell_size=20e-9, pml_layers=(12,))
        src = pf.PointDipole(position=(0.0,), component="Ez",
                             waveform=pf.GaussianPulse(freq0=freq0, fwhm=5e-15))
        mon = pf.FieldMonitor(name="s", components=("Ez",), times=[20e-15])
        sim = pf.Simulation(grid, sources=[src], monitors=[mon],
                            run_time=22e-15, backend=backend)
        return sim.run().fields["s"]["Ez"]

    a, b = run("numpy"), run("rust")
    np.testing.assert_allclose(b, a, rtol=0.0, atol=1e-12 * np.abs(a).max())


def test_rust_rejects_unsupported():
    grid = pf.Grid(size=(3e-6, 3e-6), cell_size=60e-9, pml_layers=(8, 8, 0))
    src = pf.PointDipole(position=(0.0, 0.0), component="Ez",
                         waveform=pf.GaussianPulse(freq0=pf.C_0 / 1e-6, fwhm=6e-15))
    flux = pf.FluxMonitor(name="f", plane_axis="x", plane_position=1e-6)
    sim = pf.Simulation(grid, sources=[src], monitors=[flux],
                        run_time=5e-15, backend="rust")
    with pytest.raises(NotImplementedError):
        sim.run()
    sim32 = pf.Simulation(grid, sources=[src], run_time=5e-15,
                          backend="rust", precision="float32")
    with pytest.raises(NotImplementedError):
        sim32.run()
