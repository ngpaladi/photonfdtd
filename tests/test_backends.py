"""Backend-parity tests.

The optional Numba (CPU JIT) and CuPy (GPU) backends must reproduce the
default NumPy backend bit-for-bit (up to floating-point round-off). These run
only when the backend is installed and usable, and skip cleanly otherwise -
so they exercise the real GPU/JIT code paths on a developer machine that has
them, while remaining no-ops in a minimal CI environment.

They specifically cover the moving :class:`~photonfdtd.ChargedParticle`
current injection, which is dispatched separately in the NumPy/CuPy and Numba
branches of the time loop.
"""
import numpy as np
import pytest
import photonfdtd as pf


def _particle_3d(use_numba=False, use_gpu=False):
    """Small 3D charge-in-dielectric run; returns the Ex snapshot."""
    n = 2.0
    L, dx = 2.5e-6, 60e-9
    grid = pf.Grid(size=(L, L, L), cell_size=dx, pml_layers=(8, 8, 8))
    med = pf.Box(center=(0.0, 0.0, 0.0), size=(3 * L, 3 * L, 3 * L),
                 medium=pf.Medium.from_index(n))
    p = pf.ChargedParticle(charge=2e-9, velocity=(0.9 * pf.C_0, 0.0, 0.0),
                           start=(-0.9e-6, 0.0, 0.0))
    mon = pf.FieldMonitor(name="s", components=("Ex",), times=[9e-15])
    sim = pf.Simulation(grid, structures=[med], sources=[p], monitors=[mon],
                        run_time=10e-15, use_numba=use_numba, use_gpu=use_gpu)
    return sim.run().fields["s"]["Ex"][0]


def _cherenkov_2d(use_gpu=False):
    """2D Cherenkov run (exercises the vectorised path + monitor read-back)."""
    n = 2.0
    Lx, Ly, dx = 8e-6, 6e-6, 50e-9
    grid = pf.Grid(size=(Lx, Ly), cell_size=dx, pml_layers=(12, 12, 0))
    med = pf.Box(center=(0.0, 0.0), size=(3 * Lx, 3 * Ly),
                 medium=pf.Medium.from_index(n))
    p = pf.ChargedParticle(charge=2e-9, velocity=(0.9 * pf.C_0, 0.0),
                           start=(-3e-6, 0.0))
    mon = pf.FieldMonitor(name="s", components=("Hz",), times=[22e-15])
    sim = pf.Simulation(grid, structures=[med], sources=[p], monitors=[mon],
                        run_time=24e-15, use_gpu=use_gpu)
    return sim.run().fields["s"]["Hz"][0, :, :, 0]


def test_numba_particle_matches_numpy():
    """The Numba (CPU JIT) backend reproduces the NumPy particle injection."""
    pytest.importorskip("numba")
    ref = _particle_3d(use_numba=False)
    got = _particle_3d(use_numba=True)
    den = float(np.abs(ref).max())
    assert den > 0.0 and np.isfinite(got).all()
    rel = float(np.abs(ref - got).max()) / den
    assert rel < 1e-6, f"numba diverged from numpy: rel.diff={rel:.2e}"


def test_cupy_particle_matches_numpy():
    """The CuPy (GPU) backend reproduces the NumPy particle injection."""
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device available")
    except Exception as exc:  # pragma: no cover - driver/runtime mismatch
        pytest.skip(f"CuPy present but no usable CUDA device: {exc}")

    ref = _cherenkov_2d(use_gpu=False)
    got = _cherenkov_2d(use_gpu=True)
    den = float(np.abs(ref).max())
    assert den > 0.0 and np.isfinite(got).all()
    rel = float(np.abs(ref - got).max()) / den
    assert rel < 1e-6, f"cupy diverged from numpy: rel.diff={rel:.2e}"
