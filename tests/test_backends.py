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
import warnings

import numpy as np
import pytest
import photonfdtd as pf


def _dipole_2d(size, pml, component, use_numba=False):
    """Small 2D dipole run on the given grid; returns the ``component`` snapshot."""
    freq0 = pf.C_0 / 1.0e-6
    grid = pf.Grid(size=size, cell_size=60e-9, pml_layers=pml)
    src = pf.PointDipole(position=(0.0, 0.0, 0.0), component=component,
                         waveform=pf.GaussianPulse(freq0=freq0, fwhm=6e-15))
    mon = pf.FieldMonitor(name="s", components=(component,), times=[12e-15])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # silence the sub-3D numba notice
        sim = pf.Simulation(grid, sources=[src], monitors=[mon],
                            run_time=14e-15, use_numba=use_numba)
    return sim.run().fields["s"][component][0]


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


# Cover both 2D planes and both polarisations, including the case where the
# parallel (x) axis is the one that collapses (nx=1, the yz plane), which drives
# the prange loops to a single iteration.
@pytest.mark.parametrize("label,size,pml,component", [
    ("xy-TM", (3e-6, 3e-6), (8, 8, 0), "Ez"),
    ("xy-TE", (3e-6, 3e-6), (8, 8, 0), "Ex"),
    ("yz-TM", (None, 3e-6, 3e-6), (0, 8, 8), "Ex"),
    ("yz-TE", (None, 3e-6, 3e-6), (0, 8, 8), "Ey"),
    ("xz-TM", (3e-6, None, 3e-6), (8, 0, 8), "Ey"),
    ("xz-TE", (3e-6, None, 3e-6), (8, 0, 8), "Ex"),
])
def test_numba_matches_numpy_2d(label, size, pml, component):
    """The Numba kernel must reproduce NumPy in 2D. Guards against the
    dimension-collapse bug where the 3D loop bounds went empty and silently
    skipped components differentiated along a size-1 axis.
    """
    pytest.importorskip("numba")
    ref = _dipole_2d(size, pml, component, use_numba=False)
    got = _dipole_2d(size, pml, component, use_numba=True)
    den = float(np.abs(ref).max())
    assert den > 0.0 and np.isfinite(got).all()
    rel = float(np.abs(ref - got).max()) / den
    assert rel < 1e-6, f"numba 2D {label} diverged from numpy: rel.diff={rel:.2e}"


def test_float32_matches_float64():
    """Single precision must track double precision closely (not bit-for-bit)
    and actually store float32, halving field/monitor memory.
    """
    freq0 = pf.C_0 / 1.0e-6
    grid = pf.Grid(size=(3e-6, 3e-6), cell_size=60e-9, pml_layers=(10, 10, 0))
    src = pf.PointDipole(position=(0.0, 0.0), component="Ez",
                         waveform=pf.GaussianPulse(freq0=freq0, fwhm=6e-15))

    def field(precision):
        mon = pf.FieldMonitor(name="s", components=("Ez",), times=[16e-15])
        sim = pf.Simulation(grid, sources=[src], monitors=[mon],
                            run_time=18e-15, precision=precision)
        return sim.run().fields["s"]["Ez"][0, :, :, 0]

    f64 = field("float64")
    f32 = field("float32")
    assert f32.dtype == np.float32 and f64.dtype == np.float64
    assert np.isfinite(f32).all()
    rel = float(np.abs(f64 - f32.astype(np.float64)).max()) / float(np.abs(f64).max())
    assert rel < 5e-3, f"float32 drifted from float64: rel.diff={rel:.2e}"

    with pytest.raises(ValueError):
        pf.Simulation(grid, precision="float16")


def test_per_array_precision():
    """`precision` accepts a per-array dict: monitor storage can be downcast
    independently of the (here float64) compute arrays, and individual field
    components can carry their own dtype.
    """
    freq0 = pf.C_0 / 1.0e-6
    grid = pf.Grid(size=(3e-6, 3e-6), cell_size=60e-9, pml_layers=(10, 10, 0))
    src = pf.PointDipole(position=(0.0, 0.0), component="Ez",
                         waveform=pf.GaussianPulse(freq0=freq0, fwhm=6e-15))

    def run(precision):
        mon = pf.FieldMonitor(name="s", components=("Ez",), times=[16e-15])
        sim = pf.Simulation(grid, sources=[src], monitors=[mon],
                            run_time=18e-15, precision=precision)
        return sim, sim.run().fields["s"]["Ez"]

    sim64, ez64 = run("float64")
    # Compute in float64 but store monitor snapshots in float32.
    sim_split, ez_split = run({"compute": "float64", "monitors": "float32"})
    assert sim_split.dtypes["Ex"] == np.float64        # fields still double
    assert ez64.dtype == np.float64
    assert ez_split.dtype == np.float32                 # storage halved
    # Only the stored copy is downcast - the field evolution is identical, so
    # the difference is bounded by float32 representation of the float64 result.
    rel = np.abs(ez64 - ez_split.astype(np.float64)).max() / np.abs(ez64).max()
    assert rel < 1e-6, f"monitor downcast changed the field: rel.diff={rel:.2e}"

    # Mixed field-component precision runs and stays finite.
    sim_mix, ez_mix = run({"default": "float32", "Ez": "float64"})
    assert sim_mix.dtypes["Ez"] == np.float64
    assert sim_mix.dtypes["Ex"] == np.float32
    assert np.isfinite(ez_mix).all()

    # Bad dtype values / unknown keys are rejected.
    with pytest.raises(ValueError):
        pf.Simulation(grid, precision={"Ez": "float16"})
    with pytest.raises(ValueError):
        pf.Simulation(grid, precision={"nonsense": "float32"})

    # The fused Numba kernel needs a single compute precision (only fires when
    # Numba is actually available; otherwise use_numba is silently downgraded).
    pytest.importorskip("numba")
    g3 = pf.Grid(size=(2e-6, 2e-6, 2e-6), cell_size=200e-9, pml_layers=(2, 2, 2))
    with pytest.raises(ValueError):
        pf.Simulation(g3, precision={"Ex": "float32", "Ez": "float64"},
                      use_numba=True)


def test_numba_warns_below_3d():
    """Requesting the numba backend on a sub-3D grid emits an advisory warning."""
    pytest.importorskip("numba")
    grid = pf.Grid(size=(3e-6, 3e-6), cell_size=60e-9, pml_layers=(8, 8, 0))
    with pytest.warns(UserWarning, match="use_numba"):
        pf.Simulation(grid, use_numba=True, run_time=1e-15)


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
