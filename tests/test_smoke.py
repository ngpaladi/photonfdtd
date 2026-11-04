"""Smoke tests: minimum-viable checks on FDTD and mode solver."""
import numpy as np
import pytest
import photonfdtd as pf


def test_grid_basic():
    g = pf.Grid(size=(2e-6, 2e-6), cell_size=50e-9, pml_layers=(8, 8, 0))
    assert g.ndim == 2
    assert g.shape[0] == 40 and g.shape[1] == 40 and g.shape[2] == 1


def test_mode_solver_slab_index_in_bounds():
    """Effective index of a guided mode should sit between cladding and core."""
    n_core, n_clad = 2.0, 1.0
    core = pf.Box(center=(0.0, 0.0), size=(0.5e-6, 0.3e-6),
                  medium=pf.Medium.from_index(n_core))
    ms = pf.ModeSolver(size=(4e-6, 3e-6), cell_size=20e-9,
                       structures=[core], wavelength=1.55e-6,
                       background_eps=n_clad ** 2, num_modes=2)
    res = ms.solve()
    assert res.n_eff[0] < n_core + 1e-3
    assert res.n_eff[0] > n_clad
    assert res.n_eff[0] >= res.n_eff[1]


def test_fdtd_2d_dipole_runs():
    """2D dipole simulation runs without exploding."""
    lam0 = 1.0e-6
    freq0 = pf.C_0 / lam0
    grid = pf.Grid(size=(2e-6, 2e-6), cell_size=lam0 / 15, pml_layers=(8, 8, 0))
    src = pf.PointDipole(
        position=(0.0, 0.0), component="Ez",
        waveform=pf.GaussianPulse(freq0=freq0, fwhm=8e-15),
    )
    mon = pf.FieldMonitor(name="snap", components=("Ez",), interval=20)
    sim = pf.Simulation(grid, sources=[src], monitors=[mon], run_time=40e-15)
    res = sim.run()
    assert "snap" in res.fields
    arr = res.fields["snap"]["Ez"]
    assert arr.ndim == 4  # (n_frames, nx, ny, nz)
    assert np.isfinite(arr).all()
    # field should be non-trivial in the interior
    assert np.abs(arr).max() > 0.0


def test_fdtd_3d_dipole_runs():
    """Tiny 3D simulation should also run cleanly."""
    lam0 = 1.0e-6
    freq0 = pf.C_0 / lam0
    grid = pf.Grid(size=(1.5e-6, 1.5e-6, 1.5e-6),
                   cell_size=lam0 / 10, pml_layers=(6, 6, 6))
    src = pf.PointDipole(position=(0.0, 0.0, 0.0), component="Ez",
                         waveform=pf.GaussianPulse(freq0=freq0, fwhm=6e-15))
    sim = pf.Simulation(grid, sources=[src], run_time=20e-15)
    res = sim.run()
    assert res.times.size > 0


def test_field_monitor_downsample():
    """A downsampled monitor stores exactly the strided full-resolution field,
    aligned to grid.coords[axis][::downsample], and defaults to full resolution.
    """
    lam0 = 1.0e-6
    freq0 = pf.C_0 / lam0
    grid = pf.Grid(size=(3e-6, 3e-6), cell_size=lam0 / 16, pml_layers=(8, 8, 0))
    src = pf.PointDipole(position=(0.0, 0.0), component="Ez",
                         waveform=pf.GaussianPulse(freq0=freq0, fwhm=6e-15))

    def field(ds):
        mon = pf.FieldMonitor(name="s", components=("Ez",),
                              times=[12e-15], downsample=ds)
        sim = pf.Simulation(grid, sources=[src], monitors=[mon], run_time=14e-15)
        return sim.run().fields["s"]["Ez"][0, :, :, 0]

    full = field(1)
    strided = field(2)
    # default (ds=1) is the full grid; ds=2 keeps every other cell per axis
    assert full.shape == (grid.shape[0], grid.shape[1])
    assert strided.shape == grid.coords[0][::2].shape + grid.coords[1][::2].shape
    assert np.array_equal(strided, full[::2, ::2])

    with pytest.raises(ValueError):
        pf.FieldMonitor(name="bad", downsample=0)


def test_float32_matches_float64():
    """Single precision tracks double precision closely (not bit-for-bit) and
    actually stores float32."""
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
    """Monitor storage can be downcast independently of the (float64) stepping
    arrays; the stepping arrays themselves must share one dtype."""
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
    sim_split, ez_split = run({"compute": "float64", "monitors": "float32"})
    assert sim_split.dtypes["Ex"] == np.float64        # fields still double
    assert ez64.dtype == np.float64
    assert ez_split.dtype == np.float32                 # storage halved
    rel = np.abs(ez64 - ez_split.astype(np.float64)).max() / np.abs(ez64).max()
    assert rel < 1e-6, f"monitor downcast changed the field: rel.diff={rel:.2e}"

    # Mixed stepping precision, bad dtypes and unknown keys are rejected.
    with pytest.raises(ValueError):
        pf.Simulation(grid, precision={"default": "float32", "Ez": "float64"})
    with pytest.raises(ValueError):
        pf.Simulation(grid, precision={"Ez": "float16"})
    with pytest.raises(ValueError):
        pf.Simulation(grid, precision={"nonsense": "float32"})
