# photonfdtd

A small, fully-local open-source FDTD + waveguide mode solver written in
Python/NumPy. It implements a Yee-grid FDTD time-stepper with CPML
absorbing boundaries and a 2D scalar Helmholtz mode solver, with an API
intentionally similar in spirit to Tidy3D.

This is alpha-stage software (v0.1). The pieces that exist are tested and
correct; the pieces that don't, don't. See *Status* below.

## Install

photonfdtd is not on PyPI yet, so `pip install photonfdtd` does not work.
Install it from source instead — either directly from GitHub:

```bash
pip install "git+https://github.com/ngpaladi/photonfdtd"
```

or from a checkout:

```bash
git clone https://github.com/ngpaladi/photonfdtd
cd photonfdtd
pip install -e .
```

Once a release is published to PyPI, `pip install photonfdtd` will work too.

## Quick start

### A point dipole radiating in 2D vacuum

```python
import photonfdtd as pf

lam0 = 1.0e-6
freq0 = pf.C_0 / lam0
dx = lam0 / 20

grid = pf.Grid(size=(4e-6, 4e-6), cell_size=dx, pml_layers=(12, 12, 0))
src = pf.PointDipole(
    position=(0.0, 0.0),
    component="Ez",
    waveform=pf.GaussianPulse(freq0=freq0, fwhm=10e-15),
)
mon = pf.FieldMonitor(name="snap", components=("Ez",), interval=20)

sim = pf.Simulation(grid, sources=[src], monitors=[mon], run_time=200e-15)
result = sim.run()
```

`result.fields["snap"]["Ez"]` is a `(n_frames, ny, nz)` array of field snapshots.

### Solving a slab waveguide mode

```python
import photonfdtd as pf

clad = pf.Medium.from_index(1.0)
core = pf.Medium.from_index(2.0)
slab = pf.Box(center=(0.0, 0.0), size=(0.5e-6, 0.3e-6), medium=core)

ms = pf.ModeSolver(
    size=(4e-6, 3e-6),
    cell_size=20e-9,
    structures=[slab],
    wavelength=1.55e-6,
    num_modes=2,
)
result = ms.solve()
print(result.n_eff)            # array of effective indices
```

## What v0.2 actually does

- **Yee-grid FDTD** in 1D / 2D / 3D, with the Courant-stable time step
  selected automatically.
- **CPML** absorbing boundaries (Roden & Gedney 2000, kappa = 1) on any
  number of axes.
- **Isotropic non-dispersive dielectric media** stamped per cell.
- **Geometry primitives**: axis-aligned `Box` and arbitrary-polygon
  `PolySlab` (a polygon in xy extruded between two z bounds).
- **Sources**: soft point-dipole (`PointDipole`), distributed line/area
  mode injection (`ModeSource`), an energy-normalised `SinglePhotonSource`
  whose amplitude is set so the launched wavepacket carries exactly
  $h\,\nu$ of total electromagnetic energy, and a moving-charge
  `ChargedParticle` current source that emits **Cherenkov radiation** when
  it outruns the local phase velocity (see `examples/04_cherenkov.py`).
- **Monitors**: time-domain field snapshots (`FieldMonitor`), Poynting flux
  (`FluxMonitor`), and a frequency-domain `DFTMonitor` that accumulates a
  running Fourier transform at chosen frequencies so storage scales with the
  number of frequencies rather than the number of timesteps (routinely
  50-1000x smaller than an equivalent time-domain monitor, exact at those
  frequencies).
- **Memory efficiency for large volumes**: per-array `float32`/`float64`
  precision, the permittivity grid released during stepping, a shared curl
  scratch buffer that avoids per-step full-domain temporaries, and
  `FieldMonitor(compression=...)`, which streams snapshots to disk as
  per-frame-scaled, quantised, compressed blocks - keeping RAM flat regardless
  of recording length and shrinking stored data ~10-30x (8-bit) or ~5-7x
  (16-bit) versus an uncompressed float64 monitor.
- **Out-of-core stepping** (`sim.run(out_of_core=True, tile_cells=...)`): for a
  volume whose fields do not fit in RAM, the six field arrays, `ce_field` and
  the CPML state are memory-mapped to disk and each timestep sweeps the domain
  in slabs along one axis, so **peak RAM is bounded by the tile size, not the
  grid**. Reproduces the in-core result to machine precision. NumPy backend,
  point sources and `FieldMonitor` (incl. `compression=`) supported; see
  `photonfdtd.outofcore`.
- **2D scalar Helmholtz mode solver** (eigenvalue problem in beta^2).
- **gdsfactory adapter** (`from_gdsfactory`) that reads a layout
  `Component`, maps its layers onto user-supplied materials, and returns a
  pre-built `Simulation`.
- **Optional acceleration backends** for time-stepping: a Numba JIT path and
  a CuPy GPU path (`use_gpu=True`). The GPU path uses only generic CuPy array
  ops, so it runs on either an **NVIDIA** GPU (CuPy's CUDA build,
  `cupy-cuda12x`) or an **AMD** GPU (CuPy's ROCm build, `cupy-rocm-5-0`).
  Both are optional; without them the plain NumPy core runs.
- **Differentiable JAX backend** (`use_jax=True`): a pure-functional
  reimplementation of the Yee + CPML step under `jax.lax.scan`, JIT-compiled
  through XLA (CPU/GPU/TPU from one path) and matching the in-core solver to
  machine precision. Because the stepper is a pure function of the
  permittivity, `pf.jax_value_and_grad_eps(sim, loss)` returns the gradient of
  any monitor-based scalar w.r.t. `eps_r` — a **time-domain adjoint for
  gradient-based inverse design / topology optimization**. See
  `photonfdtd.jaxbackend`. (`pip install "photonfdtd[jax]"`.)

## What v0.2 does *not* do (yet)

Items below are intentional out-of-scope and tracked in the issue
tracker:

- Dispersive media (Lorentz / Drude / Debye) - planned next, via the
  auxiliary-differential-equation method.
- Full-vectorial 2D mode solving with E and H couplings.
- Anisotropic media.
- Sub-cell averaging for polygon edges.
- Total-field / scattered-field plane wave injection.

## Roadmap

- v0.3: dispersive media (Lorentz pole-residue), full-vectorial mode
  solver, anisotropic materials.
- v0.4: Numba/JAX backend for ~10-100x speed-up.

## License

MIT.
