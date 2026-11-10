# photonfdtd

A small, fully-local open-source FDTD + waveguide mode solver for integrated
photonics: a Yee-grid FDTD time-stepper with CPML absorbing boundaries,
dispersive materials, anisotropic subpixel smoothing, and a full-vectorial
waveguide mode solver, with an API intentionally similar in spirit to Tidy3D.
The time loop runs on JAX/XLA, one code path for CPU, GPU and TPU, and the whole
solver is differentiable. Everything runs on your own machine.

This is alpha-stage software. The pieces that exist are tested and correct; the
pieces that don't, don't. See *Known limitations* below.

**Documentation:** https://www.noahpaladino.com/photonfdtd/

## Install

```bash
pip install photonfdtd
```

or from source, either directly from GitHub:

```bash
pip install "git+https://github.com/ngpaladi/photonfdtd"
```

or from a checkout:

```bash
git clone https://github.com/ngpaladi/photonfdtd
cd photonfdtd
pip install -e .
```

The PyPI wheel is pure Python and pulls in JAX (the CPU `jaxlib`). For an
NVIDIA GPU, install the CUDA build of JAX alongside it:

```bash
pip install "jax[cuda12]"
```

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

## What it does today

- **Yee-grid FDTD** in 1D / 2D / 3D, with the Courant-stable time step
  selected automatically.
- **CPML** absorbing boundaries (Roden & Gedney 2000, kappa = 1) on any
  number of axes, and **periodic** boundaries per axis
  (`Grid(periodic=(False, True, True))`) for one unit cell of a grating,
  photonic crystal or metasurface. A grid edge with neither is a perfect
  electric conductor.
- **Isotropic dielectric media, non-dispersive or dispersive.** Non-dispersive
  media stamp a single `eps_r` per cell; dispersive media (`DispersiveMedium`,
  with `.lorentz` / `.drude` / `.sellmeier` constructors) carry Lorentz, Drude
  and Sellmeier poles advanced by the auxiliary-differential-equation (ADE)
  method during stepping. A cited material library (`pf.silica()`,
  `pf.silicon()`, `pf.silicon_nitride()`, `pf.lithium_niobate()`, `pf.gold()`,
  `pf.silver()`) reproduces each source's published index — see
  [`docs/material_data.md`](docs/material_data.md).
- **Anisotropic subpixel smoothing** (`Simulation(subpixel=True)`): partially
  filled interface cells get an effective diagonal permittivity tensor
  (harmonic mean normal to the interface, arithmetic mean tangential), removing
  the staircase error and restoring second-order boundary accuracy
  (Farjadpour et al. 2006). See [`docs/accuracy.rst`](docs/accuracy.rst).
- **Geometry primitives**: axis-aligned `Box` and arbitrary-polygon
  `PolySlab` (a polygon in xy extruded between two z bounds).
- **Sources**: soft point-dipole (`PointDipole`), distributed line/area
  mode injection (`ModeSource`, and the one-way `UniModeSource` built from
  equivalence-principle current sheets), one-way **plane waves**
  (`PlaneWave`) and **Gaussian beams** (`GaussianBeam`) from a Huygens
  current sheet along any axis, calibrated to the requested peak field
  (second-order accurate, backward leakage ~2e-4 at 20 cells per
  wavelength; on a transversely periodic cell the wave stays exactly uniform
  and a half-space reproduces the Fresnel transmission to <1%), an energy-normalised
  `SinglePhotonSource` whose amplitude is set so the launched wavepacket
  carries exactly $h\,\nu$ of total electromagnetic energy, and a moving-charge
  `ChargedParticle` current source that emits **Cherenkov radiation** when
  it outruns the local phase velocity (`examples/04_cherenkov.py`), and whose
  evanescent near field below threshold matches the analytic
  `exp(-k_x |y| / gamma)` law to a few percent, the coupling that drives a
  Cherenkov radiator in a hollow tube (`examples/05_cherenkov_tube.py`).
- **Monitors**: time-domain field snapshots (`FieldMonitor`), Poynting flux
  (`FluxMonitor`), and a frequency-domain `DFTMonitor` that accumulates a
  running Fourier transform at chosen frequencies so storage scales with the
  number of frequencies rather than the number of timesteps (routinely
  50-1000x smaller than an equivalent time-domain monitor, exact at those
  frequencies).
- **Near-to-far-field projection** (`project_far_field`): turn plane
  `DFTMonitor`s on a closed box (or one open plane) into the far field and
  radiant intensity at any directions, in 2-D or 3-D, from Love's
  equivalence currents with each sample at its exact Yee position. A 3-D
  dipole gives `sin²θ` to <1% at 20 cells per wavelength; the far-field power
  matches the near-field Poynting flux to 0.2%. The same projection gives the
  angular spectrum of Cherenkov light leaving a radiator.
- **Mode-decomposition S-parameters** (`s_parameters`, `mode_amplitudes`):
  project the frequency-domain fields on a memory-light port plane (any axis via
  `DFTMonitor(plane_axis=..., plane_position=...)`) onto a solved mode to get
  complex forward/backward modal amplitudes and hence S-parameters. A mode down
  a straight lossless guide gives `|S21|≈1`; the overlap works on NumPy or JAX
  arrays, so `|S_ij|²` is differentiable through the adjoint — the PIC inverse-design
  objective.
- **Photon-count estimates** (`photon_spectrum`, `frank_tamm_spectrum`,
  `frank_tamm_photons`): turn a plane `DFTMonitor` into `dU/df` and `dN/df`
  (photons per Hz) via Parseval, integrate to a count over any band, and rescale
  a run made with an exaggerated charge to one electron (`rescale_charge`).
  Analytic references: the 3-D Frank–Tamm yield and its 2-D line-charge
  counterpart, both with the charge cloud's form factor. A 2-D Cherenkov run
  matches the line-charge spectrum to within 1% from 1.5 to 4 µm.
- **Full-vectorial waveguide mode solver** (`ModeSolver`): a finite-difference
  eigenproblem for the transverse fields on the Yee grid that distinguishes TE
  from TM and captures high-index-contrast boundary effects. Validated against
  the exact symmetric-slab dispersion to better than 0.01% on both TE0 and TM0
  (Zhu & Brown 2002; Fallahkhair et al. 2008).
- **Layout-scale methods** for whole circuits (`docs/layout_scale.rst`):
  - the **2.5-D effective-index method** (`EffectiveIndex2D`, `slab_modes`)
    collapses a vertical layer stack to effective indices so a patterned layer
    becomes an ordinary 2-D run (1-D slab solver validated against the
    analytic dispersion, second-order convergent);
  - **eigenmode expansion** (`eme_2d`, `Section`, `sections_from_eps`) for
    piecewise-uniform 2-D devices — local modes, power-orthonormal interface
    matching, Redheffer star product; cost set by mode count, not device
    length; a width step's reflection reproduces the Fresnel value;
  - **unconditionally-stable ADI-FDTD** (`ADISimulation2D`) for the 2-D
    `(Ez, Hx, Hy)` system, stable and bounded at 8x the explicit CFL step;
    the time loop is a JAX `lax.scan` with each sweep's line solves batched
    into one tridiagonal solve.
- **Differentiable PIC inverse design.** A substrate/etched-core/cladding stack
  driven by a 2-D etch density `ρ(x,y)` (`EtchedCore`) maps — conic filter →
  `tanh` projection → anisotropic subpixel sidewalls → 3-D ε — entirely in JAX,
  so `value_and_grad_density` returns `d(loss)/d(ρ)` through the full time
  evolution (topology optimization; gradient matches finite differences to
  ~1e-8). Adjoint memory is kept in check by two-level gradient checkpointing,
  for closed/periodic domains an O(1)-in-timesteps **reversible adjoint**
  (`jax_value_and_grad_eps_reversible`, plus a PML-shell-tape variant), and for
  grids too big for the device a **disk-tiled adjoint**
  (`jax_value_and_grad_eps_tiled`). Dispersive media are differentiable on
  every path (the reversible ones need lossless poles, e.g. Sellmeier glass).
- **Active devices.** Switch light with voltages and heaters
  (`photonfdtd.active`): electro-optic crystals (x/z-cut lithium niobate,
  LiTaO3, BaTiO3, EO polymers), thermo-optic heaters and silicon free-carrier
  dispersion, from built-in electrostatic and thermal solvers
  (`photonfdtd.statics`). Get `V_pi * L` and `P_pi` from the mode solver, sweep
  a switch's voltage in FDTD, optimize drive voltages by gradient descent, or
  step an RF drive in time. Anisotropic (diagonal-tensor) and conductive media
  work in FDTD runs, the mode solver and the checkpointed design gradients
  (`value_and_grad_params`); see the guide for the paths that refuse them.
- **Geometry optimization and chip training.** Optimize a `DesignRegion` box
  inside a real layout (fixed waveguides around it) or a `ShapeDesign`'s
  parametric polygons (widths, gaps, tapers), with a compile-once gradient
  (`make_value_and_grad_params`), Adam or any optax optimizer (`minimize`),
  beta continuation, robust eroded/dilated objectives, and GDS export
  (`export_gds`, optional `gdstk`). `train` fits a chip to a dataset like a
  neural network: the chip is linear, so one simulation per input port gives its
  transfer matrix and a training step costs N forward + N adjoint runs however
  big the batch. Every design gradient can use the checkpointed, reversible,
  PML-reversible or disk-tiled adjoint (`method=`), and dispersive layouts
  around the design region are supported. See `docs/optimization.rst`.
- **Memory planning**: per-array `float32`/`float64` precision (the stepping
  arrays share one dtype; monitor storage may differ) and a memory planner
  (`estimate_memory`, `recommend_mode`, `format_report`, `available_memory`)
  that sizes a run against the device before it OOMs.
- **gdsfactory adapter** (`from_gdsfactory`) that reads a layout
  `Component`, maps its layers onto user-supplied materials, and returns a
  pre-built `Simulation`.

## Running on JAX

Every time loop — `Simulation.run()`, the adjoints, and `ADISimulation2D` —
is a JIT-compiled `jax.lax.scan`, so the same code runs on CPU, GPU or TPU
depending on which `jaxlib` is installed. Float64 runs switch on
`jax_enable_x64` for you; `precision="float32"` halves memory and roughly
doubles throughput. See [`docs/backends.rst`](docs/backends.rst).

### Running on GPU

With a CUDA `jaxlib` present (`pip install "jax[cuda12]"`) the forward run and
both adjoints run on-device. Validated on an NVIDIA RTX 4080 (CUDA 12). If JAX
falls back to CPU with a "cuSPARSE not found"-style error, the nvidia CUDA
libraries just aren't on the loader path — add them, e.g.
`export LD_LIBRARY_PATH=$(python -c "import nvidia,glob,os;print(':'.join(glob.glob(os.path.dirname(nvidia.__file__)+'/*/lib')))")`.
Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` if you want JAX to allocate VRAM on
demand rather than grabbing most of the card up front.

## Known limitations

- A grid whose fields don't fit in device memory runs disk-tiled
  (`sim.run(disk_tiled=True)`, or automatically with a warning): fields live in
  memmaps on disk and stream through the device in x-slabs, several steps per
  load. It matches the in-memory result to ~1e-14 and is a few times slower
  (2.3–5.6× on a 3.5M-cell grid). The tiled adjoint
  (`jax_value_and_grad_eps_tiled`) checkpoints the state to disk, about
  2·√(n_steps / steps_per_tile) copies, and costs about two forward runs plus
  the VJPs; it does not support subpixel smoothing. Monitor outputs and the
  returned gradient stay in RAM, so use `plane_z=` / `downsample=` /
  `DFTMonitor` to keep the outputs small. `estimate_memory` sizes every mode
  before you start.
- The effective-index method assumes the vertical mode is separable from the
  in-plane confinement — good for routing, poor for deep sub-wavelength gaps
  and out-of-plane diffraction. EME is 2-D, single-polarisation, straight-axis
  only. ADI-FDTD is 2-D `(Ez, Hx, Hy)` with PEC or a conductivity absorber,
  not CPML.
- The reversible adjoint with CPML is exact but only saves memory when the PML
  shell is a small fraction (under ~5%) of the volume; for ordinary PML runs
  use checkpointing.
- Anisotropic media are diagonal tensors in the grid frame (no rotated
  crystal axes), and neither they nor dispersive media combine with subpixel
  smoothing.
- Plane waves are normal-incidence only: oblique incidence on a periodic cell
  needs Bloch-periodic (complex) fields, which the engine does not step, and
  there is no total-field/scattered-field box. Periodic boundaries run in
  memory (not disk-tiled).
- One device per run: no multi-GPU domain decomposition. Grids are uniform
  per axis, and there are no symmetry planes.

## Releasing (maintainers)

Releases publish to PyPI automatically via
[`.github/workflows/publish.yml`](.github/workflows/publish.yml) using PyPI
**Trusted Publishing** (OIDC — no API token stored). To cut a release:

1. Bump the version in `pyproject.toml` **and** `src/photonfdtd/__init__.py`
   (kept in sync by `tests/test_version.py`).
2. Publish a GitHub Release tagged `vX.Y.Z` matching that version. The workflow
   runs the tests, guards tag-vs-version, builds the sdist + wheel, and
   publishes.

The docs build and deploy from `main` via
[`.github/workflows/docs.yml`](.github/workflows/docs.yml) with
`sphinx-build -W`, so a docstring that doesn't parse as reStructuredText fails
the build.

## License

MIT.
