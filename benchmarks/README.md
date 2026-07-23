# Backend benchmarks

## Building the Rust backend

The experimental `backend="rust"` stepping core lives in `rust/src/lib.rs`
(PyO3 + rayon). Build and install it into the source tree with:

```bash
cd rust
cargo build --release
cp target/release/lib_photonfdtd_rs.so ../src/photonfdtd/_photonfdtd_rs.so
```

(`maturin develop --release` works too, if you have it.) The parity suite is
`tests/test_rust.py`; it skips cleanly when the extension is not built.

## Elliptic (focusing) grating coupler, 2D plan view

`bench_grating_coupler.py` runs a focusing grating coupler by the
effective-index method: a single-mode access waveguide opening into a slab
wedge crossed by the confocal-ellipse grating-line family
`q*lam = n_slab*r - n_c*sin(theta)*x`. 0.56M cells (880x640 at 25 nm),
6295 steps, float64, CPML on all sides.

```bash
python benchmarks/bench_grating_coupler.py rust|jax|numpy [outdir]
python benchmarks/make_gifs.py <outdir> rust jax        # field movies
```

Measured 2026-07-23 on a 24-core machine (JAX 0.10.2 on **CPU** - see the
caveat below), one backend per process:

| backend | wall (s) | Mcell-steps/s | peak RSS (MB) |
|---------|---------:|--------------:|--------------:|
| rust    |      7.7 |           462 |           258 |
| jax     |     15.6 |           227 |           858 |
| numpy   |    147.9 |            24 |           281 |

All three agree on the final field frame to ~2e-14 relative (double-precision
round-off over 6295 steps).

## Peak memory at whole-chip scale

`bench_memory.py` steps a 0.5 mm x 25 um strip at 25 nm (20M cells, 2D) a few
dozen steps and reports peak RSS:

| backend | peak RSS (GB) |
|---------|--------------:|
| rust    |          1.27 |
| numpy   |          1.72 |
| jax     |          2.31 |

The Rust core stores its CPML psi state compacted to the PML slabs, so its
resident set is essentially the floor for an in-core float64 run: six field
arrays + the update coefficient (7 x 8 bytes/cell ~ 1.19 GB here). Extrapolating,
a 30 GB machine holds a ~500M-cell 2D domain in core (e.g. ~5 x 1.6 mm at
25 nm); beyond that, `run(out_of_core=True)` remains the escape hatch.

## Caveat: JAX on GPU

This machine's JAX CUDA plugin is currently broken (missing `nvidia-*-cu12`
wheels), so the JAX numbers above are XLA-on-CPU. On a working GPU install
the JAX backend has measured ~40x over NumPy on large 3D runs (see
`AUTO_JAX_MIN_CELL_STEPS` notes), which would beat the CPU Rust core - the
Rust backend's niche is fast + minimal-memory CPU runs, not outrunning a GPU.
