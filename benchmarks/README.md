# Benchmarks

## Elliptic (focusing) grating coupler, 2D plan view

`bench_grating_coupler.py` runs a focusing grating coupler by the
effective-index method: a single-mode access waveguide opening into a slab
wedge crossed by the confocal-ellipse grating-line family
`q*lam = n_slab*r - n_c*sin(theta)*x`. 0.56M cells (880x640 at 25 nm),
6295 steps, CPML on all sides.

```bash
python benchmarks/bench_grating_coupler.py [outdir] [float64|float32]
python benchmarks/make_gifs.py <outdir> jax jax_f32     # field movies
```

Measured 2026-07-23 on a 24-core machine with an RTX 4080 (JAX 0.10.2),
one precision per process. Walls include the XLA compile (~1.2 s f64 /
~0.5 s f32; there is no persistent compilation cache configured), shown
separately as "steady":

| device     | prec | wall (s) | steady (s) | Mcell-steps/s | peak RSS (MB) |
|------------|------|---------:|-----------:|--------------:|--------------:|
| GPU        | f64  |      3.8 |       ~2.6 |           928 |          1248 |
| GPU        | f32  |      1.9 |       ~1.4 |          1894 |          1068 |
| CPU        | f64  |     15.6 |       15.6 |           227 |           858 |

The f32 run agrees with f64 on the final field frame to single-precision
round-off. (This small domain is largely cache-resident on the CPU.)

## Layout-scale methods demo

`demo_eim_eme_adi.py` renders `demo_eim_eme_adi.png`: the EIM slab mode, an
EME taper sweep, and an ADI-FDTD resonator stepped at 4x the explicit CFL
limit.

```bash
python benchmarks/demo_eim_eme_adi.py [outdir]
```

## Notes: GPU environment

The JAX CUDA plugin needs the `nvidia-*-cu12` wheels' lib dirs on
`LD_LIBRARY_PATH` on this machine (installed via
`pip install --user "jax[cuda12]"`). Set `XLA_PYTHON_CLIENT_PREALLOCATE=false`
for honest VRAM measurements; otherwise JAX reserves most of the card up front.
