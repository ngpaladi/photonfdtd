# Backend benchmarks

## Building the Rust backends

The experimental `backend="rust"` (CPU, rayon) and `backend="rust-cuda"`
(GPU, cudarc/NVRTC) stepping cores live in `rust/src/`. Build and install
into the source tree with:

```bash
cd rust
cargo build --release --features cuda    # drop --features cuda for CPU-only
cp target/release/lib_photonfdtd_rs.so ../src/photonfdtd/_photonfdtd_rs.so
```

(`maturin develop --release` works too, if you have it.) The `cuda` feature
pins `cudarc/cuda-12020` for a CUDA 12.2 driver - adjust to yours. The
parity suite is `tests/test_rust.py`; CUDA tests skip cleanly without a GPU.

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

Measured 2026-07-23 on a 24-core machine with an RTX 4080 (JAX 0.10.2),
one backend per process. JAX walls include XLA compile (~1.2 s f64 / ~0.5 s
f32; there is no persistent compilation cache configured), shown separately
as "steady":

| backend        | prec | wall (s) | steady (s) | Mcell-steps/s | peak RSS (MB) |
|----------------|------|---------:|-----------:|--------------:|--------------:|
| rust-cuda      | f64  |     0.73 |       0.73 |          4830 |           383 |
| rust-cuda      | f32  |     0.48 |       0.48 |          7452 |           378 |
| jax (GPU)      | f64  |      3.8 |       ~2.6 |           928 |          1248 |
| jax (GPU)      | f32  |      1.9 |       ~1.4 |          1894 |          1068 |
| rust (CPU)     | f64  |      7.7 |        7.7 |           462 |           258 |
| jax (CPU)      | f64  |     15.6 |       15.6 |           227 |           858 |
| numpy          | f64  |    147.9 |      147.9 |            24 |           281 |

All f64 backends agree on the final field frame to ~2e-14 relative
(double-precision round-off over 6295 steps); f32 runs agree to ~5e-6, i.e.
single-precision round-off. The Rust CUDA margin over XLA comes from fused
H/E kernels (each array is touched once per pass) and in-place updates;
both saturate the same VRAM bandwidth ceiling in principle.

## Peak memory at whole-chip scale

`bench_memory.py` steps a 0.5 mm x 25 um strip at 25 nm (20M cells, 2D,
float64) a few dozen steps and reports peak memory (VRAM polled with
nvidia-smi):

| backend   | peak RSS (GB) | peak VRAM (GB) |
|-----------|--------------:|---------------:|
| rust-cuda |          0.51 |           1.33 |
| rust      |          1.27 |              - |
| numpy     |          1.72 |              - |
| jax (GPU) |          1.65 |           3.33 |

Both Rust cores store their CPML psi state compacted to the PML slabs, so
the resident set is essentially the floor for an in-core run: six field
arrays + the update coefficient (7 x 8 bytes/cell ~ 1.19 GB here; XLA's
dense psi carries ~19 arrays/cell). Extrapolating, a 16 GB GPU holds a
~230M-cell f64 domain (~460M cells at f32, e.g. ~11 x 1 mm of 2D chip at
25 nm) on `rust-cuda`, vs ~90M cells for JAX-GPU; a 30 GB host holds
~500M cells f64 on the CPU core. Beyond that, `run(out_of_core=True)`
remains the escape hatch.

## Notes: GPU environment

The JAX CUDA plugin needs the `nvidia-*-cu12` wheels' lib dirs on
`LD_LIBRARY_PATH` on this machine (installed via
`pip install --user "jax[cuda12]"`). `rust-cuda` needs only the CUDA driver
plus NVRTC (from the system toolkit or the `nvidia-cuda-nvrtc-cu12` wheel).

## Next step for beyond-VRAM domains

`rust-cuda` currently requires the domain to fit in VRAM. The planned
extension is pinned-host-memory double buffering over the GPU's DMA copy
engines with *temporal blocking* (advance each slab T~32-64 steps per visit
with a light-cone halo), which amortizes the PCIe transfer cost by T and
keeps near-GPU throughput on host-RAM-sized (billion-cell f32) domains.
