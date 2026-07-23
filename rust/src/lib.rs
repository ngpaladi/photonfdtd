//! Rust FDTD stepping core for photonfdtd.
//!
//! The Yee+CPML update lives in `kernel.rs` as fused H/E passes, generic
//! over f32/f64; `blocked.rs` adds ghost-zone temporal blocking for the
//! RAM-bound regime; `cuda.rs` (feature `cuda`) provides the GPU steppers.
//! This module is the PyO3 surface: it wraps the caller's numpy arrays in
//! raw views and runs the time loop, injecting precomputed soft-source
//! waveform tables, so a whole run crosses the Python boundary only at
//! monitor-record steps.
//!
//! Memory layout: CPML psi state is stored **compacted to the PML slabs** -
//! along each psi array's derivative axis only the cells whose CPML `c`
//! coefficient is nonzero are allocated, addressed through a per-axis index
//! map (`-1` = bulk, no psi). Cells with `c == 0` never develop psi
//! (`p <- b*p + 0` starting from zero), so this is exactly equivalent to a
//! dense update while shrinking the stepping state to the six fields + the
//! `ce` coefficient + thin PML strips.
//!
//! No fastmath anywhere: arithmetic is evaluated exactly as written, so f64
//! results track the NumPy reference to double-precision round-off.

use numpy::{Element, PyArray1, PyArray2, PyArray3, PyArrayMethods};
use pyo3::prelude::*;

mod blocked;
mod kernel;

#[cfg(feature = "cuda")]
mod cuda;

use kernel::{CPtr, Ptr, Real, Sources, State};

fn view3<T: Element>(a: &Bound<'_, PyArray3<T>>) -> PyResult<Ptr<T>> {
    Ok(Ptr(unsafe { a.as_slice_mut() }?.as_mut_ptr()))
}

fn cvec1<T: Element>(py: Python<'_>, v: &[Py<PyArray1<T>>]) -> PyResult<[CPtr<T>; 3]> {
    Ok([
        CPtr(unsafe { v[0].bind(py).as_slice() }?.as_ptr()),
        CPtr(unsafe { v[1].bind(py).as_slice() }?.as_ptr()),
        CPtr(unsafe { v[2].bind(py).as_slice() }?.as_ptr()),
    ])
}

/// Generic implementation shared by the f64/f32 entry points.
#[allow(clippy::too_many_arguments)]
fn step_range_impl<T: Real>(
    py: Python<'_>,
    state: State<T>,
    srcs: Sources<T>,
    step0: usize,
    n_sub: usize,
    t_block: usize,
    tile_rows: usize,
) {
    py.allow_threads(|| {
        if t_block > 1 {
            blocked::run_blocked(&state, &srcs, step0, n_sub, t_block, tile_rows);
        } else {
            for step in step0..step0 + n_sub {
                blocked::step_parallel(&state, &srcs, step);
            }
        }
    });
}

macro_rules! make_step_range {
    ($name:ident, $ty:ty) => {
        /// Advance the simulation by `n_sub` steps starting at global step
        /// `step0`. Per step: H update, H sources, E update, E sources -
        /// the exact ordering of `Simulation.run`. `t_block > 1` enables
        /// ghost-zone temporal blocking with x-slab tiles of `tile_rows`
        /// core rows (see blocked.rs); results are bit-identical either way.
        #[pyfunction]
        #[allow(clippy::too_many_arguments)]
        fn $name(
            py: Python<'_>,
            ex: &Bound<'_, PyArray3<$ty>>, ey: &Bound<'_, PyArray3<$ty>>,
            ez: &Bound<'_, PyArray3<$ty>>, hx: &Bound<'_, PyArray3<$ty>>,
            hy: &Bound<'_, PyArray3<$ty>>, hz: &Bound<'_, PyArray3<$ty>>,
            psi: Vec<Py<PyArray3<$ty>>>,
            b_e: Vec<Py<PyArray1<$ty>>>, c_e: Vec<Py<PyArray1<$ty>>>,
            b_h: Vec<Py<PyArray1<$ty>>>, c_h: Vec<Py<PyArray1<$ty>>>,
            maps_e: Vec<Py<PyArray1<i32>>>, maps_h: Vec<Py<PyArray1<i32>>>,
            ce_field: &Bound<'_, PyArray3<$ty>>,
            ch_field: f64,
            dx: f64, dy: f64, dz: f64,
            src_comp: &Bound<'_, PyArray1<i64>>,
            src_idx: &Bound<'_, PyArray2<i64>>,
            src_vals: &Bound<'_, PyArray2<$ty>>,
            step0: usize,
            n_sub: usize,
            t_block: usize,
            tile_rows: usize,
        ) -> PyResult<()> {
            let dims = ex.dims();
            let mut psi_p = [Ptr(core::ptr::null_mut()); 12];
            for (n, a) in psi.iter().enumerate() {
                let b = a.bind(py);
                psi_p[n] = Ptr(unsafe { b.as_slice_mut() }?.as_mut_ptr());
            }
            let n_src = src_comp.dims()[0];
            let srcs = Sources {
                comp: unsafe { src_comp.as_slice() }?.to_vec(),
                idx: unsafe { src_idx.as_slice() }?.to_vec(),
                row: (0..n_src).collect(),
                vals: CPtr(unsafe { src_vals.as_slice() }?.as_ptr()),
                n_steps_total: src_vals.dims()[1],
            };
            let state = State::<$ty> {
                ex: view3(ex)?, ey: view3(ey)?, ez: view3(ez)?,
                hx: view3(hx)?, hy: view3(hy)?, hz: view3(hz)?,
                psi: psi_p,
                b_e: cvec1(py, &b_e)?, c_e: cvec1(py, &c_e)?,
                b_h: cvec1(py, &b_h)?, c_h: cvec1(py, &c_h)?,
                me: cvec1(py, &maps_e)?, mh: cvec1(py, &maps_h)?,
                // Compact psi extents, read off the compact array shapes.
                pe_y: psi[0].bind(py).dims()[1], // psi_Ex_y (nx, pe_y, nz)
                pe_z: psi[1].bind(py).dims()[2], // psi_Ex_z (nx, ny, pe_z)
                ph_y: psi[6].bind(py).dims()[1], // psi_Hx_y
                ph_z: psi[7].bind(py).dims()[2], // psi_Hx_z
                ce: CPtr(unsafe { ce_field.as_slice() }?.as_ptr()),
                ch: <$ty as Real>::from_f64(ch_field),
                nx: dims[0], ny: dims[1], nz: dims[2],
                inv_dx: <$ty as Real>::from_f64(1.0 / dx),
                inv_dy: <$ty as Real>::from_f64(1.0 / dy),
                inv_dz: <$ty as Real>::from_f64(1.0 / dz),
            };
            step_range_impl::<$ty>(py, state, srcs, step0, n_sub, t_block, tile_rows);
            Ok(())
        }
    };
}

make_step_range!(step_range_f64, f64);
make_step_range!(step_range_f32, f32);

#[pymodule]
fn _photonfdtd_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(step_range_f64, m)?)?;
    m.add_function(wrap_pyfunction!(step_range_f32, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    #[cfg(feature = "cuda")]
    {
        m.add_class::<cuda::CudaStepper>()?;
        m.add_class::<cuda::StreamingCudaStepper>()?;
    }
    m.add("CUDA_BUILT", cfg!(feature = "cuda"))?;
    Ok(())
}
