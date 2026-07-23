//! CUDA stepping core (`CudaStepper`), feature `cuda`.
//!
//! The same Yee+CPML update as the CPU kernel in `lib.rs`, as two CUDA
//! kernels (H pass, E pass) plus a tiny soft-source injection kernel,
//! compiled at first use with NVRTC for the local GPU. All state - fields,
//! PML-slab-compacted psi, coefficients, source tables - lives in device
//! memory for the whole run; the host is touched only when a monitor
//! records (`read_field`). Data layout, loop bounds, and arithmetic mirror
//! the CPU kernel exactly, so f64 results track the NumPy reference to
//! double round-off. Both f32 and f64 are supported (`new_f32` / `new_f64`);
//! consumer GPUs throttle f64 ALU hard, but FDTD is bandwidth-bound, so f64
//! remains usable while f32 doubles the effective bandwidth and halves
//! memory.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
    ValidAsZeroBits,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// CUDA C source; `real` is defined by the compile (float / double).
const KERNEL_SRC: &str = r#"
typedef REAL real;

extern "C" __global__ void update_h(
    const real* __restrict__ ex, const real* __restrict__ ey, const real* __restrict__ ez,
    real* __restrict__ hx, real* __restrict__ hy, real* __restrict__ hz,
    real* __restrict__ psi_hx_y, real* __restrict__ psi_hx_z,
    real* __restrict__ psi_hy_z, real* __restrict__ psi_hy_x,
    real* __restrict__ psi_hz_x, real* __restrict__ psi_hz_y,
    const real* __restrict__ bx_h, const real* __restrict__ cx_h,
    const real* __restrict__ by_h, const real* __restrict__ cy_h,
    const real* __restrict__ bz_h, const real* __restrict__ cz_h,
    const int* __restrict__ mh_x, const int* __restrict__ mh_y, const int* __restrict__ mh_z,
    int nx, int ny, int nz, int ph_y, int ph_z,
    real ch_field, real inv_dx, real inv_dy, real inv_dz)
{
    const long long n = (long long)nx * ny * nz;
    const long long sx = (long long)ny * nz;      // x stride
    const int fx = nx > 1 ? nx - 1 : 1;
    const int fy = ny > 1 ? ny - 1 : 1;
    const int fz = nz > 1 ? nz - 1 : 1;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         t < n; t += (long long)gridDim.x * blockDim.x) {
        const int k = (int)(t % nz);
        const int j = (int)((t / nz) % ny);
        const int i = (int)(t / sx);
        const long long id = t;
        // Hx at (i, j+1/2, k+1/2): full i, forward j, forward k
        if (j < fy && k < fz) {
            real cy = (real)0, cz = (real)0;
            if (ny > 1) {
                real d = (ez[id + nz] - ez[id]) * inv_dy;
                cy = d;
                int m = mh_y[j];
                if (m >= 0) {
                    long long pid = ((long long)i * ph_y + m) * nz + k;
                    real p = by_h[j] * psi_hx_y[pid] + cy_h[j] * d;
                    psi_hx_y[pid] = p;
                    cy += p;
                }
            }
            if (nz > 1) {
                real d = (ey[id + 1] - ey[id]) * inv_dz;
                cz = d;
                int m = mh_z[k];
                if (m >= 0) {
                    long long pid = ((long long)i * ny + j) * ph_z + m;
                    real p = bz_h[k] * psi_hx_z[pid] + cz_h[k] * d;
                    psi_hx_z[pid] = p;
                    cz += p;
                }
            }
            hx[id] -= ch_field * (cy - cz);
        }
        // Hy at (i+1/2, j, k+1/2): forward i, full j, forward k
        if (i < fx && k < fz) {
            real cz = (real)0, cx = (real)0;
            if (nz > 1) {
                real d = (ex[id + 1] - ex[id]) * inv_dz;
                cz = d;
                int m = mh_z[k];
                if (m >= 0) {
                    long long pid = ((long long)i * ny + j) * ph_z + m;
                    real p = bz_h[k] * psi_hy_z[pid] + cz_h[k] * d;
                    psi_hy_z[pid] = p;
                    cz += p;
                }
            }
            if (nx > 1) {
                real d = (ez[id + sx] - ez[id]) * inv_dx;
                cx = d;
                int m = mh_x[i];
                if (m >= 0) {
                    long long pid = ((long long)m * ny + j) * nz + k;
                    real p = bx_h[i] * psi_hy_x[pid] + cx_h[i] * d;
                    psi_hy_x[pid] = p;
                    cx += p;
                }
            }
            hy[id] -= ch_field * (cz - cx);
        }
        // Hz at (i+1/2, j+1/2, k): forward i, forward j, full k
        if (i < fx && j < fy) {
            real cx = (real)0, cy = (real)0;
            if (nx > 1) {
                real d = (ey[id + sx] - ey[id]) * inv_dx;
                cx = d;
                int m = mh_x[i];
                if (m >= 0) {
                    long long pid = ((long long)m * ny + j) * nz + k;
                    real p = bx_h[i] * psi_hz_x[pid] + cx_h[i] * d;
                    psi_hz_x[pid] = p;
                    cx += p;
                }
            }
            if (ny > 1) {
                real d = (ex[id + nz] - ex[id]) * inv_dy;
                cy = d;
                int m = mh_y[j];
                if (m >= 0) {
                    long long pid = ((long long)i * ph_y + m) * nz + k;
                    real p = by_h[j] * psi_hz_y[pid] + cy_h[j] * d;
                    psi_hz_y[pid] = p;
                    cy += p;
                }
            }
            hz[id] -= ch_field * (cx - cy);
        }
    }
}

extern "C" __global__ void update_e(
    real* __restrict__ ex, real* __restrict__ ey, real* __restrict__ ez,
    const real* __restrict__ hx, const real* __restrict__ hy, const real* __restrict__ hz,
    real* __restrict__ psi_ex_y, real* __restrict__ psi_ex_z,
    real* __restrict__ psi_ey_z, real* __restrict__ psi_ey_x,
    real* __restrict__ psi_ez_x, real* __restrict__ psi_ez_y,
    const real* __restrict__ bx_e, const real* __restrict__ cx_e,
    const real* __restrict__ by_e, const real* __restrict__ cy_e,
    const real* __restrict__ bz_e, const real* __restrict__ cz_e,
    const int* __restrict__ me_x, const int* __restrict__ me_y, const int* __restrict__ me_z,
    const real* __restrict__ ce_field,
    int nx, int ny, int nz, int pe_y, int pe_z,
    real inv_dx, real inv_dy, real inv_dz)
{
    const long long n = (long long)nx * ny * nz;
    const long long sx = (long long)ny * nz;
    const int bx = nx > 1 ? 1 : 0;
    const int by = ny > 1 ? 1 : 0;
    const int bz = nz > 1 ? 1 : 0;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         t < n; t += (long long)gridDim.x * blockDim.x) {
        const int k = (int)(t % nz);
        const int j = (int)((t / nz) % ny);
        const int i = (int)(t / sx);
        const long long id = t;
        // Ex at (i+1/2, j, k): full i, backward j, backward k
        if (j >= by && k >= bz) {
            real cy = (real)0, cz = (real)0;
            if (ny > 1) {
                real d = (hz[id] - hz[id - nz]) * inv_dy;
                cy = d;
                int m = me_y[j];
                if (m >= 0) {
                    long long pid = ((long long)i * pe_y + m) * nz + k;
                    real p = by_e[j] * psi_ex_y[pid] + cy_e[j] * d;
                    psi_ex_y[pid] = p;
                    cy += p;
                }
            }
            if (nz > 1) {
                real d = (hy[id] - hy[id - 1]) * inv_dz;
                cz = d;
                int m = me_z[k];
                if (m >= 0) {
                    long long pid = ((long long)i * ny + j) * pe_z + m;
                    real p = bz_e[k] * psi_ex_z[pid] + cz_e[k] * d;
                    psi_ex_z[pid] = p;
                    cz += p;
                }
            }
            ex[id] += ce_field[id] * (cy - cz);
        }
        // Ey at (i, j+1/2, k): backward i, full j, backward k
        if (i >= bx && k >= bz) {
            real cz = (real)0, cx = (real)0;
            if (nz > 1) {
                real d = (hx[id] - hx[id - 1]) * inv_dz;
                cz = d;
                int m = me_z[k];
                if (m >= 0) {
                    long long pid = ((long long)i * ny + j) * pe_z + m;
                    real p = bz_e[k] * psi_ey_z[pid] + cz_e[k] * d;
                    psi_ey_z[pid] = p;
                    cz += p;
                }
            }
            if (nx > 1) {
                real d = (hz[id] - hz[id - sx]) * inv_dx;
                cx = d;
                int m = me_x[i];
                if (m >= 0) {
                    long long pid = ((long long)m * ny + j) * nz + k;
                    real p = bx_e[i] * psi_ey_x[pid] + cx_e[i] * d;
                    psi_ey_x[pid] = p;
                    cx += p;
                }
            }
            ey[id] += ce_field[id] * (cz - cx);
        }
        // Ez at (i, j, k+1/2): backward i, backward j, full k
        if (i >= bx && j >= by) {
            real cx = (real)0, cy = (real)0;
            if (nx > 1) {
                real d = (hy[id] - hy[id - sx]) * inv_dx;
                cx = d;
                int m = me_x[i];
                if (m >= 0) {
                    long long pid = ((long long)m * ny + j) * nz + k;
                    real p = bx_e[i] * psi_ez_x[pid] + cx_e[i] * d;
                    psi_ez_x[pid] = p;
                    cx += p;
                }
            }
            if (ny > 1) {
                real d = (hx[id] - hx[id - nz]) * inv_dy;
                cy = d;
                int m = me_y[j];
                if (m >= 0) {
                    long long pid = ((long long)i * pe_y + m) * nz + k;
                    real p = by_e[j] * psi_ez_y[pid] + cy_e[j] * d;
                    psi_ez_y[pid] = p;
                    cy += p;
                }
            }
            ez[id] += ce_field[id] * (cx - cy);
        }
    }
}

extern "C" __global__ void inject(
    real* __restrict__ ex, real* __restrict__ ey, real* __restrict__ ez,
    real* __restrict__ hx, real* __restrict__ hy, real* __restrict__ hz,
    const int* __restrict__ comp, const int* __restrict__ idx,
    const real* __restrict__ vals,
    int n_src, long long n_steps_total, long long step, int e_pass, int ny, int nz)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_src) return;
    int c = comp[s];
    if ((c < 3) != (e_pass != 0)) return;
    long long id = ((long long)idx[3 * s] * ny + idx[3 * s + 1]) * nz + idx[3 * s + 2];
    real v = vals[(long long)s * n_steps_total + step];
    real* f = c == 0 ? ex : c == 1 ? ey : c == 2 ? ez : c == 3 ? hx : c == 4 ? hy : hz;
    atomicAdd(&f[id], v);
}
"#;

fn cerr<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(format!("CUDA error: {e}"))
}

struct Inner<T> {
    stream: Arc<CudaStream>,
    update_h: CudaFunction,
    update_e: CudaFunction,
    inject: CudaFunction,
    // device state
    fields: Vec<CudaSlice<T>>,          // Ex,Ey,Ez,Hx,Hy,Hz
    psi: Vec<CudaSlice<T>>,             // 12, compact (may be zero-length)
    b_e: Vec<CudaSlice<T>>, c_e: Vec<CudaSlice<T>>,
    b_h: Vec<CudaSlice<T>>, c_h: Vec<CudaSlice<T>>,
    maps_e: Vec<CudaSlice<i32>>, maps_h: Vec<CudaSlice<i32>>,
    ce_field: CudaSlice<T>,
    src_comp: CudaSlice<i32>,
    src_idx: CudaSlice<i32>,
    src_vals: CudaSlice<T>,
    // scalars
    nx: i32, ny: i32, nz: i32,
    pe_y: i32, pe_z: i32, ph_y: i32, ph_z: i32,
    ch_field: T,
    inv_dx: T, inv_dy: T, inv_dz: T,
    n_src: i32,
    n_steps_total: i64,
    n_cells: usize,
}

impl<T: DeviceRepr + ValidAsZeroBits + Copy + Default> Inner<T> {
    fn run_steps(&mut self, step0: usize, n_sub: usize) -> PyResult<()> {
        let cfg = LaunchConfig::for_num_elems(self.n_cells as u32);
        let src_cfg = LaunchConfig::for_num_elems(self.n_src.max(1) as u32);
        for step in step0..step0 + n_sub {
            let step_i = step as i64;
            {
                let (f, p) = (&mut self.fields, &mut self.psi);
                let (e0, e1, e2, h0, h1, h2) = split6(f);
                let (p6, p7, p8, p9, p10, p11) = seg6(p, 6);
                let mut l = self.stream.launch_builder(&self.update_h);
                l.arg(e0).arg(e1).arg(e2).arg(h0).arg(h1).arg(h2);
                l.arg(p6).arg(p7).arg(p8).arg(p9).arg(p10).arg(p11);
                l.arg(&self.b_h[0]).arg(&self.c_h[0]).arg(&self.b_h[1]).arg(&self.c_h[1]);
                l.arg(&self.b_h[2]).arg(&self.c_h[2]);
                l.arg(&self.maps_h[0]).arg(&self.maps_h[1]).arg(&self.maps_h[2]);
                l.arg(&self.nx).arg(&self.ny).arg(&self.nz).arg(&self.ph_y).arg(&self.ph_z);
                l.arg(&self.ch_field).arg(&self.inv_dx).arg(&self.inv_dy).arg(&self.inv_dz);
                unsafe { l.launch(cfg) }.map_err(cerr)?;
            }
            if self.n_src > 0 {
                self.launch_inject(src_cfg, step_i, 0)?;
            }
            {
                let (f, p) = (&mut self.fields, &mut self.psi);
                let (e0, e1, e2, h0, h1, h2) = split6(f);
                let (p0, p1, p2, p3, p4, p5) = seg6(p, 0);
                let mut l = self.stream.launch_builder(&self.update_e);
                l.arg(e0).arg(e1).arg(e2).arg(h0).arg(h1).arg(h2);
                l.arg(p0).arg(p1).arg(p2).arg(p3).arg(p4).arg(p5);
                l.arg(&self.b_e[0]).arg(&self.c_e[0]).arg(&self.b_e[1]).arg(&self.c_e[1]);
                l.arg(&self.b_e[2]).arg(&self.c_e[2]);
                l.arg(&self.maps_e[0]).arg(&self.maps_e[1]).arg(&self.maps_e[2]);
                l.arg(&self.ce_field);
                l.arg(&self.nx).arg(&self.ny).arg(&self.nz).arg(&self.pe_y).arg(&self.pe_z);
                l.arg(&self.inv_dx).arg(&self.inv_dy).arg(&self.inv_dz);
                unsafe { l.launch(cfg) }.map_err(cerr)?;
            }
            if self.n_src > 0 {
                self.launch_inject(src_cfg, step_i, 1)?;
            }
        }
        self.stream.synchronize().map_err(cerr)?;
        Ok(())
    }

    fn launch_inject(&mut self, cfg: LaunchConfig, step: i64, e_pass: i32) -> PyResult<()> {
        let (e0, e1, e2, h0, h1, h2) = split6(&mut self.fields);
        let mut l = self.stream.launch_builder(&self.inject);
        l.arg(e0).arg(e1).arg(e2).arg(h0).arg(h1).arg(h2);
        l.arg(&self.src_comp).arg(&self.src_idx).arg(&self.src_vals);
        l.arg(&self.n_src).arg(&self.n_steps_total).arg(&step).arg(&e_pass);
        l.arg(&self.ny).arg(&self.nz);
        unsafe { l.launch(cfg) }.map_err(cerr)?;
        Ok(())
    }

    fn read_field(&self, comp: usize) -> PyResult<Vec<T>> {
        self.stream.clone_dtoh(&self.fields[comp]).map_err(cerr)
    }
}

/// Split the six field buffers into distinct mutable refs for launch args.
fn split6<T>(v: &mut [CudaSlice<T>]) -> (
    &mut CudaSlice<T>, &mut CudaSlice<T>, &mut CudaSlice<T>,
    &mut CudaSlice<T>, &mut CudaSlice<T>, &mut CudaSlice<T>,
) {
    let (a, rest) = v.split_at_mut(1);
    let (b, rest) = rest.split_at_mut(1);
    let (c, rest) = rest.split_at_mut(1);
    let (d, rest) = rest.split_at_mut(1);
    let (e, rest) = rest.split_at_mut(1);
    (&mut a[0], &mut b[0], &mut c[0], &mut d[0], &mut e[0], &mut rest[0])
}

/// Six consecutive psi buffers starting at `off` as distinct mutable refs.
fn seg6<T>(v: &mut [CudaSlice<T>], off: usize) -> (
    &mut CudaSlice<T>, &mut CudaSlice<T>, &mut CudaSlice<T>,
    &mut CudaSlice<T>, &mut CudaSlice<T>, &mut CudaSlice<T>,
) {
    split6(&mut v[off..off + 6])
}

enum Stepper {
    F64(Inner<f64>),
    F32(Inner<f32>),
}

/// GPU FDTD stepper: all state device-resident; `run_steps` advances the
/// time loop, `read_field` downloads one component for monitor recording.
#[pyclass]
pub struct CudaStepper {
    inner: Stepper,
    shape: (usize, usize, usize),
}

fn build_module(
    real: &str,
) -> PyResult<(Arc<CudaContext>, Arc<CudaStream>, CudaFunction, CudaFunction, CudaFunction)> {
    let ctx = CudaContext::new(0).map_err(cerr)?;
    let stream = ctx.default_stream();
    // Target the device's own architecture so f64 atomicAdd (sm_60+) exists.
    let (major, minor) = (
        ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(cerr)?,
        ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(cerr)?,
    );
    let arch = format!("compute_{major}{minor}");
    let arch_static: &'static str = Box::leak(arch.into_boxed_str());
    let opts = CompileOptions {
        arch: Some(arch_static),
        options: vec![format!("-DREAL={real}")],
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(KERNEL_SRC, opts)
        .map_err(|e| PyRuntimeError::new_err(format!("NVRTC compile failed: {e:?}")))?;
    let module = ctx.load_module(ptx).map_err(cerr)?;
    Ok((
        ctx.clone(),
        stream,
        module.load_function("update_h").map_err(cerr)?,
        module.load_function("update_e").map_err(cerr)?,
        module.load_function("inject").map_err(cerr)?,
    ))
}

macro_rules! make_ctor {
    ($name:ident, $ty:ty, $real:literal, $variant:ident) => {
        #[allow(clippy::too_many_arguments)]
        fn $name(
            ce_field: PyReadonlyArray3<'_, $ty>,
            b_e: Vec<PyReadonlyArray1<'_, $ty>>, c_e: Vec<PyReadonlyArray1<'_, $ty>>,
            b_h: Vec<PyReadonlyArray1<'_, $ty>>, c_h: Vec<PyReadonlyArray1<'_, $ty>>,
            maps_e: Vec<PyReadonlyArray1<'_, i32>>, maps_h: Vec<PyReadonlyArray1<'_, i32>>,
            ch_field: $ty, dx: $ty, dy: $ty, dz: $ty,
            src_comp: PyReadonlyArray1<'_, i32>,
            src_idx: PyReadonlyArray2<'_, i32>,
            src_vals: PyReadonlyArray2<'_, $ty>,
        ) -> PyResult<CudaStepper> {
            let dims = ce_field.as_array().raw_dim();
            let (nx, ny, nz) = (dims[0], dims[1], dims[2]);
            let n_cells = nx * ny * nz;
            let (_ctx, stream, update_h, update_e, inject) = build_module($real)?;

            let up1 = |a: &PyReadonlyArray1<'_, $ty>| -> PyResult<CudaSlice<$ty>> {
                stream.clone_htod(a.as_slice()?).map_err(cerr)
            };
            let upi = |a: &PyReadonlyArray1<'_, i32>| -> PyResult<CudaSlice<i32>> {
                stream.clone_htod(a.as_slice()?).map_err(cerr)
            };
            // Compact psi extents = number of PML cells (map >= 0) per axis.
            let pcount = |a: &PyReadonlyArray1<'_, i32>| -> PyResult<usize> {
                Ok(a.as_slice()?.iter().filter(|&&m| m >= 0).count())
            };
            let (pe_x, pe_y, pe_z) =
                (pcount(&maps_e[0])?, pcount(&maps_e[1])?, pcount(&maps_e[2])?);
            let (ph_x, ph_y, ph_z) =
                (pcount(&maps_h[0])?, pcount(&maps_h[1])?, pcount(&maps_h[2])?);

            let fields = (0..6)
                .map(|_| stream.alloc_zeros::<$ty>(n_cells).map_err(cerr))
                .collect::<PyResult<Vec<CudaSlice<$ty>>>>()?;
            // Order matches the CPU backend's psi list (E first, then H).
            let psi_sizes = [
                nx * pe_y * nz, nx * ny * pe_z, nx * ny * pe_z, pe_x * ny * nz,
                pe_x * ny * nz, nx * pe_y * nz, nx * ph_y * nz, nx * ny * ph_z,
                nx * ny * ph_z, ph_x * ny * nz, ph_x * ny * nz, nx * ph_y * nz,
            ];
            let psi = psi_sizes
                .iter()
                .map(|&s: &usize| stream.alloc_zeros::<$ty>(s.max(1)).map_err(cerr))
                .collect::<PyResult<Vec<CudaSlice<$ty>>>>()?;

            let inner = Inner::<$ty> {
                stream: stream.clone(),
                update_h, update_e, inject,
                fields, psi,
                b_e: b_e.iter().map(up1).collect::<PyResult<Vec<_>>>()?,
                c_e: c_e.iter().map(up1).collect::<PyResult<Vec<_>>>()?,
                b_h: b_h.iter().map(up1).collect::<PyResult<Vec<_>>>()?,
                c_h: c_h.iter().map(up1).collect::<PyResult<Vec<_>>>()?,
                maps_e: maps_e.iter().map(upi).collect::<PyResult<Vec<_>>>()?,
                maps_h: maps_h.iter().map(upi).collect::<PyResult<Vec<_>>>()?,
                ce_field: stream.clone_htod(ce_field.as_slice()?).map_err(cerr)?,
                src_comp: stream.clone_htod(src_comp.as_slice()?).map_err(cerr)?,
                src_idx: stream.clone_htod(src_idx.as_slice()?).map_err(cerr)?,
                src_vals: stream.clone_htod(src_vals.as_slice()?).map_err(cerr)?,
                nx: nx as i32, ny: ny as i32, nz: nz as i32,
                pe_y: pe_y as i32, pe_z: pe_z as i32,
                ph_y: ph_y as i32, ph_z: ph_z as i32,
                ch_field,
                inv_dx: (1.0 as $ty) / dx, inv_dy: (1.0 as $ty) / dy,
                inv_dz: (1.0 as $ty) / dz,
                n_src: src_comp.as_array().raw_dim()[0] as i32,
                n_steps_total: src_vals.as_array().raw_dim()[1] as i64,
                n_cells,
            };
            Ok(CudaStepper { inner: Stepper::$variant(inner), shape: (nx, ny, nz) })
        }
    };
}

make_ctor!(ctor_f64, f64, "double", F64);
make_ctor!(ctor_f32, f32, "float", F32);

#[pymethods]
impl CudaStepper {
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn new_f64(
        ce_field: PyReadonlyArray3<'_, f64>,
        b_e: Vec<PyReadonlyArray1<'_, f64>>, c_e: Vec<PyReadonlyArray1<'_, f64>>,
        b_h: Vec<PyReadonlyArray1<'_, f64>>, c_h: Vec<PyReadonlyArray1<'_, f64>>,
        maps_e: Vec<PyReadonlyArray1<'_, i32>>, maps_h: Vec<PyReadonlyArray1<'_, i32>>,
        ch_field: f64, dx: f64, dy: f64, dz: f64,
        src_comp: PyReadonlyArray1<'_, i32>,
        src_idx: PyReadonlyArray2<'_, i32>,
        src_vals: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<CudaStepper> {
        ctor_f64(ce_field, b_e, c_e, b_h, c_h, maps_e, maps_h,
                 ch_field, dx, dy, dz, src_comp, src_idx, src_vals)
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn new_f32(
        ce_field: PyReadonlyArray3<'_, f32>,
        b_e: Vec<PyReadonlyArray1<'_, f32>>, c_e: Vec<PyReadonlyArray1<'_, f32>>,
        b_h: Vec<PyReadonlyArray1<'_, f32>>, c_h: Vec<PyReadonlyArray1<'_, f32>>,
        maps_e: Vec<PyReadonlyArray1<'_, i32>>, maps_h: Vec<PyReadonlyArray1<'_, i32>>,
        ch_field: f32, dx: f32, dy: f32, dz: f32,
        src_comp: PyReadonlyArray1<'_, i32>,
        src_idx: PyReadonlyArray2<'_, i32>,
        src_vals: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<CudaStepper> {
        ctor_f32(ce_field, b_e, c_e, b_h, c_h, maps_e, maps_h,
                 ch_field, dx, dy, dz, src_comp, src_idx, src_vals)
    }

    /// Advance `n_sub` steps starting at global step `step0` (device-side).
    fn run_steps(&mut self, py: Python<'_>, step0: usize, n_sub: usize) -> PyResult<()> {
        py.allow_threads(|| match &mut self.inner {
            Stepper::F64(s) => s.run_steps(step0, n_sub),
            Stepper::F32(s) => s.run_steps(step0, n_sub),
        })
    }

    /// Download one field component (0..6 = Ex,Ey,Ez,Hx,Hy,Hz) as a
    /// (nx, ny, nz) numpy array in the stepper's dtype.
    fn read_field(&self, py: Python<'_>, comp: usize) -> PyResult<PyObject> {
        let (nx, ny, nz) = self.shape;
        Ok(match &self.inner {
            Stepper::F64(s) => {
                let v = s.read_field(comp)?;
                v.into_pyarray(py).reshape([nx, ny, nz])?.into_any().unbind()
            }
            Stepper::F32(s) => {
                let v = s.read_field(comp)?;
                v.into_pyarray(py).reshape([nx, ny, nz])?.into_any().unbind()
            }
        })
    }

    /// Device name + free/total memory (MB), for logging.
    #[staticmethod]
    fn device_info() -> PyResult<(String, usize, usize)> {
        let ctx = CudaContext::new(0).map_err(cerr)?;
        let name = ctx.name().map_err(cerr)?;
        let (free, total) = cudarc::driver::result::mem_get_info().map_err(cerr)?;
        Ok((name, free / (1 << 20), total / (1 << 20)))
    }
}
