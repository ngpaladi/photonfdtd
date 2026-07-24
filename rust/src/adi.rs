//! ADI-FDTD stepping core (2-D, `(Ez, Hx, Hy)`), feature-independent.
//!
//! A port of the unconditionally-stable ADI scheme in
//! `photonfdtd/adi.py`. Each full step is two implicit sub-steps: sub-step 1
//! solves an independent tridiagonal system per grid *row* (x implicit), then
//! advances H explicitly; sub-step 2 solves one per *column* (y implicit).
//! The lines are independent, so rayon parallelises the Thomas solves across
//! them - the tridiagonal sweep is the natural unit of work here, exactly as
//! the fused Yee update was for the explicit core. Generic over f32/f64; no
//! fastmath, so f64 tracks the NumPy reference to double round-off.

use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::kernel::Real;

const MU_0: f64 = 1.256_637_062_12e-6;

/// Solve the interior tridiagonal system `diag[k] x[k] + off (x[k-1]+x[k+1]) =
/// rhs[k]` for k in 0..n (constant off-diagonal), Thomas algorithm. `cp`/`dp`
/// are caller-provided scratch of length n.
#[inline]
fn thomas<T: Real>(diag: &[T], off: T, rhs: &[T], cp: &mut [T], out: &mut [T]) {
    let n = diag.len();
    if n == 0 {
        return;
    }
    cp[0] = off / diag[0];
    out[0] = rhs[0] / diag[0];
    for k in 1..n {
        let m = diag[k] - off * cp[k - 1];
        cp[k] = off / m;
        out[k] = (rhs[k] - off * out[k - 1]) / m;
    }
    for k in (0..n - 1).rev() {
        out[k] = out[k] - cp[k] * out[k + 1];
    }
}

/// All-`Copy` view of the field/coefficient buffers + geometry. Passed by
/// value (moved) into the rayon closures so they capture the whole `Copy`
/// struct, never a reference to a bare `*mut T` field (which is not `Sync`).
#[derive(Clone, Copy)]
struct Grid<T> {
    ez: *mut T,           // (nx, ny)
    hx: *mut T,           // (nx, ny-1)
    hy: *mut T,           // (nx-1, ny)
    eps: *const T,        // (nx, ny), = eps_r * EPS_0
    sig_e: *const T,      // (nx, ny)
    sig_m: *const T,      // (nx, ny)
    nx: usize,
    ny: usize,
    dx: T,
    dy: T,
    dt: T,
    absorber: bool,
}
unsafe impl<T: Send> Send for Grid<T> {}
unsafe impl<T: Sync> Sync for Grid<T> {}

impl<T: Real> Grid<T> {
    #[inline(always)]
    fn ez_at(&self, i: usize, j: usize) -> T {
        unsafe { *self.ez.add(i * self.ny + j) }
    }
    #[inline(always)]
    fn hx_at(&self, i: usize, j: usize) -> T {
        unsafe { *self.hx.add(i * (self.ny - 1) + j) }
    }
    #[inline(always)]
    fn hy_at(&self, i: usize, j: usize) -> T {
        unsafe { *self.hy.add(i * self.ny + j) }
    }
    #[inline(always)]
    fn eps_at(&self, i: usize, j: usize) -> T {
        unsafe { *self.eps.add(i * self.ny + j) }
    }
    #[inline(always)]
    fn sige_at(&self, i: usize, j: usize) -> T {
        unsafe { *self.sig_e.add(i * self.ny + j) }
    }
    #[inline(always)]
    fn sigm_at(&self, i: usize, j: usize) -> T {
        unsafe { *self.sig_m.add(i * self.ny + j) }
    }
}

/// One full ADI step. `ez_star` is an (nx*ny) scratch buffer reused across
/// steps; the H arrays are updated in place.
fn adi_step<T: Real>(g: Grid<T>, ez_star: &mut [T], step: usize, src: &SrcTable<T>) {
    let (nx, ny) = (g.nx, g.ny);
    let two = T::from_f64(2.0);
    let s = g.dt / two;
    let c = s * s / T::from_f64(MU_0);
    let inv_dx = T::from_f64(1.0) / g.dx;
    let inv_dy = T::from_f64(1.0) / g.dy;
    let cxx = c * inv_dx * inv_dx;
    let cyy = c * inv_dy * inv_dy;
    let s_mu = s / T::from_f64(MU_0);
    let half_dt = s;

    // ===== sub-step 1: implicit in x (one tridiagonal per row j) ===== //
    // For each interior row j, solve over i = 1..nx-1.
    let ez_ptr = Ptr(ez_star.as_mut_ptr());
    (0..ny).into_par_iter().for_each(move |j| {
        let n = if nx >= 2 { nx - 2 } else { 0 };
        if n == 0 {
            return;
        }
        let mut diag = vec![T::ZERO; n];
        let mut rhs = vec![T::ZERO; n];
        let mut cp = vec![T::ZERO; n];
        let mut out = vec![T::ZERO; n];
        for (idx, i) in (1..nx - 1).enumerate() {
            let dhy_dx = (g.hy_at(i, j) - g.hy_at(i - 1, j)) * inv_dx;
            let dhx_dy = if j >= 1 && j + 1 < ny {
                (g.hx_at(i, j) - g.hx_at(i, j - 1)) * inv_dy
            } else {
                T::ZERO
            };
            diag[idx] = g.eps_at(i, j) + two * cxx + g.sige_at(i, j) * half_dt;
            rhs[idx] = g.eps_at(i, j) * g.ez_at(i, j) + s * (dhy_dx - dhx_dy);
        }
        thomas(&diag, -cxx, &rhs, &mut cp, &mut out);
        for (idx, i) in (1..nx - 1).enumerate() {
            unsafe { ez_ptr.set(i * ny + j, out[idx]) };
        }
    });
    // PEC/edge rows stay zero.
    for j in 0..ny {
        ez_star[j] = T::ZERO;
        ez_star[(nx - 1) * ny + j] = T::ZERO;
    }
    src.inject(&g, ez_star, 2 * step);        // (n+0.5) sample

    // explicit H at the half step: Hy from dEz*/dx (star, in ez_star), Hx from
    // dEz^n/dy (g.ez still holds Ez^n until the copy below).
    update_hy(g, ez_star.as_ptr(), s_mu);
    update_hx_from(g, move |i, j| (g.ez_at(i, j + 1) - g.ez_at(i, j)) * inv_dy, s_mu);

    // commit Ez* into Ez (Ez now holds the star field for sub-step 2)
    unsafe {
        std::ptr::copy_nonoverlapping(ez_star.as_ptr(), g.ez, nx * ny);
    }

    // ===== sub-step 2: implicit in y (one tridiagonal per column i) ===== //
    (0..nx).into_par_iter().for_each(move |i| {
        let n = if ny >= 2 { ny - 2 } else { 0 };
        if n == 0 {
            return;
        }
        let mut diag = vec![T::ZERO; n];
        let mut rhs = vec![T::ZERO; n];
        let mut cp = vec![T::ZERO; n];
        let mut out = vec![T::ZERO; n];
        for (idx, j) in (1..ny - 1).enumerate() {
            let dhy_dx = if i >= 1 && i + 1 < nx {
                (g.hy_at(i, j) - g.hy_at(i - 1, j)) * inv_dx
            } else {
                T::ZERO
            };
            let dhx_dy = (g.hx_at(i, j) - g.hx_at(i, j - 1)) * inv_dy;
            diag[idx] = g.eps_at(i, j) + two * cyy + g.sige_at(i, j) * half_dt;
            rhs[idx] = g.eps_at(i, j) * g.ez_at(i, j) + s * (dhy_dx - dhx_dy);
        }
        thomas(&diag, -cyy, &rhs, &mut cp, &mut out);
        for (idx, j) in (1..ny - 1).enumerate() {
            unsafe { ez_ptr.set(i * ny + j, out[idx]) };
        }
    });
    for i in 0..nx {
        ez_star[i * ny] = T::ZERO;
        ez_star[i * ny + ny - 1] = T::ZERO;
    }
    src.inject(&g, ez_star, 2 * step + 1);    // (n+1.0) sample

    // explicit H: Hy from dEz*/dx (the star field, still in g.ez), Hx from
    // dEz^{n+1}/dy (the new field, in ez_star).
    update_hy(g, g.ez as *const T, s_mu);    // star gradient
    let new_ptr = CPtr(ez_star.as_ptr());
    update_hx_from(g, move |i, j| {
        let a = unsafe { new_ptr.at(i * ny + (j + 1)) };
        let b = unsafe { new_ptr.at(i * ny + j) };
        (a - b) * inv_dy
    }, s_mu);

    // commit new Ez.
    unsafe {
        std::ptr::copy_nonoverlapping(ez_star.as_ptr(), g.ez, nx * ny);
    }
}

/// Hy += s_mu * dEz/dx, where Ez gradient is read from the array pointed to by
/// `ez_src` (nx*ny). Absorber applies the matched magnetic-loss update.
fn update_hy<T: Real>(g: Grid<T>, ez_src: *const T, s_mu: T) {
    let (nx, ny) = (g.nx, g.ny);
    let inv_dx = T::from_f64(1.0) / g.dx;
    let s = s_mu * T::from_f64(MU_0);           // = dt/2
    let ptr = Ptr(g.hy);
    let src = CPtr(ez_src);
    (0..nx - 1).into_par_iter().for_each(move |i| {
        for j in 0..ny {
            let d = unsafe { src.at((i + 1) * ny + j) - src.at(i * ny + j) } * inv_dx;
            let id = i * ny + j;
            let cur = unsafe { ptr.get(id) };
            let nv = if g.absorber {
                let sm = (g.sigm_at(i, j) + g.sigm_at(i + 1, j)) * T::from_f64(0.5);
                let a = T::from_f64(1.0) - sm * s / T::from_f64(MU_0);
                let b = T::from_f64(1.0) + sm * s / T::from_f64(MU_0);
                (a / b) * cur + (s_mu / b) * d
            } else {
                cur + s_mu * d
            };
            unsafe { ptr.set(id, nv) };
        }
    });
}

/// Hx -= s_mu * dEz/dy, gradient supplied by `grad(i, j)`.
fn update_hx_from<T, F>(g: Grid<T>, grad: F, s_mu: T)
where
    T: Real,
    F: Fn(usize, usize) -> T + Sync + Send,
{
    let (nx, ny) = (g.nx, g.ny);
    let s = s_mu * T::from_f64(MU_0);
    let ptr = Ptr(g.hx);
    (0..nx).into_par_iter().for_each(move |i| {
        for j in 0..ny - 1 {
            let d = grad(i, j);
            let id = i * (ny - 1) + j;
            let cur = unsafe { ptr.get(id) };
            let nv = if g.absorber {
                let sm = (g.sigm_at(i, j) + g.sigm_at(i, j + 1)) * T::from_f64(0.5);
                let a = T::from_f64(1.0) - sm * s / T::from_f64(MU_0);
                let b = T::from_f64(1.0) + sm * s / T::from_f64(MU_0);
                (a / b) * cur - (s_mu / b) * d
            } else {
                cur - s_mu * d
            };
            unsafe { ptr.set(id, nv) };
        }
    });
}

#[derive(Clone, Copy)]
struct Ptr<T>(*mut T);
unsafe impl<T: Send> Send for Ptr<T> {}
unsafe impl<T: Sync> Sync for Ptr<T> {}
impl<T: Copy> Ptr<T> {
    #[inline(always)]
    unsafe fn set(self, i: usize, v: T) {
        *self.0.add(i) = v;
    }
    #[inline(always)]
    unsafe fn get(self, i: usize) -> T {
        *self.0.add(i)
    }
}

#[derive(Clone, Copy)]
struct CPtr<T>(*const T);
unsafe impl<T: Sync> Send for CPtr<T> {}
unsafe impl<T: Sync> Sync for CPtr<T> {}
impl<T: Copy> CPtr<T> {
    #[inline(always)]
    unsafe fn at(self, i: usize) -> T {
        *self.0.add(i)
    }
}

struct SrcTable<T> {
    ij: Vec<(usize, usize)>,
    vals: Vec<T>,          // (n_src, 2*steps) row-major
    n_steps2: usize,       // 2*steps
    n_src: usize,
}

impl<T: Real> SrcTable<T> {
    fn inject(&self, g: &Grid<T>, ez: &mut [T], col: usize) {
        let half_dt = g.dt / T::from_f64(2.0);
        for s in 0..self.n_src {
            let (i, j) = self.ij[s];
            let v = self.vals[s * self.n_steps2 + col];
            ez[i * g.ny + j] = ez[i * g.ny + j] + v * half_dt / g.eps_at(i, j);
        }
    }
}

macro_rules! make_adi {
    ($name:ident, $ty:ty) => {
        /// Run `steps` ADI steps, recording snapshots at `rec_steps` and probe
        /// time series. See `photonfdtd/adi.py::_run_adi_rust` for the arg map.
        #[pyfunction]
        #[allow(clippy::too_many_arguments)]
        fn $name(
            py: Python<'_>,
            ez: &Bound<'_, PyArray2<$ty>>,
            hx: &Bound<'_, PyArray2<$ty>>,
            hy: &Bound<'_, PyArray2<$ty>>,
            eps: &Bound<'_, PyArray2<$ty>>,
            sig_e: &Bound<'_, PyArray2<$ty>>,
            sig_m: &Bound<'_, PyArray2<$ty>>,
            dx: f64, dy: f64, dt: f64,
            src_ij: &Bound<'_, PyArray2<i64>>,
            src_vals: &Bound<'_, PyArray2<$ty>>,
            n_src: usize,
            rec_steps: &Bound<'_, PyArray1<i64>>,
            snaps: &Bound<'_, PyArray3<$ty>>,
            probe_ij: &Bound<'_, PyArray2<i64>>,
            probe_out: &Bound<'_, PyArray2<$ty>>,
            steps: usize,
            absorber: i32,
        ) -> PyResult<()> {
            let dims = ez.dims();
            let (nx, ny) = (dims[0], dims[1]);
            let g = Grid::<$ty> {
                ez: unsafe { ez.as_slice_mut() }?.as_mut_ptr(),
                hx: unsafe { hx.as_slice_mut() }?.as_mut_ptr(),
                hy: unsafe { hy.as_slice_mut() }?.as_mut_ptr(),
                eps: unsafe { eps.as_slice() }?.as_ptr(),
                sig_e: unsafe { sig_e.as_slice() }?.as_ptr(),
                sig_m: unsafe { sig_m.as_slice() }?.as_ptr(),
                nx, ny,
                dx: <$ty as Real>::from_f64(dx),
                dy: <$ty as Real>::from_f64(dy),
                dt: <$ty as Real>::from_f64(dt),
                absorber: absorber != 0,
            };
            let ij = unsafe { src_ij.as_slice() }?;
            let src = SrcTable::<$ty> {
                ij: (0..n_src).map(|s| (ij[2 * s] as usize, ij[2 * s + 1] as usize)).collect(),
                vals: unsafe { src_vals.as_slice() }?.to_vec(),
                n_steps2: if n_src > 0 { src_vals.dims()[1] } else { 0 },
                n_src,
            };
            let rec = unsafe { rec_steps.as_slice() }?.to_vec();
            let pij = unsafe { probe_ij.as_slice() }?.to_vec();
            let n_probe = probe_ij.dims()[0];
            let snaps_ptr = Ptr(unsafe { snaps.as_slice_mut() }?.as_mut_ptr());
            let probe_ptr = Ptr(unsafe { probe_out.as_slice_mut() }?.as_mut_ptr());

            let ezp = Ptr(g.ez);
            py.allow_threads(move || {
                let mut ez_star = vec![<$ty>::default(); nx * ny];
                let mut rec_idx = 0usize;
                for step in 0..steps {
                    adi_step(g, &mut ez_star, step, &src);
                    if rec_idx < rec.len() && rec[rec_idx] as usize == step {
                        let base = rec_idx * nx * ny;
                        for k in 0..nx * ny {
                            unsafe { snaps_ptr.set(base + k, ezp.get(k)) };
                        }
                        rec_idx += 1;
                    }
                    for p in 0..n_probe {
                        let (pi, pj) = (pij[2 * p] as usize, pij[2 * p + 1] as usize);
                        unsafe { probe_ptr.set(p * steps + step, g.ez_at(pi, pj)) };
                    }
                }
            });
            Ok(())
        }
    };
}

make_adi!(adi_step_range_f64, f64);
make_adi!(adi_step_range_f32, f32);

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(adi_step_range_f64, m)?)?;
    m.add_function(wrap_pyfunction!(adi_step_range_f32, m)?)?;
    Ok(())
}
