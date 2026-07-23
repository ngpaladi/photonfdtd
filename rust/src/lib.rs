//! Rust FDTD stepping kernel for photonfdtd.
//!
//! A port of the fused Numba kernel in `photonfdtd/simulation.py`
//! (`_update_fields_numba`): one complete H+E Yee update for 1D/2D/3D
//! domains with CPML, parallelised over the x axis with rayon. On top of the
//! per-step kernel it also runs the *time loop* itself (`step_range`),
//! injecting precomputed soft-source waveform tables, so a whole run crosses
//! the Python boundary only at monitor-record steps.
//!
//! Memory layout: unlike the Numba/JAX paths (dense full-domain psi, 12
//! extra arrays), the CPML convolutional state here is stored **compacted to
//! the PML slabs** - along each psi array's derivative axis only the cells
//! whose CPML `c` coefficient is nonzero are allocated, addressed through a
//! per-axis index map (`-1` = bulk, no psi). Cells with `c == 0` never
//! develop psi (`p <- b*p + 0` starting from zero), so this is exactly
//! equivalent to the dense update while shrinking the stepping state to the
//! six fields + the `ce` coefficient + thin PML strips - the leanest
//! backend in the package, sized for whole-chip domains.
//!
//! No fastmath: arithmetic is evaluated exactly as written, so results track
//! the NumPy reference to double-precision round-off.

use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

#[cfg(feature = "cuda")]
mod cuda;

/// Raw mutable pointer wrapper so rayon threads can write disjoint x-slices
/// of the same array. Safety: every parallel loop below writes only cells
/// whose x index belongs to its own iteration, and reads either arrays not
/// written in the same pass (E during the H pass and vice versa) or its own
/// disjoint slice (psi).
#[derive(Clone, Copy)]
struct Ptr(*mut f64);
unsafe impl Send for Ptr {}
unsafe impl Sync for Ptr {}

impl Ptr {
    #[inline(always)]
    unsafe fn at(self, idx: usize) -> f64 {
        *self.0.add(idx)
    }
    #[inline(always)]
    unsafe fn set(self, idx: usize, v: f64) {
        *self.0.add(idx) = v;
    }
    #[inline(always)]
    unsafe fn add_assign(self, idx: usize, v: f64) {
        *self.0.add(idx) += v;
    }
}

/// Read-only f64 array pointer (coefficients, source tables).
#[derive(Clone, Copy)]
struct CPtr(*const f64);
unsafe impl Send for CPtr {}
unsafe impl Sync for CPtr {}

impl CPtr {
    #[inline(always)]
    unsafe fn at(self, idx: usize) -> f64 {
        *self.0.add(idx)
    }
}

/// Read-only i32 array pointer (PML compact-index maps; -1 = bulk).
#[derive(Clone, Copy)]
struct MPtr(*const i32);
unsafe impl Send for MPtr {}
unsafe impl Sync for MPtr {}

impl MPtr {
    #[inline(always)]
    unsafe fn at(self, idx: usize) -> i32 {
        *self.0.add(idx)
    }
}

struct Kernel {
    // fields, shape (nx, ny, nz) C-contiguous
    ex: Ptr, ey: Ptr, ez: Ptr, hx: Ptr, hy: Ptr, hz: Ptr,
    // CPML psi state, compacted along the derivative axis (see module docs):
    //   psi_ex_y (nx, pe_y, nz)   psi_ex_z (nx, ny, pe_z)
    //   psi_ey_z (nx, ny, pe_z)   psi_ey_x (pe_x, ny, nz)
    //   psi_ez_x (pe_x, ny, nz)   psi_ez_y (nx, pe_y, nz)
    //   psi_hx_y (nx, ph_y, nz)   psi_hx_z (nx, ny, ph_z)
    //   psi_hy_z (nx, ny, ph_z)   psi_hy_x (ph_x, ny, nz)
    //   psi_hz_x (ph_x, ny, nz)   psi_hz_y (nx, ph_y, nz)
    psi_ex_y: Ptr, psi_ex_z: Ptr, psi_ey_z: Ptr, psi_ey_x: Ptr,
    psi_ez_x: Ptr, psi_ez_y: Ptr, psi_hx_y: Ptr, psi_hx_z: Ptr,
    psi_hy_z: Ptr, psi_hy_x: Ptr, psi_hz_x: Ptr, psi_hz_y: Ptr,
    // CPML 1D coefficients (E and H staggered variants per axis)
    bx_e: CPtr, cx_e: CPtr, by_e: CPtr, cy_e: CPtr, bz_e: CPtr, cz_e: CPtr,
    bx_h: CPtr, cx_h: CPtr, by_h: CPtr, cy_h: CPtr, bz_h: CPtr, cz_h: CPtr,
    // PML compact-index maps per axis (len n_axis; -1 in the bulk) and the
    // compact y/z extents (number of nonzero-c cells along that axis; the
    // x-compacted psi arrays index (m*ny + j)*nz + k and need no extent).
    me_x: MPtr, me_y: MPtr, me_z: MPtr,
    mh_x: MPtr, mh_y: MPtr, mh_z: MPtr,
    pe_y: usize, pe_z: usize,
    ph_y: usize, ph_z: usize,
    // update coefficients
    ce_field: CPtr,      // dt / (eps_r * EPS_0), per cell
    ch_field: f64,       // dt / MU_0
    // geometry
    nx: usize, ny: usize, nz: usize,
    inv_dx: f64, inv_dy: f64, inv_dz: f64,
}

impl Kernel {
    #[inline(always)]
    fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        (i * self.ny + j) * self.nz + k
    }

    /// H half of the Yee update (forward differences of E).
    fn update_h(&self) {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let fx = if nx > 1 { nx - 1 } else { 1 };
        let fy = if ny > 1 { ny - 1 } else { 1 };
        let fz = if nz > 1 { nz - 1 } else { 1 };

        // Hx at (i, j+1/2, k+1/2): full i, forward j (dEz/dy), forward k (dEy/dz)
        (0..nx).into_par_iter().for_each(|i| unsafe {
            for j in 0..fy {
                let mj = if ny > 1 { self.mh_y.at(j) } else { -1 };
                for k in 0..fz {
                    let id = self.idx(i, j, k);
                    let mut cy = 0.0;
                    if ny > 1 {
                        let d = (self.ez.at(self.idx(i, j + 1, k)) - self.ez.at(id)) * self.inv_dy;
                        cy = d;
                        if mj >= 0 {
                            let pid = (i * self.ph_y + mj as usize) * nz + k;
                            let p = self.by_h.at(j) * self.psi_hx_y.at(pid) + self.cy_h.at(j) * d;
                            self.psi_hx_y.set(pid, p);
                            cy += p;
                        }
                    }
                    let mut cz = 0.0;
                    if nz > 1 {
                        let d = (self.ey.at(self.idx(i, j, k + 1)) - self.ey.at(id)) * self.inv_dz;
                        cz = d;
                        let mk = self.mh_z.at(k);
                        if mk >= 0 {
                            let pid = (i * ny + j) * self.ph_z + mk as usize;
                            let p = self.bz_h.at(k) * self.psi_hx_z.at(pid) + self.cz_h.at(k) * d;
                            self.psi_hx_z.set(pid, p);
                            cz += p;
                        }
                    }
                    self.hx.add_assign(id, -self.ch_field * (cy - cz));
                }
            }
        });
        // Hy at (i+1/2, j, k+1/2): forward i (dEz/dx), full j, forward k (dEx/dz)
        (0..fx).into_par_iter().for_each(|i| unsafe {
            let mi = if nx > 1 { self.mh_x.at(i) } else { -1 };
            for j in 0..ny {
                for k in 0..fz {
                    let id = self.idx(i, j, k);
                    let mut cz = 0.0;
                    if nz > 1 {
                        let d = (self.ex.at(self.idx(i, j, k + 1)) - self.ex.at(id)) * self.inv_dz;
                        cz = d;
                        let mk = self.mh_z.at(k);
                        if mk >= 0 {
                            let pid = (i * ny + j) * self.ph_z + mk as usize;
                            let p = self.bz_h.at(k) * self.psi_hy_z.at(pid) + self.cz_h.at(k) * d;
                            self.psi_hy_z.set(pid, p);
                            cz += p;
                        }
                    }
                    let mut cx = 0.0;
                    if nx > 1 {
                        let d = (self.ez.at(self.idx(i + 1, j, k)) - self.ez.at(id)) * self.inv_dx;
                        cx = d;
                        if mi >= 0 {
                            let pid = (mi as usize * ny + j) * nz + k;
                            let p = self.bx_h.at(i) * self.psi_hy_x.at(pid) + self.cx_h.at(i) * d;
                            self.psi_hy_x.set(pid, p);
                            cx += p;
                        }
                    }
                    self.hy.add_assign(id, -self.ch_field * (cz - cx));
                }
            }
        });
        // Hz at (i+1/2, j+1/2, k): forward i (dEy/dx), forward j (dEx/dy), full k
        (0..fx).into_par_iter().for_each(|i| unsafe {
            let mi = if nx > 1 { self.mh_x.at(i) } else { -1 };
            for j in 0..fy {
                let mj = if ny > 1 { self.mh_y.at(j) } else { -1 };
                for k in 0..nz {
                    let id = self.idx(i, j, k);
                    let mut cx = 0.0;
                    if nx > 1 {
                        let d = (self.ey.at(self.idx(i + 1, j, k)) - self.ey.at(id)) * self.inv_dx;
                        cx = d;
                        if mi >= 0 {
                            let pid = (mi as usize * ny + j) * nz + k;
                            let p = self.bx_h.at(i) * self.psi_hz_x.at(pid) + self.cx_h.at(i) * d;
                            self.psi_hz_x.set(pid, p);
                            cx += p;
                        }
                    }
                    let mut cy = 0.0;
                    if ny > 1 {
                        let d = (self.ex.at(self.idx(i, j + 1, k)) - self.ex.at(id)) * self.inv_dy;
                        cy = d;
                        if mj >= 0 {
                            let pid = (i * self.ph_y + mj as usize) * nz + k;
                            let p = self.by_h.at(j) * self.psi_hz_y.at(pid) + self.cy_h.at(j) * d;
                            self.psi_hz_y.set(pid, p);
                            cy += p;
                        }
                    }
                    self.hz.add_assign(id, -self.ch_field * (cx - cy));
                }
            }
        });
    }

    /// E half of the Yee update (backward differences of H).
    fn update_e(&self) {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let bx = usize::from(nx > 1);
        let by = usize::from(ny > 1);
        let bz = usize::from(nz > 1);

        // Ex at (i+1/2, j, k): full i, backward j (dHz/dy), backward k (dHy/dz)
        (0..nx).into_par_iter().for_each(|i| unsafe {
            for j in by..ny {
                let mj = if ny > 1 { self.me_y.at(j) } else { -1 };
                for k in bz..nz {
                    let id = self.idx(i, j, k);
                    let mut cy = 0.0;
                    if ny > 1 {
                        let d = (self.hz.at(id) - self.hz.at(self.idx(i, j - 1, k))) * self.inv_dy;
                        cy = d;
                        if mj >= 0 {
                            let pid = (i * self.pe_y + mj as usize) * nz + k;
                            let p = self.by_e.at(j) * self.psi_ex_y.at(pid) + self.cy_e.at(j) * d;
                            self.psi_ex_y.set(pid, p);
                            cy += p;
                        }
                    }
                    let mut cz = 0.0;
                    if nz > 1 {
                        let d = (self.hy.at(id) - self.hy.at(self.idx(i, j, k - 1))) * self.inv_dz;
                        cz = d;
                        let mk = self.me_z.at(k);
                        if mk >= 0 {
                            let pid = (i * ny + j) * self.pe_z + mk as usize;
                            let p = self.bz_e.at(k) * self.psi_ex_z.at(pid) + self.cz_e.at(k) * d;
                            self.psi_ex_z.set(pid, p);
                            cz += p;
                        }
                    }
                    self.ex.add_assign(id, self.ce_field.at(id) * (cy - cz));
                }
            }
        });
        // Ey at (i, j+1/2, k): backward i (dHz/dx), full j, backward k (dHx/dz)
        (bx..nx).into_par_iter().for_each(|i| unsafe {
            let mi = if nx > 1 { self.me_x.at(i) } else { -1 };
            for j in 0..ny {
                for k in bz..nz {
                    let id = self.idx(i, j, k);
                    let mut cz = 0.0;
                    if nz > 1 {
                        let d = (self.hx.at(id) - self.hx.at(self.idx(i, j, k - 1))) * self.inv_dz;
                        cz = d;
                        let mk = self.me_z.at(k);
                        if mk >= 0 {
                            let pid = (i * ny + j) * self.pe_z + mk as usize;
                            let p = self.bz_e.at(k) * self.psi_ey_z.at(pid) + self.cz_e.at(k) * d;
                            self.psi_ey_z.set(pid, p);
                            cz += p;
                        }
                    }
                    let mut cx = 0.0;
                    if nx > 1 {
                        let d = (self.hz.at(id) - self.hz.at(self.idx(i - 1, j, k))) * self.inv_dx;
                        cx = d;
                        if mi >= 0 {
                            let pid = (mi as usize * ny + j) * nz + k;
                            let p = self.bx_e.at(i) * self.psi_ey_x.at(pid) + self.cx_e.at(i) * d;
                            self.psi_ey_x.set(pid, p);
                            cx += p;
                        }
                    }
                    self.ey.add_assign(id, self.ce_field.at(id) * (cz - cx));
                }
            }
        });
        // Ez at (i, j, k+1/2): backward i (dHy/dx), backward j (dHx/dy), full k
        (bx..nx).into_par_iter().for_each(|i| unsafe {
            let mi = if nx > 1 { self.me_x.at(i) } else { -1 };
            for j in by..ny {
                let mj = if ny > 1 { self.me_y.at(j) } else { -1 };
                for k in 0..nz {
                    let id = self.idx(i, j, k);
                    let mut cx = 0.0;
                    if nx > 1 {
                        let d = (self.hy.at(id) - self.hy.at(self.idx(i - 1, j, k))) * self.inv_dx;
                        cx = d;
                        if mi >= 0 {
                            let pid = (mi as usize * ny + j) * nz + k;
                            let p = self.bx_e.at(i) * self.psi_ez_x.at(pid) + self.cx_e.at(i) * d;
                            self.psi_ez_x.set(pid, p);
                            cx += p;
                        }
                    }
                    let mut cy = 0.0;
                    if ny > 1 {
                        let d = (self.hx.at(id) - self.hx.at(self.idx(i, j - 1, k))) * self.inv_dy;
                        cy = d;
                        if mj >= 0 {
                            let pid = (i * self.pe_y + mj as usize) * nz + k;
                            let p = self.by_e.at(j) * self.psi_ez_y.at(pid) + self.cy_e.at(j) * d;
                            self.psi_ez_y.set(pid, p);
                            cy += p;
                        }
                    }
                    self.ez.add_assign(id, self.ce_field.at(id) * (cx - cy));
                }
            }
        });
    }

    /// One full step: H update, H sources, E update, E sources - the exact
    /// ordering of `Simulation.run`.
    #[inline]
    fn step(
        &self,
        step: usize,
        n_src: usize,
        comp: &[i64],
        idx: &[i64],
        vals: CPtr,
        n_steps_total: usize,
    ) {
        self.update_h();
        self.inject(step, n_src, comp, idx, vals, n_steps_total, false);
        self.update_e();
        self.inject(step, n_src, comp, idx, vals, n_steps_total, true);
    }

    #[inline]
    fn inject(
        &self,
        step: usize,
        n_src: usize,
        comp: &[i64],
        idx: &[i64],
        vals: CPtr,
        n_steps_total: usize,
        e_pass: bool,
    ) {
        for s in 0..n_src {
            let c = comp[s];
            let is_e = c < 3;
            if is_e != e_pass {
                continue;
            }
            let (i, j, k) = (idx[3 * s] as usize, idx[3 * s + 1] as usize, idx[3 * s + 2] as usize);
            let id = self.idx(i, j, k);
            let v = unsafe { vals.at(s * n_steps_total + step) };
            let f = match c {
                0 => self.ex, 1 => self.ey, 2 => self.ez,
                3 => self.hx, 4 => self.hy, _ => self.hz,
            };
            unsafe { f.add_assign(id, v) };
        }
    }
}

/// Advance the simulation by `n_sub` steps starting at global step `step0`.
///
/// Soft sources are precomputed waveform tables: `src_comp[s]` in 0..6 codes
/// (Ex,Ey,Ez,Hx,Hy,Hz), `src_idx[s] = (i,j,k)`, `src_vals[s, step]` the
/// additive value at `step`. `maps_e` / `maps_h` are the per-axis PML
/// compact-index maps (i32, -1 in the bulk) matching the compact psi array
/// shapes (see `Kernel`). The caller records monitors between calls (a chunk
/// ends *after* the last requested step, matching the NumPy path's
/// record-at-end-of-step timing).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn step_range(
    py: Python<'_>,
    ex: &Bound<'_, PyArray3<f64>>, ey: &Bound<'_, PyArray3<f64>>, ez: &Bound<'_, PyArray3<f64>>,
    hx: &Bound<'_, PyArray3<f64>>, hy: &Bound<'_, PyArray3<f64>>, hz: &Bound<'_, PyArray3<f64>>,
    psi_ex_y: &Bound<'_, PyArray3<f64>>, psi_ex_z: &Bound<'_, PyArray3<f64>>,
    psi_ey_z: &Bound<'_, PyArray3<f64>>, psi_ey_x: &Bound<'_, PyArray3<f64>>,
    psi_ez_x: &Bound<'_, PyArray3<f64>>, psi_ez_y: &Bound<'_, PyArray3<f64>>,
    psi_hx_y: &Bound<'_, PyArray3<f64>>, psi_hx_z: &Bound<'_, PyArray3<f64>>,
    psi_hy_z: &Bound<'_, PyArray3<f64>>, psi_hy_x: &Bound<'_, PyArray3<f64>>,
    psi_hz_x: &Bound<'_, PyArray3<f64>>, psi_hz_y: &Bound<'_, PyArray3<f64>>,
    b_e: Vec<Py<PyArray1<f64>>>, c_e: Vec<Py<PyArray1<f64>>>,
    b_h: Vec<Py<PyArray1<f64>>>, c_h: Vec<Py<PyArray1<f64>>>,
    maps_e: Vec<Py<PyArray1<i32>>>, maps_h: Vec<Py<PyArray1<i32>>>,
    ce_field: &Bound<'_, PyArray3<f64>>,
    ch_field: f64,
    dx: f64, dy: f64, dz: f64,
    src_comp: &Bound<'_, PyArray1<i64>>,
    src_idx: &Bound<'_, PyArray2<i64>>,
    src_vals: &Bound<'_, PyArray2<f64>>,
    step0: usize,
    n_sub: usize,
) -> PyResult<()> {
    let dims = ex.dims();
    let (nx, ny, nz) = (dims[0], dims[1], dims[2]);

    // Raw pointers to the (C-contiguous, caller-guaranteed) array buffers.
    // The GIL is released for the compute; Python-side code never touches
    // these arrays while step_range runs (single caller, sequential).
    macro_rules! p3 {
        ($a:expr) => {{
            let ro = unsafe { $a.as_slice_mut() }?;
            Ptr(ro.as_mut_ptr())
        }};
    }
    macro_rules! c1 {
        ($v:expr, $i:expr) => {
            CPtr(unsafe { $v[$i].bind(py).as_slice() }?.as_ptr())
        };
    }
    macro_rules! m1 {
        ($v:expr, $i:expr) => {
            MPtr(unsafe { $v[$i].bind(py).as_slice() }?.as_ptr())
        };
    }
    let kern = Kernel {
        ex: p3!(ex), ey: p3!(ey), ez: p3!(ez),
        hx: p3!(hx), hy: p3!(hy), hz: p3!(hz),
        psi_ex_y: p3!(psi_ex_y), psi_ex_z: p3!(psi_ex_z),
        psi_ey_z: p3!(psi_ey_z), psi_ey_x: p3!(psi_ey_x),
        psi_ez_x: p3!(psi_ez_x), psi_ez_y: p3!(psi_ez_y),
        psi_hx_y: p3!(psi_hx_y), psi_hx_z: p3!(psi_hx_z),
        psi_hy_z: p3!(psi_hy_z), psi_hy_x: p3!(psi_hy_x),
        psi_hz_x: p3!(psi_hz_x), psi_hz_y: p3!(psi_hz_y),
        bx_e: c1!(b_e, 0), cx_e: c1!(c_e, 0),
        by_e: c1!(b_e, 1), cy_e: c1!(c_e, 1),
        bz_e: c1!(b_e, 2), cz_e: c1!(c_e, 2),
        bx_h: c1!(b_h, 0), cx_h: c1!(c_h, 0),
        by_h: c1!(b_h, 1), cy_h: c1!(c_h, 1),
        bz_h: c1!(b_h, 2), cz_h: c1!(c_h, 2),
        me_x: m1!(maps_e, 0), me_y: m1!(maps_e, 1), me_z: m1!(maps_e, 2),
        mh_x: m1!(maps_h, 0), mh_y: m1!(maps_h, 1), mh_z: m1!(maps_h, 2),
        pe_y: psi_ex_y.dims()[1], pe_z: psi_ex_z.dims()[2],
        ph_y: psi_hx_y.dims()[1], ph_z: psi_hx_z.dims()[2],
        ce_field: CPtr(unsafe { ce_field.as_slice() }?.as_ptr()),
        ch_field,
        nx, ny, nz,
        inv_dx: 1.0 / dx, inv_dy: 1.0 / dy, inv_dz: 1.0 / dz,
    };

    let n_src = src_comp.dims()[0];
    let comp = unsafe { src_comp.as_slice() }?.to_vec();
    let idx = unsafe { src_idx.as_slice() }?.to_vec();     // (n_src, 3) flat
    let vals_arr = unsafe { src_vals.as_slice() }?;
    let n_steps_total = src_vals.dims()[1];
    let vals = CPtr(vals_arr.as_ptr());

    py.allow_threads(|| {
        for step in step0..step0 + n_sub {
            kern.step(step, n_src, &comp, &idx, vals, n_steps_total);
        }
    });
    Ok(())
}

#[pymodule]
fn _photonfdtd_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(step_range, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    #[cfg(feature = "cuda")]
    m.add_class::<cuda::CudaStepper>()?;
    m.add("CUDA_BUILT", cfg!(feature = "cuda"))?;
    Ok(())
}
