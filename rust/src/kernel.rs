//! Generic fused Yee+CPML update kernels, shared by the full-domain parallel
//! stepper, the temporally-blocked (ghost-zone) stepper, and - structurally -
//! by the CUDA kernels, which mirror this code in CUDA C.
//!
//! The H pass updates Hx/Hy/Hz in a single sweep (one read of each E array
//! instead of two across the old per-component loops), and likewise the E
//! pass; per-cell arithmetic is unchanged, so results are bit-identical to
//! the unfused kernels. Everything is generic over f32/f64.
//!
//! CPML psi state is compacted to the PML slabs via per-axis index maps
//! (-1 = bulk); see lib.rs for the layout notes.

/// Scalar type of the stepping arrays.
pub trait Real:
    Copy
    + Send
    + Sync
    + PartialEq
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Neg<Output = Self>
    + core::ops::AddAssign
{
    const ZERO: Self;
    fn from_f64(x: f64) -> Self;
}

impl Real for f64 {
    const ZERO: Self = 0.0;
    #[inline(always)]
    fn from_f64(x: f64) -> f64 {
        x
    }
}

impl Real for f32 {
    const ZERO: Self = 0.0;
    #[inline(always)]
    fn from_f64(x: f64) -> f32 {
        x as f32
    }
}

/// Raw mutable pointer wrapper so rayon threads can write disjoint x-slices
/// of the same array. Safety: every caller writes only cells whose x index
/// belongs to its own row range, and reads either arrays not written in the
/// same pass (E during the H pass and vice versa) or its own disjoint slice.
pub struct Ptr<T>(pub *mut T);
unsafe impl<T: Send> Send for Ptr<T> {}
unsafe impl<T: Sync> Sync for Ptr<T> {}
impl<T> Clone for Ptr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Ptr<T> {}

impl<T: Copy> Ptr<T> {
    #[inline(always)]
    pub unsafe fn at(self, idx: usize) -> T {
        *self.0.add(idx)
    }
    #[inline(always)]
    pub unsafe fn set(self, idx: usize, v: T) {
        *self.0.add(idx) = v;
    }
}

/// Read-only pointer wrapper (coefficients, index maps, source tables).
pub struct CPtr<T>(pub *const T);
unsafe impl<T: Sync> Send for CPtr<T> {}
unsafe impl<T: Sync> Sync for CPtr<T> {}
impl<T> Clone for CPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for CPtr<T> {}

impl<T: Copy> CPtr<T> {
    #[inline(always)]
    pub unsafe fn at(self, idx: usize) -> T {
        *self.0.add(idx)
    }
}

/// Indices of the compact psi arrays in `State::psi`, matching the Python
/// driver's list order.
pub const PSI_EX_Y: usize = 0; // (nx, pe_y, nz)
pub const PSI_EX_Z: usize = 1; // (nx, ny, pe_z)
pub const PSI_EY_Z: usize = 2; // (nx, ny, pe_z)
pub const PSI_EY_X: usize = 3; // (pe_x, ny, nz)
pub const PSI_EZ_X: usize = 4; // (pe_x, ny, nz)
pub const PSI_EZ_Y: usize = 5; // (nx, pe_y, nz)
pub const PSI_HX_Y: usize = 6; // (nx, ph_y, nz)
pub const PSI_HX_Z: usize = 7; // (nx, ny, ph_z)
pub const PSI_HY_Z: usize = 8; // (nx, ny, ph_z)
pub const PSI_HY_X: usize = 9; // (ph_x, ny, nz)
pub const PSI_HZ_X: usize = 10; // (ph_x, ny, nz)
pub const PSI_HZ_Y: usize = 11; // (nx, ph_y, nz)

/// One domain's (or one tile's) stepping state: raw views over the field,
/// psi, coefficient, and map arrays, plus the geometry scalars.
pub struct State<T> {
    pub ex: Ptr<T>,
    pub ey: Ptr<T>,
    pub ez: Ptr<T>,
    pub hx: Ptr<T>,
    pub hy: Ptr<T>,
    pub hz: Ptr<T>,
    pub psi: [Ptr<T>; 12],
    pub b_e: [CPtr<T>; 3],
    pub c_e: [CPtr<T>; 3],
    pub b_h: [CPtr<T>; 3],
    pub c_h: [CPtr<T>; 3],
    /// Compact-index maps (-1 = bulk), E and H staggered variants per axis.
    pub me: [CPtr<i32>; 3],
    pub mh: [CPtr<i32>; 3],
    /// Compact psi extents along y and z (x-compacted psi needs no extent:
    /// it indexes (m*ny + j)*nz + k).
    pub pe_y: usize,
    pub pe_z: usize,
    pub ph_y: usize,
    pub ph_z: usize,
    pub ce: CPtr<T>,
    pub ch: T,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub inv_dx: T,
    pub inv_dy: T,
    pub inv_dz: T,
}

impl<T: Copy> Clone for State<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: Copy> Copy for State<T> {}

impl<T: Real> State<T> {
    #[inline(always)]
    fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        (i * self.ny + j) * self.nz + k
    }

    /// Fused H pass (Hx+Hy+Hz) over rows [i0, i1). Serial; callers
    /// parallelise by handing out disjoint row ranges.
    pub fn update_h_rows(&self, i0: usize, i1: usize) {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let fx = if nx > 1 { nx - 1 } else { 1 };
        let fy = if ny > 1 { ny - 1 } else { 1 };
        let fz = if nz > 1 { nz - 1 } else { 1 };
        for i in i0..i1 {
            let mi = if nx > 1 { unsafe { self.mh[0].at(i) } } else { -1 };
            for j in 0..ny {
                let mj = if ny > 1 { unsafe { self.mh[1].at(j) } } else { -1 };
                for k in 0..nz {
                    unsafe {
                        let id = self.idx(i, j, k);
                        // Hx at (i, j+1/2, k+1/2)
                        if j < fy && k < fz {
                            let mut cy = T::ZERO;
                            if ny > 1 {
                                let d = (self.ez.at(id + nz) - self.ez.at(id)) * self.inv_dy;
                                cy = d;
                                if mj >= 0 {
                                    let pid = (i * self.ph_y + mj as usize) * nz + k;
                                    let ps = self.psi[PSI_HX_Y];
                                    let p = self.b_h[1].at(j) * ps.at(pid)
                                        + self.c_h[1].at(j) * d;
                                    ps.set(pid, p);
                                    cy += p;
                                }
                            }
                            let mut cz = T::ZERO;
                            if nz > 1 {
                                let d = (self.ey.at(id + 1) - self.ey.at(id)) * self.inv_dz;
                                cz = d;
                                let mk = self.mh[2].at(k);
                                if mk >= 0 {
                                    let pid = (i * ny + j) * self.ph_z + mk as usize;
                                    let ps = self.psi[PSI_HX_Z];
                                    let p = self.b_h[2].at(k) * ps.at(pid)
                                        + self.c_h[2].at(k) * d;
                                    ps.set(pid, p);
                                    cz += p;
                                }
                            }
                            self.hx.set(id, self.hx.at(id) + (-self.ch) * (cy - cz));
                        }
                        // Hy at (i+1/2, j, k+1/2)
                        if i < fx && k < fz {
                            let mut cz = T::ZERO;
                            if nz > 1 {
                                let d = (self.ex.at(id + 1) - self.ex.at(id)) * self.inv_dz;
                                cz = d;
                                let mk = self.mh[2].at(k);
                                if mk >= 0 {
                                    let pid = (i * ny + j) * self.ph_z + mk as usize;
                                    let ps = self.psi[PSI_HY_Z];
                                    let p = self.b_h[2].at(k) * ps.at(pid)
                                        + self.c_h[2].at(k) * d;
                                    ps.set(pid, p);
                                    cz += p;
                                }
                            }
                            let mut cx = T::ZERO;
                            if nx > 1 {
                                let d = (self.ez.at(id + ny * nz) - self.ez.at(id)) * self.inv_dx;
                                cx = d;
                                if mi >= 0 {
                                    let pid = (mi as usize * ny + j) * nz + k;
                                    let ps = self.psi[PSI_HY_X];
                                    let p = self.b_h[0].at(i) * ps.at(pid)
                                        + self.c_h[0].at(i) * d;
                                    ps.set(pid, p);
                                    cx += p;
                                }
                            }
                            self.hy.set(id, self.hy.at(id) + (-self.ch) * (cz - cx));
                        }
                        // Hz at (i+1/2, j+1/2, k)
                        if i < fx && j < fy {
                            let mut cx = T::ZERO;
                            if nx > 1 {
                                let d = (self.ey.at(id + ny * nz) - self.ey.at(id)) * self.inv_dx;
                                cx = d;
                                if mi >= 0 {
                                    let pid = (mi as usize * ny + j) * nz + k;
                                    let ps = self.psi[PSI_HZ_X];
                                    let p = self.b_h[0].at(i) * ps.at(pid)
                                        + self.c_h[0].at(i) * d;
                                    ps.set(pid, p);
                                    cx += p;
                                }
                            }
                            let mut cy = T::ZERO;
                            if ny > 1 {
                                let d = (self.ex.at(id + nz) - self.ex.at(id)) * self.inv_dy;
                                cy = d;
                                if mj >= 0 {
                                    let pid = (i * self.ph_y + mj as usize) * nz + k;
                                    let ps = self.psi[PSI_HZ_Y];
                                    let p = self.b_h[1].at(j) * ps.at(pid)
                                        + self.c_h[1].at(j) * d;
                                    ps.set(pid, p);
                                    cy += p;
                                }
                            }
                            self.hz.set(id, self.hz.at(id) + (-self.ch) * (cx - cy));
                        }
                    }
                }
            }
        }
    }

    /// Fused E pass (Ex+Ey+Ez) over rows [i0, i1). Serial.
    pub fn update_e_rows(&self, i0: usize, i1: usize) {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let bx = usize::from(nx > 1);
        let by = usize::from(ny > 1);
        let bz = usize::from(nz > 1);
        for i in i0..i1 {
            let mi = if nx > 1 { unsafe { self.me[0].at(i) } } else { -1 };
            for j in 0..ny {
                let mj = if ny > 1 { unsafe { self.me[1].at(j) } } else { -1 };
                for k in 0..nz {
                    unsafe {
                        let id = self.idx(i, j, k);
                        // Ex at (i+1/2, j, k)
                        if j >= by && k >= bz {
                            let mut cy = T::ZERO;
                            if ny > 1 {
                                let d = (self.hz.at(id) - self.hz.at(id - nz)) * self.inv_dy;
                                cy = d;
                                if mj >= 0 {
                                    let pid = (i * self.pe_y + mj as usize) * nz + k;
                                    let ps = self.psi[PSI_EX_Y];
                                    let p = self.b_e[1].at(j) * ps.at(pid)
                                        + self.c_e[1].at(j) * d;
                                    ps.set(pid, p);
                                    cy += p;
                                }
                            }
                            let mut cz = T::ZERO;
                            if nz > 1 {
                                let d = (self.hy.at(id) - self.hy.at(id - 1)) * self.inv_dz;
                                cz = d;
                                let mk = self.me[2].at(k);
                                if mk >= 0 {
                                    let pid = (i * ny + j) * self.pe_z + mk as usize;
                                    let ps = self.psi[PSI_EX_Z];
                                    let p = self.b_e[2].at(k) * ps.at(pid)
                                        + self.c_e[2].at(k) * d;
                                    ps.set(pid, p);
                                    cz += p;
                                }
                            }
                            self.ex.set(id, self.ex.at(id) + self.ce.at(id) * (cy - cz));
                        }
                        // Ey at (i, j+1/2, k)
                        if i >= bx && k >= bz {
                            let mut cz = T::ZERO;
                            if nz > 1 {
                                let d = (self.hx.at(id) - self.hx.at(id - 1)) * self.inv_dz;
                                cz = d;
                                let mk = self.me[2].at(k);
                                if mk >= 0 {
                                    let pid = (i * ny + j) * self.pe_z + mk as usize;
                                    let ps = self.psi[PSI_EY_Z];
                                    let p = self.b_e[2].at(k) * ps.at(pid)
                                        + self.c_e[2].at(k) * d;
                                    ps.set(pid, p);
                                    cz += p;
                                }
                            }
                            let mut cx = T::ZERO;
                            if nx > 1 {
                                let d = (self.hz.at(id) - self.hz.at(id - ny * nz)) * self.inv_dx;
                                cx = d;
                                if mi >= 0 {
                                    let pid = (mi as usize * ny + j) * nz + k;
                                    let ps = self.psi[PSI_EY_X];
                                    let p = self.b_e[0].at(i) * ps.at(pid)
                                        + self.c_e[0].at(i) * d;
                                    ps.set(pid, p);
                                    cx += p;
                                }
                            }
                            self.ey.set(id, self.ey.at(id) + self.ce.at(id) * (cz - cx));
                        }
                        // Ez at (i, j, k+1/2)
                        if i >= bx && j >= by {
                            let mut cx = T::ZERO;
                            if nx > 1 {
                                let d = (self.hy.at(id) - self.hy.at(id - ny * nz)) * self.inv_dx;
                                cx = d;
                                if mi >= 0 {
                                    let pid = (mi as usize * ny + j) * nz + k;
                                    let ps = self.psi[PSI_EZ_X];
                                    let p = self.b_e[0].at(i) * ps.at(pid)
                                        + self.c_e[0].at(i) * d;
                                    ps.set(pid, p);
                                    cx += p;
                                }
                            }
                            let mut cy = T::ZERO;
                            if ny > 1 {
                                let d = (self.hx.at(id) - self.hx.at(id - nz)) * self.inv_dy;
                                cy = d;
                                if mj >= 0 {
                                    let pid = (i * self.pe_y + mj as usize) * nz + k;
                                    let ps = self.psi[PSI_EZ_Y];
                                    let p = self.b_e[1].at(j) * ps.at(pid)
                                        + self.c_e[1].at(j) * d;
                                    ps.set(pid, p);
                                    cy += p;
                                }
                            }
                            self.ez.set(id, self.ez.at(id) + self.ce.at(id) * (cx - cy));
                        }
                    }
                }
            }
        }
    }
}

/// Precomputed soft-source table for one State (global or tile-local `i`).
pub struct Sources<T> {
    pub comp: Vec<i64>,
    pub idx: Vec<i64>,   // (n_src, 3) flattened, i already local
    pub row: Vec<usize>, // row of each source in the waveform table
    pub vals: CPtr<T>,   // (rows, n_steps_total), indexed by *global* step
    pub n_steps_total: usize,
}

impl<T: Real> Sources<T> {
    /// Add each source's value for `step` onto its cell (H or E family).
    pub fn inject(&self, s: &State<T>, step: usize, e_pass: bool) {
        for n in 0..self.comp.len() {
            let c = self.comp[n];
            if (c < 3) != e_pass {
                continue;
            }
            let (i, j, k) = (
                self.idx[3 * n] as usize,
                self.idx[3 * n + 1] as usize,
                self.idx[3 * n + 2] as usize,
            );
            let id = s.idx(i, j, k);
            let v = unsafe { self.vals.at(self.row[n] * self.n_steps_total + step) };
            let f = match c {
                0 => s.ex,
                1 => s.ey,
                2 => s.ez,
                3 => s.hx,
                4 => s.hy,
                _ => s.hz,
            };
            unsafe { f.set(id, f.at(id) + v) };
        }
    }
}
