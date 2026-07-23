//! Temporal (ghost-zone) blocking for the CPU stepper.
//!
//! The RAM-bound regime streams every array through memory once per step.
//! This module instead advances the domain in x-slab tiles, `t_block` steps
//! per visit: each tile copies its slab plus a halo into a thread-local
//! buffer, evolves it T steps entirely in cache, and writes the core back.
//! Per T steps the global arrays are touched ~twice instead of 2T times.
//!
//! Correctness: a tile buffer's edge rows go wrong from the first local step
//! (the kernel's bounds guards treat them as domain edges), and the error
//! propagates inward at one row per step - the light cone. With a halo of
//! `T + 2` rows the written-back core is untouched by it, so results are
//! bit-identical to plain stepping (same per-cell arithmetic, same order).
//! The parity tests run this path against the NumPy reference.
//!
//! Concurrency: tiles only ever read their *own* core rows from the global
//! arrays - halo rows are read from per-boundary snapshot strips taken
//! before the tile pass - so tiles are fully independent and writebacks
//! (each tile writes only its own core) race with nothing.

use rayon::prelude::*;

use crate::kernel::{CPtr, Ptr, Real, Sources, State};

/// Per-array view for slab copy/writeback: `row_len` bytes... elements per
/// x row, and for x-compacted psi arrays the global map that says which x
/// rows exist.
struct ArrView<T> {
    ptr: Ptr<T>,
    row_len: usize,
    x_map: Option<CPtr<i32>>,
}

fn views<T: Real>(s: &State<T>) -> [ArrView<T>; 18] {
    let (ny, nz) = (s.ny, s.nz);
    let f = |ptr: Ptr<T>| ArrView { ptr, row_len: ny * nz, x_map: None };
    [
        f(s.ex), f(s.ey), f(s.ez), f(s.hx), f(s.hy), f(s.hz),
        ArrView { ptr: s.psi[0], row_len: s.pe_y * nz, x_map: None },
        ArrView { ptr: s.psi[1], row_len: ny * s.pe_z, x_map: None },
        ArrView { ptr: s.psi[2], row_len: ny * s.pe_z, x_map: None },
        ArrView { ptr: s.psi[3], row_len: ny * nz, x_map: Some(s.me[0]) },
        ArrView { ptr: s.psi[4], row_len: ny * nz, x_map: Some(s.me[0]) },
        ArrView { ptr: s.psi[5], row_len: s.pe_y * nz, x_map: None },
        ArrView { ptr: s.psi[6], row_len: s.ph_y * nz, x_map: None },
        ArrView { ptr: s.psi[7], row_len: ny * s.ph_z, x_map: None },
        ArrView { ptr: s.psi[8], row_len: ny * s.ph_z, x_map: None },
        ArrView { ptr: s.psi[9], row_len: ny * nz, x_map: Some(s.mh[0]) },
        ArrView { ptr: s.psi[10], row_len: ny * nz, x_map: Some(s.mh[0]) },
        ArrView { ptr: s.psi[11], row_len: s.ph_y * nz, x_map: None },
    ]
}

/// Copy global row `r` of `v` into `dst` (one row). Returns false if the
/// row does not exist (x-compacted array, bulk row).
unsafe fn copy_row_out<T: Real>(v: &ArrView<T>, r: usize, dst: &mut [T]) -> bool {
    let off = match v.x_map {
        None => r * v.row_len,
        Some(m) => {
            let mm = m.at(r);
            if mm < 0 {
                return false;
            }
            mm as usize * v.row_len
        }
    };
    core::ptr::copy_nonoverlapping(v.ptr.0.add(off), dst.as_mut_ptr(), v.row_len);
    true
}

/// Write one row back into global row `r` of `v`.
unsafe fn copy_row_in<T: Real>(v: &ArrView<T>, r: usize, src: &[T]) {
    let off = match v.x_map {
        None => r * v.row_len,
        Some(m) => {
            let mm = m.at(r);
            if mm < 0 {
                return;
            }
            mm as usize * v.row_len
        }
    };
    core::ptr::copy_nonoverlapping(src.as_ptr(), v.ptr.0.add(off), v.row_len);
}

struct Tile {
    a: usize,   // core start
    b: usize,   // core end
    lo: usize,  // buffer start (a - halo, clamped)
    hi: usize,  // buffer end (b + halo, clamped)
}

/// Snapshot strip around one internal tile boundary: rows [lo, hi) of every
/// evolving array, stored per array as consecutive existing rows.
struct Strip<T> {
    lo: usize,
    rows: Vec<Vec<T>>, // 18 arrays x (existing rows * row_len)
}

fn take_strip<T: Real>(v: &[ArrView<T>; 18], lo: usize, hi: usize) -> Strip<T> {
    let mut rows = Vec::with_capacity(18);
    for view in v.iter() {
        let mut buf: Vec<T> = Vec::new();
        let mut tmp = vec![T::ZERO; view.row_len];
        for r in lo..hi {
            if unsafe { copy_row_out(view, r, &mut tmp) } {
                buf.extend_from_slice(&tmp);
            }
        }
        rows.push(buf);
    }
    let _ = hi;
    Strip { lo, rows }
}

impl<T: Real> Strip<T> {
    /// Row `r` of array `ai` inside this strip (must exist).
    fn row(&self, v: &ArrView<T>, ai: usize, r: usize) -> &[T] {
        // Position = number of existing rows in [self.lo, r).
        let pos = match v.x_map {
            None => r - self.lo,
            Some(m) => (self.lo..r)
                .filter(|&q| unsafe { m.at(q) } >= 0)
                .count(),
        };
        &self.rows[ai][pos * v.row_len..(pos + 1) * v.row_len]
    }
}

/// Advance `n_sub` steps from `step0` with ghost-zone temporal blocking.
/// Falls back to the caller for plain stepping when blocking cannot help
/// (the caller checks tile geometry first via `worthwhile`).
#[allow(clippy::too_many_arguments)]
pub fn run_blocked<T: Real>(
    g: &State<T>,
    srcs: &Sources<T>,
    step0: usize,
    n_sub: usize,
    t_block: usize,
    tile_rows: usize,
) {
    let mut done = 0;
    while done < n_sub {
        let t = t_block.min(n_sub - done);
        let halo = t + 2;
        if t < 2 || g.nx < 2 * (tile_rows.max(2 * halo)) {
            // Too small to tile: plain parallel steps for the remainder.
            for s in step0 + done..step0 + n_sub {
                step_parallel(g, srcs, s);
            }
            return;
        }
        run_chunk(g, srcs, step0 + done, t, halo, tile_rows.max(2 * halo));
        done += t;
    }
}

/// One plain (unblocked) step, rayon-parallel over rows. Used by the
/// fallback above and exported for lib.rs.
pub fn step_parallel<T: Real>(g: &State<T>, srcs: &Sources<T>, step: usize) {
    par_rows(g.nx, |i0, i1| g.update_h_rows(i0, i1));
    srcs.inject(g, step, false);
    par_rows(g.nx, |i0, i1| g.update_e_rows(i0, i1));
    srcs.inject(g, step, true);
}

fn par_rows(nx: usize, f: impl Fn(usize, usize) + Sync) {
    // Chunk rows so each rayon task is substantial (few tasks per thread).
    let chunk = (nx / (rayon::current_num_threads() * 4)).max(1);
    (0..nx.div_ceil(chunk)).into_par_iter().for_each(|c| {
        let i0 = c * chunk;
        f(i0, (i0 + chunk).min(nx));
    });
}

fn run_chunk<T: Real>(
    g: &State<T>,
    srcs: &Sources<T>,
    chunk_start: usize,
    t: usize,
    halo: usize,
    tile_rows: usize,
) {
    let nx = g.nx;
    let v = views(g);

    let mut tiles = Vec::new();
    let mut a = 0;
    while a < nx {
        let b = (a + tile_rows).min(nx);
        tiles.push(Tile { a, b, lo: a.saturating_sub(halo), hi: (b + halo).min(nx) });
        a = b;
    }

    // Snapshot strips around each internal boundary, before any writeback.
    let strips: Vec<Strip<T>> = tiles[1..]
        .par_iter()
        .map(|tl| take_strip(&v, tl.a.saturating_sub(halo), (tl.a + halo).min(nx)))
        .collect();

    tiles.par_iter().enumerate().for_each(|(ti, tile)| {
        evolve_tile(g, srcs, &v, tile,
                    if ti > 0 { Some(&strips[ti - 1]) } else { None },
                    if ti + 1 < tiles.len() { Some(&strips[ti]) } else { None },
                    chunk_start, t);
    });
}

#[allow(clippy::too_many_arguments)]
fn evolve_tile<T: Real>(
    g: &State<T>,
    srcs: &Sources<T>,
    v: &[ArrView<T>; 18],
    tile: &Tile,
    left: Option<&Strip<T>>,
    right: Option<&Strip<T>>,
    chunk_start: usize,
    t: usize,
) {
    let (ny, nz) = (g.ny, g.nz);
    let (lo, hi, a, b) = (tile.lo, tile.hi, tile.a, tile.b);
    let lnx = hi - lo;

    // Local x maps + compact extent for the x-compacted psi arrays.
    let mut lme_x = vec![-1i32; lnx];
    let mut lmh_x = vec![-1i32; lnx];
    let (mut n_e, mut n_h) = (0i32, 0i32);
    for il in 0..lnx {
        if unsafe { g.me[0].at(lo + il) } >= 0 {
            lme_x[il] = n_e;
            n_e += 1;
        }
        if unsafe { g.mh[0].at(lo + il) } >= 0 {
            lmh_x[il] = n_h;
            n_h += 1;
        }
    }

    // Local buffers for the 18 evolving arrays.
    let local_len = |ai: usize| -> usize {
        match v[ai].x_map {
            None => lnx * v[ai].row_len,
            Some(_) => {
                let n = if ai == 3 || ai == 4 { n_e } else { n_h };
                (n.max(1) as usize) * v[ai].row_len
            }
        }
    };
    let mut bufs: Vec<Vec<T>> = (0..18).map(|ai| vec![T::ZERO; local_len(ai)]).collect();

    // Fill: core rows from global (only this tile touches them), halo rows
    // from the boundary snapshot strips.
    for ai in 0..18 {
        let view = &v[ai];
        let rl = view.row_len;
        let mut lrow = 0usize; // next local row index (compact for x-mapped)
        for r in lo..hi {
            let exists = match view.x_map {
                None => true,
                Some(m) => (unsafe { m.at(r) }) >= 0,
            };
            if !exists {
                continue;
            }
            let dst = &mut bufs[ai][lrow * rl..(lrow + 1) * rl];
            if r >= a && r < b {
                unsafe { copy_row_out(view, r, dst) };
            } else if r < a {
                dst.copy_from_slice(left.unwrap().row(view, ai, r));
            } else {
                dst.copy_from_slice(right.unwrap().row(view, ai, r));
            }
            lrow += 1;
        }
    }

    // Local 1D x coefficient slices.
    let bx_e: Vec<T> = (lo..hi).map(|r| unsafe { g.b_e[0].at(r) }).collect();
    let cx_e: Vec<T> = (lo..hi).map(|r| unsafe { g.c_e[0].at(r) }).collect();
    let bx_h: Vec<T> = (lo..hi).map(|r| unsafe { g.b_h[0].at(r) }).collect();
    let cx_h: Vec<T> = (lo..hi).map(|r| unsafe { g.c_h[0].at(r) }).collect();

    // Tile-local sources (any source inside the buffer, halo included -
    // its influence on the core is inside the light cone).
    let mut ls = Sources {
        comp: Vec::new(), idx: Vec::new(), row: Vec::new(),
        vals: srcs.vals, n_steps_total: srcs.n_steps_total,
    };
    for n in 0..srcs.comp.len() {
        let gi = srcs.idx[3 * n] as usize;
        if gi >= lo && gi < hi {
            ls.comp.push(srcs.comp[n]);
            ls.idx.extend_from_slice(&[
                (gi - lo) as i64, srcs.idx[3 * n + 1], srcs.idx[3 * n + 2],
            ]);
            ls.row.push(srcs.row[n]);
        }
    }

    let mut psi_ptrs = [Ptr(core::ptr::null_mut()); 12];
    for pi in 0..12 {
        psi_ptrs[pi] = Ptr(bufs[6 + pi].as_mut_ptr());
    }
    let local = State {
        ex: Ptr(bufs[0].as_mut_ptr()), ey: Ptr(bufs[1].as_mut_ptr()),
        ez: Ptr(bufs[2].as_mut_ptr()), hx: Ptr(bufs[3].as_mut_ptr()),
        hy: Ptr(bufs[4].as_mut_ptr()), hz: Ptr(bufs[5].as_mut_ptr()),
        psi: psi_ptrs,
        b_e: [CPtr(bx_e.as_ptr()), g.b_e[1], g.b_e[2]],
        c_e: [CPtr(cx_e.as_ptr()), g.c_e[1], g.c_e[2]],
        b_h: [CPtr(bx_h.as_ptr()), g.b_h[1], g.b_h[2]],
        c_h: [CPtr(cx_h.as_ptr()), g.c_h[1], g.c_h[2]],
        me: [CPtr(lme_x.as_ptr()), g.me[1], g.me[2]],
        mh: [CPtr(lmh_x.as_ptr()), g.mh[1], g.mh[2]],
        pe_y: g.pe_y, pe_z: g.pe_z, ph_y: g.ph_y, ph_z: g.ph_z,
        ce: CPtr(unsafe { g.ce.0.add(lo * ny * nz) }),
        ch: g.ch,
        nx: lnx, ny, nz,
        inv_dx: g.inv_dx, inv_dy: g.inv_dy, inv_dz: g.inv_dz,
    };

    // Evolve T steps entirely inside the local buffer.
    for step in chunk_start..chunk_start + t {
        local.update_h_rows(0, lnx);
        ls.inject(&local, step, false);
        local.update_e_rows(0, lnx);
        ls.inject(&local, step, true);
    }

    // Write the core rows back.
    for ai in 0..18 {
        let view = &v[ai];
        let rl = view.row_len;
        let mut lrow = 0usize;
        for r in lo..hi {
            let exists = match view.x_map {
                None => true,
                Some(m) => (unsafe { m.at(r) }) >= 0,
            };
            if !exists {
                continue;
            }
            if r >= a && r < b {
                let src = &bufs[ai][lrow * rl..(lrow + 1) * rl];
                unsafe { copy_row_in(view, r, src) };
            }
            lrow += 1;
        }
    }
}
