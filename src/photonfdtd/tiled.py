"""Disk-tiled forward stepping, for grids whose fields do not fit in memory.

The in-memory engine (:mod:`photonfdtd.jaxbackend`) keeps every field, CPML
``psi`` and coefficient array resident on the device. :func:`run_tiled` keeps
them in memory-mapped files on disk instead, splits the domain into slabs of
``tile_cells`` planes along x, and advances one slab at a time on the JAX
device. Peak device memory is set by the slab, not the grid. The OS page cache
gives a free middle tier: a grid that fits in host RAM but not on the GPU is
streamed from RAM rather than from the disk.

Temporal blocking
-----------------
Each slab is loaded with a halo of ``h = steps_per_tile + 1`` planes on each
side and advanced ``steps_per_tile`` (K) steps in one compiled ``lax.scan``
before its interior is written back. The slab's edge planes lack their outside
neighbours, so they go wrong, and the error spreads inward by at most one plane
per step. After K steps the interior, ``h`` planes in from each edge, is still
exact, and so is every monitor sample taken there along the way. Disk traffic
is therefore one read and one write of the grid per K steps, rather than per
step, at the cost of recomputing the halos.

Slabs are processed left to right in one buffer: the old values of the ``h``
planes left of the write frontier, which the next slab's halo needs, are kept
in RAM, so the update is equivalent to a synchronous one. The x axis is padded
by ``h`` zero planes on each side (and rounded up to whole slabs). A per-plane
update mask reproduces the in-memory engine's boundary planes exactly, so the
results match :func:`~photonfdtd.jaxbackend.run_jax` to floating-point
reassociation (about 1e-12 relative in float64).

Scope
-----
Forward runs only (no adjoint), with one precision for all stepping arrays.
Supported: point and expanded soft sources, ``ChargedParticle`` charges,
subpixel smoothing, ``FieldMonitor``, ``DFTMonitor`` and ``FluxMonitor``.
Dispersive media raise ``NotImplementedError``. ``eps_r`` is still stamped
in host RAM (one grid-sized array, plus three more while the subpixel tensor
is written out, against the ~20 the in-memory run holds), and monitor outputs
are held in RAM, so record planes or downsampled volumes on a large grid.
"""
from __future__ import annotations
import functools
import math
import os
import shutil
import tempfile
from typing import Optional
import numpy as np

from .constants import EPS_0
from .jaxbackend import (_build_static, _monitor_plan, _particle_plan,
                         _validate, _enable_x64_if_needed)

_FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_TANG = {0: ("Ey", "Ez", "Hy", "Hz"),
         1: ("Ez", "Ex", "Hz", "Hx"),
         2: ("Ex", "Ey", "Hx", "Hy")}


def _half_offset(comp, ax):
    """True if ``comp`` sits half a cell off a plane normal to ``ax``."""
    a = "xyz".index(comp[1])
    return (a == ax) == (comp[0] == "E")


def disk_bytes(sim, tile_cells: int, steps_per_tile: int) -> int:
    """Bytes of scratch files :func:`run_tiled` writes for ``sim``."""
    static = _build_static(sim)
    nx, ny, nz = static["shape"]
    h = steps_per_tile + 1
    px = int(math.ceil(nx / tile_cells)) * tile_cells + 2 * h
    n_psi = sum(len(c["terms"]) for c in static["comps"])
    n_ce = 4 if sim.subpixel else 1
    return (6 + n_psi + n_ce) * px * ny * nz * np.dtype(static["npdt"]).itemsize


def tile_bytes(sim, tile_cells: int, steps_per_tile: int) -> int:
    """Estimated peak device bytes of one slab (fields, psi, coefficients,
    double-buffered across the step, plus monitor buffers)."""
    static = _build_static(sim)
    nx, ny, nz = static["shape"]
    L = tile_cells + 2 * (steps_per_tile + 1)
    item = np.dtype(static["npdt"]).itemsize
    n_psi = sum(len(c["terms"]) for c in static["comps"])
    n_ce = 4 if sim.subpixel else 1
    plane = ny * nz
    per = (2 * (6 + n_psi) + n_ce) * L * plane * item
    for m in sim.monitors:
        name = type(m).__name__
        if name == "FieldMonitor":
            nz_m = 1 if m.plane_z is not None else nz
            per += (steps_per_tile + 1) * len(m.components) * L * ny * nz_m * item
        elif name == "DFTMonitor":
            pl = m.plane()
            cells = L * plane
            if pl is not None:
                cells = cells // (L if pl[0] == 0 else static["shape"][pl[0]])
            per += len(m.freqs) * len(m.components) * cells * 2 * max(item, 8)
    return int(per)


def choose_tile_cells(sim, budget_bytes: float, steps_per_tile: int = 1) -> int:
    """Largest ``tile_cells`` whose slab fits ``budget_bytes``, or 0 if even
    a one-plane slab does not. :func:`run_tiled` sizes slabs against half the
    device budget (headroom for XLA temporaries), as does the planner."""
    nx = sim.grid.shape[0]
    if tile_bytes(sim, 1, steps_per_tile) > budget_bytes:
        return 0
    lo, hi = 1, nx
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if tile_bytes(sim, mid, steps_per_tile) <= budget_bytes:
            lo = mid
        else:
            hi = mid - 1
    return lo


def run_tiled(sim, tile_cells: Optional[int] = None, steps_per_tile: int = 8,
              workdir: Optional[str] = None, budget_bytes: Optional[float] = None):
    """Run ``sim`` forward with disk-backed fields, tiled along x.

    Parameters
    ----------
    sim : Simulation
    tile_cells : int, optional
        Interior x-planes per slab. Default: the largest that fits half of
        ``budget_bytes``, with ``steps_per_tile`` reduced if needed so the
        slab is not mostly halo.
    steps_per_tile : int
        Time steps per slab load (temporal blocking). Larger values cut disk
        traffic but widen the recomputed halo (``steps_per_tile + 1`` planes
        per side).
    workdir : str, optional
        Directory for the scratch memmaps (default: a new temporary directory,
        removed afterwards, as are the files in a given ``workdir``).
    budget_bytes : float, optional
        Device memory budget used to pick ``tile_cells`` (default: what
        :func:`photonfdtd.memory.available_memory` reports).
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from .simulation import Result

    _validate(sim)
    if sim._has_dispersion:
        raise NotImplementedError(
            "disk-tiled stepping does not support dispersive media.")
    _enable_x64_if_needed(sim)
    K = int(steps_per_tile)
    if K < 1:
        raise ValueError("steps_per_tile must be >= 1")
    if sim.grid.shape[0] < 2:
        raise ValueError("disk-tiled stepping tiles along x, which is collapsed")
    if tile_cells is None:
        if budget_bytes is None:
            from .memory import available_memory
            budget_bytes = available_memory()[0] or float("inf")
        # Shrink the temporal block until a slab fits; prefer at least as
        # many interior planes as halo planes, so halo recompute stays < 2/3.
        while True:
            tile_cells = choose_tile_cells(sim, 0.5 * budget_bytes, K)
            if tile_cells >= K + 1 or K == 1:
                break
            K = max(1, K // 2)
        if tile_cells == 0:
            raise MemoryError("even a one-plane slab does not fit the device")
    T = int(tile_cells)
    if T < 1:
        raise ValueError("tile_cells must be >= 1")

    static = _build_static(sim)
    plans = _monitor_plan(sim)
    pplans = _particle_plan(sim, static)
    grid = sim.grid
    nx, ny, nz = static["shape"]
    dt = float(static["dt"])
    n_steps = static["n_steps"]
    npdt = np.dtype(static["npdt"])
    cdt_np = np.complex128 if npdt == np.float64 else np.complex64
    h = K + 1
    n_tiles = int(math.ceil(nx / T))
    L = T + 2 * h
    px = n_tiles * T + 2 * h                 # padded x extent
    comps = static["comps"]
    terms = [t for c in comps for t in c["terms"]]

    # ------------------------------------------------------------------ #
    # Scratch memmaps: fields, psi (full padded layout), E coefficients.
    # ------------------------------------------------------------------ #
    own_dir = workdir is None
    wd = tempfile.mkdtemp(prefix="photonfdtd_tiled_") if own_dir else workdir
    os.makedirs(wd, exist_ok=True)
    need = disk_bytes(sim, T, K)
    free = shutil.disk_usage(wd).free
    if need > free:
        if own_dir:
            shutil.rmtree(wd, ignore_errors=True)
        raise OSError(f"disk-tiled run needs {need / 1e9:.2f} GB of scratch in "
                      f"{wd}, only {free / 1e9:.2f} GB free; pass workdir=")
    files = []

    def mm(name, eps=None):
        """A zeroed padded memmap; with ``eps``, filled slab by slab with the
        E-update coefficient ``dt / (eps * EPS_0)`` (no grid-sized temporary)."""
        path = os.path.join(wd, name + ".npy")
        files.append(path)
        a = np.lib.format.open_memmap(path, mode="w+", dtype=npdt,
                                      shape=(px, ny, nz))
        if eps is not None:
            for g0 in range(0, nx, T):
                g1 = min(nx, g0 + T)
                a[h + g0:h + g1] = (dt / (np.asarray(eps[g0:g1], dtype=npdt)
                                          * EPS_0)).astype(npdt)
        return a

    try:
        disk = {c: mm(c) for c in _FIELDS}
        for t in terms:
            disk["psi_" + t["key"]] = mm("psi_" + t["key"])
        if sim.subpixel:
            smoothed = sim._smoothed_eps_components()
            ce_disk = {c: mm("ce_" + c, e)
                       for c, e in zip(("Ex", "Ey", "Ez"), smoothed)}
            del smoothed
            ce_disk["particle"] = mm("ce_particle", sim.eps_r)
        else:
            shared = mm("ce", sim.eps_r)
            ce_disk = {c: shared for c in ("Ex", "Ey", "Ez", "particle")}

        # -------------------------------------------------------------- #
        # Padded per-plane x data: update masks and x-axis CPML coefficients.
        # -------------------------------------------------------------- #
        def xpad(vals_on_real_planes):
            out = np.zeros(px, dtype=npdt)
            out[h:h + nx] = vals_on_real_planes
            return out

        xmask, yzmask = {}, {}
        for c in comps:
            reg = c["region"]
            rx = np.zeros(nx); rx[reg[0]] = 1.0
            xmask[c["Fkey"]] = xpad(rx)
            m = np.ones((1, ny, nz), dtype=npdt)
            m[:, :, :] = 0.0
            m[(slice(None),) + tuple(reg[1:])] = 1.0
            yzmask[c["Fkey"]] = m
        tcoef = {}
        for c in comps:
            b_src = sim._b_e if c["is_E"] else sim._b_h
            c_src = sim._c_e if c["is_E"] else sim._c_h
            for t in c["terms"]:
                ax = t["ax"]
                if ax == 0:
                    tcoef[t["key"]] = (xpad(np.asarray(b_src[0])),
                                       xpad(np.asarray(c_src[0])))
                else:
                    shp = [1, 1, 1]; shp[ax] = -1
                    tcoef[t["key"]] = (
                        np.asarray(b_src[ax], dtype=npdt).reshape(shp),
                        np.asarray(c_src[ax], dtype=npdt).reshape(shp))

        # Sources grouped per component: indices + full waveform tables.
        def src_table(entries):
            out = {}
            for comp in {e[0] for e in entries}:
                es = [e for e in entries if e[0] == comp]
                out[comp] = (np.array([e[1] + h for e in es], np.int32),
                             np.array([e[2] for e in es], np.int32),
                             np.array([e[3] for e in es], np.int32),
                             np.stack([e[4] for e in es]).astype(npdt))
            return out
        srcH, srcE = src_table(static["src_H"]), src_table(static["src_E"])

        # -------------------------------------------------------------- #
        # Monitor bookkeeping (host-side outputs, device-side block buffers).
        # -------------------------------------------------------------- #
        mons = []
        result = Result(times=np.arange(n_steps) * dt)
        for p in plans:
            m = p["m"]
            info = dict(kind=p["kind"], m=m)
            if p["kind"] == "field":
                ds = m.downsample
                zsel = (slice(p["plane"][1], p["plane"][1] + 1)
                        if p["plane"] is not None else slice(None, None, ds))
                info.update(rec=p["rec"], ds=ds, zsel=zsel,
                            slot=np.full(n_steps, -1, np.int64))
                for r, s in enumerate(p["rec"]):
                    info["slot"][s] = r
                nzm = len(range(nz)[zsel])
                out_shape = (len(p["rec"]), len(range(0, nx, ds)),
                             len(range(0, ny, ds)), nzm)
                info["out"] = {c: np.zeros(out_shape, npdt) for c in m.components}
            elif p["kind"] == "dft":
                ds = m.downsample
                plane = p["plane"]
                sel = [slice(None), slice(None, None, ds), slice(None, None, ds)]
                if plane is not None and plane[0] != 0:
                    sel[plane[0]] = slice(plane[1], plane[1] + 1)
                info.update(ds=ds, plane=plane, sel=tuple(sel),
                            omega=2 * np.pi * np.asarray(m.freqs, float))
                xs = 1 if (plane is not None and plane[0] == 0) else \
                    len(range(0, nx, ds))
                ys = len(range(ny)[sel[1]])
                zs = len(range(nz)[sel[2]])
                info["out"] = {c: np.zeros((len(m.freqs), xs, ys, zs), cdt_np)
                               for c in m.components}
            else:
                ax = {"x": 0, "y": 1, "z": 2}[m.plane_axis]
                coord = grid.coords[ax]
                idx = 0 if coord.size == 1 else int(
                    np.argmin(np.abs(coord - m.plane_position)))
                info.update(ax=ax, idx=idx, lo=max(idx - 1, 0),
                            collapsed=coord.size == 1, total=0.0)
            mons.append(info)

        # -------------------------------------------------------------- #
        # The compiled K-step slab update.
        # -------------------------------------------------------------- #
        neg_ch = -float(static["ch_field"])
        cell = [float(d) for d in static["cell"]]
        d_xmask = {k: jnp.asarray(v) for k, v in xmask.items()}
        d_yz = {k: jnp.asarray(v) for k, v in yzmask.items()}
        d_coef = {k: (jnp.asarray(b), jnp.asarray(c)) for k, (b, c) in tcoef.items()}
        d_srcH = {k: tuple(jnp.asarray(a) for a in v) for k, v in srcH.items()}
        d_srcE = {k: tuple(jnp.asarray(a) for a in v) for k, v in srcE.items()}
        d_part = [dict(R=pp["R"], axes=pp["axes"], coef3=[float(c) for c in pp["coef3"]],
                       starts=jnp.asarray(pp["starts"]),
                       wblocks=jnp.asarray(pp["wblocks"])) for pp in pplans]
        cdt = jnp.complex128 if npdt == np.float64 else jnp.complex64

        def diff(src, ax, is_E, dl):
            hi = [slice(None)] * 3; hi[ax] = slice(1, None)
            lo = [slice(None)] * 3; lo[ax] = slice(0, -1)
            d = (src[tuple(hi)] - src[tuple(lo)]) / dl
            pad = [(0, 0)] * 3
            pad[ax] = (1, 0) if is_E else (0, 1)
            return jnp.pad(d, pad)

        def xslice(a, b0):
            return lax.dynamic_slice_in_dim(a, b0, L, 0).reshape(-1, 1, 1)

        def apply_pass(f, psi, ce, b0, is_E):
            f = dict(f); psi = dict(psi)
            for c in comps:
                if c["is_E"] != is_E:
                    continue
                curl = None
                for t in c["terms"]:
                    ax = t["ax"]
                    d = diff(f[t["src"]], ax, is_E, cell[ax])
                    b, cc = d_coef[t["key"]]
                    if ax == 0:
                        b, cc = xslice(b, b0), xslice(cc, b0)
                    p = b * psi[t["key"]] + cc * d
                    psi[t["key"]] = p
                    cterm = d + p
                    curl = (cterm if t["sign"] > 0 else -cterm) if curl is None \
                        else (curl + cterm if t["sign"] > 0 else curl - cterm)
                if curl is None:
                    continue
                mask = xslice(d_xmask[c["Fkey"]], b0) * d_yz[c["Fkey"]]
                k = ce[c["Fkey"]] if is_E else neg_ch
                f[c["Fkey"]] = f[c["Fkey"]] + (k * curl) * mask
            return f, psi

        def add_sources(f, table, n, b0):
            for comp, (I, J, Kk, V) in table.items():
                li = I - b0
                li = jnp.where((li >= 0) & (li < L), li, L)
                f[comp] = f[comp].at[li, J, Kk].add(V[:, n], mode="drop")
            return f

        def deposit(f, ce_p, n, b0):
            for pp in d_part:
                Rx, Ry, Rz = pp["R"]
                st = pp["starts"][n]
                ix = st[0] + h - b0 + jnp.arange(Rx)
                ix = jnp.where((ix >= 0) & (ix < L), ix, L)
                iy = st[1] + jnp.arange(Ry)
                iz = st[2] + jnp.arange(Rz)
                idx = (ix[:, None, None], iy[None, :, None], iz[None, None, :])
                ce_blk = ce_p.at[idx].get(mode="fill", fill_value=0)
                wb = pp["wblocks"][n]
                for a in pp["axes"]:
                    comp = "xyz"[a]
                    f["E" + comp] = f["E" + comp].at[idx].add(
                        pp["coef3"][a] * wb * ce_blk, mode="drop")
            return f

        def plane_at(F, ax, i):
            return lax.dynamic_index_in_dim(F, i, ax, keepdims=True)

        def flux_now(f, info, args, core):
            """Instantaneous flux through ``info``'s plane, interior planes only."""
            ax = info["ax"]
            E1, E2, H1, H2 = _TANG[ax]
            if ax == 0:
                li, llo, inside = args
                def pa(F): return plane_at(F, 0, li)
                def pv(F): return 0.5 * (plane_at(F, 0, llo) + plane_at(F, 0, li))
                w = inside
            else:
                sl = [slice(None)] * 3
                sl[ax] = slice(info["idx"], info["idx"] + 1)
                sl_lo = list(sl)
                sl_lo[ax] = slice(info["lo"], info["lo"] + 1)
                def pa(F, sl=tuple(sl)): return F[sl]
                if info["collapsed"]:
                    pv = pa
                else:
                    def pv(F, sl=tuple(sl), sll=tuple(sl_lo)):
                        return 0.5 * (F[sll] + F[sl])
                w = core
            S = pa(f[E1]) * pv(f[H2]) - pa(f[E2]) * pv(f[H1])
            dA = 1.0
            for a in range(3):
                if a != ax and static["shape"][a] > 1:
                    dA *= cell[a]
            return jnp.sum(S * w) * dA

        def step_block(state, ce, b0, n0, mon_args, k):
            core = jnp.arange(L)
            core = ((core >= h) & (core < L - h)).astype(npdt).reshape(-1, 1, 1)

            def body(carry, s):
                f, psi, bufs = carry
                n = n0 + s
                f, psi = apply_pass(f, psi, ce, b0, False)
                f = add_sources(f, d_srcH, n, b0)
                pre = {i: flux_now(f, info, mon_args[i], core)    # time-centred
                       for i, info in enumerate(mons) if info["kind"] == "flux"}
                f, psi = apply_pass(f, psi, ce, b0, True)
                f = add_sources(f, d_srcE, n, b0)
                if d_part:
                    f = deposit(f, ce["particle"], n, b0)
                bufs = list(bufs)
                for i, info in enumerate(mons):
                    m = info["m"]
                    if info["kind"] == "field":
                        slot = mon_args[i][s]
                        bufs[i] = {c: bufs[i][c].at[slot].set(
                            f[c][:, ::info["ds"], info["zsel"]], mode="drop")
                            for c in m.components}
                    elif info["kind"] == "dft":
                        active = (n % m.interval == 0).astype(npdt)
                        plane = info["plane"]
                        new = {}
                        for c in m.components:
                            F = f[c]
                            if plane is not None and plane[0] == 0:
                                li, llo = mon_args[i]
                                yz = (slice(None),) + info["sel"][1:]
                                smp = plane_at(F, 0, li)[yz]
                                if _half_offset(c, 0):
                                    smp = 0.5 * (smp + plane_at(F, 0, llo)[yz])
                            else:
                                smp = F[info["sel"]]
                                if plane is not None and plane[1] > 0 and \
                                        _half_offset(c, plane[0]):
                                    prev = list(info["sel"])
                                    prev[plane[0]] = slice(plane[1] - 1, plane[1])
                                    smp = 0.5 * (smp + F[tuple(prev)])
                            tt = (n + (1.0 if c[0] == "E" else 0.5)) * dt
                            ph = jnp.exp(1j * jnp.asarray(info["omega"]) * tt).astype(cdt)
                            new[c] = bufs[i][c] + active * ph.reshape(-1, 1, 1, 1) \
                                * smp.astype(cdt)[None] * dt
                        bufs[i] = new
                    else:
                        post = flux_now(f, info, mon_args[i], core)
                        bufs[i] = bufs[i] + 0.5 * (pre[i] + post) * dt
                return (f, psi, tuple(bufs)), None

            f, psi, bufs = state
            (f, psi, bufs), _ = lax.scan(body, (f, psi, bufs), jnp.arange(k))
            core_sl = slice(h, L - h)
            return ({c: v[core_sl] for c, v in f.items()},
                    {c: v[core_sl] for c, v in psi.items()}, bufs)

        jitted = {}

        def run_block(k, *args):
            if k not in jitted:                    # one compile per chunk length
                jitted[k] = jax.jit(functools.partial(step_block, k=k))
            return jitted[k](*args)

        # -------------------------------------------------------------- #
        # Time loop: chunks of K steps, slabs left to right.
        # -------------------------------------------------------------- #
        names = list(_FIELDS) + ["psi_" + t["key"] for t in terms]
        for n0 in range(0, n_steps, K):
            k = min(K, n_steps - n0)
            rolling = {nm: np.zeros((h, ny, nz), npdt) for nm in names}
            for t_i in range(n_tiles):
                b0 = t_i * T                       # padded block start
                front = b0 + h                     # write frontier (padded)
                blk = {nm: np.concatenate([rolling[nm], disk[nm][front:front + T + h]])
                       for nm in names}
                loaded = {}
                for key, arr in ce_disk.items():
                    if id(arr) not in loaded:
                        loaded[id(arr)] = jnp.asarray(arr[b0:b0 + L])
                ce = {key: loaded[id(arr)] for key, arr in ce_disk.items()}
                # Per-monitor block arguments and buffers.
                margs, bufs = [], []
                gc0, gc1 = b0, b0 + T              # real-cell core range
                for info in mons:
                    m = info["m"]
                    if info["kind"] == "field":
                        sl = info["slot"][n0:n0 + k]
                        rec_here = sl[sl >= 0]
                        r0 = int(rec_here[0]) if rec_here.size else 0
                        local = np.where(sl >= 0, sl - r0, k + 1).astype(np.int32)
                        margs.append(jnp.asarray(local))
                        shp = (k + 1, L, len(range(0, ny, info["ds"])),
                               len(range(nz)[info["zsel"]]))
                        bufs.append({c: jnp.zeros(shp, npdt) for c in m.components})
                        info["_r0"] = r0
                        info["_nrec"] = int(rec_here.size)
                    elif info["kind"] == "dft":
                        pl = info["plane"]
                        if pl is not None and pl[0] == 0:
                            ci = pl[1]
                            lo = ci - 1 if ci > 0 else ci
                            margs.append((ci + h - b0, lo + h - b0))
                            xs = 1
                        else:
                            margs.append(None)
                            xs = L
                        ys = len(range(ny)[info["sel"][1]])
                        zs = len(range(nz)[info["sel"][2]])
                        bufs.append({c: jnp.zeros((len(m.freqs), xs, ys, zs), cdt)
                                     for c in m.components})
                    else:
                        if info["ax"] == 0:
                            ci = info["idx"]
                            inside = 1.0 if gc0 <= ci < gc1 else 0.0
                            lo = info["lo"]
                            margs.append((ci + h - b0, lo + h - b0, inside))
                        else:
                            margs.append(None)
                        bufs.append(jnp.asarray(0.0, npdt))
                state = ({c: jnp.asarray(blk[c]) for c in _FIELDS},
                         {t["key"]: jnp.asarray(blk["psi_" + t["key"]]) for t in terms},
                         tuple(bufs))
                fo, po, bo = run_block(k, state, ce, b0, n0, tuple(margs))
                # Keep the old planes the next slab's left halo needs.
                rolling = {nm: blk[nm][T:T + h].copy() for nm in names}
                for c in _FIELDS:
                    disk[c][front:front + T] = np.asarray(fo[c])
                for t in terms:
                    disk["psi_" + t["key"]][front:front + T] = np.asarray(po[t["key"]])
                _collect(mons, bo, gc0, gc1, nx, h, T)

        for info in mons:
            m = info["m"]
            if info["kind"] == "field":
                result.fields[m.name] = {c: v.astype(sim.dtypes["monitors"])
                                         for c, v in info["out"].items()}
                result.monitor_times[m.name] = np.asarray(info["rec"], float) * dt
            elif info["kind"] == "dft":
                result.dft[m.name] = info["out"]
                result.dft_freqs[m.name] = np.asarray(m.freqs, float)
            else:
                result.flux[m.name] = float(info["total"])
        return result
    finally:
        disk = ce_disk = None                      # drop the maps before unlinking
        for p in files:
            try:
                os.remove(p)
            except OSError:
                pass
        if own_dir:
            shutil.rmtree(wd, ignore_errors=True)


def _collect(mons, bufs, gc0, gc1, nx, h, T):
    """Add one slab's monitor buffers (interior planes only) to the outputs."""
    g1 = min(gc1, nx)
    if g1 <= gc0:
        return
    for info, b in zip(mons, bufs):
        if info["kind"] == "field":
            nrec = info.get("_nrec", 0)
            if nrec == 0:
                continue
            ds = info["ds"]
            xs = np.arange(gc0, g1)
            xs = xs[xs % ds == 0]
            r0 = info["_r0"]
            for c, arr in b.items():
                arr = np.asarray(arr)[:nrec, xs - gc0 + h]
                info["out"][c][r0:r0 + nrec, xs // ds] = arr
        elif info["kind"] == "dft":
            pl = info["plane"]
            if pl is not None and pl[0] == 0:
                if gc0 <= pl[1] < g1:
                    for c, arr in b.items():
                        info["out"][c] += np.asarray(arr)
                continue
            ds = info["ds"]
            xs = np.arange(gc0, g1)
            xs = xs[xs % ds == 0]
            for c, arr in b.items():
                info["out"][c][:, xs // ds] += np.asarray(arr)[:, xs - gc0 + h]
        else:
            info["total"] += float(np.asarray(b))
