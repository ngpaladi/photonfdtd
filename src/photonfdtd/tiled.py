"""Disk-tiled stepping, for grids whose fields do not fit in memory.

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
reassociation (about 1e-14 relative in float64).

Dispersive media
----------------
The pole polarization lives only on the dispersive cells, as in the in-memory
engine: a ``(2, npole, 3, n_disp)`` memmap with the cells sorted by x, so each
slab's cells are one contiguous range. The ADE update is cell-local and needs
no extra halo.

Adjoint
-------
:func:`value_and_grad_eps_tiled` differentiates a monitor loss w.r.t. the
per-cell ``eps_r``. One chunk of K steps maps the whole state to the next
state; its vector-Jacobian product is assembled slab by slab with
``jax.vjp`` of the same compiled slab update, seeded with the cotangent of
the slab's interior outputs and its monitor samples, and scatter-added over
the slab's full (halo-including) input range. The chunk input states come
from two-level checkpointing on disk: every ``checkpoint_every`` chunks the
state is kept, and the chunks in between are recomputed segment by segment
during the backward pass (about 2x the forward compute in total, plus the
VJPs).

Design gradients use the same machinery with the design's own per-component
coefficients and return a cotangent per component, through
``make_value_and_grad_params(..., method="tiled")``.

Scope
-----
One precision for all stepping arrays. Supported: point and expanded soft
sources, ``ChargedParticle`` charges, dispersive media, subpixel smoothing
(forward only), ``FieldMonitor``, ``DFTMonitor`` and ``FluxMonitor``.
``eps_r`` is still stamped in host RAM (one grid-sized array, plus three more
while the subpixel tensor is written out, against the ~20 the in-memory run
holds), as are the dispersive-cell tables, the returned gradient, and the
monitor outputs, so record planes or downsampled volumes on a large grid.
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
                         _build_ade_jax, _validate, _enable_x64_if_needed)

_FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_E = ("Ex", "Ey", "Ez")
_TANG = {0: ("Ey", "Ez", "Hy", "Hz"),
         1: ("Ez", "Ex", "Hz", "Hx"),
         2: ("Ex", "Ey", "Hx", "Hy")}


def _half_offset(comp, ax):
    """True if ``comp`` sits half a cell off a plane normal to ``ax``."""
    a = "xyz".index(comp[1])
    return (a == ax) == (comp[0] == "E")


def _n_poles_cells(sim, static):
    """``(npole, n_disp)`` of the dispersive media in ``sim`` (zeros if none)."""
    ade = _build_ade_jax(sim, static)
    return (0, 0) if ade is None else (ade["npole"], ade["n_disp"])


def _sizing_info(sim):
    """What the size estimates need: the static plan and ``(npole, n_disp)``."""
    static = _build_static(sim)
    return static, _n_poles_cells(sim, static)


def _state_bytes(sim, static, px, poles):
    """Bytes of one full padded state: fields, psi and polarization."""
    _, ny, nz = static["shape"]
    item = np.dtype(static["npdt"]).itemsize
    n_psi = sum(len(c["terms"]) for c in static["comps"])
    npole, n_disp = poles
    return ((6 + n_psi) * px * ny * nz + 2 * npole * 3 * n_disp) * item


def disk_bytes(sim, tile_cells: int, steps_per_tile: int, n_states: int = 1) -> int:
    """Bytes of scratch files a disk-tiled run writes for ``sim``.

    ``n_states`` is the number of full states kept at once: 1 for
    :func:`run_tiled`; see :func:`adjoint_disk_bytes` for the adjoint.
    """
    static = _build_static(sim)
    nx, ny, nz = static["shape"]
    h = steps_per_tile + 1
    px = int(math.ceil(nx / tile_cells)) * tile_cells + 2 * h
    n_ce = 4 if sim.per_component else 1
    item = np.dtype(static["npdt"]).itemsize
    state = _state_bytes(sim, static, px, _n_poles_cells(sim, static))
    return n_states * state + n_ce * px * ny * nz * item


def _adjoint_states(n_steps, steps_per_tile, checkpoint_every=None):
    """``(n_chunks, checkpoint_every, n_states)`` of the tiled adjoint: the
    kept checkpoints, the recomputed segment, the working state and the two
    cotangent buffers."""
    nc = max(1, int(math.ceil(n_steps / steps_per_tile)))
    C = int(checkpoint_every) if checkpoint_every else int(math.ceil(math.sqrt(nc)))
    C = max(1, min(C, nc))
    return nc, C, int(math.ceil(nc / C)) + (C - 1) + 1 + 2


def adjoint_disk_bytes(sim, tile_cells: int, steps_per_tile: int,
                       checkpoint_every: Optional[int] = None) -> int:
    """Bytes of scratch files :func:`value_and_grad_eps_tiled` writes (the
    states of :func:`_adjoint_states`, the coefficient and its cotangent)."""
    _, _, n_states = _adjoint_states(sim.n_steps, steps_per_tile, checkpoint_every)
    static = _build_static(sim)
    nx, ny, nz = static["shape"]
    h = steps_per_tile + 1
    px = int(math.ceil(nx / tile_cells)) * tile_cells + 2 * h
    item = np.dtype(static["npdt"]).itemsize
    return disk_bytes(sim, tile_cells, steps_per_tile, n_states) \
        + px * ny * nz * item


def tile_bytes(sim, tile_cells: int, steps_per_tile: int,
               adjoint: bool = False, _info=None) -> int:
    """Estimated peak device bytes of one slab (fields, psi, coefficients,
    double-buffered across the step, plus monitor buffers). With ``adjoint``,
    also the slab VJP's per-step residuals and cotangents."""
    static, (npole, n_disp) = _info or _sizing_info(sim)
    nx, ny, nz = static["shape"]
    L = tile_cells + 2 * (steps_per_tile + 1)
    item = np.dtype(static["npdt"]).itemsize
    n_psi = sum(len(c["terms"]) for c in static["comps"])
    n_ce = 4 if sim.per_component else 1
    plane = ny * nz
    state = (6 + n_psi) * L * plane * item \
        + 2 * npole * 3 * min(n_disp, L * plane) * item
    per = 2 * state + n_ce * L * plane * item
    if adjoint:
        per += (steps_per_tile + 2) * state + n_ce * L * plane * item
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


def choose_tile_cells(sim, budget_bytes: float, steps_per_tile: int = 1,
                      adjoint: bool = False) -> int:
    """Largest ``tile_cells`` whose slab fits ``budget_bytes``, or 0 if even
    a one-plane slab does not. The tiled runs size slabs against half the
    device budget (headroom for XLA temporaries), as does the planner."""
    nx = sim.grid.shape[0]
    info = _sizing_info(sim)
    if tile_bytes(sim, 1, steps_per_tile, adjoint, info) > budget_bytes:
        return 0
    lo, hi = 1, nx
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if tile_bytes(sim, mid, steps_per_tile, adjoint, info) <= budget_bytes:
            lo = mid
        else:
            hi = mid - 1
    return lo


class _Tiled:
    """Shared machinery of the disk-tiled forward run and adjoint: scratch
    memmaps, per-plane data, the compiled K-step slab update and its VJP, and
    the monitor bookkeeping. Use as a context manager (it removes its files).
    """

    def __init__(self, sim, tile_cells, steps_per_tile, workdir, budget_bytes,
                 adjoint=False, ce=None):
        import jax.numpy as jnp
        from .simulation import Result

        _validate(sim)
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
                tile_cells = choose_tile_cells(sim, 0.5 * budget_bytes, K, adjoint)
                if tile_cells >= K + 1 or K == 1:
                    break
                K = max(1, K // 2)
            if tile_cells == 0:
                raise MemoryError("even a one-plane slab does not fit the device")
        T = int(tile_cells)
        if T < 1:
            raise ValueError("tile_cells must be >= 1")

        if any(getattr(sim.grid, "periodic", ())):
            raise NotImplementedError(
                "disk-tiled stepping does not support periodic boundaries; "
                "run in memory (disk_tiled=False)")
        from .monitors import EnergyMonitor, FluxMonitor
        for m in sim.monitors:
            if isinstance(m, EnergyMonitor) or (
                    isinstance(m, FluxMonitor) and m.bounds is not None):
                raise NotImplementedError(
                    "disk-tiled stepping supports neither EnergyMonitor nor "
                    "bounded FluxMonitor; run in memory (disk_tiled=False)")
        if getattr(sim, "devices", None):
            raise NotImplementedError(
                "disk-tiled stepping runs on one device; build the Simulation "
                "with devices=None")
        if getattr(sim.grid, "has_symmetry", False):
            raise NotImplementedError(
                "disk-tiled stepping does not support symmetry planes; run in "
                "memory (disk_tiled=False)")
        if getattr(sim, "_has_magnetic", False):
            raise NotImplementedError(
                "disk-tiled stepping does not support magnetic media; run in "
                "memory (disk_tiled=False)")
        self.sim = sim
        self.static = static = _build_static(sim)
        plans = _monitor_plan(sim)
        pplans = _particle_plan(sim, static)
        ade = _build_ade_jax(sim, static)
        nx, ny, nz = static["shape"]
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dt = dt = float(static["dt"])
        self.n_steps = n_steps = static["n_steps"]
        self.npdt = npdt = np.dtype(static["npdt"])
        cdt_np = np.complex128 if npdt == np.float64 else np.complex64
        self.K, self.T = K, T
        self.h = h = K + 1
        self.n_tiles = n_tiles = int(math.ceil(nx / T))
        self.L = L = T + 2 * h
        self.px = px = n_tiles * T + 2 * h               # padded x extent
        comps = static["comps"]
        self.terms = terms = [t for c in comps for t in c["terms"]]
        self.names = list(_FIELDS) + ["psi_" + t["key"] for t in terms]

        # Scratch directory (removed on close if we made it).
        self.own_dir = workdir is None
        self.wd = tempfile.mkdtemp(prefix="photonfdtd_tiled_") if self.own_dir \
            else workdir
        os.makedirs(self.wd, exist_ok=True)
        self.files = []
        self._n_mm = 0

        # Dispersive cells, sorted by x, as padded-x index ranges per slab.
        self.ade = ade
        if ade is not None:
            I, J, Kk = (np.asarray(a, np.int64) for a in ade["idx"])
            self.pol_shape = (2, ade["npole"], 3, ade["n_disp"])
            self.ade_Ip = Ip = I + h
            assert np.all(np.diff(Ip) >= 0)
            self.ade_J, self.ade_K = J, Kk
            self.ade_C = np.asarray(ade["C"], npdt)
            b0s = np.arange(n_tiles) * T
            self.M = max(1, int((np.searchsorted(Ip, b0s + L)
                                 - np.searchsorted(Ip, b0s)).max()))
        self.result = Result(times=np.arange(n_steps) * dt)

        # ------------------------------------------------------------------ #
        # E-update coefficients on disk.
        # ------------------------------------------------------------------ #
        def ce_mm(name, eps, coeff=False):
            a = self._mm(name, (px, ny, nz))
            for g0 in range(0, nx, T):
                g1 = min(nx, g0 + T)
                blk = np.asarray(eps[g0:g1], dtype=npdt)
                a[h + g0:h + g1] = blk if coeff else \
                    (dt / (blk * EPS_0)).astype(npdt)
            return a

        if ce is not None:
            # Caller-supplied per-component coefficients (a design); moving
            # charges deposit with the Ex coefficient, as in the in-memory run.
            self.ce_disk = {c: ce_mm("ce_" + c, ce[c], coeff=True) for c in _E}
            self.ce_disk["particle"] = self.ce_disk["Ex"]
        elif sim.subpixel:
            smoothed = sim._smoothed_eps_components()
            self.ce_disk = {c: ce_mm("ce_" + c, e) for c, e in zip(_E, smoothed)}
            del smoothed
            self.ce_disk["particle"] = ce_mm("ce_particle", sim.eps_r)
        else:
            shared = ce_mm("ce", sim.eps_r)
            self.ce_disk = {c: shared for c in _E + ("particle",)}

        # ------------------------------------------------------------------ #
        # Padded per-plane x data: update masks and x-axis CPML coefficients.
        # ------------------------------------------------------------------ #
        def xpad(vals_on_real_planes):
            out = np.zeros(px, dtype=npdt)
            out[h:h + nx] = vals_on_real_planes
            return out

        xmask, yzmask = {}, {}
        for c in comps:
            reg = c["region"]
            rx = np.zeros(nx); rx[reg[0]] = 1.0
            xmask[c["Fkey"]] = xpad(rx)
            m = np.zeros((1, ny, nz), dtype=npdt)
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

        # Sources, already grouped per component (see _build_static), with x
        # moved to padded indices; V is the separable SourceTable.
        def src_table(entries):
            return {comp: (I + h, J, Kk, V) for comp, I, J, Kk, V in entries}
        srcH, srcE = src_table(static["src_H"]), src_table(static["src_E"])

        # ------------------------------------------------------------------ #
        # Monitor bookkeeping (host-side outputs).
        # ------------------------------------------------------------------ #
        self.mons = mons = []
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
                coord = sim.grid.coords[ax]
                idx = 0 if coord.size == 1 else int(
                    np.argmin(np.abs(coord - m.plane_position)))
                info.update(ax=ax, idx=idx, lo=max(idx - 1, 0),
                            collapsed=coord.size == 1, total=0.0)
            mons.append(info)

        # ------------------------------------------------------------------ #
        # The compiled K-step slab update.
        # ------------------------------------------------------------------ #
        neg_ch = -float(static["ch_field"])
        cell = [float(d) for d in static["cell"]]
        d_xmask = {k: jnp.asarray(v) for k, v in xmask.items()}
        d_yz = {k: jnp.asarray(v) for k, v in yzmask.items()}
        d_coef = {k: (jnp.asarray(b), jnp.asarray(c)) for k, (b, c) in tcoef.items()}
        d_srcH = {k: tuple(jnp.asarray(a) for a in v[:3]) + (v[3].device(),)
                  for k, v in srcH.items()}
        d_srcE = {k: tuple(jnp.asarray(a) for a in v[:3]) + (v[3].device(),)
                  for k, v in srcE.items()}
        d_part = [dict(R=pp["R"], axes=pp["axes"], coef3=[float(c) for c in pp["coef3"]],
                       starts=jnp.asarray(pp["starts"]),
                       wblocks=jnp.asarray(pp["wblocks"])) for pp in pplans]
        cdt = jnp.complex128 if npdt == np.float64 else jnp.complex64
        if ade is not None:
            ade_A = jnp.asarray(np.asarray(ade["A"], npdt)).reshape(-1, 1, 1)
            ade_B = jnp.asarray(np.asarray(ade["B"], npdt)).reshape(-1, 1, 1)

        def diff(src, ax, is_E, dl):
            hi = [slice(None)] * 3; hi[ax] = slice(1, None)
            lo = [slice(None)] * 3; lo[ax] = slice(0, -1)
            d = (src[tuple(hi)] - src[tuple(lo)]) / dl
            pad = [(0, 0)] * 3
            pad[ax] = (1, 0) if is_E else (0, 1)
            return jnp.pad(d, pad)

        def xslice(a, b0):
            from jax import lax
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
                f[comp] = f[comp].at[li, J, Kk].add(V[n], mode="drop")
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
            from jax import lax
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

        def step_block(state, ce, b0, n0, mon_args, ade_args, k, remat=False):
            """Advance one slab ``k`` steps. ``state`` is (fields, psi, pol,
            monitor buffers) on the L-plane slab; returns the interior planes
            of fields and psi, the slab's full polarization block and the
            buffers."""
            import jax
            from jax import lax
            core = jnp.arange(L)
            core = ((core >= h) & (core < L - h)).astype(npdt).reshape(-1, 1, 1)
            if ade is not None:
                li, jj, kk, Cb = ade_args
                aidx = (li, jj, kk)
                Cb = Cb[:, None, :]
                inv = [ce[c].at[aidx].get(mode="fill", fill_value=0) / dt for c in _E]

            def body(carry, s):
                f, psi, pol, bufs = carry
                n = n0 + s
                f, psi = apply_pass(f, psi, ce, b0, False)
                f = add_sources(f, d_srcH, n, b0)
                if ade is not None:
                    En = jnp.stack([f[c].at[aidx].get(mode="fill", fill_value=0)
                                    for c in _E])
                pre = {i: flux_now(f, info, mon_args[i], core)    # time-centred
                       for i, info in enumerate(mons) if info["kind"] == "flux"}
                f, psi = apply_pass(f, psi, ce, b0, True)
                if ade is not None:
                    P0, P1 = pol
                    Pn = ade_A * P0 + ade_B * P1 + Cb * En[None]
                    delta = jnp.sum(Pn - P0, axis=0)
                    for a, c in enumerate(_E):
                        f[c] = f[c].at[aidx].add(-inv[a] * delta[a], mode="drop")
                    pol = (Pn, P0)
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
                                li_, llo = mon_args[i]
                                yz = (slice(None),) + info["sel"][1:]
                                smp = plane_at(F, 0, li_)[yz]
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
                return (f, psi, pol, tuple(bufs)), None

            if remat:                          # the VJP keeps one carry per step
                body = jax.checkpoint(body)
            (f, psi, pol, bufs), _ = lax.scan(body, state, jnp.arange(k))
            core_sl = slice(h, L - h)
            return ({c: v[core_sl] for c, v in f.items()},
                    {c: v[core_sl] for c, v in psi.items()}, pol, bufs)

        def vjp_block(state, ce, b0, n0, mon_args, ade_args, cot, k):
            import jax
            fn = lambda st, c: step_block(st, c, b0, n0, mon_args, ade_args, k,
                                          remat=True)
            _, pull = jax.vjp(fn, state, ce)
            return pull(cot)

        self._step_block = step_block
        self._vjp_block = vjp_block
        self._jit = {}
        self._cdt = cdt

    # ---------------------------------------------------------------------- #
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.ce_disk = None                        # drop the maps before unlinking
        for p in self.files:
            try:
                os.remove(p)
            except OSError:
                pass
        if self.own_dir:
            shutil.rmtree(self.wd, ignore_errors=True)

    def check_disk(self, need):
        free = shutil.disk_usage(self.wd).free
        if need > free:
            raise OSError(f"disk-tiled run needs {need / 1e9:.2f} GB of scratch "
                          f"in {self.wd}, only {free / 1e9:.2f} GB free; pass "
                          "workdir=")

    def _mm(self, name, shape):
        """A zeroed scratch memmap."""
        self._n_mm += 1
        path = os.path.join(self.wd, f"{self._n_mm:03d}_{name}.npy")
        self.files.append(path)
        return np.lib.format.open_memmap(path, mode="w+", dtype=self.npdt,
                                         shape=shape)

    def new_state(self, tag):
        """A zeroed full state on disk: padded fields and psi, polarization."""
        st = {nm: self._mm(f"{tag}_{nm}", (self.px, self.ny, self.nz))
              for nm in self.names}
        if self.ade is not None:
            st["pol"] = self._mm(f"{tag}_pol", self.pol_shape)
        return st

    @staticmethod
    def zero(state):
        for a in state.values():
            for i in range(a.shape[0]):            # plane by plane: no big temporaries
                a[i] = 0

    def _compiled(self, kind, k):
        import jax
        key = (kind, k)
        if key not in self._jit:                   # one compile per chunk length
            fn = self._step_block if kind == "fwd" else self._vjp_block
            self._jit[key] = jax.jit(functools.partial(fn, k=k))
        return self._jit[key]

    # ---------------------------------------------------------------------- #
    def _tile(self, t_i, n0, k):
        """Host-side per-slab arguments: geometry, monitor arguments and zeroed
        buffers, and the slab's dispersive cells."""
        import jax.numpy as jnp
        h, T, L, ny, nz = self.h, self.T, self.L, self.ny, self.nz
        b0 = t_i * T
        tile = dict(b0=b0, front=b0 + h, gc0=b0, gc1=b0 + T)
        margs, bufs, meta = [], [], []
        for info in self.mons:
            m = info["m"]
            if info["kind"] == "field":
                sl = info["slot"][n0:n0 + k]
                rec_here = sl[sl >= 0]
                r0 = int(rec_here[0]) if rec_here.size else 0
                local = np.where(sl >= 0, sl - r0, k + 1).astype(np.int32)
                margs.append(jnp.asarray(local))
                shp = (k + 1, L, len(range(0, ny, info["ds"])),
                       len(range(nz)[info["zsel"]]))
                bufs.append({c: jnp.zeros(shp, self.npdt) for c in m.components})
                meta.append(dict(r0=r0, nrec=int(rec_here.size)))
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
                bufs.append({c: jnp.zeros((len(m.freqs), xs, ys, zs), self._cdt)
                             for c in m.components})
                meta.append(None)
            else:
                if info["ax"] == 0:
                    ci = info["idx"]
                    inside = 1.0 if tile["gc0"] <= ci < tile["gc1"] else 0.0
                    margs.append((ci + h - b0, info["lo"] + h - b0, inside))
                else:
                    margs.append(None)
                bufs.append(jnp.asarray(0.0, self.npdt))
                meta.append(None)
        tile.update(margs=tuple(margs), bufs=tuple(bufs), meta=meta)
        if self.ade is not None:
            Ip = self.ade_Ip
            s = lambda x: int(np.searchsorted(Ip, x))
            lo, hi = s(b0), s(b0 + L)
            M, n = self.M, hi - lo
            li = np.full(M, L, np.int32); li[:n] = Ip[lo:hi] - b0
            jj = np.zeros(M, np.int32); jj[:n] = self.ade_J[lo:hi]
            kk = np.zeros(M, np.int32); kk[:n] = self.ade_K[lo:hi]
            Cb = np.zeros((self.ade_C.shape[0], M), self.npdt)
            Cb[:, :n] = self.ade_C[:, lo:hi]
            tile.update(lo=lo, hi=hi, n_roll=s(b0 + h) - lo,
                        roll=(s(b0 + T) - lo, s(b0 + T + h) - lo),
                        core=(s(b0 + h), s(b0 + h + T)),
                        ade_args=tuple(jnp.asarray(a) for a in (li, jj, kk, Cb)))
        else:
            tile["ade_args"] = None
        return tile

    def _ce_block(self, b0):
        import jax.numpy as jnp
        loaded = {}
        for key, arr in self.ce_disk.items():
            if id(arr) not in loaded:
                loaded[id(arr)] = jnp.asarray(arr[b0:b0 + self.L])
        return {key: loaded[id(arr)] for key, arr in self.ce_disk.items()}

    def _pol_block(self, pol_src, tile):
        """The slab's polarization block, zero-padded to M cells."""
        blk = np.zeros(self.pol_shape[:3] + (self.M,), self.npdt)
        blk[..., :tile["hi"] - tile["lo"]] = pol_src[..., tile["lo"]:tile["hi"]]
        return blk

    def forward_chunk(self, src, dst, n0, collect=True):
        """Advance the full state ``src`` by one chunk of K steps into ``dst``
        (which may be ``src``: the update is synchronous either way), adding
        the monitor samples to the outputs if ``collect``."""
        import jax.numpy as jnp
        h, T, ny, nz = self.h, self.T, self.ny, self.nz
        k = min(self.K, self.n_steps - n0)
        run = self._compiled("fwd", k)
        rolling = {nm: np.zeros((h, ny, nz), self.npdt) for nm in self.names}
        roll_pol = None
        for t_i in range(self.n_tiles):
            tile = self._tile(t_i, n0, k)
            b0, front = tile["b0"], tile["front"]
            blk = {nm: np.concatenate([rolling[nm], src[nm][front:front + T + h]])
                   for nm in self.names}
            if self.ade is not None:
                pblk = self._pol_block(src["pol"], tile)
                if tile["n_roll"]:
                    pblk[..., :tile["n_roll"]] = roll_pol
                pol = (jnp.asarray(pblk[0]), jnp.asarray(pblk[1]))
            else:
                pol = ()
            state = ({c: jnp.asarray(blk[c]) for c in _FIELDS},
                     {t["key"]: jnp.asarray(blk["psi_" + t["key"]]) for t in self.terms},
                     pol, tile["bufs"])
            fo, po, polo, bo = run(state, self._ce_block(b0), b0, n0,
                                   tile["margs"], tile["ade_args"])
            # Keep the old planes the next slab's left halo needs.
            rolling = {nm: blk[nm][T:T + h].copy() for nm in self.names}
            for c in _FIELDS:
                dst[c][front:front + T] = np.asarray(fo[c])
            for t in self.terms:
                dst["psi_" + t["key"]][front:front + T] = np.asarray(po[t["key"]])
            if self.ade is not None:
                r0, r1 = tile["roll"]
                roll_pol = pblk[..., r0:r1].copy()
                c0, c1 = tile["core"]
                lo = tile["lo"]
                dst["pol"][..., c0:c1] = np.stack(
                    [np.asarray(polo[0]), np.asarray(polo[1])])[..., c0 - lo:c1 - lo]
            if collect:
                _collect(self.mons, bo, tile, self.nx, h)

    def vjp_chunk(self, src, bar_out, bar_in, bar_ce, n0, g):
        """Pull the cotangent ``bar_out`` of the state after chunk ``n0`` back
        through it: the state cotangent goes to ``bar_in`` (zeroed by the
        caller), the coefficient cotangent is added to ``bar_ce``. ``src`` is
        the chunk's input state; ``g`` the loss cotangent of the outputs."""
        import jax.numpy as jnp
        h, T, L = self.h, self.T, self.L
        k = min(self.K, self.n_steps - n0)
        pull = self._compiled("vjp", k)
        for t_i in range(self.n_tiles):
            tile = self._tile(t_i, n0, k)
            b0, front = tile["b0"], tile["front"]
            if self.ade is not None:
                pblk = self._pol_block(src["pol"], tile)
                pol = (jnp.asarray(pblk[0]), jnp.asarray(pblk[1]))
                lo, (c0, c1) = tile["lo"], tile["core"]
                pcot = np.zeros_like(pblk)
                pcot[..., c0 - lo:c1 - lo] = bar_out["pol"][..., c0:c1]
                pcot = (jnp.asarray(pcot[0]), jnp.asarray(pcot[1]))
            else:
                pol = pcot = ()
            state = ({c: jnp.asarray(src[c][b0:b0 + L]) for c in _FIELDS},
                     {t["key"]: jnp.asarray(src["psi_" + t["key"]][b0:b0 + L])
                      for t in self.terms},
                     pol, tile["bufs"])
            cot = ({c: jnp.asarray(bar_out[c][front:front + T]) for c in _FIELDS},
                   {t["key"]: jnp.asarray(bar_out["psi_" + t["key"]][front:front + T])
                    for t in self.terms},
                   pcot, _buf_cotangent(self.mons, tile, g, self.nx, h, tile["bufs"]))
            bar_state, bar_c = pull(state, self._ce_block(b0), b0, n0,
                                    tile["margs"], tile["ade_args"], cot)
            fb, pb, polb, _ = bar_state
            for c in _FIELDS:
                bar_in[c][b0:b0 + L] += np.asarray(fb[c])
            for t in self.terms:
                bar_in["psi_" + t["key"]][b0:b0 + L] += np.asarray(pb[t["key"]])
            if self.ade is not None:
                n = tile["hi"] - tile["lo"]
                bar_in["pol"][..., tile["lo"]:tile["hi"]] += np.stack(
                    [np.asarray(polb[0]), np.asarray(polb[1])])[..., :n]
            for key, v in bar_c.items():       # shared arrays share a buffer
                bar_ce[key][b0:b0 + L] += np.asarray(v)

    def outputs(self):
        """The collected monitor outputs as a :class:`Result`."""
        result = self.result
        for info in self.mons:
            m = info["m"]
            if info["kind"] == "field":
                result.fields[m.name] = {c: v.astype(self.sim.dtypes["monitors"])
                                         for c, v in info["out"].items()}
                result.monitor_times[m.name] = np.asarray(info["rec"], float) * self.dt
            elif info["kind"] == "dft":
                result.dft[m.name] = info["out"]
                result.dft_freqs[m.name] = np.asarray(m.freqs, float)
            else:
                result.flux[m.name] = float(info["total"])
        return result


def _collect(mons, bufs, tile, nx, h):
    """Add one slab's monitor buffers (interior planes only) to the outputs."""
    gc0 = tile["gc0"]
    g1 = min(tile["gc1"], nx)
    if g1 <= gc0:
        return
    for info, b, meta in zip(mons, bufs, tile["meta"]):
        if info["kind"] == "field":
            nrec = meta["nrec"]
            if nrec == 0:
                continue
            ds = info["ds"]
            xs = np.arange(gc0, g1)
            xs = xs[xs % ds == 0]
            r0 = meta["r0"]
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


def _buf_cotangent(mons, tile, g, nx, h, bufs):
    """Cotangent of one slab's monitor buffers: the transpose of
    :func:`_collect`, reading the loss cotangent ``g`` of the outputs."""
    import jax.numpy as jnp
    gc0 = tile["gc0"]
    g1 = min(tile["gc1"], nx)
    out = []
    for info, b, meta in zip(mons, bufs, tile["meta"]):
        m = info["m"]
        if info["kind"] == "flux":
            out.append(jnp.asarray(g["flux"][m.name], b.dtype))
            continue
        cot = {c: np.zeros(v.shape, v.dtype) for c, v in b.items()}
        if g1 > gc0:
            ds = info.get("ds", 1)
            xs = np.arange(gc0, g1)
            xs = xs[xs % ds == 0]
            if info["kind"] == "field":
                nrec, r0 = meta["nrec"], meta["r0"]
                for c in cot:
                    cot[c][:nrec, xs - gc0 + h] = \
                        np.asarray(g["fields"][m.name][c])[r0:r0 + nrec, xs // ds]
            else:
                pl = info["plane"]
                for c in cot:
                    gc = np.asarray(g["dft"][m.name][c])
                    if pl is not None and pl[0] == 0:
                        if gc0 <= pl[1] < g1:
                            cot[c][...] = gc
                    else:
                        cot[c][:, xs - gc0 + h] = gc[:, xs // ds]
        out.append({c: jnp.asarray(v) for c, v in cot.items()})
    return tuple(out)


def run_tiled(sim, tile_cells: Optional[int] = None, steps_per_tile: int = 8,
              workdir: Optional[str] = None, budget_bytes: Optional[float] = None,
              ce=None):
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
    if ce is None and getattr(sim, "_has_anisotropy", False):
        from .jaxbackend import update_coefficients
        ce = update_coefficients(sim)[0]
    with _Tiled(sim, tile_cells, steps_per_tile, workdir, budget_bytes,
                ce=ce) as tr:
        tr.check_disk(disk_bytes(sim, tr.T, tr.K))
        state = tr.new_state("s")
        for n0 in range(0, tr.n_steps, tr.K):
            tr.forward_chunk(state, state, n0)
        return tr.outputs()


def _out_jax(result):
    """The loss-facing monitor-output dict (as in
    :func:`photonfdtd.jaxbackend.value_and_grad_eps`) from a tiled Result."""
    import jax.numpy as jnp
    return {"fields": {n: {c: jnp.asarray(v) for c, v in d.items()}
                       for n, d in result.fields.items()},
            "dft": {n: {c: jnp.asarray(v) for c, v in d.items()}
                    for n, d in result.dft.items()},
            "flux": {n: jnp.asarray(v) for n, v in result.flux.items()}}


class _TiledAdjoint:
    """A disk-tiled forward run that can then be pulled back.

    ``forward()`` runs the simulation, keeping chunk checkpoints on disk, and
    returns the loss-facing output dict; ``backward(g)`` takes that dict's
    cotangent and returns the cotangent of the E-update coefficients,
    ``{'Ex','Ey','Ez'}`` host arrays of the grid's shape (arrays that share a
    coefficient - e.g. all three without subpixel smoothing - share one,
    total, cotangent). ``ce`` optionally supplies the coefficients (a design);
    by default they come from ``sim`` as in :func:`run_tiled`. Close it (or
    use it as a context manager) to remove the scratch files.
    """

    def __init__(self, sim, ce=None, tile_cells=None, steps_per_tile=8,
                 workdir=None, budget_bytes=None, checkpoint_every=None):
        self.tr = tr = _Tiled(sim, tile_cells, steps_per_tile, workdir,
                              budget_bytes, adjoint=True, ce=ce)
        try:
            self.nc, self.C, _ = _adjoint_states(tr.n_steps, tr.K,
                                                 checkpoint_every)
            n_ce = len({id(a) for a in tr.ce_disk.values()})
            plane = tr.px * tr.ny * tr.nz * tr.npdt.itemsize
            tr.check_disk(adjoint_disk_bytes(sim, tr.T, tr.K, self.C)
                          + 2 * (n_ce - 1) * plane)
        except BaseException:
            tr.close()
            raise
        self._done = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.tr.close()

    def forward(self):
        tr, nc, C, K = self.tr, self.nc, self.C, self.tr.K
        self.n_ck = n_ck = int(math.ceil(nc / C))
        # States at chunks 0, C, 2C, ... are kept in their own files; the
        # chunks in between step a working state in place.
        self.ck = ck = [tr.new_state(f"ck{j}") for j in range(n_ck)]
        work = tr.new_state("w")
        cur = ck[0]
        for c in range(nc):
            nxt = ck[(c + 1) // C] if (c + 1) % C == 0 and c + 1 < nc else work
            tr.forward_chunk(cur, nxt, c * K)
            cur = nxt
        self.result = tr.outputs()
        return _out_jax(self.result)

    def backward(self, g, copy=True):
        """Cotangent of the coefficients; ``copy=False`` returns views of the
        scratch memmaps (valid until :meth:`close`)."""
        if self._done:
            raise RuntimeError("backward() can only be called once")
        self._done = True
        tr, nc, C, K, ck = self.tr, self.nc, self.C, self.tr.K, self.ck
        seg = [None] + [tr.new_state(f"seg{i}") for i in range(1, C)]
        bar, bar_new = tr.new_state("bar"), tr.new_state("barn")
        bufs = {}
        for key, arr in tr.ce_disk.items():
            if id(arr) not in bufs:
                bufs[id(arr)] = tr._mm(f"bar_ce{len(bufs)}", (tr.px, tr.ny, tr.nz))
        bar_ce = {key: bufs[id(arr)] for key, arr in tr.ce_disk.items()}
        for s in range(self.n_ck - 1, -1, -1):
            c0, c1 = s * C, min((s + 1) * C, nc)
            seg[0] = ck[s]
            for i in range(1, c1 - c0):        # recompute the segment's inputs
                tr.forward_chunk(seg[i - 1], seg[i], (c0 + i - 1) * K,
                                 collect=False)
            for c in range(c1 - 1, c0 - 1, -1):
                tr.zero(bar_new)
                tr.vjp_chunk(seg[c - c0], bar, bar_new, bar_ce, c * K, g)
                bar, bar_new = bar_new, bar
        h, nx = tr.h, tr.nx
        out, copied = {}, {}
        for c in _E:
            a = bar_ce[c]
            if id(a) not in copied:
                copied[id(a)] = np.array(a[h:h + nx]) if copy else a[h:h + nx]
            out[c] = copied[id(a)]
        return out

    def ce(self, comp="Ex"):
        """The coefficient of ``comp`` on the real planes (a memmap view)."""
        tr = self.tr
        return tr.ce_disk[comp][tr.h:tr.h + tr.nx]


def value_and_grad_eps_tiled(sim, loss, tile_cells: Optional[int] = None,
                             steps_per_tile: int = 8,
                             workdir: Optional[str] = None,
                             budget_bytes: Optional[float] = None,
                             checkpoint_every: Optional[int] = None):
    """``(value, d loss / d eps_r)`` with disk-backed state, tiled along x.

    Same contract as :func:`photonfdtd.jax_value_and_grad_eps` (``loss`` maps
    the monitor-output dict to a scalar; the gradient is a host array of the
    grid's shape), for grids whose adjoint does not fit on the device. See the
    module docstring for the method. ``tile_cells``, ``steps_per_tile``,
    ``workdir`` and ``budget_bytes`` are as in :func:`run_tiled` (slabs are
    sized for the VJP, which holds about ``steps_per_tile`` slab states).

    ``checkpoint_every`` is the number of K-step chunks between kept states
    (default ``sqrt`` of the chunk count, which minimises disk use). Set it to
    1 to keep every chunk's input state and skip the recomputation, if the
    disk allows (:func:`adjoint_disk_bytes`).

    Subpixel smoothing is not supported (``eps_r`` does not parameterize the
    smoothed tensor), as in the in-memory gradient. Dispersive media are: in
    their cells ``eps_r`` is ``eps_inf``, with the poles held fixed.
    """
    import jax
    if sim._has_anisotropy or sim._has_conductivity:
        raise NotImplementedError(
            "value_and_grad_eps_tiled differentiates a scalar eps_r; anisotropic "
            "or conductive media need value_and_grad_params")
    if sim.subpixel:
        raise NotImplementedError(
            "value_and_grad_eps_tiled differentiates the scalar per-cell eps_r, "
            "which does not parameterize a subpixel-smoothed tensor.")
    with _TiledAdjoint(sim, None, tile_cells, steps_per_tile, workdir,
                       budget_bytes, checkpoint_every) as ta:
        value, pull = jax.vjp(loss, ta.forward())
        (g,) = pull(np.ones((), np.asarray(value).dtype))
        bar = ta.backward(g, copy=False)["Ex"]   # shared coefficient: total
        ce, T, npdt = ta.ce("Ex"), ta.tr.T, ta.tr.npdt
        eps = np.asarray(sim.eps_r)
        grad = np.empty(sim.grid.shape, npdt)
        for g0 in range(0, ta.tr.nx, T):         # d ce / d eps_r = -ce / eps_r
            g1 = min(ta.tr.nx, g0 + T)
            grad[g0:g1] = -bar[g0:g1] * ce[g0:g1] / np.asarray(eps[g0:g1], npdt)
        return float(value), grad
