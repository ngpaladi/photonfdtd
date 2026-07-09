"""Out-of-core (disk-backed) FDTD time stepping via domain-decomposition tiling.

The in-core solver holds the six field arrays plus the E-update coefficient
resident in RAM. For a volume whose fields do not fit in memory,
:func:`run_out_of_core` streams them: every full-grid array (``Ex..Hz``,
``ce_field`` and the CPML ``psi`` state) lives in a memory-mapped file on disk,
and each timestep sweeps the domain in slabs along axis 0, loading only
``tile_cells`` planes (plus a one-cell halo) into RAM at a time. Peak RAM is set
by the tile size, not the grid size.

Correctness
-----------
Each Yee update is elementwise per cell, and reads a field that is not modified
during its pass (the H update reads E; the E update reads the freshly-updated
H). So a complete H-pass over every tile followed by a complete E-pass over
every tile reproduces the in-core result. The forward differences a tile's H
update needs at its high-x face read one plane of the next tile's E; the
backward differences the E update needs at its low-x face read one plane of the
previous tile's already-updated H. Those single-plane halos are the only
cross-tile coupling. The per-cell arithmetic mirrors the fused Numba kernel's
``cy - cz`` formulation, so results track the in-core backends to floating
reordering (~1e-11 relative), like the Numba backend itself.

Scope (v1)
----------
NumPy backend, a single uniform precision, point/soft sources and
``FieldMonitor`` (including ``compression=``). ``DFTMonitor``, ``FluxMonitor``,
``ChargedParticle`` sources and the GPU/Numba backends raise a clear error. The
CPML ``psi`` state is stored full-grid on disk for simplicity; a PML-slab-only
layout is a natural follow-up to cut that disk use.
"""
from __future__ import annotations
import os
import tempfile
from typing import Dict, List, Optional
import numpy as np

from .constants import EPS_0, MU_0
from .monitors import FieldMonitor, FluxMonitor, DFTMonitor
from .storage import CompressedFieldSeries, get_codec


def _memmap(path, shape, dtype):
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _bcast(a1d, axis):
    """Reshape a 1-D coefficient array to broadcast along ``axis`` of a 3-D block."""
    return a1d.reshape([-1 if i == axis else 1 for i in range(3)])


def _cpml(d, psi_view, b1d, c1d, axis):
    """Advance one CPML psi block and return ``d + psi`` (the corrected term).

    ``b1d``/``c1d`` are the 1-D coefficients along ``axis`` already sliced to the
    region's extent on that axis; ``psi_view`` is the full-grid psi restricted to
    the same region (a memmap slice, written back in place).
    """
    bb = _bcast(b1d, axis)
    cc = _bcast(c1d, axis)
    p = np.asarray(psi_view)
    p = bb * p + cc * d
    psi_view[...] = p
    return d + p


def run_out_of_core(sim, tile_cells: int, workdir: Optional[str] = None):
    """Run ``sim`` with disk-backed fields, tiled along axis 0.

    Parameters
    ----------
    sim : Simulation
        A fully-built simulation (NumPy backend, uniform precision).
    tile_cells : int
        Number of x-planes held in RAM per tile (>= 1). Smaller = less RAM.
    workdir : str, optional
        Directory for temporary memmaps (default: a fresh temp dir, removed at
        the end).
    """
    from .simulation import Result  # avoid import cycle

    if sim.use_gpu or sim.use_numba:
        raise NotImplementedError(
            "out-of-core stepping supports the NumPy backend only; "
            "drop use_gpu / use_numba."
        )
    if sim.particle_sources:
        raise NotImplementedError(
            "out-of-core stepping does not yet support ChargedParticle sources."
        )
    for m in sim.monitors:
        if isinstance(m, (DFTMonitor, FluxMonitor)):
            raise NotImplementedError(
                f"out-of-core stepping does not yet support {type(m).__name__}; "
                "use a FieldMonitor (compression= is supported)."
            )
    if len({sim.dtypes[k] for k in
            ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "eps_r")}) > 1:
        raise NotImplementedError(
            "out-of-core stepping requires one uniform precision for all "
            "stepping arrays."
        )
    if int(tile_cells) < 1:
        raise ValueError("tile_cells must be >= 1")
    tile_cells = int(tile_cells)

    grid = sim.grid
    nx, ny, nz = grid.shape
    dx, dy, dz = (d if d > 0 else 1.0 for d in grid.cell_size)
    dt = sim.dt
    n_steps = sim.n_steps
    dtype = sim.dtypes["Ex"]
    mon_dtype = sim.dtypes["monitors"]

    be = [np.asarray(sim._b_e[a], dtype=dtype) for a in range(3)]
    ce = [np.asarray(sim._c_e[a], dtype=dtype) for a in range(3)]
    bh = [np.asarray(sim._b_h[a], dtype=dtype) for a in range(3)]
    ch = [np.asarray(sim._c_h[a], dtype=dtype) for a in range(3)]
    ch_field = dtype(dt / MU_0)

    # Reduced-axis extents: forward differences span n-1 cells, backward
    # differences start at 1, exactly as the in-core / Numba paths.
    fy = ny - 1 if ny > 1 else None      # y forward-diff last index / None if flat
    fz = nz - 1 if nz > 1 else None

    tmp = workdir or tempfile.mkdtemp(prefix="photonfdtd_ooc_")
    made_tmp = workdir is None
    created: List[str] = []

    def mm(name, zero=True):
        path = os.path.join(tmp, name + ".npy")
        created.append(path)
        arr = _memmap(path, (nx, ny, nz), dtype)
        if zero:
            arr[...] = 0
        return arr

    F = ce_field = None
    psi: Dict[str, np.ndarray] = {}
    try:
        F = {c: mm(c) for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
        ce_field = mm("ce_field", zero=False)
        eps = sim.eps_r
        for x0 in range(0, nx, tile_cells):
            x1 = min(x0 + tile_cells, nx)
            ce_field[x0:x1] = dt / (np.asarray(eps[x0:x1], dtype=dtype) * EPS_0)
        for key in ("Hx_y", "Hx_z", "Hy_z", "Hy_x", "Hz_x", "Hz_y",
                    "Ex_y", "Ex_z", "Ey_z", "Ey_x", "Ez_x", "Ez_y"):
            psi[key] = mm("psi_" + key)

        tiles = [(x0, min(x0 + tile_cells, nx)) for x0 in range(0, nx, tile_cells)]

        # ---------------- H pass over one tile [x0:x1] -------------------- #
        def h_pass(x0, x1):
            xh = min(x1 + 1, nx)                 # high-x halo for d/dx terms
            Ex = np.asarray(F["Ex"][x0:xh]); Ey = np.asarray(F["Ey"][x0:xh])
            Ez = np.asarray(F["Ez"][x0:xh])
            w = x1 - x0

            # Hx (i, j+1/2, k+1/2): no d/dx.  cy=dEz/dy, cz=dEy/dz
            if fy is not None or fz is not None:
                ry = slice(0, fy) if fy is not None else slice(None)
                rz = slice(0, fz) if fz is not None else slice(None)
                reg = (slice(x0, x1), ry, rz)
                cy = cz = None
                if fy is not None:
                    d = (Ez[:w, 1:, :] - Ez[:w, :-1, :]) / dy
                    if fz is not None:
                        d = d[:, :, :fz]
                    cy = _cpml(d, psi["Hx_y"][reg], bh[1][ry], ch[1][ry], 1)
                if fz is not None:
                    d = (Ey[:w, :, 1:] - Ey[:w, :, :-1]) / dz
                    if fy is not None:
                        d = d[:, :fy, :]
                    cz = _cpml(d, psi["Hx_z"][reg], bh[2][rz], ch[2][rz], 2)
                _acc(F["Hx"], reg, -ch_field, cy, cz)

            # Hy (i+1/2, j, k+1/2): cz=dEx/dz, cx=dEz/dx
            if nx > 1 and (fz is not None or True):
                wx = x1 - 1 - x0 if x1 == nx else x1 - x0   # forward d/dx write width
                wx = max(wx, 0)
                rz = slice(0, fz) if fz is not None else slice(None)
                reg = (slice(x0, x0 + wx), slice(None), rz)
                cz = cx = None
                if fz is not None and wx > 0:
                    d = (Ex[:wx, :, 1:] - Ex[:wx, :, :-1]) / dz
                    cz = _cpml(d, psi["Hy_z"][reg], bh[2][rz], ch[2][rz], 2)
                if wx > 0:
                    d = (Ez[1:wx + 1, :, :] - Ez[:wx, :, :]) / dx
                    if fz is not None:
                        d = d[:, :, :fz]
                    cx = _cpml(d, psi["Hy_x"][reg], bh[0][slice(x0, x0 + wx)],
                               ch[0][slice(x0, x0 + wx)], 0)
                _acc(F["Hy"], reg, -ch_field, cz, cx)

            # Hz (i+1/2, j+1/2, k): cx=dEy/dx, cy=dEx/dy
            if nx > 1:
                wx = x1 - 1 - x0 if x1 == nx else x1 - x0
                wx = max(wx, 0)
                ry = slice(0, fy) if fy is not None else slice(None)
                reg = (slice(x0, x0 + wx), ry, slice(None))
                cx = cy = None
                if wx > 0:
                    d = (Ey[1:wx + 1, :, :] - Ey[:wx, :, :]) / dx
                    if fy is not None:
                        d = d[:, :fy, :]
                    cx = _cpml(d, psi["Hz_x"][reg], bh[0][slice(x0, x0 + wx)],
                               ch[0][slice(x0, x0 + wx)], 0)
                if fy is not None and wx > 0:
                    d = (Ex[:wx, 1:, :] - Ex[:wx, :-1, :]) / dy
                    cy = _cpml(d, psi["Hz_y"][reg], bh[1][ry], ch[1][ry], 1)
                _acc(F["Hz"], reg, -ch_field, cx, cy)

        # ---------------- E pass over one tile [x0:x1] -------------------- #
        def e_pass(x0, x1):
            xl = max(x0 - 1, 0)                  # low-x halo for backward d/dx
            Hx = np.asarray(F["Hx"][xl:x1]); Hy = np.asarray(F["Hy"][xl:x1])
            Hz = np.asarray(F["Hz"][xl:x1])
            off = x0 - xl                        # local index of global x0
            w = x1 - x0

            by = slice(1, ny) if ny > 1 else slice(None)   # backward-diff region
            bz = slice(1, nz) if nz > 1 else slice(None)

            # Ex (i+1/2, j, k): no d/dx.  cy=dHz/dy, cz=dHy/dz
            if ny > 1 or nz > 1:
                reg = (slice(x0, x1), by, bz)
                cy = cz = None
                Hzs = Hz[off:off + w]; Hys = Hy[off:off + w]
                if ny > 1:
                    d = (Hzs[:, 1:, :] - Hzs[:, :-1, :]) / dy
                    if nz > 1:
                        d = d[:, :, 1:]
                    cy = _cpml(d, psi["Ex_y"][reg], be[1][by], ce[1][by], 1)
                if nz > 1:
                    d = (Hys[:, :, 1:] - Hys[:, :, :-1]) / dz
                    if ny > 1:
                        d = d[:, 1:, :]
                    cz = _cpml(d, psi["Ex_z"][reg], be[2][bz], ce[2][bz], 2)
                _acc(F["Ex"], reg, ce_field[reg], cy, cz)

            # Ey (i, j+1/2, k): cz=dHx/dz, cx=dHz/dx
            if nx > 1:
                xs = max(x0, 1)                  # region starts at global x=1
                wx = x1 - xs
                if wx > 0:
                    reg = (slice(xs, x1), slice(None), bz)
                    cz = cx = None
                    a = xs - xl
                    if nz > 1:
                        Hxs = Hx[a:a + wx]
                        d = (Hxs[:, :, 1:] - Hxs[:, :, :-1]) / dz
                        cz = _cpml(d, psi["Ey_z"][reg], be[2][bz], ce[2][bz], 2)
                    d = (Hz[a:a + wx] - Hz[a - 1:a - 1 + wx]) / dx
                    if nz > 1:
                        d = d[:, :, 1:]
                    cx = _cpml(d, psi["Ey_x"][reg], be[0][slice(xs, x1)],
                               ce[0][slice(xs, x1)], 0)
                    _acc(F["Ey"], reg, ce_field[reg], cz, cx)

            # Ez (i, j, k+1/2): cx=dHy/dx, cy=dHx/dy
            if nx > 1:
                xs = max(x0, 1)
                wx = x1 - xs
                if wx > 0:
                    reg = (slice(xs, x1), by, slice(None))
                    cx = cy = None
                    a = xs - xl
                    d = (Hy[a:a + wx] - Hy[a - 1:a - 1 + wx]) / dx
                    if ny > 1:
                        d = d[:, 1:, :]
                    cx = _cpml(d, psi["Ez_x"][reg], be[0][slice(xs, x1)],
                               ce[0][slice(xs, x1)], 0)
                    if ny > 1:
                        Hxs = Hx[a:a + wx]
                        d = (Hxs[:, 1:, :] - Hxs[:, :-1, :]) / dy
                        cy = _cpml(d, psi["Ez_y"][reg], be[1][by], ce[1][by], 1)
                    _acc(F["Ez"], reg, ce_field[reg], cx, cy)

        def _acc(Fc, reg, k, ca, cb):
            curl = None
            if ca is not None:
                curl = ca
            if cb is not None:
                curl = -cb if curl is None else curl - cb
            if curl is None:
                return
            Fc[reg] = np.asarray(Fc[reg]) + k * curl

        # ---------------- monitors (FieldMonitor only) -------------------- #
        result = Result(times=np.arange(n_steps) * dt)
        rec_fields: Dict[str, Dict[str, object]] = {}
        rec_times: Dict[str, list] = {}
        rec_steps: Dict[str, set] = {}
        rec_n: Dict[str, int] = {}
        rec_count: Dict[str, int] = {}
        rec_zslice: Dict[str, slice] = {}
        rec_codec: Dict[str, object] = {}
        for m in sim.monitors:
            if not isinstance(m, FieldMonitor):
                continue
            rec_fields[m.name] = {c: None for c in m.components}
            rec_times[m.name] = []
            rec_count[m.name] = 0
            rec_steps[m.name] = ({int(round(t / dt)) for t in m.times}
                                 if m.times is not None
                                 else set(range(0, n_steps, m.interval)))
            rec_n[m.name] = sum(1 for s in rec_steps[m.name] if 0 <= s < n_steps)
            if m.plane_z is not None:
                zi = int(np.argmin(np.abs(np.asarray(grid.coords[2]) - m.plane_z)))
                rec_zslice[m.name] = slice(zi, zi + 1)
            else:
                rec_zslice[m.name] = slice(None)
            if m.compression is not None:
                rec_codec[m.name] = get_codec(m.compression)

        source_cells = [(src, *grid.index_at(src.position)) for src in sim.sources]

        def record(m, step):
            ds = m.downsample
            zsl = rec_zslice[m.name]
            idx = rec_count[m.name]
            store = rec_fields[m.name]
            compressed = m.compression is not None
            for c in m.components:
                src = F[c]
                if zsl != slice(None):
                    snap = np.asarray(src[::ds, ::ds, zsl])
                elif ds > 1:
                    snap = np.asarray(src[::ds, ::ds, ::ds])
                else:
                    snap = np.array(src[:])
                arr = store[c]
                if compressed:
                    if arr is None:
                        arr = CompressedFieldSeries(snap.shape, mon_dtype,
                                                    rec_codec[m.name],
                                                    bits=m.compression_bits)
                        store[c] = arr
                    arr.append(snap)
                    continue
                if arr is None:
                    arr = np.empty((rec_n[m.name],) + snap.shape, dtype=mon_dtype)
                    store[c] = arr
                arr[idx] = snap
            rec_times[m.name].append(step * dt)
            rec_count[m.name] += 1

        # ---------------- time loop --------------------------------------- #
        for step in range(n_steps):
            if sim.progress_callback is not None and \
                    step % max(n_steps // 100, 1) == 0:
                sim.progress_callback(step, n_steps)
            t_h = (step + 0.5) * dt
            t_e = (step + 1.0) * dt

            for (x0, x1) in tiles:
                h_pass(x0, x1)
            for src, i, j, k in source_cells:
                if src.component[0] == "H":
                    F[src.component][i, j, k] += \
                        src.amplitude * src.waveform(np.array([t_h]))[0]

            for (x0, x1) in tiles:
                e_pass(x0, x1)
            for src, i, j, k in source_cells:
                if src.component[0] == "E":
                    F[src.component][i, j, k] += \
                        src.amplitude * src.waveform(np.array([t_e]))[0]

            for m in sim.monitors:
                if isinstance(m, FieldMonitor) and step in rec_steps[m.name]:
                    record(m, step)

        if sim.progress_callback is not None:
            sim.progress_callback(n_steps, n_steps)

        for m in sim.monitors:
            if not isinstance(m, FieldMonitor):
                continue
            if rec_count[m.name] > 0:
                if m.compression is not None:
                    for c in m.components:
                        rec_fields[m.name][c].finalize()
                result.fields[m.name] = {c: rec_fields[m.name][c]
                                         for c in m.components}
                result.monitor_times[m.name] = np.array(rec_times[m.name])
            else:
                result.fields[m.name] = {c: np.zeros((0,) + grid.shape)
                                         for c in m.components}
                result.monitor_times[m.name] = np.array([])
        return result
    finally:
        F = ce_field = None
        psi.clear()
        if made_tmp:
            for p in created:
                try:
                    os.unlink(p)
                except OSError:
                    pass
            try:
                os.rmdir(tmp)
            except OSError:
                pass
