"""Rust backend: the fused Yee+CPML kernel compiled from ``rust/src/lib.rs``.

The compiled extension (``photonfdtd._photonfdtd_rs``) runs the *time loop*
itself in chunks: soft-source waveforms are precomputed into per-step tables
(exactly as the JAX backend does in ``_build_static``), and Python is
re-entered only at monitor-record steps to copy snapshots out. The kernel is
a port of the Numba ``_update_fields_numba`` body (same maths, no fastmath),
parallelised over x with rayon, so results track the NumPy reference to
double-precision round-off.

Build the extension with::

    cd rust && cargo build --release
    cp target/release/lib_photonfdtd_rs.so ../src/photonfdtd/_photonfdtd_rs.so

Supported: 1D/2D/3D, CPML, point/soft sources (hence ModeSource /
SinglePhotonSource via expansion), FieldMonitor (interval/times, downsample,
plane_z) and DFTMonitor. Not supported (raises): float32 precision,
dispersive media, subpixel smoothing, ChargedParticle, FluxMonitor (it needs
every-step accumulation, which would defeat the chunked loop), and
FieldMonitor(compression=).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .constants import EPS_0, MU_0
from .monitors import FieldMonitor, FluxMonitor, DFTMonitor

_COMP_CODE = {"Ex": 0, "Ey": 1, "Ez": 2, "Hx": 3, "Hy": 4, "Hz": 5}

_RS = None


def _rust_module():
    global _RS
    if _RS is None:
        try:
            from . import _photonfdtd_rs as _RS
        except ImportError as e:
            raise RuntimeError(
                "backend='rust' but the compiled extension photonfdtd."
                "_photonfdtd_rs is missing. Build it with 'cargo build "
                "--release' in the rust/ directory and copy "
                "target/release/lib_photonfdtd_rs.so to "
                "src/photonfdtd/_photonfdtd_rs.so."
            ) from e
    return _RS


def rust_available() -> bool:
    try:
        _rust_module()
        return True
    except RuntimeError:
        return False


def _validate(sim) -> None:
    from .simulation import _COMPUTE_KEYS
    dts = {np.dtype(sim.dtypes[k]) for k in _COMPUTE_KEYS}
    if dts != {np.dtype(np.float64)}:
        raise NotImplementedError(
            "the Rust backend runs in float64 only; use precision='float64' "
            "(monitor storage precision may still differ)."
        )
    if sim.subpixel:
        raise NotImplementedError(
            "subpixel=True is not supported on the Rust backend; use the "
            "NumPy or JAX backend."
        )
    if getattr(sim, "_has_dispersion", False):
        raise NotImplementedError(
            "dispersive media are not supported on the Rust backend; use the "
            "NumPy or JAX backend."
        )
    if sim.particle_sources:
        raise NotImplementedError(
            "ChargedParticle sources are not supported on the Rust backend; "
            "use the NumPy, Numba, or JAX backend."
        )
    for m in sim.monitors:
        if isinstance(m, FluxMonitor):
            raise NotImplementedError(
                "FluxMonitor needs every-step accumulation and is not "
                "supported on the Rust backend; use a DFTMonitor port plane "
                "or the NumPy/JAX backend."
            )
        if isinstance(m, FieldMonitor) and m.compression is not None:
            raise NotImplementedError(
                "FieldMonitor(compression=) is a host-side store not wired "
                "to the Rust backend; drop compression for backend='rust'."
            )


def run_rust(sim):
    """Run ``sim`` on the Rust backend and return a Result."""
    rs = _rust_module()
    from .simulation import Result

    _validate(sim)
    grid = sim.grid
    nx, ny, nz = grid.shape
    dx, dy, dz = (d if d > 0 else 1.0 for d in grid.cell_size)
    dt = sim.dt
    n_steps = sim.n_steps
    mon_dtype = sim.dtypes["monitors"]

    # Field arrays, C-contiguous float64 (the kernel indexes flat
    # (i*ny + j)*nz + k).
    z = lambda: np.zeros((nx, ny, nz), dtype=np.float64)
    Ex, Ey, Ez, Hx, Hy, Hz = z(), z(), z(), z(), z(), z()

    ce_field = np.ascontiguousarray(dt / (np.asarray(sim.eps_r, dtype=np.float64) * EPS_0))
    ch_field = dt / MU_0
    b_e = [np.ascontiguousarray(a, dtype=np.float64) for a in sim._b_e]
    c_e = [np.ascontiguousarray(a, dtype=np.float64) for a in sim._c_e]
    b_h = [np.ascontiguousarray(a, dtype=np.float64) for a in sim._b_h]
    c_h = [np.ascontiguousarray(a, dtype=np.float64) for a in sim._c_h]

    # CPML psi state, compacted to the PML slabs: along each psi array's
    # derivative axis only the cells with a nonzero CPML `c` coefficient are
    # stored, addressed via a per-axis compact-index map (-1 = bulk, no psi).
    # Cells with c == 0 never develop psi, so this is exactly equivalent to
    # dense storage while keeping the stepping state at ~6 field arrays +
    # ce_field + thin PML strips - sized for whole-chip domains.
    def _pml_map(c_arr):
        m = np.full(c_arr.size, -1, dtype=np.int32)
        idx = np.flatnonzero(c_arr != 0.0)
        m[idx] = np.arange(idx.size, dtype=np.int32)
        return m, idx.size

    maps_e, maps_h, pe, ph = [], [], [], []
    for ax in range(3):
        m, n = _pml_map(c_e[ax]); maps_e.append(m); pe.append(n)
        m, n = _pml_map(c_h[ax]); maps_h.append(m); ph.append(n)

    zc = lambda shape: np.zeros(shape, dtype=np.float64)
    psi = [
        zc((nx, pe[1], nz)),   # psi_Ex_y
        zc((nx, ny, pe[2])),   # psi_Ex_z
        zc((nx, ny, pe[2])),   # psi_Ey_z
        zc((pe[0], ny, nz)),   # psi_Ey_x
        zc((pe[0], ny, nz)),   # psi_Ez_x
        zc((nx, pe[1], nz)),   # psi_Ez_y
        zc((nx, ph[1], nz)),   # psi_Hx_y
        zc((nx, ny, ph[2])),   # psi_Hx_z
        zc((nx, ny, ph[2])),   # psi_Hy_z
        zc((ph[0], ny, nz)),   # psi_Hy_x
        zc((ph[0], ny, nz)),   # psi_Hz_x
        zc((nx, ph[1], nz)),   # psi_Hz_y
    ]

    # Precomputed soft-source tables: value added to (comp, i, j, k) at each
    # step, evaluated at the component's Yee time (H at t+dt/2, E at t+dt).
    n_src = len(sim.sources)
    src_comp = np.zeros(n_src, dtype=np.int64)
    src_idx = np.zeros((n_src, 3), dtype=np.int64)
    src_vals = np.zeros((n_src, max(n_steps, 1)), dtype=np.float64)
    steps = np.arange(n_steps)
    for s, src in enumerate(sim.sources):
        i, j, k = grid.index_at(src.position)
        src_comp[s] = _COMP_CODE[src.component]
        src_idx[s] = (i, j, k)
        t = (steps + (0.5 if src.component[0] == "H" else 1.0)) * dt
        src_vals[s] = np.asarray(src.amplitude * src.waveform(t), dtype=np.float64)

    comps = {"Ex": Ex, "Ey": Ey, "Ez": Ez, "Hx": Hx, "Hy": Hy, "Hz": Hz}

    # ---- monitor bookkeeping (mirrors the NumPy path) ---- #
    result = Result(times=np.arange(n_steps) * dt)
    rec_fields: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
    rec_times: Dict[str, list] = {}
    rec_steps: Dict[str, set] = {}
    rec_count: Dict[str, int] = {}
    rec_zslice: Dict[str, slice] = {}
    dft_accum: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
    dft_omega: Dict[str, np.ndarray] = {}
    dft_steps: Dict[str, set] = {}
    dft_plane: Dict[str, Optional[tuple]] = {}
    for m in sim.monitors:
        if isinstance(m, FieldMonitor):
            rec_fields[m.name] = {c: None for c in m.components}
            rec_times[m.name] = []
            rec_count[m.name] = 0
            if m.times is not None:
                rec_steps[m.name] = {int(round(t / dt)) for t in m.times
                                     if 0 <= int(round(t / dt)) < n_steps}
            else:
                rec_steps[m.name] = set(range(0, n_steps, m.interval))
            if m.plane_z is not None:
                zi = int(np.argmin(np.abs(np.asarray(grid.coords[2]) - m.plane_z)))
                rec_zslice[m.name] = slice(zi, zi + 1)
            else:
                rec_zslice[m.name] = slice(None)
        elif isinstance(m, DFTMonitor):
            dft_accum[m.name] = {c: None for c in m.components}
            dft_omega[m.name] = 2.0 * np.pi * np.asarray(m.freqs, dtype=np.float64)
            dft_steps[m.name] = set(range(0, n_steps, m.interval))
            result.dft_freqs[m.name] = np.asarray(m.freqs, dtype=np.float64)
            pl = m.plane()
            if pl is not None:
                ax, pos = pl
                ci = int(np.argmin(np.abs(np.asarray(grid.coords[ax]) - pos)))
                dft_plane[m.name] = (ax, ci)
            else:
                dft_plane[m.name] = None

    def snap_view(field, ds, zsl):
        if zsl != slice(None):
            return field[::ds, ::ds, zsl]
        if ds > 1:
            return field[::ds, ::ds, ::ds]
        return field

    def snap_view_plane(field, ds, plane):
        sl = [slice(None, None, ds) if ds > 1 else slice(None) for _ in range(3)]
        if plane is not None:
            ax, ci = plane
            sl[ax] = slice(ci, ci + 1)
        return field[tuple(sl)]

    def record_field(m, step):
        store = rec_fields[m.name]
        idx = rec_count[m.name]
        n_rec = sum(1 for s in rec_steps[m.name] if 0 <= s < n_steps)
        for c in m.components:
            snap = snap_view(comps[c], m.downsample, rec_zslice[m.name])
            arr = store[c]
            if arr is None:
                arr = np.empty((n_rec,) + snap.shape, dtype=mon_dtype)
                store[c] = arr
            arr[idx] = snap
        rec_times[m.name].append(step * dt)
        rec_count[m.name] += 1

    def record_dft(m, step):
        t_e = (step + 1.0) * dt
        t_h = (step + 0.5) * dt
        omega = dft_omega[m.name]
        store = dft_accum[m.name]
        for c in m.components:
            snap = snap_view_plane(comps[c], m.downsample, dft_plane[m.name])
            t = t_e if c[0] == "E" else t_h
            phase = np.exp(1j * omega * t).reshape((-1,) + (1,) * snap.ndim)
            acc = store[c]
            if acc is None:
                acc = np.zeros((omega.size,) + snap.shape, dtype=np.complex128)
                store[c] = acc
            acc += phase * snap[None, ...] * dt

    # Steps after which Python must look at the fields.
    stop_steps = sorted(set().union(*rec_steps.values(), *dft_steps.values())
                        if (rec_steps or dft_steps) else set())

    progress_interval = max(n_steps // 100, 1)
    pos = 0

    def _advance(to_step_exclusive):
        """Run steps [pos, to_step_exclusive) in Rust (chunked for progress)."""
        nonlocal pos
        while pos < to_step_exclusive:
            n_sub = min(to_step_exclusive - pos,
                        progress_interval if sim.progress_callback else
                        to_step_exclusive - pos)
            rs.step_range(Ex, Ey, Ez, Hx, Hy, Hz, *psi,
                          b_e, c_e, b_h, c_h, maps_e, maps_h,
                          ce_field, ch_field,
                          dx, dy, dz, src_comp, src_idx, src_vals,
                          pos, n_sub)
            pos += n_sub
            if sim.progress_callback is not None:
                sim.progress_callback(min(pos, n_steps), n_steps)

    for s in stop_steps:
        _advance(s + 1)          # record fires after step s completes
        for m in sim.monitors:
            if isinstance(m, FieldMonitor) and s in rec_steps[m.name]:
                record_field(m, s)
            elif isinstance(m, DFTMonitor) and s in dft_steps[m.name]:
                record_dft(m, s)
    _advance(n_steps)
    if sim.progress_callback is not None:
        sim.progress_callback(n_steps, n_steps)

    # ---- assemble Result (mirrors the NumPy path) ---- #
    for m in sim.monitors:
        if isinstance(m, FieldMonitor):
            if rec_count[m.name] > 0:
                result.fields[m.name] = {c: rec_fields[m.name][c]
                                         for c in m.components}
                result.monitor_times[m.name] = np.array(rec_times[m.name])
            else:
                result.fields[m.name] = {c: np.zeros((0,) + grid.shape)
                                         for c in m.components}
                result.monitor_times[m.name] = np.array([])
        elif isinstance(m, DFTMonitor):
            out = {}
            for c in m.components:
                acc = dft_accum[m.name][c]
                out[c] = acc if acc is not None else \
                    np.zeros((len(m.freqs),) + grid.shape, dtype=np.complex128)
            result.dft[m.name] = out
    return result
