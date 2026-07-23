"""Rust backends: compiled Yee+CPML stepping cores from ``rust/src/``.

Two engines share one Python driver (source tables, chunked time loop,
monitor recording):

* ``backend="rust"`` - CPU (rayon), float64. The whole time loop runs
  natively; Python is re-entered only at monitor-record steps.
* ``backend="rust-cuda"`` - GPU (CUDA, feature ``cuda``), float64 or
  float32. All state is device-resident; the host is touched only when a
  monitor records.

Both store CPML psi state compacted to the PML slabs (cells whose CPML `c`
coefficient is nonzero never develop psi, so this is exactly equivalent to
dense storage), keeping the stepping state at six field arrays + the update
coefficient + thin PML strips - sized for whole-chip domains.

Build (CPU only / with CUDA)::

    cd rust && cargo build --release [--features cuda]
    cp target/release/lib_photonfdtd_rs.so ../src/photonfdtd/_photonfdtd_rs.so

Supported: 1D/2D/3D, CPML, point/soft sources (hence ModeSource /
SinglePhotonSource via expansion), FieldMonitor (interval/times, downsample,
plane_z) and DFTMonitor. Not supported (raises): dispersive media, subpixel
smoothing, ChargedParticle, FluxMonitor (needs every-step accumulation),
FieldMonitor(compression=), and float32 on the CPU core (CUDA-only).
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


def cuda_available() -> bool:
    """Whether the extension was built with the CUDA feature *and* a usable
    device is present."""
    try:
        rs = _rust_module()
    except RuntimeError:
        return False
    if not getattr(rs, "CUDA_BUILT", False):
        return False
    try:
        rs.CudaStepper.device_info()
        return True
    except Exception:
        return False


def _validate(sim, cuda: bool) -> None:
    from .simulation import _COMPUTE_KEYS
    dts = {np.dtype(sim.dtypes[k]) for k in _COMPUTE_KEYS}
    if cuda:
        if dts not in ({np.dtype(np.float64)}, {np.dtype(np.float32)}):
            raise NotImplementedError(
                "backend='rust-cuda' needs one uniform compute precision "
                "('float32' or 'float64'; monitor storage may differ)."
            )
    elif dts != {np.dtype(np.float64)}:
        raise NotImplementedError(
            "the Rust CPU backend runs in float64 only; use "
            "precision='float64' (monitor storage precision may differ), or "
            "backend='rust-cuda' for float32."
        )
    if sim.subpixel:
        raise NotImplementedError(
            "subpixel=True is not supported on the Rust backends; use the "
            "NumPy or JAX backend."
        )
    if getattr(sim, "_has_dispersion", False):
        raise NotImplementedError(
            "dispersive media are not supported on the Rust backends; use "
            "the NumPy or JAX backend."
        )
    if sim.particle_sources:
        raise NotImplementedError(
            "ChargedParticle sources are not supported on the Rust backends; "
            "use the NumPy, Numba, or JAX backend."
        )
    for m in sim.monitors:
        if isinstance(m, FluxMonitor):
            raise NotImplementedError(
                "FluxMonitor needs every-step accumulation and is not "
                "supported on the Rust backends; use a DFTMonitor port plane "
                "or the NumPy/JAX backend."
            )
        if isinstance(m, FieldMonitor) and m.compression is not None:
            raise NotImplementedError(
                "FieldMonitor(compression=) is a host-side store not wired "
                "to the Rust backends; drop compression."
            )


# ------------------------------------------------------------------ #
# Static tables shared by both engines.
# ------------------------------------------------------------------ #
def _pml_map(c_arr):
    """Compact-index map along one axis: position of each nonzero-c cell in
    the compact psi array, -1 in the bulk."""
    m = np.full(c_arr.size, -1, dtype=np.int32)
    idx = np.flatnonzero(c_arr != 0.0)
    m[idx] = np.arange(idx.size, dtype=np.int32)
    return m


def _build_tables(sim, npdt):
    grid = sim.grid
    dt = sim.dt
    n_steps = sim.n_steps
    t = dict(
        ce_field=np.ascontiguousarray(
            dt / (np.asarray(sim.eps_r, dtype=np.float64) * EPS_0), dtype=npdt),
        ch_field=npdt(dt / MU_0),
        b_e=[np.ascontiguousarray(a, dtype=npdt) for a in sim._b_e],
        c_e=[np.ascontiguousarray(a, dtype=npdt) for a in sim._c_e],
        b_h=[np.ascontiguousarray(a, dtype=npdt) for a in sim._b_h],
        c_h=[np.ascontiguousarray(a, dtype=npdt) for a in sim._c_h],
    )
    t["maps_e"] = [_pml_map(a) for a in t["c_e"]]
    t["maps_h"] = [_pml_map(a) for a in t["c_h"]]

    # Soft-source tables: value added to (comp, i, j, k) at each step,
    # evaluated at the component's Yee time (H at t+dt/2, E at t+dt).
    n_src = len(sim.sources)
    steps = np.arange(n_steps)
    t["src_comp"] = np.zeros(n_src, dtype=np.int64)
    t["src_idx"] = np.zeros((n_src, 3), dtype=np.int64)
    t["src_vals"] = np.zeros((n_src, max(n_steps, 1)), dtype=npdt)
    for s, src in enumerate(sim.sources):
        i, j, k = grid.index_at(src.position)
        t["src_comp"][s] = _COMP_CODE[src.component]
        t["src_idx"][s] = (i, j, k)
        tt = (steps + (0.5 if src.component[0] == "H" else 1.0)) * dt
        t["src_vals"][s] = np.asarray(src.amplitude * src.waveform(tt), dtype=npdt)
    return t


class _CpuEngine:
    """Rayon CPU core: fields/psi live as numpy arrays, stepped in place."""

    def __init__(self, sim):
        rs = _rust_module()
        self._rs = rs
        nx, ny, nz = sim.grid.shape
        self._t = t = _build_tables(sim, np.float64)
        self._dxyz = tuple(d if d > 0 else 1.0 for d in sim.grid.cell_size)
        z = lambda shape: np.zeros(shape, dtype=np.float64)
        self._fields = [z((nx, ny, nz)) for _ in range(6)]
        pe = [int((m >= 0).sum()) for m in t["maps_e"]]
        ph = [int((m >= 0).sum()) for m in t["maps_h"]]
        self._psi = [
            z((nx, pe[1], nz)), z((nx, ny, pe[2])),   # psi_Ex_y, psi_Ex_z
            z((nx, ny, pe[2])), z((pe[0], ny, nz)),   # psi_Ey_z, psi_Ey_x
            z((pe[0], ny, nz)), z((nx, pe[1], nz)),   # psi_Ez_x, psi_Ez_y
            z((nx, ph[1], nz)), z((nx, ny, ph[2])),   # psi_Hx_y, psi_Hx_z
            z((nx, ny, ph[2])), z((ph[0], ny, nz)),   # psi_Hy_z, psi_Hy_x
            z((ph[0], ny, nz)), z((nx, ph[1], nz)),   # psi_Hz_x, psi_Hz_y
        ]

    def advance(self, step0: int, n_sub: int) -> None:
        t = self._t
        dx, dy, dz = self._dxyz
        self._rs.step_range(*self._fields, *self._psi,
                            t["b_e"], t["c_e"], t["b_h"], t["c_h"],
                            t["maps_e"], t["maps_h"],
                            t["ce_field"], float(t["ch_field"]),
                            dx, dy, dz,
                            t["src_comp"], t["src_idx"], t["src_vals"],
                            step0, n_sub)

    def get(self, comp: str) -> np.ndarray:
        return self._fields[_COMP_CODE[comp]]


class _CudaEngine:
    """CUDA core: all state device-resident; components are downloaded (and
    cached until the next advance) only when a monitor records."""

    def __init__(self, sim, npdt):
        import sys
        rs = _rust_module()
        if not getattr(rs, "CUDA_BUILT", False):
            raise RuntimeError(
                "backend='rust-cuda' but the extension was built without the "
                "CUDA feature. Rebuild with 'cargo build --release --features "
                "cuda' in rust/."
            )
        t = _build_tables(sim, npdt)
        dx, dy, dz = (npdt(d) if d > 0 else npdt(1.0) for d in sim.grid.cell_size)
        ctor = rs.CudaStepper.new_f64 if npdt == np.float64 else rs.CudaStepper.new_f32
        self._stepper = ctor(
            t["ce_field"], t["b_e"], t["c_e"], t["b_h"], t["c_h"],
            [m for m in t["maps_e"]], [m for m in t["maps_h"]],
            npdt(t["ch_field"]), dx, dy, dz,
            t["src_comp"].astype(np.int32),
            t["src_idx"].astype(np.int32),
            t["src_vals"])
        name, free_mb, total_mb = rs.CudaStepper.device_info()
        if sim.verbose:
            print(f"[photonfdtd] Rust CUDA backend, device: {name} "
                  f"({free_mb}/{total_mb} MB free)", file=sys.stderr, flush=True)
        self._cache: Dict[str, np.ndarray] = {}

    def advance(self, step0: int, n_sub: int) -> None:
        self._cache.clear()
        self._stepper.run_steps(step0, n_sub)

    def get(self, comp: str) -> np.ndarray:
        if comp not in self._cache:
            self._cache[comp] = self._stepper.read_field(_COMP_CODE[comp])
        return self._cache[comp]


# ------------------------------------------------------------------ #
# Shared driver: chunked time loop + monitor recording.
# ------------------------------------------------------------------ #
def _run_loop(sim, engine):
    from .simulation import Result

    grid = sim.grid
    dt = sim.dt
    n_steps = sim.n_steps
    mon_dtype = sim.dtypes["monitors"]

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
            snap = snap_view(engine.get(c), m.downsample, rec_zslice[m.name])
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
            snap = snap_view_plane(engine.get(c), m.downsample, dft_plane[m.name])
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
        """Run steps [pos, to_step_exclusive) (chunked for progress)."""
        nonlocal pos
        while pos < to_step_exclusive:
            n_sub = min(to_step_exclusive - pos,
                        progress_interval if sim.progress_callback else
                        to_step_exclusive - pos)
            engine.advance(pos, n_sub)
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


def run_rust(sim):
    """Run ``sim`` on the Rust CPU backend and return a Result."""
    _validate(sim, cuda=False)
    return _run_loop(sim, _CpuEngine(sim))


def run_rust_cuda(sim):
    """Run ``sim`` on the Rust CUDA backend and return a Result."""
    _validate(sim, cuda=True)
    npdt = np.dtype(sim.dtypes["Ex"]).type
    return _run_loop(sim, _CudaEngine(sim, npdt))
