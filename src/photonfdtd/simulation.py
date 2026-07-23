"""Simulation - the FDTD time-stepping engine.

Workflow

    sim = Simulation(grid, structures, sources, monitors, run_time)
    result = sim.run()
    snapshot = result.fields["my_mon"]["Ez"][time_index]

The solver operates on a uniform Yee grid and supports 1D, 2D, and 3D
simulations. The dimensionality is determined by which axes of `grid.shape`
exceed 1: collapsed axes naturally drop out of the curl operations.

Numerical details
-----------------
- The time step is set to 0.99 of the Courant limit by default.
- Absorbing boundaries are CPML with kappa=1 (Roden & Gedney 2000).
- Sources are soft additive: they add to one field component at one cell.
- Materials are isotropic dielectrics: non-dispersive (eps_r per cell) or
  dispersive (Lorentz/Drude/Sellmeier poles advanced by the ADE method; see
  _build_ade / photonfdtd.materials). Anisotropic subpixel smoothing of
  interfaces is available via Simulation(subpixel=True) (photonfdtd.smoothing).

Yee staggering (all six components live on different sub-grids):

    Ex[i,j,k] at (i+1/2,  j,      k     )
    Ey[i,j,k] at (i,      j+1/2,  k     )
    Ez[i,j,k] at (i,      j,      k+1/2 )
    Hx[i,j,k] at (i,      j+1/2,  k+1/2 )
    Hy[i,j,k] at (i+1/2,  j,      k+1/2 )
    Hz[i,j,k] at (i+1/2,  j+1/2,  k     )

H is updated at half-integer times from forward differences of E; E is
updated at integer times from backward differences of H.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import math
import os
import warnings
import numpy as np

from .constants import C_0, EPS_0, MU_0
from .grid import Grid
from .geometry import Box
from .sources import PointDipole, ChargedParticle
from .monitors import FieldMonitor, FluxMonitor, DFTMonitor
from .storage import CompressedFieldSeries, get_codec

from .boundaries import cpml_coeffs_1axis, CPMLParams

# Optional CuPy GPU backend.
try:
    import cupy as _cp
    _GPU_AVAILABLE = True
except ImportError:
    _cp = None
    _GPU_AVAILABLE = False

# Optional Numba CPU backend (JIT-compiled parallel loops).
try:
    from numba import njit, prange as _prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

if _NUMBA_AVAILABLE:
    from numba import njit, prange as _prange

    @njit(parallel=True, cache=True, fastmath=True)
    def _update_fields_numba(
        Ex, Ey, Ez, Hx, Hy, Hz,
        psi_Ex_y, psi_Ex_z, psi_Ey_z, psi_Ey_x,
        psi_Ez_x, psi_Ez_y, psi_Hx_y, psi_Hx_z,
        psi_Hy_z, psi_Hy_x, psi_Hz_x, psi_Hz_y,
        bx_e, cx_e, by_e, cy_e, bz_e, cz_e,
        bx_h, cx_h, by_h, cy_h, bz_h, cz_h,
        ce_field, ch_field,
        dx, dy, dz,
    ):
        """One complete H+E Yee update for 1D/2D/3D domains, Numba-parallel over x.

        Loop bounds adapt to collapsed (size-1) axes: a forward difference needs
        the neighbour ``+1`` cell, so it runs over ``n-1`` cells when ``n > 1``
        but over the single cell when ``n == 1`` (with that derivative term set
        to zero). Backward differences likewise start at index 1 only when the
        axis is resolved. This reduces exactly to the dense 3D update when all
        three axes are resolved, and mirrors the NumPy reference path's
        reduced-dimension handling otherwise.
        """
        nx, ny, nz = Ex.shape
        # Forward-difference extents (need the i+1/j+1/k+1 neighbour).
        fx = nx - 1 if nx > 1 else 1
        fy = ny - 1 if ny > 1 else 1
        fz = nz - 1 if nz > 1 else 1
        # Backward-difference starts (need the i-1/j-1/k-1 neighbour).
        bx = 1 if nx > 1 else 0
        by = 1 if ny > 1 else 0
        bz = 1 if nz > 1 else 0

        # ---- H update (forward differences of E) ----
        # Hx at (i, j+1/2, k+1/2): full i, forward j (dEz/dy), forward k (dEy/dz)
        for i in _prange(nx):
            for j in range(fy):
                for k in range(fz):
                    if ny > 1:
                        dEz_dy = (Ez[i, j+1, k] - Ez[i, j, k]) / dy
                        psi_Hx_y[i, j, k] = by_h[j] * psi_Hx_y[i, j, k] + cy_h[j] * dEz_dy
                        cy = dEz_dy + psi_Hx_y[i, j, k]
                    else:
                        cy = 0.0
                    if nz > 1:
                        dEy_dz = (Ey[i, j, k+1] - Ey[i, j, k]) / dz
                        psi_Hx_z[i, j, k] = bz_h[k] * psi_Hx_z[i, j, k] + cz_h[k] * dEy_dz
                        cz = dEy_dz + psi_Hx_z[i, j, k]
                    else:
                        cz = 0.0
                    Hx[i, j, k] -= ch_field * (cy - cz)
        # Hy at (i+1/2, j, k+1/2): forward i (dEz/dx), full j, forward k (dEx/dz)
        for i in _prange(fx):
            for j in range(ny):
                for k in range(fz):
                    if nz > 1:
                        dEx_dz = (Ex[i, j, k+1] - Ex[i, j, k]) / dz
                        psi_Hy_z[i, j, k] = bz_h[k] * psi_Hy_z[i, j, k] + cz_h[k] * dEx_dz
                        cz = dEx_dz + psi_Hy_z[i, j, k]
                    else:
                        cz = 0.0
                    if nx > 1:
                        dEz_dx = (Ez[i+1, j, k] - Ez[i, j, k]) / dx
                        psi_Hy_x[i, j, k] = bx_h[i] * psi_Hy_x[i, j, k] + cx_h[i] * dEz_dx
                        cx = dEz_dx + psi_Hy_x[i, j, k]
                    else:
                        cx = 0.0
                    Hy[i, j, k] -= ch_field * (cz - cx)
        # Hz at (i+1/2, j+1/2, k): forward i (dEy/dx), forward j (dEx/dy), full k
        for i in _prange(fx):
            for j in range(fy):
                for k in range(nz):
                    if nx > 1:
                        dEy_dx = (Ey[i+1, j, k] - Ey[i, j, k]) / dx
                        psi_Hz_x[i, j, k] = bx_h[i] * psi_Hz_x[i, j, k] + cx_h[i] * dEy_dx
                        cx = dEy_dx + psi_Hz_x[i, j, k]
                    else:
                        cx = 0.0
                    if ny > 1:
                        dEx_dy = (Ex[i, j+1, k] - Ex[i, j, k]) / dy
                        psi_Hz_y[i, j, k] = by_h[j] * psi_Hz_y[i, j, k] + cy_h[j] * dEx_dy
                        cy = dEx_dy + psi_Hz_y[i, j, k]
                    else:
                        cy = 0.0
                    Hz[i, j, k] -= ch_field * (cx - cy)

        # ---- E update (backward differences of H) ----
        # Ex at (i+1/2, j, k): full i, backward j (dHz/dy), backward k (dHy/dz)
        for i in _prange(nx):
            for j in range(by, ny):
                for k in range(bz, nz):
                    if ny > 1:
                        dHz_dy = (Hz[i, j, k] - Hz[i, j-1, k]) / dy
                        psi_Ex_y[i, j, k] = by_e[j] * psi_Ex_y[i, j, k] + cy_e[j] * dHz_dy
                        cy = dHz_dy + psi_Ex_y[i, j, k]
                    else:
                        cy = 0.0
                    if nz > 1:
                        dHy_dz = (Hy[i, j, k] - Hy[i, j, k-1]) / dz
                        psi_Ex_z[i, j, k] = bz_e[k] * psi_Ex_z[i, j, k] + cz_e[k] * dHy_dz
                        cz = dHy_dz + psi_Ex_z[i, j, k]
                    else:
                        cz = 0.0
                    Ex[i, j, k] += ce_field[i, j, k] * (cy - cz)
        # Ey at (i, j+1/2, k): backward i (dHz/dx), full j, backward k (dHx/dz)
        for i in _prange(bx, nx):
            for j in range(ny):
                for k in range(bz, nz):
                    if nz > 1:
                        dHx_dz = (Hx[i, j, k] - Hx[i, j, k-1]) / dz
                        psi_Ey_z[i, j, k] = bz_e[k] * psi_Ey_z[i, j, k] + cz_e[k] * dHx_dz
                        cz = dHx_dz + psi_Ey_z[i, j, k]
                    else:
                        cz = 0.0
                    if nx > 1:
                        dHz_dx = (Hz[i, j, k] - Hz[i-1, j, k]) / dx
                        psi_Ey_x[i, j, k] = bx_e[i] * psi_Ey_x[i, j, k] + cx_e[i] * dHz_dx
                        cx = dHz_dx + psi_Ey_x[i, j, k]
                    else:
                        cx = 0.0
                    Ey[i, j, k] += ce_field[i, j, k] * (cz - cx)
        # Ez at (i, j, k+1/2): backward i (dHy/dx), backward j (dHx/dy), full k
        for i in _prange(bx, nx):
            for j in range(by, ny):
                for k in range(nz):
                    if nx > 1:
                        dHy_dx = (Hy[i, j, k] - Hy[i-1, j, k]) / dx
                        psi_Ez_x[i, j, k] = bx_e[i] * psi_Ez_x[i, j, k] + cx_e[i] * dHy_dx
                        cx = dHy_dx + psi_Ez_x[i, j, k]
                    else:
                        cx = 0.0
                    if ny > 1:
                        dHx_dy = (Hx[i, j, k] - Hx[i, j-1, k]) / dy
                        psi_Ez_y[i, j, k] = by_e[j] * psi_Ez_y[i, j, k] + cy_e[j] * dHx_dy
                        cy = dHx_dy + psi_Ez_y[i, j, k]
                    else:
                        cy = 0.0
                    Ez[i, j, k] += ce_field[i, j, k] * (cx - cy)


def _get_backend(use_gpu: bool):
    """Return the array module (cupy or numpy) to use for field arrays."""
    if use_gpu:
        if not _GPU_AVAILABLE:
            raise RuntimeError(
                "use_gpu=True but CuPy is not installed or its build does not "
                "match the running driver. Install the CuPy build for your GPU: "
                "'pip install cupy-cuda12x' for an NVIDIA/CUDA GPU, or "
                "'pip install cupy-rocm-5-0' for an AMD/ROCm GPU. Only generic "
                "CuPy array ops are used, so either backend works."
            )
        return _cp
    return np


def _contig_nonzero_blocks(arr1d):
    """Maximal contiguous runs of nonzero entries, as (start, stop) pairs.

    Used to locate the PML slabs along an axis from its CPML coefficient array
    (which is nonzero only inside the PML layers).
    """
    nz = np.flatnonzero(np.asarray(arr1d) != 0.0)
    if nz.size == 0:
        return []
    blocks = []
    start = prev = int(nz[0])
    for idx in nz[1:]:
        idx = int(idx)
        if idx != prev + 1:
            blocks.append((start, prev + 1))
            start = idx
        prev = idx
    blocks.append((start, prev + 1))
    return blocks


# ---------------------------------------------------------------------------- #
# Per-array precision control.
#
# `precision` may be a single dtype string applied to everything (the common
# case, fully back-compatible), or a dict that sets the dtype of individual
# arrays. Addressable array keys:
#
#     'Ex','Ey','Ez','Hx','Hy','Hz'  - the six field components
#     'eps_r'                        - per-cell permittivity / E-update coeff
#     'psi'                          - CPML convolutional state + coefficients
#     'monitors'                     - stored monitor snapshots
#
# plus group aliases that fan out to several arrays at once (most specific key
# wins: individual > e_fields/h_fields > fields > compute > default):
#
#     'default'  - fallback for any key not otherwise set (default 'float64')
#     'compute'  - every stepping array (all fields + eps_r + psi); the natural
#                  "compute precision" knob, paired with 'monitors' for storage
#     'fields'   - all six field components
#     'e_fields' / 'h_fields' - the E or H components only
#
# Example - run the solver in double precision but halve monitor memory:
#     precision={'compute': 'float64', 'monitors': 'float32'}
# Example - mixed fields:
#     precision={'default': 'float32', 'Ez': 'float64'}
# ---------------------------------------------------------------------------- #
_FIELD_KEYS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_E_KEYS = ("Ex", "Ey", "Ez")
_H_KEYS = ("Hx", "Hy", "Hz")
_ARRAY_KEYS = _FIELD_KEYS + ("eps_r", "psi", "monitors")
_COMPUTE_KEYS = _FIELD_KEYS + ("eps_r", "psi")   # all arrays except monitor storage
_GROUP_ALIASES = ("default", "compute", "fields", "e_fields", "h_fields")
_VALID_DTYPES = {"float32": np.float32, "float64": np.float64}


def _resolve_precision(precision) -> Dict[str, np.dtype]:
    """Resolve the `precision` argument to a {array_key: numpy dtype} map.

    `precision` is either a dtype string (applied to every array) or a dict
    keyed by individual array names and/or group aliases (see module notes).
    """
    if isinstance(precision, str):
        if precision not in _VALID_DTYPES:
            raise ValueError("precision must be 'float32' or 'float64'")
        d = _VALID_DTYPES[precision]
        return {k: d for k in _ARRAY_KEYS}

    if not isinstance(precision, dict):
        raise TypeError(
            "precision must be a str ('float32'/'float64') or a dict mapping "
            "array keys to dtype strings"
        )

    allowed = set(_ARRAY_KEYS) | set(_GROUP_ALIASES)
    unknown = set(precision) - allowed
    if unknown:
        raise ValueError(
            f"unknown precision key(s) {sorted(unknown)}; "
            f"allowed keys: {sorted(allowed)}"
        )
    for k, v in precision.items():
        if v not in _VALID_DTYPES:
            raise ValueError(
                f"precision[{k!r}] must be 'float32' or 'float64', got {v!r}"
            )

    def pick(*keys, fallback):
        for k in keys:                       # first listed key present wins
            if k in precision:
                return _VALID_DTYPES[precision[k]]
        return fallback

    default = _VALID_DTYPES[precision.get("default", "float64")]
    out: Dict[str, np.dtype] = {}
    for k in _FIELD_KEYS:
        grp = "e_fields" if k in _E_KEYS else "h_fields"
        out[k] = pick(k, grp, "fields", "compute", fallback=default)
    out["eps_r"] = pick("eps_r", "compute", fallback=default)
    out["psi"] = pick("psi", "compute", fallback=default)
    out["monitors"] = pick("monitors", fallback=default)
    return out


@dataclass
class Result:
    times: np.ndarray
    fields: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    monitor_times: Dict[str, np.ndarray] = field(default_factory=dict)
    flux: Dict[str, float] = field(default_factory=dict)
    # DFTMonitor output: dft[name][component] is a complex array of shape
    # (n_freq, *snapshot_shape); dft_freqs[name] holds the frequencies (Hz).
    dft: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    dft_freqs: Dict[str, np.ndarray] = field(default_factory=dict)


# Auto-backend threshold. With backend="auto", dispatch to JAX once the run is
# at least this many cell-steps (grid cells x timesteps). Below it the NumPy core
# is already sub-second and beats JAX's one-time XLA compile; above it, XLA
# fusion (and the GPU, if present) wins by a wide margin - measured ~4x on CPU
# and ~40x on GPU on a ~7e8 cell-step 3-D run. Module-level so it is tunable.
AUTO_JAX_MIN_CELL_STEPS = 5e7

_JAX_AVAILABLE: Optional[bool] = None


def _jax_available() -> bool:
    """Whether the JAX backend can be imported (cached)."""
    global _JAX_AVAILABLE
    if _JAX_AVAILABLE is None:
        try:
            import jax  # noqa: F401
            _JAX_AVAILABLE = True
        except Exception:
            _JAX_AVAILABLE = False
    return _JAX_AVAILABLE


class Simulation:
    def __init__(
        self,
        grid: Grid,
        structures: Sequence[Box] = (),
        sources: Sequence[PointDipole] = (),
        monitors: Sequence = (),
        run_time: float = 1e-12,
        courant: float = 0.99,
        cpml_params: CPMLParams = CPMLParams(),
        verbose: bool = False,
        use_gpu: bool = False,
        use_numba: bool = False,
        use_jax: bool = False,
        precision="float64",
        subpixel: bool = False,
        subpixel_factor: int = 3,
        backend: str = "auto",
    ) -> None:
        # Working dtype(s) for fields, CPML state, coefficients and monitor
        # storage. float32 halves memory and roughly doubles throughput (FDTD is
        # memory-bandwidth bound) at single precision; the default float64 is
        # bit-for-bit unchanged. `precision` may be a single dtype string or a
        # per-array dict - see _resolve_precision / module notes above.
        self.precision = precision
        self.dtypes = _resolve_precision(precision)
        # Representative compute dtype, kept for back-compat (== the Ex dtype).
        self.dtype = self.dtypes["Ex"]
        # The CuPy in-core GPU backend is superseded by the JAX backend, which
        # runs on the GPU (via XLA) *and* is differentiable. Prefer use_jax=True
        # for GPU work; use_gpu (CuPy) is retained as an optional legacy path
        # (e.g. AMD/ROCm, or GPU out-of-core) and deprecated.
        if use_gpu:
            warnings.warn(
                "use_gpu (CuPy backend) is deprecated in favour of use_jax=True, "
                "which runs on the GPU through XLA and is differentiable. CuPy is "
                "kept as an optional legacy path (AMD/ROCm, GPU out-of-core).",
                DeprecationWarning, stacklevel=2,
            )
        self.use_gpu = use_gpu and _GPU_AVAILABLE
        self.use_numba = use_numba and _NUMBA_AVAILABLE and not self.use_gpu
        # The JAX backend is a separate functional stepper (see jaxbackend.py);
        # it is exclusive of the CuPy/Numba paths and dispatched from run().
        self.use_jax = bool(use_jax)
        if self.use_jax and (self.use_gpu or self.use_numba):
            raise ValueError("use_jax is exclusive of use_gpu / use_numba.")
        # Backend selection. "auto" (the default) dispatches to the JAX backend
        # when it is installed and the run is big enough for XLA fusion / GPU to
        # pay off (see _use_jax_backend), otherwise the NumPy core. Explicit
        # values ("numpy"/"jax") force the choice; the use_jax / use_gpu /
        # use_numba booleans, if set, still win for back-compatibility.
        if backend not in ("auto", "numpy", "jax", "rust", "rust-cuda"):
            raise ValueError(
                "backend must be 'auto', 'numpy', 'jax', 'rust', or 'rust-cuda'")
        self.backend = backend
        # The fused Numba kernel is a single specialization over all stepping
        # arrays; mixing dtypes among them has no real use case and unclear
        # promotion semantics, so require one compute dtype there. Monitor
        # storage precision may still differ (it is handled outside the kernel).
        if self.use_numba and len({self.dtypes[k] for k in _COMPUTE_KEYS}) > 1:
            raise ValueError(
                "use_numba=True requires a single precision for all stepping "
                "arrays (fields, eps_r, psi); only 'monitors' may differ. "
                "Use a uniform 'compute' precision or drop use_numba."
            )
        if self.use_numba and grid.ndim < 3:
            warnings.warn(
                "use_numba: the JIT field-update kernel is intended for large "
                "3D problems. It now produces correct results in 1D/2D as well, "
                "but the one-time compilation cost usually outweighs any speedup "
                "there - the default NumPy backend is typically as fast or "
                "faster for sub-3D grids.",
                stacklevel=2,
            )
        # Anisotropic subpixel smoothing of material interfaces (see
        # photonfdtd.smoothing). Off by default so results are bit-for-bit
        # unchanged. When on, the E-update uses a per-component (diagonal
        # permittivity tensor) coefficient instead of one shared scalar; this
        # is supported on the vectorized NumPy/CuPy path only.
        self.subpixel = bool(subpixel)
        self.subpixel_factor = int(subpixel_factor)
        if self.subpixel and self.use_numba:
            raise ValueError(
                "subpixel=True is not supported with the Numba backend "
                "(its fused kernel assumes one scalar per-cell coefficient). "
                "Use the default NumPy backend, the CuPy GPU backend, or JAX."
            )
        self._xp = _get_backend(self.use_gpu)
        self.grid = grid
        self.structures = list(structures)
        # Dispersive media: any structure whose medium carries poles activates
        # the auxiliary-differential-equation (ADE) polarization stepping. Only
        # the vectorized NumPy/CuPy path supports it (the fused Numba/JAX kernels
        # assume a single instantaneous per-cell coefficient).
        self._has_dispersion = any(
            getattr(getattr(s, "medium", None), "is_dispersive", False)
            for s in self.structures
        )
        if self._has_dispersion:
            if self.use_numba:
                raise ValueError(
                    "Dispersive media are not supported with the Numba backend; "
                    "use the default NumPy backend, the CuPy GPU backend, or JAX."
                )
            if self.subpixel:
                raise NotImplementedError(
                    "Subpixel smoothing combined with dispersive media is not "
                    "yet supported. Use one or the other."
                )
        self._ade: List[dict] = []
        # Moving current sources (e.g. ChargedParticle) are stepped specially in
        # the time loop rather than expanded into fixed-cell soft dipoles.
        self.particle_sources: List[ChargedParticle] = []
        expanded: List[PointDipole] = []
        for s in sources:
            if isinstance(s, PointDipole):
                expanded.append(s)
            elif isinstance(s, ChargedParticle):
                self._register_particle(s)
            elif hasattr(s, "expand"):
                expanded.extend(s.expand(grid))
            else:
                raise TypeError(
                    f"Unsupported source type: {type(s).__name__}. "
                    "Pass a PointDipole, ModeSource, SinglePhotonSource, "
                    "or ChargedParticle."
                )
        self.sources = expanded
        self.monitors = list(monitors)
        self.run_time = float(run_time)
        self.courant = float(courant)
        self.cpml_params = cpml_params
        self.verbose = verbose
        # Optional progress hook called as progress_callback(step, n_steps)
        # periodically during run() (and once more at completion). Left as a
        # settable attribute rather than a constructor arg so callers can attach
        # it to an already-built Simulation (e.g. one made by from_gdsfactory).
        # A None callback is simply skipped; exceptions from it are not caught,
        # so keep it cheap and non-throwing.
        self.progress_callback: Optional[Callable[[int, int], None]] = None

        # Per-cell relative permittivity (background = 1) – always on CPU first
        # so geometry primitives can stamp into it, then moved to the backend.
        # Stored in the 'eps_r' working dtype: at float64 this is bit-for-bit the
        # same float64 grid as before; at float32 it halves what is otherwise the
        # single largest array (eps_r outlives the ce_field build and was
        # previously kept at float64 regardless of precision).
        #
        # eps_r is a full-domain array that the time loop reads exactly once (to
        # build the E-update coefficient ce_field) and never again. It is fully
        # determined by grid + structures, so rather than keep it resident
        # alongside ce_field for the whole run we release it after ce_field is
        # built and regenerate it on demand (deterministic re-stamp) via the
        # `eps_r` property. This keeps one fewer full-domain array live during
        # stepping on large volumes.
        self._eps_r: Optional[np.ndarray] = None
        # True once a caller assigns a custom eps_r grid that cannot be
        # regenerated from `structures`; such a grid is never released.
        self._eps_r_custom = False
        self._materialize_eps_r()

        # Time step from the Courant condition.
        inv_dl2 = 0.0
        for d in grid.cell_size:
            if d > 0:
                inv_dl2 += 1.0 / d ** 2
        self.dt = self.courant / (C_0 * math.sqrt(inv_dl2))
        self.n_steps = int(math.ceil(self.run_time / self.dt))

        self._build_cpml()

    # ------------------------------------------------------------------ #
    # Relative permittivity grid. Held as a private array that run() releases
    # once the E-update coefficient has been derived from it (see __init__);
    # the property regenerates it from grid + structures if accessed while
    # released, so `sim.eps_r` behaves as a plain attribute to callers while
    # not sitting resident alongside ce_field during a large-volume run.
    # ------------------------------------------------------------------ #
    def _materialize_eps_r(self) -> np.ndarray:
        eps = np.ones(self.grid.shape, dtype=self.dtypes["eps_r"])
        for s in self.structures:
            s.stamp(self.grid, eps)
        self._eps_r = eps
        return eps

    def _smoothed_eps_components(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Anisotropic subpixel-smoothed diagonal permittivity (eps_xx, eps_yy,
        eps_zz), one array per E-field component. Requires structures; a
        caller-supplied custom scalar eps_r cannot be smoothed (there is no
        geometry to super-sample) and falls back to that scalar for all three.
        """
        from .smoothing import smooth_permittivity
        if self._eps_r_custom:
            eps = self.eps_r
            return (eps, eps, eps)
        return smooth_permittivity(
            self.grid, self.structures, background_eps=1.0,
            factor=self.subpixel_factor, dtype=self.dtypes["eps_r"],
        )

    # ------------------------------------------------------------------ #
    # Dispersive-media (ADE) state.
    #
    # Each dispersive material occupies a set of cells (built by stamping the
    # structures in order, later structures overwriting earlier ones). For that
    # material we hold, over those cells only, the auxiliary polarization P of
    # every pole at the current and previous time step. Each step advances P by
    # the central-difference recursion  P^{n+1} = a P^n + b P^{n-1} + c E^n  and
    # corrects E by the polarization increment (see _ade_apply). This is the
    # standard ADE-FDTD scheme (Taflove & Hagness, ch. 9).
    # ------------------------------------------------------------------ #
    def _build_ade(self, dt: float, xp, eps_dtype) -> None:
        self._ade = []
        if not self._has_dispersion:
            return
        grid = self.grid
        # Per-cell dispersive-material id (-1 = background / non-dispersive).
        id_grid = np.full(grid.shape, -1, dtype=np.int32)
        media: List = []
        med_id: Dict[int, int] = {}
        for s in self.structures:
            med = getattr(s, "medium", None)
            mask = s.region_mask(grid)
            if getattr(med, "is_dispersive", False):
                key = id(med)
                if key not in med_id:
                    med_id[key] = len(media)
                    media.append(med)
                id_grid[mask] = med_id[key]
            else:
                id_grid[mask] = -1        # a later plain medium overrides

        max_w0dt = 0.0
        for mid, med in enumerate(media):
            max_w0dt = max(max_w0dt, med.max_pole_omega() * dt)
        # Explicit ADE poles are stable only while omega0 * dt < 2 (the discrete
        # pole stays on the unit circle). Deep-UV Sellmeier / high-energy Lorentz
        # poles violate this on a grid tuned for near-IR/optical work.
        if max_w0dt >= 2.0:
            raise ValueError(
                f"A dispersive medium has a pole with omega0*dt = {max_w0dt:.2f} "
                ">= 2, which makes the explicit ADE update unstable at this "
                "resolution. That pole is too high in frequency for the grid to "
                "resolve. Options: use a much finer grid (smaller dt), restrict "
                "the medium to poles inside the simulation band, or if you only "
                "need the correct index at one wavelength use "
                "DispersiveMedium.at_wavelength(lambda) to get a fixed-index "
                "Medium instead."
            )

        for mid, med in enumerate(media):
            idx = np.nonzero(id_grid == mid)
            if idx[0].size == 0:
                continue
            n_cells = idx[0].size
            abc = []
            for p in med.poles:
                denom = 1.0 + p.gamma * dt
                a = (2.0 - p.omega0 ** 2 * dt ** 2) / denom
                b = (p.gamma * dt - 1.0) / denom
                c = (EPS_0 * p.strength * dt ** 2) / denom
                abc.append((a, b, c))
            bidx = tuple(xp.asarray(i) for i in idx)
            npole = len(med.poles)
            Pcur = [[xp.zeros(n_cells, dtype=eps_dtype) for _ in range(3)]
                    for _ in range(npole)]
            Pprev = [[xp.zeros(n_cells, dtype=eps_dtype) for _ in range(3)]
                     for _ in range(npole)]
            self._ade.append({
                "idx": bidx,
                "abc": abc,
                "Pcur": Pcur,
                "Pprev": Pprev,
                "inv_eps": 1.0 / (EPS_0 * med.eps_inf),
                "En": [None, None, None],
            })

    def _ade_capture_E(self, E_comps) -> None:
        """Snapshot E^n over each dispersive region before the curl E-update."""
        for st in self._ade:
            idx = st["idx"]
            for c in range(3):
                st["En"][c] = E_comps[c][idx].copy()

    def _ade_apply(self, E_comps) -> None:
        """Advance each pole's polarization and correct E by its increment.

        Runs after the curl E-update, using the captured E^n so the recursion is
        the intended  P^{n+1} = a P^n + b P^{n-1} + c E^n.
        """
        for st in self._ade:
            idx = st["idx"]
            inv = st["inv_eps"]
            abc = st["abc"]
            Pcur = st["Pcur"]
            Pprev = st["Pprev"]
            for c in range(3):
                En = st["En"][c]
                delta = None
                for p, (a, b, cc) in enumerate(abc):
                    pc = Pcur[p][c]
                    pp = Pprev[p][c]
                    pnew = a * pc + b * pp + cc * En
                    incr = pnew - pc
                    delta = incr if delta is None else delta + incr
                    Pprev[p][c] = pc
                    Pcur[p][c] = pnew
                if delta is not None:
                    E_comps[c][idx] -= inv * delta

    @property
    def eps_r(self) -> np.ndarray:
        if self._eps_r is None:
            self._materialize_eps_r()
        return self._eps_r

    @eps_r.setter
    def eps_r(self, value) -> None:
        # Allow callers to stamp a custom permittivity grid. Once set
        # explicitly it is no longer regenerated from `structures`, and run()
        # will not release it (there would be no way to reconstruct it).
        self._eps_r = value
        self._eps_r_custom = True

    # ------------------------------------------------------------------ #
    # Attach sources / monitors after construction. Useful when the
    # Simulation was produced by an adapter (e.g. from_gdsfactory) and the
    # user adds the excitation later.
    # ------------------------------------------------------------------ #
    def add_source(self, source) -> None:
        if isinstance(source, PointDipole):
            self.sources.append(source)
        elif isinstance(source, ChargedParticle):
            self._register_particle(source)
        elif hasattr(source, "expand"):
            self.sources.extend(source.expand(self.grid))
        else:
            raise TypeError(
                f"Unsupported source type: {type(source).__name__}"
            )

    def _register_particle(self, p: ChargedParticle) -> None:
        """Validate and store a moving charged-particle current source."""
        for axis, v in enumerate(p.velocity3):
            if v != 0.0 and self.grid.shape[axis] == 1:
                raise ValueError(
                    f"ChargedParticle has a velocity component on axis {axis}, "
                    "which is collapsed (size 1) in this grid - it has nowhere "
                    "to move or radiate. Drop that component or add that axis."
                )
        if p.speed == 0.0:
            raise ValueError("ChargedParticle velocity must be non-zero.")
        self.particle_sources.append(p)

    def _inject_particle_currents(self, t_now, Ex, Ey, Ez, ce_field, xp) -> None:
        """Deposit each moving charge's current onto the E-field at time ``t_now``.

        A charge ``q`` moving at velocity ``v`` is a current density
        ``J = q v`` localised at the particle, contributing ``dE = -dt/eps * J``
        to Ampere's law. Since ``ce_field = dt/(eps_r*EPS_0)``, the per-cell
        deposit is ``dE = -(q*v/dV) * weight * ce_field`` - reusing ``ce_field``
        keeps the deposition in the working dtype and needs no separate eps copy. The charge is smeared over a Gaussian cloud (width
        ``cloud_cells``) to band-limit the radiated spectrum, and the cloud is
        deposited only while the particle *centre* sits in the physical
        (non-PML) interior, so it stops radiating into the PML once it exits
        even if ``t_stop`` has not been reached. (Within ~``radius`` cells of
        the interior edge the Gaussian tail still leaks a sub-percent fraction
        into the PML; launch a few cells clear of the PML to avoid it.)
        """
        if not self.particle_sources:
            return
        grid = self.grid
        # Cell "volume" of the active axes (collapsed axes count as 1 m, so in
        # 2-D the charge is per-unit-length, matching the flux convention).
        dV = 1.0
        for a in range(3):
            if grid.shape[a] > 1:
                dV *= grid.cell_size[a]
        E_comp = (Ex, Ey, Ez)

        for p in self.particle_sources:
            if not (p.t_start <= t_now <= p.t_stop):
                continue
            pos = p.position_at(t_now)
            sub = []          # per-axis index slices of the deposition stencil
            waxes = []        # per-axis normalised Gaussian weights
            inside = True
            for a in range(3):
                n = grid.shape[a]
                coord = grid.coords[a]
                if n == 1:
                    sub.append(slice(0, 1))
                    waxes.append(np.array([1.0]))
                    continue
                npml = grid.pml_layers[a]
                lo_i, hi_i = npml, n - 1 - npml
                # Skip entirely once the particle leaves the physical interior,
                # so we never dump current into (or past) the PML.
                if pos[a] < coord[lo_i] or pos[a] > coord[hi_i]:
                    inside = False
                    break
                d = grid.cell_size[a]
                sigma = max(p.cloud_cells, 1e-6) * d
                radius = max(1, int(math.ceil(3.0 * p.cloud_cells)))
                ic = int(round((pos[a] - coord[0]) / d))
                i0 = max(0, ic - radius)
                i1 = min(n, ic + radius + 1)
                w = np.exp(-0.5 * ((coord[i0:i1] - pos[a]) / sigma) ** 2)
                wsum = w.sum()
                if wsum == 0.0:
                    inside = False
                    break
                sub.append(slice(i0, i1))
                waxes.append(w / wsum)      # normalise so the cloud carries all of q*v
            if not inside:
                continue

            # Separable Gaussian -> outer product weight block (sums to 1).
            wblock = (waxes[0].reshape(-1, 1, 1)
                      * waxes[1].reshape(1, -1, 1)
                      * waxes[2].reshape(1, 1, -1))
            wblock = xp.asarray(wblock, dtype=ce_field.dtype)
            ce_sub = ce_field[sub[0], sub[1], sub[2]]
            for a in range(3):
                va = p.velocity3[a]
                if va == 0.0 or grid.shape[a] == 1:
                    continue
                coef = -(p.charge * va) / dV
                E_comp[a][sub[0], sub[1], sub[2]] += coef * wblock * ce_sub

    def add_monitor(self, monitor) -> None:
        self.monitors.append(monitor)

    def _build_cpml(self) -> None:
        xp = self._xp
        be, ce, bh, ch = [], [], [], []
        for ax in range(3):
            n_cells = self.grid.shape[ax]
            dl = self.grid.cell_size[ax]
            n_pml = self.grid.pml_layers[ax] if n_cells > 1 else 0
            be_ax, ce_ax, bh_ax, ch_ax = cpml_coeffs_1axis(
                n_cells, dl if n_cells > 1 else 1.0, n_pml, self.dt, self.cpml_params
            )
            # Move CPML coefficients to GPU if using CuPy
            be.append(xp.asarray(be_ax)); ce.append(xp.asarray(ce_ax))
            bh.append(xp.asarray(bh_ax)); ch.append(xp.asarray(ch_ax))
        self._b_e = be; self._c_e = ce
        self._b_h = bh; self._c_h = ch

    # ------------------------------------------------------------------ #
    # Backend dispatch.
    # ------------------------------------------------------------------ #
    def _jax_compatible(self) -> bool:
        """Whether the JAX backend can run this sim as configured (auto falls
        back to NumPy otherwise)."""
        # JAX requires one uniform precision across the stepping arrays.
        if len({self.dtypes[k] for k in _FIELD_KEYS + ("eps_r",)}) > 1:
            return False
        # The disk-side compressed FieldMonitor store is host-only.
        for m in self.monitors:
            if isinstance(m, FieldMonitor) and m.compression is not None:
                return False
        return True

    def _use_jax_backend(self, out_of_core: bool) -> bool:
        """Resolve whether to run on JAX for this call. Explicit backend choices
        win; ``backend="auto"`` picks JAX when it is installed, the sim is
        JAX-compatible, and the run is large enough for it to pay off."""
        if self.use_gpu or self.use_numba:
            return False                       # explicit CuPy / Numba backend
        if self.use_jax or self.backend == "jax":
            return True                        # explicit JAX
        if self.backend == "numpy":
            return False
        # backend == "auto":
        if out_of_core or not _jax_available() or not self._jax_compatible():
            return False
        work = int(np.prod(self.grid.shape)) * int(self.n_steps)
        if work < AUTO_JAX_MIN_CELL_STEPS:
            return False
        # Machine-spec aware: don't route to JAX if it wouldn't fit the device
        # it would run on (GPU VRAM if a GPU is present, else host RAM). A JAX
        # run that overflows VRAM OOMs, so falling back to the NumPy core - and
        # nudging toward out_of_core - is the safer default.
        from .memory import estimate_memory, available_memory
        need = estimate_memory(self)["working_set"] * 2   # margin for XLA scratch
        budget, kind = available_memory()
        if budget is not None and need > budget:
            warnings.warn(
                f"backend='auto': this run needs ~{need / 1e9:.1f} GB but only "
                f"~{budget / 1e9:.1f} GB of {kind.upper()} memory is available, "
                "so it stays on the NumPy backend. For a run this large, use "
                "run(out_of_core=True) (and use_gpu=True for GPU tiling).",
                stacklevel=3,
            )
            return False
        return True

    # ------------------------------------------------------------------ #
    def run(self, out_of_core: bool = False, tile_cells: Optional[int] = None,
            ooc_workdir: Optional[str] = None) -> Result:
        """Time-step the simulation and return the recorded monitor data.

        Parameters
        ----------
        out_of_core : bool
            If True, stream the field arrays to disk and step the domain in
            slabs along axis 0 so peak RAM is bounded by ``tile_cells`` planes
            rather than the whole grid (see :mod:`photonfdtd.outofcore`). With
            ``use_gpu=True`` each disk-backed tile is processed on the GPU so
            peak *device* memory is one tile, not the grid (GPU/host/disk
            hierarchy). A single uniform precision, point/soft sources and
            ``FieldMonitor`` (incl. ``compression=``) are supported; other
            monitors / sources / the Numba backend raise a clear error.
        tile_cells : int, optional
            Planes held in RAM per tile when ``out_of_core`` (default: ~1/8 of
            the x-extent, at least 1). Smaller = less RAM, more sweeps.
        ooc_workdir : str, optional
            Directory for the temporary memmap files (default: a fresh temp dir
            removed at the end).
        """
        if out_of_core:
            if self.subpixel:
                raise NotImplementedError(
                    "subpixel=True is not yet supported with out_of_core=True."
                )
            if self._has_dispersion:
                raise NotImplementedError(
                    "Dispersive media are not yet supported with out_of_core=True."
                )
            from .outofcore import run_out_of_core
            if tile_cells is None:
                tile_cells = max(self.grid.shape[0] // 8, 1)
            return run_out_of_core(self, tile_cells, workdir=ooc_workdir)

        if self.backend in ("rust", "rust-cuda"):
            # Compiled Rust stepping cores (rust/src/): the whole time loop
            # runs natively with PML-slab-compacted psi state - the
            # lowest-memory in-core backends. "rust" is the rayon CPU core;
            # "rust-cuda" keeps all state GPU-resident (float32 or float64).
            # Explicit opt-in only (not part of "auto") while experimental.
            from .rustbackend import run_rust, run_rust_cuda
            return (run_rust_cuda if self.backend == "rust-cuda"
                    else run_rust)(self)

        if self._use_jax_backend(out_of_core):
            from .jaxbackend import run_jax
            return run_jax(self)

        xp = self._xp          # numpy or cupy
        nx, ny, nz = self.grid.shape
        dx, dy, dz = (d if d > 0 else 1.0 for d in self.grid.cell_size)
        dt = self.dt
        n_steps = self.n_steps

        if self.use_gpu:
            import sys
            devname = xp.cuda.runtime.getDeviceProperties(0)["name"].decode()
            print(f"[photonfdtd] GPU backend (CuPy {xp.__version__}), "
                  f"device: {devname}", file=sys.stderr, flush=True)

        # Per-array dtypes (a single string `precision` makes them all equal).
        dt_ = self.dtypes
        dtype = self.dtype          # representative compute dtype (== Ex's)
        psi_dtype = dt_["psi"]
        eps_dtype = dt_["eps_r"]
        mon_dtype = dt_["monitors"]

        # Field arrays allocated on the chosen backend, each in its own dtype.
        Ex = xp.zeros((nx, ny, nz), dtype=dt_["Ex"])
        Ey = xp.zeros((nx, ny, nz), dtype=dt_["Ey"])
        Ez = xp.zeros((nx, ny, nz), dtype=dt_["Ez"])
        Hx = xp.zeros((nx, ny, nz), dtype=dt_["Hx"])
        Hy = xp.zeros((nx, ny, nz), dtype=dt_["Hy"])
        Hz = xp.zeros((nx, ny, nz), dtype=dt_["Hz"])

        # CPML convolutional state (psi). The Numba kernel uses dense, full-domain
        # psi arrays; the vectorized numpy/cupy path instead stores psi only in
        # the PML slabs (built below) to cut peak memory on large volumes.
        if self.use_numba:
            z = lambda: xp.zeros((nx, ny, nz), dtype=psi_dtype)
            psi_Ex_y = z(); psi_Ex_z = z()
            psi_Ey_z = z(); psi_Ey_x = z()
            psi_Ez_x = z(); psi_Ez_y = z()
            psi_Hx_y = z(); psi_Hx_z = z()
            psi_Hy_z = z(); psi_Hy_x = z()
            psi_Hz_x = z(); psi_Hz_y = z()

        # Per-cell E-update coefficient dt/(eps_r*EPS_0), in the 'eps_r' dtype.
        # eps_r already carries that dtype, so at float64 this evaluates exactly
        # as before; at float32 the intermediates stay single-precision (no
        # transient full-domain float64 temporaries).
        ce_field = (dt / (xp.asarray(self.eps_r) * EPS_0)).astype(eps_dtype, copy=False)
        ch_field = dt_["Hx"](dt / MU_0)        # for H updates (mu_r = 1)

        # Per-component E-update coefficients. Without subpixel smoothing all
        # three reference the one scalar ce_field, so the vectorized update is
        # bit-for-bit identical to the historical single-coefficient path. With
        # smoothing on, each E component gets the coefficient built from its own
        # diagonal permittivity tensor entry (eps_xx -> Ex, eps_yy -> Ey,
        # eps_zz -> Ez); see photonfdtd.smoothing.
        if self.subpixel:
            exx, eyy, ezz = self._smoothed_eps_components()
            ce_comp = tuple(
                (dt / (xp.asarray(e) * EPS_0)).astype(eps_dtype, copy=False)
                for e in (exx, eyy, ezz)
            )
        else:
            ce_comp = (ce_field, ce_field, ce_field)

        # Dispersive-media auxiliary state (empty unless a structure is
        # dispersive). Built here because the pole recursion coefficients depend
        # on dt. ce_field/ce_comp already carry eps_inf for these cells, so the
        # curl update is the instantaneous response and the ADE only adds the
        # pole polarization increment each step.
        self._build_ade(dt, xp, eps_dtype)

        # eps_r is not referenced again in the time loop - ce_field carries all
        # the per-cell material information the stepper needs. Release the
        # full-domain eps_r array so it does not remain resident alongside
        # ce_field (and the six field arrays) for the whole run; it is
        # regenerated from grid + structures on demand via the eps_r property.
        # A caller-supplied custom grid cannot be reconstructed, so keep it.
        if not self._eps_r_custom:
            self._eps_r = None

        # Monitor bookkeeping.
        result = Result(times=np.arange(n_steps) * dt)
        # Snapshots are written straight into a preallocated (n_rec, *snap) array
        # rather than appended to a Python list and np.stack-ed at the end - the
        # final stack would briefly double the monitor's peak memory. The output
        # array is allocated lazily on the first recorded step (when the snapshot
        # shape/dtype is known) and filled by row index thereafter.
        rec_fields: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
        rec_times: Dict[str, list] = {}
        rec_step_list: Dict[str, set] = {}
        rec_n: Dict[str, int] = {}      # total rows to record (known up front)
        rec_count: Dict[str, int] = {}  # rows recorded so far (next write index)
        rec_zslice: Dict[str, slice] = {}  # single-plane restriction (plane_z), else whole z
        rec_codec: Dict[str, object] = {}  # compression codec for streamed monitors
        for m in self.monitors:
            if isinstance(m, FieldMonitor):
                rec_fields[m.name] = {c: None for c in m.components}
                rec_times[m.name] = []
                rec_count[m.name] = 0
                if m.compression is not None:
                    rec_codec[m.name] = get_codec(m.compression)
                if m.times is not None:
                    rec_step_list[m.name] = {int(round(t / dt)) for t in m.times}
                else:
                    rec_step_list[m.name] = set(range(0, n_steps, m.interval))
                # Distinct in-range steps that will actually fire during the loop.
                rec_n[m.name] = sum(1 for s in rec_step_list[m.name]
                                    if 0 <= s < n_steps)
                if m.plane_z is not None:
                    zi = int(np.argmin(np.abs(np.asarray(self.grid.coords[2]) - m.plane_z)))
                    rec_zslice[m.name] = slice(zi, zi + 1)  # size-1 z axis kept
                else:
                    rec_zslice[m.name] = slice(None)
            elif isinstance(m, FluxMonitor):
                result.flux[m.name] = 0.0

        # DFTMonitor bookkeeping: a running Fourier transform accumulated in
        # complex128 on the backend. Storage is (n_freq, *snap) per component -
        # independent of the number of timesteps, unlike a FieldMonitor.
        dft_accum: Dict[str, Dict[str, Optional[object]]] = {}
        dft_omega: Dict[str, np.ndarray] = {}      # angular frequencies (rad/s)
        dft_steps: Dict[str, set] = {}
        dft_plane: Dict[str, Optional[Tuple[int, int]]] = {}
        for m in self.monitors:
            if isinstance(m, DFTMonitor):
                dft_accum[m.name] = {c: None for c in m.components}
                dft_omega[m.name] = 2.0 * np.pi * np.asarray(m.freqs, dtype=np.float64)
                dft_steps[m.name] = set(range(0, n_steps, m.interval))
                result.dft_freqs[m.name] = np.asarray(m.freqs, dtype=np.float64)
                pl = m.plane()
                if pl is not None:
                    ax, pos = pl
                    ci = int(np.argmin(np.abs(np.asarray(self.grid.coords[ax]) - pos)))
                    dft_plane[m.name] = (ax, ci)
                else:
                    dft_plane[m.name] = None

        # Pre-locate source cells.
        source_cells = [(src, *self.grid.index_at(src.position)) for src in self.sources]

        # Helper to pull an array from the backend to a CPU numpy array.
        def to_cpu(arr):
            if self.use_gpu:
                return arr.get()         # cupy → numpy (np.asarray is rejected by cupy>=13)
            return arr

        # Record one FieldMonitor snapshot into its preallocated output array.
        # Shared by the NumPy/CuPy and Numba paths. The strided/plane slice is
        # taken on the backend (so the GPU transfers only the kept subset), then
        # copied into the destination row by assignment - no per-step list growth
        # and no end-of-run np.stack.
        comps = {"Ex": Ex, "Ey": Ey, "Ez": Ez, "Hx": Hx, "Hy": Hy, "Hz": Hz}

        # Backend (possibly strided/plane-restricted) view of one component,
        # shared by the time-domain and DFT recording paths.
        def snap_view(field, ds, zsl):
            if zsl != slice(None):
                return field[::ds, ::ds, zsl]
            if ds > 1:
                return field[::ds, ::ds, ::ds]
            return field

        # General single-plane view on any axis (for DFTMonitor port planes).
        # Keeps a size-1 axis so the reduced array stays 3-D. The slice is taken
        # on the backend, so on the GPU only the plane is materialised/kept.
        def snap_view_plane(field, ds, plane):
            sl = [slice(None, None, ds) if ds > 1 else slice(None)
                  for _ in range(3)]
            if plane is not None:
                ax, ci = plane
                sl[ax] = slice(ci, ci + 1)
            return field[tuple(sl)]

        def record_monitor(m, step):
            ds = m.downsample
            zsl = rec_zslice[m.name]
            idx = rec_count[m.name]
            store = rec_fields[m.name]
            compressed = m.compression is not None
            for c in m.components:
                snap = to_cpu(snap_view(comps[c], ds, zsl))
                arr = store[c]
                if compressed:
                    # Stream this frame to a disk-backed compressed series so
                    # snapshots never accumulate in RAM. The series is created
                    # lazily on the first frame (snapshot shape now known).
                    if arr is None:
                        arr = CompressedFieldSeries(snap.shape, mon_dtype,
                                                    rec_codec[m.name],
                                                    bits=m.compression_bits)
                        store[c] = arr
                    arr.append(snap)
                    continue
                if arr is None:
                    # Stored in the 'monitors' dtype (independent of the field's
                    # own precision), so monitor memory can be halved even on a
                    # float64 run; the assignment below casts the snapshot.
                    arr = np.empty((rec_n[m.name],) + snap.shape, dtype=mon_dtype)
                    store[c] = arr
                arr[idx] = snap      # assignment copies/casts into the preallocated row
            rec_times[m.name].append(step * dt)
            rec_count[m.name] += 1

        # Accumulate one step into a DFTMonitor's running transform. Each
        # component is sampled at its own Yee time (E at t_e, H at t_h) so the
        # E/H relative phase is physical. Kept on the backend (stays on-device
        # for CuPy) and in complex128 to bound accumulation drift over long runs.
        # The (n_freq, *snap) accumulator holds all spectral information in place
        # of the O(n_steps) snapshots a FieldMonitor would store.
        def record_dft(m, t_e, t_h):
            ds = m.downsample
            plane = dft_plane[m.name]
            omega = dft_omega[m.name]
            store = dft_accum[m.name]
            for c in m.components:
                snap = snap_view_plane(comps[c], ds, plane)
                t = t_e if c[0] == "E" else t_h
                # exp(+i*omega*t) per frequency, broadcast over the snapshot axes.
                phase = xp.asarray(np.exp(1j * omega * t))
                phase = phase.reshape((-1,) + (1,) * snap.ndim)
                acc = store[c]
                if acc is None:
                    acc = xp.zeros((omega.size,) + tuple(snap.shape),
                                   dtype=xp.complex128)
                    store[c] = acc
                # F(omega) += f(t) * exp(i*omega*t) * dt  (trapezoid-free Riemann sum)
                acc += phase * snap[None, ...] * dt

        # ============================================================== #
        # Helpers for axis-aware finite differences (uniform-grid central)
        # ============================================================== #
        def d_fwd(arr, axis, dl):
            """Forward first difference along axis: arr[1:] - arr[:-1].

            Uses plain slicing (cheap array views) rather than ``xp.take`` with an
            index array, which would allocate two full gather copies every call.
            The slice is built from ``axis`` so it is correct for axes 1 and 2,
            not just axis 0.
            """
            if arr.shape[axis] <= 1:
                return None
            hi = [slice(None)] * arr.ndim
            lo = [slice(None)] * arr.ndim
            hi[axis] = slice(1, None)
            lo[axis] = slice(0, -1)
            return (arr[tuple(hi)] - arr[tuple(lo)]) / dl

        # Pre-compute 1D CPML arrays for the Numba path (they must be plain numpy).
        if self.use_numba:
            _nb_be = [np.asarray(a, dtype=dtype) for a in self._b_e]
            _nb_ce = [np.asarray(a, dtype=dtype) for a in self._c_e]
            _nb_bh = [np.asarray(a, dtype=dtype) for a in self._b_h]
            _nb_ch = [np.asarray(a, dtype=dtype) for a in self._c_h]
            _nb_ce_field = np.asarray(ce_field)
            print(f"[photonfdtd] Numba CPU backend (parallel JIT), "
                  f"compiling on first step...")

        # ============================================================== #
        # Vectorized (numpy/cupy) update with slab-restricted CPML psi.
        #
        # Each E/H component is curl = termA - termB of two first-differences.
        # We split the update into a full-domain *bulk* curl (no psi) plus a
        # *PML correction* applied only on the thin PML slabs, which is exactly
        # equivalent (ce*((a+psiA)-(b+psiB)) = ce*(a-b) + ce*psiA - ce*psiB) but
        # lets psi be stored only where it is nonzero (the PML layers).
        #
        # Each entry: (F, region, K_region, k_scalar, is_E, terms) where `terms`
        # is a list of (deriv_axis, source, other_axis, sign, blocks) and each
        # block is (slab_slices, b_block, c_block, psi_slab).
        # ============================================================== #
        cell = (dx, dy, dz)
        gshape = self.grid.shape

        def _crop(ax, low):
            return (slice(1, None) if low else slice(0, -1)) if gshape[ax] > 1 \
                else slice(None)

        vterms_H = []
        vterms_E = []
        if not self.use_numba:
            # (F, aligned_axis, (axA, srcA), (axB, srcB), is_E)
            _table = [
                (Hx, 0, (1, Ez), (2, Ey), False, vterms_H),
                (Hy, 1, (2, Ex), (0, Ez), False, vterms_H),
                (Hz, 2, (0, Ey), (1, Ex), False, vterms_H),
                (Ex, 0, (1, Hz), (2, Hy), True,  vterms_E),
                (Ey, 1, (2, Hx), (0, Hz), True,  vterms_E),
                (Ez, 2, (0, Hy), (1, Hx), True,  vterms_E),
            ]
            for F, _aax, (axA, srcA), (axB, srcB), is_E, bucket in _table:
                region = [slice(None), slice(None), slice(None)]
                region[axA] = _crop(axA, is_E)
                region[axB] = _crop(axB, is_E)
                region = tuple(region)
                rshape = F[region].shape
                # For E terms the aligned axis _aax is the component index
                # (Ex->0, Ey->1, Ez->2), so ce_comp[_aax] is that component's
                # own coefficient (equal to the shared ce_field unless subpixel
                # smoothing is on).
                K_region = ce_comp[_aax][region] if is_E else None
                k_scalar = None if is_E else -ch_field
                b_src = self._b_e if is_E else self._b_h
                c_src = self._c_e if is_E else self._c_h
                terms = []
                for ax, src, other_ax, sign in ((axA, srcA, axB, 1.0),
                                                (axB, srcB, axA, -1.0)):
                    if gshape[ax] <= 1:
                        continue                      # collapsed axis -> term absent
                    csl = _crop(ax, is_E)
                    # coefficients may live on the GPU; pull the tiny 1D arrays
                    # to the host to locate PML blocks and slice them.
                    _b = b_src[ax]; _c = c_src[ax]
                    b1 = (_b.get() if hasattr(_b, "get") else np.asarray(_b))[csl]
                    c1 = (_c.get() if hasattr(_c, "get") else np.asarray(_c))[csl]
                    blocks = []
                    for s0, s1 in _contig_nonzero_blocks(c1):
                        slab = [slice(None), slice(None), slice(None)]
                        slab[ax] = slice(s0, s1)
                        rs = [1, 1, 1]; rs[ax] = s1 - s0
                        bb = xp.asarray(b1[s0:s1], dtype=psi_dtype).reshape(rs)
                        cc = xp.asarray(c1[s0:s1], dtype=psi_dtype).reshape(rs)
                        pshape = list(rshape); pshape[ax] = s1 - s0
                        blocks.append((tuple(slab), bb, cc,
                                       xp.zeros(tuple(pshape), dtype=psi_dtype)))
                    terms.append((ax, src, other_ax, sign, blocks))
                bucket.append((F, region, K_region, k_scalar, is_E, terms))

        # Shared scratch for the bulk curl, reused across all six component
        # updates and every timestep, so the vectorized path allocates no
        # per-step full-domain `curl` / `K*curl` temporaries - only the two
        # forward-difference derivatives (needed as views by the PML slab
        # correction) are transient. On a large volume this trims the stepper's
        # peak by roughly one full-domain array and eliminates per-step
        # allocation churn. Enabled only when every stepping array shares one
        # dtype (the common case - a string `precision`); a mixed-precision
        # dict falls back to the allocating path. The scratch path is
        # bit-for-bit identical to it: cbuf = dA - dB == dA + (-dB), then
        # cbuf *= K reproduces K*curl exactly.
        # Set PHOTONFDTD_NO_SCRATCH=1 to force the allocating path (benchmarking
        # / parity debugging); results are identical either way.
        _compute_dtypes = {dt_[k] for k in _FIELD_KEYS} | {eps_dtype}
        use_scratch = (not self.use_numba) and len(_compute_dtypes) == 1 \
            and not os.environ.get("PHOTONFDTD_NO_SCRATCH")
        curl_scratch = xp.empty(gshape, dtype=dt_["Ex"]) if use_scratch else None

        def apply_update(vlist):
            for F, region, K_region, k_scalar, is_E, terms in vlist:
                derivs = []
                for ax, src, other_ax, sign, blocks in terms:
                    d = d_fwd(src, ax, cell[ax])
                    if d is not None and gshape[other_ax] > 1:
                        osl = [slice(None), slice(None), slice(None)]
                        osl[other_ax] = _crop(other_ax, is_E)
                        d = d[tuple(osl)]
                    derivs.append(d)
                # bulk curl (no psi)
                if use_scratch:
                    # Accumulate dA - dB in place into a shared buffer, then fold
                    # in the E/H coefficient and add to the field - no new
                    # full-domain temporaries.
                    cbuf = curl_scratch[region]
                    empty = True
                    for (ax, src, other_ax, sign, blocks), d in zip(terms, derivs):
                        if d is None:
                            continue
                        if empty:
                            if sign > 0:
                                xp.copyto(cbuf, d)
                            else:
                                xp.negative(d, out=cbuf)
                            empty = False
                        elif sign > 0:
                            xp.add(cbuf, d, out=cbuf)
                        else:
                            xp.subtract(cbuf, d, out=cbuf)
                    if empty:
                        continue
                    Freg = F[region]
                    if is_E:
                        xp.multiply(cbuf, K_region, out=cbuf)
                    else:
                        xp.multiply(cbuf, k_scalar, out=cbuf)
                    Freg += cbuf
                else:
                    curl = None
                    for (ax, src, other_ax, sign, blocks), d in zip(terms, derivs):
                        if d is None:
                            continue
                        contrib = d if sign > 0 else -d
                        curl = contrib if curl is None else curl + contrib
                    if curl is None:
                        continue
                    Freg = F[region]
                    Freg += (K_region * curl) if is_E else (k_scalar * curl)
                # PML correction on the thin slabs only (uses the derivative
                # views and psi state; independent of the bulk-curl temporary).
                for (ax, src, other_ax, sign, blocks), d in zip(terms, derivs):
                    if d is None:
                        continue
                    for slab, bb, cc, psi in blocks:
                        psi[...] = bb * psi + cc * d[slab]
                        k = K_region[slab] if is_E else k_scalar
                        Freg[slab] += sign * k * psi

        # ============================================================== #
        # Main time loop
        # ============================================================== #
        # ~100 progress ticks over the whole run (cheap, and fine-grained enough
        # for a smooth UI bar without flooding a callback that may hop threads or
        # processes). Done at the loop top so it covers both the Numba and the
        # NumPy/CuPy branches (the Numba path `continue`s before the bottom).
        progress_interval = max(n_steps // 100, 1)
        for step in range(n_steps):
            if self.progress_callback is not None and step % progress_interval == 0:
                self.progress_callback(step, n_steps)
            t_h = (step + 0.5) * dt
            t_e = (step + 1.0) * dt

            if self.use_numba:
                # Numba-JIT path: one call does all 6 field components + CPML.
                _update_fields_numba(
                    Ex, Ey, Ez, Hx, Hy, Hz,
                    psi_Ex_y, psi_Ex_z, psi_Ey_z, psi_Ey_x,
                    psi_Ez_x, psi_Ez_y, psi_Hx_y, psi_Hx_z,
                    psi_Hy_z, psi_Hy_x, psi_Hz_x, psi_Hz_y,
                    # E CPML coefficients (first in function signature)
                    _nb_be[0], _nb_ce[0], _nb_be[1], _nb_ce[1], _nb_be[2], _nb_ce[2],
                    # H CPML coefficients (second)
                    _nb_bh[0], _nb_ch[0], _nb_bh[1], _nb_ch[1], _nb_bh[2], _nb_ch[2],
                    _nb_ce_field, ch_field,
                    dx, dy, dz,
                )
                # Sources and monitors continue below (plain numpy indexing, fast)
                H_map = {"Hx": Hx, "Hy": Hy, "Hz": Hz}
                for src, i, j, k in source_cells:
                    if src.component[0] == "H":
                        val = src.amplitude * src.waveform(np.array([t_h]))[0]
                        H_map[src.component][i, j, k] += val
                E_map = {"Ex": Ex, "Ey": Ey, "Ez": Ez}
                for src, i, j, k in source_cells:
                    if src.component[0] == "E":
                        val = src.amplitude * src.waveform(np.array([t_e]))[0]
                        E_map[src.component][i, j, k] += val
                self._inject_particle_currents(t_e, Ex, Ey, Ez, ce_field, xp)
                for m in self.monitors:
                    if isinstance(m, FieldMonitor):
                        if step in rec_step_list[m.name]:
                            record_monitor(m, step)
                    elif isinstance(m, DFTMonitor):
                        if step in dft_steps[m.name]:
                            record_dft(m, t_e, t_h)
                if self.verbose and step % max(n_steps // 20, 1) == 0:
                    emax = float(max(abs(Ex).max(), abs(Ey).max(), abs(Ez).max()))
                    print(f"  step {step}/{n_steps}  t={step*dt*1e15:6.1f} fs  |E|max={emax:.3e}")
                continue   # skip the NumPy/CuPy branch below

            # -------- H update (forward differences of E) -------- #
            apply_update(vterms_H)

            # -------- H-component sources -------- #
            H_map = {"Hx": Hx, "Hy": Hy, "Hz": Hz}
            for src, i, j, k in source_cells:
                if src.component[0] != "H":
                    continue
                val = src.amplitude * src.waveform(np.array([t_h]))[0]
                H_map[src.component][i, j, k] += val

            # -------- E update (backward differences of H) -------- #
            if self._ade:
                # Capture E^n over dispersive regions before the curl update,
                # so the pole recursion uses the pre-update field.
                self._ade_capture_E((Ex, Ey, Ez))
            apply_update(vterms_E)
            if self._ade:
                # Advance polarization and subtract its increment (dispersion).
                self._ade_apply((Ex, Ey, Ez))

            # -------- E-component sources -------- #
            E_map = {"Ex": Ex, "Ey": Ey, "Ez": Ez}
            for src, i, j, k in source_cells:
                if src.component[0] != "E":
                    continue
                val = src.amplitude * src.waveform(np.array([t_e]))[0]
                E_map[src.component][i, j, k] += val

            # -------- Moving charged-particle currents -------- #
            self._inject_particle_currents(t_e, Ex, Ey, Ez, ce_field, xp)

            # -------- Monitors -------- #
            for m in self.monitors:
                if isinstance(m, FieldMonitor):
                    if step in rec_step_list[m.name]:
                        record_monitor(m, step)
                elif isinstance(m, DFTMonitor):
                    if step in dft_steps[m.name]:
                        record_dft(m, t_e, t_h)
                elif isinstance(m, FluxMonitor):
                    result.flux[m.name] += _flux_through_plane(
                        m, self.grid, Ex, Ey, Ez, Hx, Hy, Hz
                    ) * dt

            if self.verbose and step % max(n_steps // 20, 1) == 0:
                emax = float(max(float(xp.abs(Ex).max()), float(xp.abs(Ey).max()), float(xp.abs(Ez).max())))
                print(f"  step {step}/{n_steps}  t={step*dt*1e15:6.1f} fs  |E|max={emax:.3e}")

        if self.progress_callback is not None:
            self.progress_callback(n_steps, n_steps)  # 100% on completion

        for m in self.monitors:
            if isinstance(m, FieldMonitor):
                if rec_count[m.name] > 0:
                    if m.compression is not None:
                        for c in m.components:          # flush disk-backed frames
                            rec_fields[m.name][c].finalize()
                    result.fields[m.name] = {
                        c: rec_fields[m.name][c] for c in m.components
                    }
                    result.monitor_times[m.name] = np.array(rec_times[m.name])
                else:
                    result.fields[m.name] = {c: np.zeros((0,) + self.grid.shape)
                                             for c in m.components}
                    result.monitor_times[m.name] = np.array([])
            elif isinstance(m, DFTMonitor):
                out = {}
                for c in m.components:
                    acc = dft_accum[m.name][c]
                    out[c] = to_cpu(acc) if acc is not None else \
                        np.zeros((len(m.freqs),) + self.grid.shape, dtype=np.complex128)
                result.dft[m.name] = out
        return result


# -------------------------------------------------------------------------- #
def _flux_through_plane(mon: FluxMonitor, grid: Grid,
                        Ex, Ey, Ez, Hx, Hy, Hz) -> float:
    # Support both numpy and cupy arrays by using the array's own module.
    xp = np if not hasattr(Ex, 'get') else __import__('cupy')
    axis = {"x": 0, "y": 1, "z": 2}[mon.plane_axis]
    coord = grid.coords[axis]
    idx = 0 if coord.size == 1 else int(np.argmin(np.abs(coord - mon.plane_position)))
    if axis == 0:
        S = Ey[idx, :, :] * Hz[idx, :, :] - Ez[idx, :, :] * Hy[idx, :, :]
        dA = (grid.cell_size[1] if grid.shape[1] > 1 else 1.0) * \
             (grid.cell_size[2] if grid.shape[2] > 1 else 1.0)
    elif axis == 1:
        S = Ez[:, idx, :] * Hx[:, idx, :] - Ex[:, idx, :] * Hz[:, idx, :]
        dA = (grid.cell_size[0] if grid.shape[0] > 1 else 1.0) * \
             (grid.cell_size[2] if grid.shape[2] > 1 else 1.0)
    else:
        S = Ex[:, :, idx] * Hy[:, :, idx] - Ey[:, :, idx] * Hx[:, :, idx]
        dA = (grid.cell_size[0] if grid.shape[0] > 1 else 1.0) * \
             (grid.cell_size[1] if grid.shape[1] > 1 else 1.0)
    return float(xp.sum(S) * dA)
