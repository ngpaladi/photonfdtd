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
- Materials are non-dispersive isotropic dielectrics (eps_r per cell).

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
from typing import Dict, List, Optional, Sequence, Tuple
import math
import warnings
import numpy as np

from .constants import C_0, EPS_0, MU_0
from .grid import Grid
from .geometry import Box
from .sources import PointDipole, ChargedParticle
from .monitors import FieldMonitor, FluxMonitor

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
                "use_gpu=True but CuPy is not installed or has a CUDA version "
                "mismatch with the running driver.  "
                "Install: pip install cupy-cuda12x"
            )
        return _cp
    return np


@dataclass
class Result:
    times: np.ndarray
    fields: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    monitor_times: Dict[str, np.ndarray] = field(default_factory=dict)
    flux: Dict[str, float] = field(default_factory=dict)


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
    ) -> None:
        self.use_gpu = use_gpu and _GPU_AVAILABLE
        self.use_numba = use_numba and _NUMBA_AVAILABLE and not self.use_gpu
        if self.use_numba and grid.ndim < 3:
            warnings.warn(
                "use_numba: the JIT field-update kernel is intended for large "
                "3D problems. It now produces correct results in 1D/2D as well, "
                "but the one-time compilation cost usually outweighs any speedup "
                "there - the default NumPy backend is typically as fast or "
                "faster for sub-3D grids.",
                stacklevel=2,
            )
        self._xp = _get_backend(self.use_gpu)
        self.grid = grid
        self.structures = list(structures)
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

        # Per-cell relative permittivity (background = 1) – always on CPU first
        # so geometry primitives can stamp into it, then moved to the backend.
        self.eps_r = np.ones(grid.shape, dtype=np.float64)
        for s in self.structures:
            s.stamp(grid, self.eps_r)

        # Time step from the Courant condition.
        inv_dl2 = 0.0
        for d in grid.cell_size:
            if d > 0:
                inv_dl2 += 1.0 / d ** 2
        self.dt = self.courant / (C_0 * math.sqrt(inv_dl2))
        self.n_steps = int(math.ceil(self.run_time / self.dt))

        self._build_cpml()

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

    def _inject_particle_currents(self, t_now, Ex, Ey, Ez, eps_r_xp, xp) -> None:
        """Deposit each moving charge's current onto the E-field at time ``t_now``.

        A charge ``q`` moving at velocity ``v`` is a current density
        ``J = q v`` localised at the particle, contributing ``dE = -dt/eps * J``
        to Ampere's law. The charge is smeared over a Gaussian cloud (width
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
            wblock = xp.asarray(wblock)
            eps_sub = eps_r_xp[sub[0], sub[1], sub[2]]
            for a in range(3):
                va = p.velocity3[a]
                if va == 0.0 or grid.shape[a] == 1:
                    continue
                base = -(self.dt * p.charge * va) / (EPS_0 * dV)
                E_comp[a][sub[0], sub[1], sub[2]] += base * wblock / eps_sub

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
    def run(self) -> Result:
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

        # Field arrays allocated on the chosen backend
        Ex = xp.zeros((nx, ny, nz))
        Ey = xp.zeros((nx, ny, nz))
        Ez = xp.zeros((nx, ny, nz))
        Hx = xp.zeros((nx, ny, nz))
        Hy = xp.zeros((nx, ny, nz))
        Hz = xp.zeros((nx, ny, nz))

        # CPML auxiliary variables
        psi_Ex_y = xp.zeros((nx, ny, nz)); psi_Ex_z = xp.zeros((nx, ny, nz))
        psi_Ey_z = xp.zeros((nx, ny, nz)); psi_Ey_x = xp.zeros((nx, ny, nz))
        psi_Ez_x = xp.zeros((nx, ny, nz)); psi_Ez_y = xp.zeros((nx, ny, nz))
        psi_Hx_y = xp.zeros((nx, ny, nz)); psi_Hx_z = xp.zeros((nx, ny, nz))
        psi_Hy_z = xp.zeros((nx, ny, nz)); psi_Hy_x = xp.zeros((nx, ny, nz))
        psi_Hz_x = xp.zeros((nx, ny, nz)); psi_Hz_y = xp.zeros((nx, ny, nz))

        # CPML coefficient broadcasts.
        bx_e = self._b_e[0].reshape(-1, 1, 1); cx_e = self._c_e[0].reshape(-1, 1, 1)
        by_e = self._b_e[1].reshape(1, -1, 1); cy_e = self._c_e[1].reshape(1, -1, 1)
        bz_e = self._b_e[2].reshape(1, 1, -1); cz_e = self._c_e[2].reshape(1, 1, -1)
        bx_h = self._b_h[0].reshape(-1, 1, 1); cx_h = self._c_h[0].reshape(-1, 1, 1)
        by_h = self._b_h[1].reshape(1, -1, 1); cy_h = self._c_h[1].reshape(1, -1, 1)
        bz_h = self._b_h[2].reshape(1, 1, -1); cz_h = self._c_h[2].reshape(1, 1, -1)

        # Move eps_r to the backend (no-op if already numpy and backend is numpy)
        eps_r_xp = xp.asarray(self.eps_r)
        ce_field = dt / (eps_r_xp * EPS_0)    # for E updates
        ch_field = dt / MU_0                   # for H updates (mu_r = 1)

        # Monitor bookkeeping.
        result = Result(times=np.arange(n_steps) * dt)
        rec_fields: Dict[str, Dict[str, list]] = {}
        rec_times: Dict[str, list] = {}
        rec_step_list: Dict[str, set] = {}
        for m in self.monitors:
            if isinstance(m, FieldMonitor):
                rec_fields[m.name] = {c: [] for c in m.components}
                rec_times[m.name] = []
                if m.times is not None:
                    rec_step_list[m.name] = {int(round(t / dt)) for t in m.times}
                else:
                    rec_step_list[m.name] = set(range(0, n_steps, m.interval))
            elif isinstance(m, FluxMonitor):
                result.flux[m.name] = 0.0

        # Pre-locate source cells.
        source_cells = [(src, *self.grid.index_at(src.position)) for src in self.sources]

        # Helper to pull an array from the backend to a CPU numpy array.
        def to_cpu(arr):
            if self.use_gpu:
                return arr.get()         # cupy → numpy (np.asarray is rejected by cupy>=13)
            return arr

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
            _nb_be = [np.asarray(a) for a in self._b_e]
            _nb_ce = [np.asarray(a) for a in self._c_e]
            _nb_bh = [np.asarray(a) for a in self._b_h]
            _nb_ch = [np.asarray(a) for a in self._c_h]
            _nb_ce_field = np.asarray(ce_field)
            print(f"[photonfdtd] Numba CPU backend (parallel JIT), "
                  f"compiling on first step...")

        # ============================================================== #
        # Main time loop
        # ============================================================== #
        for step in range(n_steps):
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
                self._inject_particle_currents(t_e, Ex, Ey, Ez, eps_r_xp, xp)
                for m in self.monitors:
                    if isinstance(m, FieldMonitor):
                        if step in rec_step_list[m.name]:
                            comps = {"Ex": Ex, "Ey": Ey, "Ez": Ez,
                                     "Hx": Hx, "Hy": Hy, "Hz": Hz}
                            for c in m.components:
                                rec_fields[m.name][c].append(comps[c].copy())
                            rec_times[m.name].append(step * dt)
                if self.verbose and step % max(n_steps // 20, 1) == 0:
                    emax = float(max(abs(Ex).max(), abs(Ey).max(), abs(Ez).max()))
                    print(f"  step {step}/{n_steps}  t={step*dt*1e15:6.1f} fs  |E|max={emax:.3e}")
                continue   # skip the NumPy/CuPy branch below

            # -------- H update (forward differences of E) -------- #
            # Hx at (i, j+1/2, k+1/2): uses dEz/dy and dEy/dz cropped to (nx, ny-1, nz-1)
            if ny > 1 or nz > 1:
                dEz_dy = d_fwd(Ez, 1, dy)               # (nx, ny-1, nz) or None
                dEy_dz = d_fwd(Ey, 2, dz)               # (nx, ny, nz-1) or None
                if ny > 1 and nz > 1:
                    a = dEz_dy[:, :, :-1]; b_ = dEy_dz[:, :-1, :]
                    psi_Hx_y[:, :-1, :-1] = (by_h[:, :-1, :] * psi_Hx_y[:, :-1, :-1] +
                                              cy_h[:, :-1, :] * a)
                    psi_Hx_z[:, :-1, :-1] = (bz_h[:, :, :-1] * psi_Hx_z[:, :-1, :-1] +
                                              cz_h[:, :, :-1] * b_)
                    curl = (a + psi_Hx_y[:, :-1, :-1]) - (b_ + psi_Hx_z[:, :-1, :-1])
                    Hx[:, :-1, :-1] -= ch_field * curl
                elif ny > 1:                            # 2D in xy-plane (TM): k size 1
                    a = dEz_dy                          # (nx, ny-1, 1)
                    psi_Hx_y[:, :-1, :] = (by_h[:, :-1, :] * psi_Hx_y[:, :-1, :] +
                                            cy_h[:, :-1, :] * a)
                    Hx[:, :-1, :] -= ch_field * (a + psi_Hx_y[:, :-1, :])
                elif nz > 1:                            # 2D in xz-plane: j size 1
                    b_ = dEy_dz                         # (nx, 1, nz-1)
                    psi_Hx_z[:, :, :-1] = (bz_h[:, :, :-1] * psi_Hx_z[:, :, :-1] +
                                            cz_h[:, :, :-1] * b_)
                    Hx[:, :, :-1] -= ch_field * (-(b_ + psi_Hx_z[:, :, :-1]))

            # Hy at (i+1/2, j, k+1/2): uses dEx/dz and dEz/dx cropped to (nx-1, ny, nz-1)
            if nx > 1 or nz > 1:
                dEx_dz = d_fwd(Ex, 2, dz)               # (nx, ny, nz-1)
                dEz_dx = d_fwd(Ez, 0, dx)               # (nx-1, ny, nz)
                if nx > 1 and nz > 1:
                    a = dEx_dz[:-1, :, :]; b_ = dEz_dx[:, :, :-1]
                    psi_Hy_z[:-1, :, :-1] = (bz_h[:, :, :-1] * psi_Hy_z[:-1, :, :-1] +
                                              cz_h[:, :, :-1] * a)
                    psi_Hy_x[:-1, :, :-1] = (bx_h[:-1, :, :] * psi_Hy_x[:-1, :, :-1] +
                                              cx_h[:-1, :, :] * b_)
                    curl = (a + psi_Hy_z[:-1, :, :-1]) - (b_ + psi_Hy_x[:-1, :, :-1])
                    Hy[:-1, :, :-1] -= ch_field * curl
                elif nz > 1:                            # 2D yz-plane: i size 1
                    a = dEx_dz                          # (1, ny, nz-1)
                    psi_Hy_z[:, :, :-1] = (bz_h[:, :, :-1] * psi_Hy_z[:, :, :-1] +
                                            cz_h[:, :, :-1] * a)
                    Hy[:, :, :-1] -= ch_field * (a + psi_Hy_z[:, :, :-1])
                elif nx > 1:                            # 2D xz-plane: k size 1, but no Ex/Ez decoupling here
                    b_ = dEz_dx                         # (nx-1, ny, 1)
                    psi_Hy_x[:-1, :, :] = (bx_h[:-1, :, :] * psi_Hy_x[:-1, :, :] +
                                            cx_h[:-1, :, :] * b_)
                    Hy[:-1, :, :] -= ch_field * (-(b_ + psi_Hy_x[:-1, :, :]))

            # Hz at (i+1/2, j+1/2, k): uses dEy/dx and dEx/dy cropped to (nx-1, ny-1, nz)
            if nx > 1 or ny > 1:
                dEy_dx = d_fwd(Ey, 0, dx)               # (nx-1, ny, nz)
                dEx_dy = d_fwd(Ex, 1, dy)               # (nx, ny-1, nz)
                if nx > 1 and ny > 1:
                    a = dEy_dx[:, :-1, :]; b_ = dEx_dy[:-1, :, :]
                    psi_Hz_x[:-1, :-1, :] = (bx_h[:-1, :, :] * psi_Hz_x[:-1, :-1, :] +
                                              cx_h[:-1, :, :] * a)
                    psi_Hz_y[:-1, :-1, :] = (by_h[:, :-1, :] * psi_Hz_y[:-1, :-1, :] +
                                              cy_h[:, :-1, :] * b_)
                    curl = (a + psi_Hz_x[:-1, :-1, :]) - (b_ + psi_Hz_y[:-1, :-1, :])
                    Hz[:-1, :-1, :] -= ch_field * curl
                elif nx > 1:                            # 1D x-axis: only Ey along axis matters
                    a = dEy_dx
                    psi_Hz_x[:-1, :, :] = (bx_h[:-1, :, :] * psi_Hz_x[:-1, :, :] +
                                            cx_h[:-1, :, :] * a)
                    Hz[:-1, :, :] -= ch_field * (a + psi_Hz_x[:-1, :, :])
                elif ny > 1:
                    b_ = dEx_dy
                    psi_Hz_y[:, :-1, :] = (by_h[:, :-1, :] * psi_Hz_y[:, :-1, :] +
                                            cy_h[:, :-1, :] * b_)
                    Hz[:, :-1, :] -= ch_field * (-(b_ + psi_Hz_y[:, :-1, :]))

            # -------- H-component sources -------- #
            H_map = {"Hx": Hx, "Hy": Hy, "Hz": Hz}
            for src, i, j, k in source_cells:
                if src.component[0] != "H":
                    continue
                val = src.amplitude * src.waveform(np.array([t_h]))[0]
                H_map[src.component][i, j, k] += val

            # -------- E update (backward differences of H) -------- #
            # Ex at (i+1/2, j, k): uses dHz/dy and dHy/dz; valid j in [1, ny-1], k in [1, nz-1]
            if ny > 1 or nz > 1:
                dHz_dy = d_fwd(Hz, 1, dy)               # (nx, ny-1, nz) maps to j in [1, ny-1]
                dHy_dz = d_fwd(Hy, 2, dz)               # (nx, ny, nz-1) maps to k in [1, nz-1]
                if ny > 1 and nz > 1:
                    a = dHz_dy[:, :, 1:]; b_ = dHy_dz[:, 1:, :]
                    psi_Ex_y[:, 1:, 1:] = (by_e[:, 1:, :] * psi_Ex_y[:, 1:, 1:] +
                                            cy_e[:, 1:, :] * a)
                    psi_Ex_z[:, 1:, 1:] = (bz_e[:, :, 1:] * psi_Ex_z[:, 1:, 1:] +
                                            cz_e[:, :, 1:] * b_)
                    curl = (a + psi_Ex_y[:, 1:, 1:]) - (b_ + psi_Ex_z[:, 1:, 1:])
                    Ex[:, 1:, 1:] += ce_field[:, 1:, 1:] * curl
                elif ny > 1:
                    a = dHz_dy
                    psi_Ex_y[:, 1:, :] = (by_e[:, 1:, :] * psi_Ex_y[:, 1:, :] +
                                           cy_e[:, 1:, :] * a)
                    Ex[:, 1:, :] += ce_field[:, 1:, :] * (a + psi_Ex_y[:, 1:, :])
                elif nz > 1:
                    b_ = dHy_dz
                    psi_Ex_z[:, :, 1:] = (bz_e[:, :, 1:] * psi_Ex_z[:, :, 1:] +
                                           cz_e[:, :, 1:] * b_)
                    Ex[:, :, 1:] += ce_field[:, :, 1:] * (-(b_ + psi_Ex_z[:, :, 1:]))

            # Ey at (i, j+1/2, k): uses dHx/dz and dHz/dx
            if nx > 1 or nz > 1:
                dHx_dz = d_fwd(Hx, 2, dz)               # (nx, ny, nz-1)
                dHz_dx = d_fwd(Hz, 0, dx)               # (nx-1, ny, nz)
                if nx > 1 and nz > 1:
                    a = dHx_dz[1:, :, :]; b_ = dHz_dx[:, :, 1:]
                    psi_Ey_z[1:, :, 1:] = (bz_e[:, :, 1:] * psi_Ey_z[1:, :, 1:] +
                                            cz_e[:, :, 1:] * a)
                    psi_Ey_x[1:, :, 1:] = (bx_e[1:, :, :] * psi_Ey_x[1:, :, 1:] +
                                            cx_e[1:, :, :] * b_)
                    curl = (a + psi_Ey_z[1:, :, 1:]) - (b_ + psi_Ey_x[1:, :, 1:])
                    Ey[1:, :, 1:] += ce_field[1:, :, 1:] * curl
                elif nz > 1:
                    a = dHx_dz
                    psi_Ey_z[:, :, 1:] = (bz_e[:, :, 1:] * psi_Ey_z[:, :, 1:] +
                                           cz_e[:, :, 1:] * a)
                    Ey[:, :, 1:] += ce_field[:, :, 1:] * (a + psi_Ey_z[:, :, 1:])
                elif nx > 1:
                    b_ = dHz_dx
                    psi_Ey_x[1:, :, :] = (bx_e[1:, :, :] * psi_Ey_x[1:, :, :] +
                                           cx_e[1:, :, :] * b_)
                    Ey[1:, :, :] += ce_field[1:, :, :] * (-(b_ + psi_Ey_x[1:, :, :]))

            # Ez at (i, j, k+1/2): uses dHy/dx and dHx/dy
            if nx > 1 or ny > 1:
                dHy_dx = d_fwd(Hy, 0, dx)               # (nx-1, ny, nz)
                dHx_dy = d_fwd(Hx, 1, dy)               # (nx, ny-1, nz)
                if nx > 1 and ny > 1:
                    a = dHy_dx[:, 1:, :]; b_ = dHx_dy[1:, :, :]
                    psi_Ez_x[1:, 1:, :] = (bx_e[1:, :, :] * psi_Ez_x[1:, 1:, :] +
                                            cx_e[1:, :, :] * a)
                    psi_Ez_y[1:, 1:, :] = (by_e[:, 1:, :] * psi_Ez_y[1:, 1:, :] +
                                            cy_e[:, 1:, :] * b_)
                    curl = (a + psi_Ez_x[1:, 1:, :]) - (b_ + psi_Ez_y[1:, 1:, :])
                    Ez[1:, 1:, :] += ce_field[1:, 1:, :] * curl
                elif nx > 1:
                    a = dHy_dx
                    psi_Ez_x[1:, :, :] = (bx_e[1:, :, :] * psi_Ez_x[1:, :, :] +
                                           cx_e[1:, :, :] * a)
                    Ez[1:, :, :] += ce_field[1:, :, :] * (a + psi_Ez_x[1:, :, :])
                elif ny > 1:
                    b_ = dHx_dy
                    psi_Ez_y[:, 1:, :] = (by_e[:, 1:, :] * psi_Ez_y[:, 1:, :] +
                                           cy_e[:, 1:, :] * b_)
                    Ez[:, 1:, :] += ce_field[:, 1:, :] * (-(b_ + psi_Ez_y[:, 1:, :]))

            # -------- E-component sources -------- #
            E_map = {"Ex": Ex, "Ey": Ey, "Ez": Ez}
            for src, i, j, k in source_cells:
                if src.component[0] != "E":
                    continue
                val = src.amplitude * src.waveform(np.array([t_e]))[0]
                E_map[src.component][i, j, k] += val

            # -------- Moving charged-particle currents -------- #
            self._inject_particle_currents(t_e, Ex, Ey, Ez, eps_r_xp, xp)

            # -------- Monitors -------- #
            for m in self.monitors:
                if isinstance(m, FieldMonitor):
                    if step in rec_step_list[m.name]:
                        comps = {"Ex": Ex, "Ey": Ey, "Ez": Ez,
                                 "Hx": Hx, "Hy": Hy, "Hz": Hz}
                        for c in m.components:
                            # Pull to CPU before storing so monitors are always
                            # plain numpy arrays regardless of the backend.
                            rec_fields[m.name][c].append(to_cpu(comps[c]).copy())
                        rec_times[m.name].append(step * dt)
                elif isinstance(m, FluxMonitor):
                    result.flux[m.name] += _flux_through_plane(
                        m, self.grid, Ex, Ey, Ez, Hx, Hy, Hz
                    ) * dt

            if self.verbose and step % max(n_steps // 20, 1) == 0:
                emax = float(max(float(xp.abs(Ex).max()), float(xp.abs(Ey).max()), float(xp.abs(Ez).max())))
                print(f"  step {step}/{n_steps}  t={step*dt*1e15:6.1f} fs  |E|max={emax:.3e}")

        for m in self.monitors:
            if isinstance(m, FieldMonitor):
                if rec_fields[m.name][m.components[0]]:
                    result.fields[m.name] = {
                        c: np.stack(rec_fields[m.name][c], axis=0) for c in m.components
                    }
                    result.monitor_times[m.name] = np.array(rec_times[m.name])
                else:
                    result.fields[m.name] = {c: np.zeros((0,) + self.grid.shape)
                                             for c in m.components}
                    result.monitor_times[m.name] = np.array([])
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
