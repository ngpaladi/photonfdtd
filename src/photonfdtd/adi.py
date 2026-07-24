"""Unconditionally-stable ADI-FDTD (2-D), CFL-free time stepping.

Explicit Yee FDTD is bound by the Courant-Friedrichs-Lewy limit: the time step
cannot exceed ``~dx/(c*sqrt(D))``. On a photonic-circuit grid a single small
feature (a narrow gap, a thin sidewall) forces a small ``dx`` everywhere it
touches, and CFL then forces a small ``dt`` over the *whole* domain and every
timestep - even where the fields are smooth. The alternating-direction-implicit
FDTD (Namiki 1999; Zheng, Chen & Zhang 2000) removes the stability bound
entirely: it is **unconditionally stable**, so ``dt`` is set by accuracy alone.
Where CFL was the binding constraint, that is a 3-10x cut in the number of
steps.

The price is two implicit sub-steps per full step, each a set of independent
tridiagonal solves - one direction implicit, the other explicit, then swapped.
This module implements the scheme for the 2-D ``(Ez, Hx, Hy)`` polarisation
(the out-of-plane-E field of an ``xy`` grid - the polarisation the 2.5-D
effective-index reduction produces). The tridiagonal solves are the natural
unit of work for the Rust backend (each line is independent; see
``rust/src/adi.rs``), so ``backend="rust"`` parallelises them across lines.

Boundaries: ``"pec"`` (perfect electric conductor, ``Ez=0`` - a closed cavity,
used to demonstrate that the resonant frequency is ``dt``-independent) and
``"absorber"`` (a graded electric+magnetic conductivity layer, impedance
matched at normal incidence; a basic open boundary, not as reflectionless as
the explicit solver's CPML).

Scheme, per full step ``n -> n+1`` (``s = dt/2``, ``c = s^2/mu0``):

    sub-step 1 (x implicit):
        [eps - c d2/dx2] Ez* = eps Ez^n + s (dHy^n/dx - dHx^n/dy)
        Hy* = Hy^n + (s/mu0) dEz*/dx ,  Hx* = Hx^n - (s/mu0) dEz^n/dy
    sub-step 2 (y implicit):
        [eps - c d2/dy2] Ez^{n+1} = eps Ez* + s (dHy*/dx - dHx*/dy)
        Hy^{n+1} = Hy* + (s/mu0) dEz*/dx ,  Hx^{n+1} = Hx* - (s/mu0) dEz^{n+1}/dy
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .constants import C_0, EPS_0, MU_0


def _thomas(diag, lower, upper, rhs):
    """Solve a batch of tridiagonal systems along axis 0 (Thomas algorithm).

    ``diag`` is ``(n, batch)``; ``lower``/``upper`` are the (constant) sub- and
    super-diagonals as scalars; ``rhs`` is ``(n, batch)``. Returns ``(n, batch)``.
    Modifies copies only.
    """
    n = diag.shape[0]
    cp = np.empty_like(diag)
    dp = np.empty_like(rhs)
    cp[0] = upper / diag[0]
    dp[0] = rhs[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - lower * cp[i - 1]
        cp[i] = upper / m
        dp[i] = (rhs[i] - lower * dp[i - 1]) / m
    x = np.empty_like(rhs)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


@dataclass
class ADIResult:
    times: np.ndarray
    snapshots: Dict[str, np.ndarray] = field(default_factory=dict)
    probes: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class ADISource:
    """Soft additive ``Ez`` point source at a grid index."""
    i: int
    j: int
    waveform: Callable[[np.ndarray], np.ndarray]
    amplitude: float = 1.0


class ADISimulation2D:
    """2-D unconditionally-stable ADI-FDTD for the ``(Ez, Hx, Hy)`` system.

    Parameters
    ----------
    eps_r : (nx, ny) array
        Relative permittivity at the ``Ez`` nodes.
    cell_size : float or (dx, dy)
        Grid step (m).
    dt : float, optional
        Time step (s). May exceed the explicit CFL limit (that is the point).
        Defaults to ``courant_factor`` times the CFL step.
    courant_factor : float
        If ``dt`` is not given, ``dt = courant_factor * dt_CFL``. Values > 1 are
        the CFL-breaking regime (unconditionally stable here).
    boundary : {"pec", "absorber"}
        Domain truncation (see module docstring).
    pml_cells : int
        Absorber thickness in cells (``boundary="absorber"``).
    backend : {"numpy", "rust"}
        ``"rust"`` runs the parallel tridiagonal sweeps in the compiled core.
    """

    def __init__(
        self,
        eps_r: np.ndarray,
        cell_size,
        *,
        dt: Optional[float] = None,
        courant_factor: float = 1.0,
        boundary: str = "pec",
        pml_cells: int = 12,
        sources: Sequence[ADISource] = (),
        backend: str = "numpy",
        precision: str = "float64",
    ) -> None:
        self.eps_r = np.ascontiguousarray(eps_r, dtype=np.float64)
        self.nx, self.ny = self.eps_r.shape
        if isinstance(cell_size, (int, float)):
            self.dx = self.dy = float(cell_size)
        else:
            self.dx, self.dy = float(cell_size[0]), float(cell_size[1])
        if boundary not in ("pec", "absorber"):
            raise ValueError("boundary must be 'pec' or 'absorber'")
        self.boundary = boundary
        self.pml_cells = int(pml_cells)
        self.sources = list(sources)
        if backend not in ("numpy", "rust"):
            raise ValueError("backend must be 'numpy' or 'rust'")
        self.backend = backend
        self.npdt = np.float32 if precision == "float32" else np.float64

        dt_cfl = 1.0 / (C_0 * np.sqrt(1.0 / self.dx ** 2 + 1.0 / self.dy ** 2))
        self.dt = float(dt) if dt is not None else courant_factor * dt_cfl
        self.dt_cfl = dt_cfl
        self.courant_factor = self.dt / dt_cfl

        # Electric / magnetic conductivity profiles (0 for PEC).
        self.sigma_e, self.sigma_m = self._build_conductivity()

    # ------------------------------------------------------------------ #
    def _build_conductivity(self):
        nx, ny = self.nx, self.ny
        sig_e = np.zeros((nx, ny))
        if self.boundary != "absorber":
            return sig_e, sig_e.copy()
        # Graded (polynomial) conductivity ramp in the boundary layers, matched
        # magnetic conductivity sigma_m = sigma_e * mu0/eps0 for normal-incidence
        # impedance matching. Not a CPML - a simple lossy layer.
        m = 3
        npml = self.pml_cells
        eta = MU_0 / EPS_0
        sig_max = 0.8 * (m + 1) / (np.sqrt(eta) * self.dx)
        ramp = np.zeros(nx)
        prof = sig_max * (np.arange(npml)[::-1] / max(npml - 1, 1)) ** m
        ax = np.zeros(nx); ay = np.zeros(ny)
        ax[:npml] = prof; ax[-npml:] = prof[::-1]
        ay[:npml] = prof; ay[-npml:] = prof[::-1]
        sig = np.maximum(ax[:, None], ay[None, :])
        return sig, sig * (MU_0 / EPS_0)

    # ------------------------------------------------------------------ #
    def run(self, steps: int, *, snapshot_interval: int = 0,
            probes: Sequence[Tuple[int, int]] = ()) -> ADIResult:
        if self.backend == "rust":
            return self._run_rust(steps, snapshot_interval, probes)
        return self._run_numpy(steps, snapshot_interval, probes)

    # ------------------------------------------------------------------ #
    def _run_numpy(self, steps, snapshot_interval, probes) -> ADIResult:
        nx, ny = self.nx, self.ny
        dx, dy, dt = self.dx, self.dy, self.dt
        eps = self.eps_r * EPS_0
        s = dt / 2.0
        c = s * s / MU_0

        Ez = np.zeros((nx, ny))
        Hx = np.zeros((nx, ny - 1))          # at (i, j+1/2)
        Hy = np.zeros((nx - 1, ny))          # at (i+1/2, j)

        # Tridiagonal off-diagonal (constant): -c/dx^2 (x sweep), -c/dy^2 (y).
        offx = -c / dx ** 2
        offy = -c / dy ** 2
        # Interior diagonals include the eps term; PEC -> solve interior nodes.
        snaps: List[np.ndarray] = []
        snap_t: List[float] = []
        probe_vals = {f"{i},{j}": [] for (i, j) in probes}
        times = []

        for n in range(steps):
            # ---- sub-step 1: implicit in x ---- #
            dHy_dx = np.zeros((nx, ny)); dHx_dy = np.zeros((nx, ny))
            dHy_dx[1:-1, :] = (Hy[1:, :] - Hy[:-1, :])[:, :] / dx
            dHx_dy[:, 1:-1] = (Hx[:, 1:] - Hx[:, :-1])[:, :] / dy
            rhs = eps * Ez + s * (dHy_dx - dHx_dy)
            Ez_star = self._solve_x(eps, offx, rhs, c, dx)
            self._inject(Ez_star, (n + 0.5), eps, s)   # half-step source
            # explicit H at the half step
            dEz_dx = (Ez_star[1:, :] - Ez_star[:-1, :]) / dx
            dEz_dy_star = (Ez_star[:, 1:] - Ez_star[:, :-1]) / dy
            Hy_star = self._advance_hy(Hy, dEz_dx, s)
            Hx_star = self._advance_hx(Hx,
                                       (Ez[:, 1:] - Ez[:, :-1]) / dy, s)

            # ---- sub-step 2: implicit in y ---- #
            dHy_dx2 = np.zeros((nx, ny)); dHx_dy2 = np.zeros((nx, ny))
            dHy_dx2[1:-1, :] = (Hy_star[1:, :] - Hy_star[:-1, :]) / dx
            dHx_dy2[:, 1:-1] = (Hx_star[:, 1:] - Hx_star[:, :-1]) / dy
            rhs2 = eps * Ez_star + s * (dHy_dx2 - dHx_dy2)
            Ez_new = self._solve_y(eps, offy, rhs2, c, dy)
            self._inject(Ez_new, (n + 1.0), eps, s)
            dEz_dx2 = (Ez_star[1:, :] - Ez_star[:-1, :]) / dx
            dEz_dy_new = (Ez_new[:, 1:] - Ez_new[:, :-1]) / dy
            Hy_new = self._advance_hy(Hy_star, dEz_dx2, s)
            Hx_new = self._advance_hx(Hx_star, dEz_dy_new, s)

            Ez, Hx, Hy = Ez_new, Hx_new, Hy_new

            t = (n + 1) * dt
            times.append(t)
            if snapshot_interval and (n % snapshot_interval == 0):
                snaps.append(Ez.copy()); snap_t.append(t)
            for (i, j) in probes:
                probe_vals[f"{i},{j}"].append(Ez[i, j])

        res = ADIResult(times=np.asarray(times))
        if snaps:
            res.snapshots["Ez"] = np.asarray(snaps)
            res.snapshots["t"] = np.asarray(snap_t)
        res.probes = {k: np.asarray(v) for k, v in probe_vals.items()}
        return res

    # ---- per-substep helpers (also the reference for the Rust core) ---- #
    def _solve_x(self, eps, off, rhs, c, dx):
        """Solve [eps - c d2/dx2] Ez = rhs for each row, PEC/absorber ends."""
        nx, ny = self.nx, self.ny
        diag = eps + 2.0 * c / dx ** 2 + self.sigma_e * (self.dt / 2.0)
        # Interior system over i=1..nx-2 (Ez=0 on PEC ends; absorber handled by
        # the sigma term keeping ends finite but we still pin the outermost).
        d = diag[1:-1, :].copy()
        r = rhs[1:-1, :].copy()
        sol = _thomas(d, off, off, r)
        out = np.zeros((nx, ny))
        out[1:-1, :] = sol
        return out

    def _solve_y(self, eps, off, rhs, c, dy):
        nx, ny = self.nx, self.ny
        diag = eps + 2.0 * c / dy ** 2 + self.sigma_e * (self.dt / 2.0)
        d = diag[:, 1:-1].T.copy()
        r = rhs[:, 1:-1].T.copy()
        sol = _thomas(d, off, off, r).T
        out = np.zeros((nx, ny))
        out[:, 1:-1] = sol
        return out

    def _advance_hy(self, Hy, dEz_dx, s):
        if self.boundary == "absorber":
            sm = 0.5 * (self.sigma_m[1:, :] + self.sigma_m[:-1, :])
            a = 1.0 - sm * s / MU_0
            b = 1.0 + sm * s / MU_0
            return (a / b) * Hy + (s / MU_0 / b) * dEz_dx
        return Hy + (s / MU_0) * dEz_dx

    def _advance_hx(self, Hx, dEz_dy, s):
        if self.boundary == "absorber":
            sm = 0.5 * (self.sigma_m[:, 1:] + self.sigma_m[:, :-1])
            a = 1.0 - sm * s / MU_0
            b = 1.0 + sm * s / MU_0
            return (a / b) * Hx - (s / MU_0 / b) * dEz_dy
        return Hx - (s / MU_0) * dEz_dy

    def _inject(self, Ez, step_frac, eps, s):
        for src in self.sources:
            val = src.amplitude * float(src.waveform(np.array([step_frac * self.dt]))[0])
            Ez[src.i, src.j] += val * (self.dt / 2.0) / (eps[src.i, src.j])

    # ------------------------------------------------------------------ #
    def _run_rust(self, steps, snapshot_interval, probes) -> ADIResult:
        from .rustbackend import _rust_module
        rs = _rust_module()
        if not hasattr(rs, "adi_step_range_f64"):
            raise RuntimeError(
                "the Rust extension was built without the ADI kernel; rebuild "
                "with 'cargo build --release' in rust/ and reinstall the .so.")
        return _run_adi_rust(self, rs, steps, snapshot_interval, probes)


def _run_adi_rust(sim, rs, steps, snapshot_interval, probes) -> ADIResult:
    """Drive the Rust ADI stepper (mirrors ``_run_numpy``)."""
    npdt = sim.npdt
    step_fn = rs.adi_step_range_f64 if npdt == np.float64 else rs.adi_step_range_f32
    nx, ny = sim.nx, sim.ny
    Ez = np.zeros((nx, ny), dtype=npdt)
    Hx = np.zeros((nx, ny - 1), dtype=npdt)
    Hy = np.zeros((nx - 1, ny), dtype=npdt)
    eps = np.ascontiguousarray(sim.eps_r * EPS_0, dtype=npdt)
    sig_e = np.ascontiguousarray(sim.sigma_e, dtype=npdt)
    sig_m = np.ascontiguousarray(sim.sigma_m, dtype=npdt)

    # Precompute source waveform tables (Ez soft source, injected at both
    # half steps as in the NumPy path).
    n_src = max(len(sim.sources), 1)
    src_ij = np.zeros((n_src, 2), dtype=np.int64)
    # Two columns per step: the (n+0.5) and (n+1.0) sample.
    src_vals = np.zeros((n_src, 2 * max(steps, 1)), dtype=npdt)
    for s_i, src in enumerate(sim.sources):
        src_ij[s_i] = (src.i, src.j)
        fr = np.empty(2 * steps)
        for n in range(steps):
            fr[2 * n] = src.amplitude * float(src.waveform(np.array([(n + 0.5) * sim.dt]))[0])
            fr[2 * n + 1] = src.amplitude * float(src.waveform(np.array([(n + 1.0) * sim.dt]))[0])
        src_vals[s_i] = fr.astype(npdt)

    rec = sorted(set(range(0, steps, snapshot_interval))) if snapshot_interval else []
    probe_arr = np.asarray(probes, dtype=np.int64).reshape(-1, 2)
    snaps = np.zeros((len(rec), nx, ny), dtype=npdt)
    probe_out = np.zeros((max(len(probe_arr), 1), steps), dtype=npdt)

    step_fn(Ez, Hx, Hy, eps, sig_e, sig_m,
            sim.dx, sim.dy, sim.dt,
            src_ij, src_vals, len(sim.sources),
            np.asarray(rec, dtype=np.int64), snaps,
            probe_arr, probe_out, steps,
            1 if sim.boundary == "absorber" else 0)

    res = ADIResult(times=(np.arange(steps) + 1) * sim.dt)
    if len(rec):
        res.snapshots["Ez"] = snaps
        res.snapshots["t"] = (np.asarray(rec) + 1) * sim.dt
    res.probes = {f"{i},{j}": probe_out[k]
                  for k, (i, j) in enumerate(probe_arr)}
    return res
