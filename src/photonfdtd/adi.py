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
effective-index reduction produces). The time loop is a ``jax.lax.scan``; each
sub-step's independent line solves run as one batched
``jax.lax.linalg.tridiagonal_solve``.

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
    precision : {"float64", "float32"}
        Working dtype of the JAX time loop.
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
        if precision not in ("float64", "float32"):
            raise ValueError("precision must be 'float64' or 'float32'")
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
        """Advance ``steps`` full ADI steps from zero fields.

        ``snapshot_interval`` records the full ``Ez`` after every step ``n``
        with ``n % snapshot_interval == 0`` (0 = none); ``probes`` records
        ``Ez`` at the listed ``(i, j)`` indices after every step.
        """
        import jax
        import jax.numpy as jnp
        from jax.lax.linalg import tridiagonal_solve

        npdt = self.npdt
        if npdt == np.float64:
            jax.config.update("jax_enable_x64", True)
        nx, ny = self.nx, self.ny
        # Python floats (weakly typed in JAX) so float32 runs stay float32.
        dx, dy, dt = float(self.dx), float(self.dy), float(self.dt)
        s = dt / 2.0
        c = s * s / MU_0
        eps = self.eps_r * EPS_0

        # Implicit operators [eps - c d2/dl2] over the interior nodes (Ez = 0
        # pinned on the outermost nodes), one tridiagonal system per line:
        # x sweep batched over columns (ny, nx-2), y sweep over rows (nx, ny-2).
        # The absorber's electric conductivity enters the diagonal.
        damp = self.sigma_e * s
        dX = (eps + 2.0 * c / dx ** 2 + damp)[1:-1, :].T
        dY = (eps + 2.0 * c / dy ** 2 + damp)[:, 1:-1]

        def offdiag(diag, off):
            lo = np.full_like(diag, off); lo[..., 0] = 0.0
            up = np.full_like(diag, off); up[..., -1] = 0.0
            return lo, up
        loX, upX = offdiag(dX, -c / dx ** 2)
        loY, upY = offdiag(dY, -c / dy ** 2)

        # Explicit H updates H <- a H + k dEz/dl, with the magnetic conductivity
        # averaged onto the staggered H nodes (a = 1, k = s/mu0 for PEC).
        smy = 0.5 * (self.sigma_m[1:, :] + self.sigma_m[:-1, :])
        smx = 0.5 * (self.sigma_m[:, 1:] + self.sigma_m[:, :-1])
        by = 1.0 + smy * s / MU_0
        bx = 1.0 + smx * s / MU_0
        ay, ky = (1.0 - smy * s / MU_0) / by, (s / MU_0) / by
        ax, kx = (1.0 - smx * s / MU_0) / bx, (s / MU_0) / bx

        # Soft Ez sources, sampled on host at both half steps and pre-scaled by
        # (dt/2)/eps at their cell.
        src_i = np.array([src.i for src in self.sources], dtype=np.int32)
        src_j = np.array([src.j for src in self.sources], dtype=np.int32)
        t_half = (np.arange(steps) + 0.5) * dt
        t_full = (np.arange(steps) + 1.0) * dt
        v_half = np.zeros((steps, len(self.sources)))
        v_full = np.zeros((steps, len(self.sources)))
        for k, src in enumerate(self.sources):
            scale = src.amplitude * s / eps[src.i, src.j]
            v_half[:, k] = scale * np.asarray(src.waveform(t_half), float)
            v_full[:, k] = scale * np.asarray(src.waveform(t_full), float)

        rec = list(range(0, steps, snapshot_interval)) if snapshot_interval else []
        slot = np.full(steps, -1, dtype=np.int32)
        slot[rec] = np.arange(len(rec), dtype=np.int32)
        pr = np.asarray(probes, dtype=np.int32).reshape(-1, 2)

        J = lambda a: jnp.asarray(a, dtype=npdt)
        eps_j = J(eps)
        dX_j, loX_j, upX_j = J(dX), J(loX), J(upX)
        dY_j, loY_j, upY_j = J(dY), J(loY), J(upY)
        ax_j, kx_j, ay_j, ky_j = J(ax), J(kx), J(ay), J(ky)
        si, sj = jnp.asarray(src_i), jnp.asarray(src_j)
        pi, pj = jnp.asarray(pr[:, 0]), jnp.asarray(pr[:, 1])

        def solve_x(rhs):
            sol = tridiagonal_solve(loX_j, dX_j, upX_j, rhs[1:-1, :].T[..., None])
            return jnp.zeros((nx, ny), npdt).at[1:-1, :].set(sol[..., 0].T)

        def solve_y(rhs):
            sol = tridiagonal_solve(loY_j, dY_j, upY_j, rhs[:, 1:-1][..., None])
            return jnp.zeros((nx, ny), npdt).at[:, 1:-1].set(sol[..., 0])

        def curl_h(Hx, Hy):
            dHy_dx = jnp.zeros((nx, ny), npdt).at[1:-1, :].set((Hy[1:, :] - Hy[:-1, :]) / dx)
            dHx_dy = jnp.zeros((nx, ny), npdt).at[:, 1:-1].set((Hx[:, 1:] - Hx[:, :-1]) / dy)
            return dHy_dx - dHx_dy

        def step(carry, xs):
            Ez, Hx, Hy, snaps = carry
            vh, vf, k = xs
            # sub-step 1: implicit in x, explicit H at the half step
            Ez_s = solve_x(eps_j * Ez + s * curl_h(Hx, Hy)).at[si, sj].add(vh)
            dEz_dx = (Ez_s[1:, :] - Ez_s[:-1, :]) / dx
            Hy_s = ay_j * Hy + ky_j * dEz_dx
            Hx_s = ax_j * Hx - kx_j * (Ez[:, 1:] - Ez[:, :-1]) / dy
            # sub-step 2: implicit in y
            Ez_n = solve_y(eps_j * Ez_s + s * curl_h(Hx_s, Hy_s)).at[si, sj].add(vf)
            Hy_n = ay_j * Hy_s + ky_j * dEz_dx
            Hx_n = ax_j * Hx_s - kx_j * (Ez_n[:, 1:] - Ez_n[:, :-1]) / dy
            if rec:
                snaps = jax.lax.cond(k >= 0, lambda b: b.at[k].set(Ez_n),
                                     lambda b: b, snaps)
            return (Ez_n, Hx_n, Hy_n, snaps), Ez_n[pi, pj]

        @jax.jit
        def simulate(v_half, v_full, slot):
            init = (jnp.zeros((nx, ny), npdt), jnp.zeros((nx, ny - 1), npdt),
                    jnp.zeros((nx - 1, ny), npdt),
                    jnp.zeros((len(rec), nx, ny), npdt))
            (_, _, _, snaps), pvals = jax.lax.scan(step, init,
                                                   (v_half, v_full, slot))
            return snaps, pvals

        snaps, pvals = simulate(J(v_half), J(v_full), jnp.asarray(slot))

        res = ADIResult(times=t_full)
        if rec:
            res.snapshots["Ez"] = np.asarray(snaps)
            res.snapshots["t"] = (np.asarray(rec) + 1.0) * dt
        pvals = np.asarray(pvals)
        res.probes = {f"{i},{j}": pvals[:, k] for k, (i, j) in enumerate(pr)}
        return res
