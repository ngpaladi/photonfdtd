"""Diffraction orders of a periodic structure from a plane DFT monitor.

On a cell periodic along the plane's transverse axes (``Grid(periodic=...)``,
optionally with a Bloch wavevector ``k_B``), the field on a plane normal to
``p`` is a sum of plane waves ("orders")

    F(x) = sum_m  F_m  exp(i k_m x),   k_m = k_B + 2 pi m / L

and the flux through the plane splits exactly into one term per order
(Parseval): ``P_m = Re(E1_m H2_m* - E2_m H1_m*) * A``. :func:`diffraction_orders`
returns that split at one frequency, with each order's direction: an order
propagates when ``|k_m| < n k0`` and leaves at ``sin(theta) = |k_m| / (n k0)``.

Each component's Fourier coefficient is taken at its own Yee position
(components staggered half a cell along a transverse axis pick up the phase
``exp(i k_m dx/2)``), so E and H of an order pair up exactly. The sum over
orders equals :func:`photonfdtd.photon_spectrum` on the same plane.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np

from .constants import C_0
from .monitors import DFTMonitor
from .photons import _TANGENTIAL, photon_spectrum

__all__ = ["DiffractionOrders", "diffraction_orders"]


@dataclass
class DiffractionOrders:
    """Per-order split of the flux through a periodic plane at one frequency.

    Attributes
    ----------
    freq : float
        Frequency (Hz).
    orders : (n, 2) int array
        Order indices ``(m1, m2)`` along the plane's two transverse axes
        (``t_axes``); 0 on a collapsed axis.
    t_axes : tuple of int
        The two transverse grid axes.
    k : (n, 2) array
        Transverse wavevector of each order (rad/m), Bloch part included.
    propagating : (n,) bool array
        ``|k| < n k0``: the order carries power away; the rest are evanescent.
    sin_theta : (n, 2) array
        Direction sines ``k / (n k0)`` along ``t_axes`` (NaN when evanescent).
    power : (n,) array
        The order's share of ``photon_spectrum(...).energy_density`` at
        ``freq`` (same units; they sum to it).
    fraction : (n,) array
        ``power / sum(power)``.
    plane_sum : (n,) array
        The order's part of ``sum over the plane cells of
        Re(E1 H2* - E2 H1*)`` (raw DFT units); Parseval makes these add up to
        that sum exactly.
    """
    freq: float
    orders: np.ndarray
    t_axes: Tuple[int, int]
    k: np.ndarray
    propagating: np.ndarray
    sin_theta: np.ndarray
    power: np.ndarray
    fraction: np.ndarray
    plane_sum: np.ndarray

    def order(self, m1: int, m2: int = 0) -> int:
        """Row index of order ``(m1, m2)``."""
        hit = np.flatnonzero((self.orders[:, 0] == m1) & (self.orders[:, 1] == m2))
        if hit.size == 0:
            raise KeyError((m1, m2))
        return int(hit[0])


def diffraction_orders(result, sim, monitor: str, freq_index: int = 0,
                       direction: int = +1, index: float = 1.0) -> DiffractionOrders:
    """Split the flux through a plane :class:`~photonfdtd.DFTMonitor` into
    diffraction orders.

    Parameters
    ----------
    result, sim
        The run and its simulation.
    monitor : str
        A plane DFT monitor (``plane_axis``/``plane_position``,
        ``downsample=1``) recording the plane's tangential E and H, as for
        :func:`photonfdtd.photon_spectrum`. Every non-collapsed transverse
        axis must be periodic.
    freq_index : int
        Which of the monitor's frequencies.
    direction : +1 or -1
        Count flux along ``+`` or ``-`` the plane normal.
    index : float
        Refractive index of the medium at the plane (sets ``n k0``).
    """
    m = next((mm for mm in sim.monitors if mm.name == monitor), None)
    if not isinstance(m, DFTMonitor) or m.plane() is None:
        raise ValueError(f"{monitor!r} must be a plane DFTMonitor of this simulation")
    if m.downsample != 1:
        raise ValueError("diffraction_orders needs downsample=1")
    grid = sim.grid
    if getattr(grid, "has_symmetry", False):
        raise NotImplementedError("diffraction_orders does not support symmetry planes")
    p = m.plane()[0]
    t_axes = tuple(a for a in range(3) if a != p)
    for a in t_axes:
        if grid.shape[a] > 1 and not grid.periodic[a]:
            raise ValueError(f"transverse axis {'xyz'[a]} is not periodic; "
                             "diffraction orders need a periodic cell")
    names = _TANGENTIAL[p]
    missing = [c for c in names if c not in m.components]
    if missing:
        raise ValueError(f"DFTMonitor {monitor!r} must record {names}; "
                         f"missing {tuple(missing)}")
    freq = float(m.freqs[freq_index])
    bloch = getattr(grid, "bloch", (0.0, 0.0, 0.0))

    # Order wavevectors along each transverse axis.
    ks, ms = [], []
    for a in t_axes:
        n = grid.shape[a]
        if n == 1:
            ks.append(np.array([0.0])); ms.append(np.array([0]))
            continue
        L = n * grid.cell_size[a]
        mm = np.fft.fftfreq(n, d=1.0 / n).round().astype(int)
        ms.append(mm)
        ks.append(bloch[a] + 2.0 * math.pi * mm / L)

    def coeffs(comp):
        F = np.asarray(result.dft[monitor][comp])[freq_index]
        F = np.squeeze(F, axis=p).astype(complex)      # (n_t1, n_t2)
        for j, a in enumerate(t_axes):
            n = grid.shape[a]
            if n == 1:
                continue
            along = "xyz".index(comp[1]) == a
            stag = along if comp[0] == "E" else not along
            x = np.asarray(grid.coords[a]) + (0.5 * grid.cell_size[a] if stag else 0.0)
            shp = [1, 1]; shp[j] = -1
            F = F * np.exp(-1j * bloch[a] * x).reshape(shp)   # strip the Bloch phase
            F = np.fft.fft(F, axis=j) / n
            # The staggered sample sits dx/2 up: remove the order's phase there.
            if stag:
                k_rel = ks[j] - bloch[a]
                F = F * np.exp(-1j * k_rel * 0.5 * grid.cell_size[a]).reshape(shp)
        return F

    E1, E2, H1, H2 = (coeffs(c) for c in names)
    P = direction * (E1 * H2.conj() - E2 * H1.conj()).real
    K1, K2 = np.meshgrid(ks[0], ks[1], indexing="ij")
    M1, M2 = np.meshgrid(ms[0], ms[1], indexing="ij")
    k0n = 2.0 * math.pi * freq * float(index) / C_0
    kk = np.stack([K1.ravel(), K2.ravel()], axis=1)
    prop = np.hypot(kk[:, 0], kk[:, 1]) < k0n
    sin = np.where(prop[:, None], kk / k0n, np.nan)
    Pf = P.ravel()
    total = Pf.sum()
    frac = Pf / total if total != 0 else np.zeros_like(Pf)
    ref = photon_spectrum(result, sim, monitor, direction=direction).energy_density[freq_index]
    order = np.argsort(np.abs(M1.ravel()) + np.abs(M2.ravel()), kind="stable")
    return DiffractionOrders(
        freq=freq, orders=np.stack([M1.ravel(), M2.ravel()], axis=1)[order],
        t_axes=t_axes, k=kk[order], propagating=prop[order], sin_theta=sin[order],
        power=(frac * ref)[order], fraction=frac[order],
        plane_sum=direction * Pf[order] * Pf.size)
