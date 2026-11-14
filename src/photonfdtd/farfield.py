r"""Near-to-far-field projection from frequency-domain monitor planes.

The far field of everything inside a closed surface follows from the
tangential fields on it (Love's equivalence principle): the surface carries
electric and magnetic currents

.. math::

    \mathbf J = \hat n \times \mathbf H, \qquad \mathbf M = -\hat n \times \mathbf E,

with :math:`\hat n` the outward normal, whose radiation vectors

.. math::

    \mathbf N = \oint \mathbf J\, e^{-ik\hat r\cdot\mathbf r'}\,dS, \qquad
    \mathbf L = \oint \mathbf M\, e^{-ik\hat r\cdot\mathbf r'}\,dS

give the field at distance :math:`r` in direction :math:`\hat r`
(:math:`e^{-i\omega t}` convention, which is the one :class:`DFTMonitor`
produces):

.. math::

    E_\theta = \frac{ik\,e^{ikr}}{4\pi r}(\eta N_\theta + L_\phi), \qquad
    E_\phi = \frac{ik\,e^{ikr}}{4\pi r}(\eta N_\phi - L_\theta).

In 2-D (a grid collapsed on z, fields invariant along it) the surface is a
closed contour, the integrals are line integrals, and the 3-D propagator
:math:`e^{ikr}/4\pi r` becomes the large-argument 2-D Green's function
:math:`\tfrac{i}{4}\sqrt{2/\pi k\rho}\,e^{i(k\rho-\pi/4)}`.

Record each face of the surface with a plane :class:`DFTMonitor`
(``plane_axis`` / ``plane_position``) carrying the tangential E and H
components. Those monitors already put every component on the plane (normal
staggering averaged out); the in-plane half-cell staggering is kept exact by
placing each sample at its own Yee position in the phase factor. Cells inside
the PML are dropped, and the faces crop each other so the box closes exactly.
A single open plane is also accepted, e.g. the plane a
Cherenkov cone crosses, as the standard half-space approximation.

The returned :class:`FarField` holds :math:`r\,e^{-ikr}\mathbf E` (3-D) or
:math:`\sqrt\rho\,e^{-ik\rho}\mathbf E` (2-D), so it is independent of the
observation distance, and the radiant intensity
:math:`|r\mathbf E|^2/2\eta` per steradian (per radian and unit length in
2-D), in the units of the monitor spectra (the DFT of a pulse gives energy
spectral densities).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
import math
import numpy as np

from .constants import C_0, EPS_0, MU_0

ETA_0 = math.sqrt(MU_0 / EPS_0)
_AX = {"x": 0, "y": 1, "z": 2}


@dataclass
class FarField:
    """Projected far field at a set of directions and frequencies.

    Attributes
    ----------
    freqs : (n_f,) array
    directions : (n_dir, 3) array of unit vectors
    E : (n_f, n_dir, 3) complex array
        Distance-normalised far field, ``r e^{-ikr} E`` (3-D) or
        ``sqrt(rho) e^{-ik rho} E`` (2-D), Cartesian components.
    index : float
        Background refractive index the field radiates into.
    ndim : int
        2 or 3.
    """
    freqs: np.ndarray
    directions: np.ndarray
    E: np.ndarray
    index: float
    ndim: int

    @property
    def eta(self) -> float:
        return ETA_0 / self.index

    @property
    def intensity(self) -> np.ndarray:
        """Radiant intensity ``|rE|^2 / 2 eta``, shape (n_f, n_dir): per
        steradian in 3-D, per radian (and unit length along z) in 2-D."""
        return np.sum(np.abs(self.E) ** 2, axis=-1) / (2.0 * self.eta)

    def field_at(self, distance: float) -> np.ndarray:
        """The far field ``E`` itself at ``distance`` (m), (n_f, n_dir, 3)."""
        k = 2 * math.pi * self.freqs * self.index / C_0
        ph = np.exp(1j * k * distance)[:, None, None]
        scale = distance if self.ndim == 3 else math.sqrt(distance)
        return self.E * ph / scale

    def components(self) -> Tuple[np.ndarray, np.ndarray]:
        """Spherical components ``(E_theta, E_phi)`` of :attr:`E` (each
        (n_f, n_dir)), with theta from +z and phi from +x in the xy plane.
        In 2-D, ``E_theta = -E_z`` and ``E_phi`` is the in-plane field."""
        th_hat, ph_hat = _spherical_unit_vectors(self.directions)
        return (np.einsum("fdc,dc->fd", self.E, th_hat),
                np.einsum("fdc,dc->fd", self.E, ph_hat))


def spherical_directions(theta, phi) -> np.ndarray:
    """Unit vectors for every (theta, phi) pair of two equal-shape arrays
    (radians; theta from +z, phi from +x), flattened to (n, 3)."""
    th, ph = np.broadcast_arrays(np.asarray(theta, float), np.asarray(phi, float))
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph),
                     np.cos(th)], axis=-1).reshape(-1, 3)


def polar_directions(phi) -> np.ndarray:
    """In-plane unit vectors at angles ``phi`` from +x (radians), (n, 3)."""
    ph = np.asarray(phi, float).ravel()
    return np.stack([np.cos(ph), np.sin(ph), np.zeros_like(ph)], axis=-1)


def _spherical_unit_vectors(d):
    d = np.asarray(d, float)
    th = np.arccos(np.clip(d[:, 2], -1, 1))
    ph = np.arctan2(d[:, 1], d[:, 0])
    th_hat = np.stack([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph),
                       -np.sin(th)], axis=-1)
    ph_hat = np.stack([-np.sin(ph), np.cos(ph), np.zeros_like(ph)], axis=-1)
    return th_hat, ph_hat


def _parse_normal(normal: str) -> Tuple[int, float]:
    s = str(normal).strip().lower()
    if len(s) != 2 or s[0] not in "+-" or s[1] not in _AX:
        raise ValueError(f"surface normal must be '+x', '-x', ..., '-z'; got {normal!r}")
    return _AX[s[1]], (1.0 if s[0] == "+" else -1.0)


def _component_coords(grid, comp, axis):
    """Yee coordinates of ``comp`` along ``axis``: E_a is offset half a cell
    along a; H_a half a cell along the two other axes."""
    c = np.asarray(grid.coords[axis], float)
    if c.size == 1:
        return c
    a = _AX[comp[1]]
    offset = (a == axis) if comp[0] == "E" else (a != axis)
    return c + (0.5 * grid.cell_size[axis] if offset else 0.0)


def _weights(coords, lo, hi, dl):
    """Midpoint quadrature weights for samples in [lo, hi], with half
    weights at the end points (shared edges of adjacent faces)."""
    tol = 1e-6 * dl
    w = np.where((coords > lo + tol) & (coords < hi - tol), dl, 0.0)
    w = np.where((np.abs(coords - lo) <= tol) | (np.abs(coords - hi) <= tol), 0.5 * dl, w)
    return w


def project_far_field(sim, result, surfaces: Dict[str, str], directions,
                      index: float = 1.0,
                      freqs: Optional[Sequence[float]] = None) -> FarField:
    """Project plane :class:`DFTMonitor` data to the far field.

    Parameters
    ----------
    sim : Simulation
        The simulation the monitors belong to (for the grid and the PML).
    result : Result
        Its run result.
    surfaces : dict
        ``{monitor_name: outward_normal}``, e.g.
        ``{"xm": "-x", "xp": "+x", "ym": "-y", "yp": "+y"}`` for a closed
        2-D contour. Each monitor must be a plane DFT monitor
        (``plane_axis``/``plane_position``, ``downsample=1``) recording the
        tangential components it has (missing ones count as zero, which is
        exact for a polarization that does not have them, e.g. ``Ez, Hx, Hy``
        in 2-D). The faces define the box: on an axis with a ``-a`` and a
        ``+a`` face, every other face is cropped to between those two planes
        (as snapped to the grid), with half weights on the shared edges, so
        the surface closes exactly. An axis without faces is not cropped
        beyond dropping its PML cells, so a single plane projects as an
        open surface.
    directions : (n, 3) array
        Unit observation directions (:func:`spherical_directions`,
        :func:`polar_directions`). In 2-D they must lie in the xy plane.
    index : float
        Refractive index of the homogeneous medium the surface sits in.
    freqs : sequence or None
        Subset of the monitors' frequencies to project (default: all; all
        surfaces must share them).
    """
    grid = sim.grid
    if getattr(grid, "has_symmetry", False):
        raise NotImplementedError(
            "far-field projection on a grid with symmetry planes is not "
            "supported yet; run the full domain for the projection")
    collapsed = [a for a in range(3) if grid.shape[a] == 1]
    if len(collapsed) > 1:
        raise ValueError("far-field projection needs a 2-D or 3-D grid")
    if collapsed and collapsed != [2]:
        raise ValueError("2-D far-field projection expects the grid collapsed on z")
    ndim = 3 - len(collapsed)
    d = np.atleast_2d(np.asarray(directions, float))
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    if ndim == 2 and np.abs(d[:, 2]).max() > 1e-9:
        raise ValueError("2-D projection directions must lie in the xy plane")

    mons = {m.name: m for m in sim.monitors}
    all_f = None
    for name in surfaces:
        f = np.asarray(result.dft_freqs[name], float)
        if all_f is None:
            all_f = f
        elif f.shape != all_f.shape or not np.allclose(f, all_f):
            raise ValueError("all projection surfaces must share their frequencies")
    if freqs is None:
        fsel = np.arange(all_f.size)
    else:
        fsel = np.array([int(np.argmin(np.abs(all_f - fq))) for fq in freqs])
    fr = all_f[fsel]
    k = 2 * math.pi * fr * index / C_0                       # (n_f,)
    eta = ETA_0 / index

    # Snap every face to its grid plane; faces on both sides of an axis
    # bound the box along it.
    faces = []
    box = [[None, None] for _ in range(3)]
    for name, normal in surfaces.items():
        m = mons[name]
        ax, sgn = _parse_normal(normal)
        if getattr(m, "plane_axis", None) is not None:
            m_ax = _AX[m.plane_axis]
            pos = m.plane_position
        elif getattr(m, "plane_z", None) is not None:
            m_ax, pos = 2, m.plane_z
        else:
            raise ValueError(f"monitor {name!r} is not a plane DFTMonitor")
        if m_ax != ax:
            raise ValueError(f"monitor {name!r} is normal to {'xyz'[m_ax]}, "
                             f"not the surface normal {normal!r}")
        if m.downsample != 1:
            raise ValueError(f"monitor {name!r}: far-field projection needs downsample=1")
        ci = int(np.argmin(np.abs(grid.coords[ax] - pos)))
        x_plane = float(grid.coords[ax][ci])
        box[ax][0 if sgn < 0 else 1] = x_plane
        faces.append((name, ax, sgn, x_plane))
    for a in range(3):
        if (box[a][0] is None) != (box[a][1] is None):
            box[a] = [None, None]     # one-sided: an open surface on this axis
        elif box[a][0] is not None and box[a][0] >= box[a][1]:
            raise ValueError(f"the '-{'xyz'[a]}' face must lie below the '+{'xyz'[a]}' face")

    N = np.zeros((fr.size, d.shape[0], 3), complex)
    L = np.zeros_like(N)
    for name, ax, sgn, x_plane in faces:
        nhat = np.zeros(3); nhat[ax] = sgn
        data = result.dft[name]
        tan = [a for a in range(3) if a != ax]
        for comp, arr in data.items():
            ca = _AX[comp[1]]
            if ca == ax:
                continue                       # normal components carry no current
            F = np.asarray(arr)[fsel]          # (n_f, *plane shape)
            F = np.squeeze(F, axis=1 + ax)     # (n_f, n_t1, n_t2)
            # Current: J = n x H (from H), M = -n x E (from E).
            unit = np.zeros(3); unit[ca] = 1.0
            vec = np.cross(nhat, unit) * (1.0 if comp[0] == "H" else -1.0)
            # Sample coordinates and quadrature weights on the two in-plane axes.
            ws, cs = [], []
            for a in tan:
                c = _component_coords(grid, comp, a)
                if grid.shape[a] == 1:
                    ws.append(np.ones(1)); cs.append(np.zeros(1)); continue
                dl = grid.cell_size[a]
                npml = int(grid.pml_layers[a])
                lo_g = grid.coords[a][npml] if npml else -np.inf
                hi_g = grid.coords[a][-1 - npml] if npml else np.inf
                if box[a][0] is not None:
                    w = _weights(c, max(box[a][0], lo_g), min(box[a][1], hi_g), dl)
                else:
                    w = np.where((c >= lo_g) & (c <= hi_g), dl, 0.0)
                ws.append(w); cs.append(c)
            W = ws[0][:, None] * ws[1][None, :]
            if not W.any():
                continue
            i1, i2 = np.nonzero(W)
            r = np.zeros((i1.size, 3))
            r[:, ax] = x_plane
            r[:, tan[0]] = cs[0][i1]
            r[:, tan[1]] = cs[1][i2]
            phase = d @ r.T                                   # (n_dir, n_pts)
            f_pts = F[:, i1, i2] * W[i1, i2]                  # (n_f, n_pts)
            # Integral over the face, sum_pts f e^{-ik d.r}, one frequency
            # at a time so memory stays O(n_dir * n_pts).
            target = N if comp[0] == "H" else L
            for fi in range(fr.size):
                I = np.exp(-1j * k[fi] * phase) @ f_pts[fi]   # (n_dir,)
                target[fi] += I[:, None] * vec[None, :]

    th_hat, ph_hat = _spherical_unit_vectors(d)
    Nt = np.einsum("fdc,dc->fd", N, th_hat); Np = np.einsum("fdc,dc->fd", N, ph_hat)
    Lt = np.einsum("fdc,dc->fd", L, th_hat); Lp = np.einsum("fdc,dc->fd", L, ph_hat)
    if ndim == 3:
        pref = 1j * k / (4 * math.pi)
    else:
        # ik e^{ikr}/(4 pi r) integrated over z -> ik (i/4) sqrt(2/(pi k rho)) e^{i(k rho - pi/4)}
        pref = 1j * k * 0.25j * np.sqrt(2.0 / (math.pi * k)) * np.exp(-0.25j * math.pi)
    Eth = pref[:, None] * (eta * Nt + Lp)
    Eph = pref[:, None] * (eta * Np - Lt)
    E = Eth[..., None] * th_hat[None] + Eph[..., None] * ph_hat[None]
    return FarField(freqs=fr, directions=d, E=E, index=float(index), ndim=ndim)
