"""Eigenmode expansion (EME) for piecewise-uniform 2-D photonic devices.

FDTD time-steps every cell for thousands of steps; for a device built from
sections that are *uniform along the propagation axis* (straight waveguides,
step junctions, and - as a staircase of short uniform slices - tapers and
adiabatic transitions), eigenmode expansion is astronomically cheaper. Each
section is represented by the handful of local eigenmodes it supports; within
a section a mode just accumulates the phase ``exp(i beta_m L)``; and the whole
device S-matrix is the cascade of one small mode-overlap scattering matrix per
interface with the diagonal propagation matrices between them. Cost is set by
the number of modes kept, not by the device length - a millimetre-long taper
costs the same as a micron-long one.

Scope: 2-D, single polarisation, propagation along x with the transverse
profile along y. This is exactly the problem left after the 2.5-D
effective-index reduction (:mod:`photonfdtd.eim`), so EME + EIM together take a
layout-scale device to an S-matrix without ever time-stepping it. ``"Ez"``
(default) is the out-of-plane-E polarisation of a 2-D ``xy`` grid - the fields
are ``(Ez, Hx, Hy)`` and the transverse mode problem is the scalar Helmholtz
equation ``d2Ez/dy2 + (k0^2 eps(y) - beta^2) Ez = 0``; ``"Hz"`` is the dual.

The interface scattering matrix uses power-orthogonal mode normalisation and
the standard transverse-field continuity mode match; sections are cascaded with
the Redheffer star product. Validated in the tests against the FDTD backend: a
uniform guide gives ``|T|=1``/``|R|=0`` to mode-truncation error, and a step
junction's modal reflection matches an FDTD S-parameter measurement.

References
----------
* D. F. G. Gallagher & T. P. Felici, "Eigenmode expansion methods ...",
  Proc. SPIE 4987 (2003).
* A. W. Snyder & J. D. Love, *Optical Waveguide Theory*, ch. 31 (orthogonality
  and the star product).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .constants import C_0, EPS_0, MU_0


@dataclass
class Section:
    """One x-uniform slice of the device: a transverse permittivity and length.

    ``eps`` is ``eps_r(y)`` sampled on the shared transverse grid; ``length`` is
    the extent along x (m). Build these directly, or with
    :func:`sections_from_eps` from a 2-D ``eps_r(x, y)`` map.
    """
    eps: np.ndarray            #: eps_r(y), shape (ny,)
    length: float              #: section length along x (m)


@dataclass
class EMEResult:
    """Device scattering matrix and the modal data behind it."""
    wavelength: float
    #: Full 2N x 2N scattering matrix in the combined mode basis of the input
    #: (port 0) and output (port 1) sections. Ordered [in-modes, out-modes].
    S: np.ndarray
    n_eff_in: np.ndarray       #: input-section modal effective indices
    n_eff_out: np.ndarray      #: output-section modal effective indices
    num_modes: int

    def transmission(self, out_mode: int = 0, in_mode: int = 0) -> complex:
        """Complex transmission amplitude from input ``in_mode`` to output
        ``out_mode`` (forward-to-forward block of ``S``)."""
        N = self.num_modes
        return complex(self.S[N + out_mode, in_mode])

    def reflection(self, out_mode: int = 0, in_mode: int = 0) -> complex:
        """Complex reflection amplitude back into the input section."""
        return complex(self.S[out_mode, in_mode])


def _solve_modes(eps: np.ndarray, dy: float, wavelength: float, num_modes: int,
                 polarization: str):
    """Local eigenmodes of a 1-D transverse slice (``Ez`` polarisation).

    Returns ``(beta, e, h)``: propagation constants (largest first), the modal
    ``Ez`` profiles, and the transverse magnetic field ``Hy = beta/(omega mu0)
    Ez``. Modes are power-normalised so ``integral e_m h_m dy = 1`` and
    ``integral e_i h_k dy = delta_ik`` within a section, which makes the
    interface match a plain projection.
    """
    ny = eps.size
    k0 = 2.0 * np.pi / wavelength
    inv_dy2 = 1.0 / dy ** 2
    # d2Ez/dy2 + (k0^2 eps - beta^2) Ez = 0. Symmetric tridiagonal.
    main = -2.0 * inv_dy2 + k0 ** 2 * eps
    off = inv_dy2 * np.ones(ny - 1)
    w, v = np.linalg.eigh(np.diag(main) + np.diag(off, 1) + np.diag(off, -1))

    order = np.argsort(-w)[:num_modes]
    beta2 = w[order]
    beta = np.sqrt(beta2.astype(complex))          # evanescent -> +i|beta|
    beta = np.where(beta.imag < 0, -beta, beta)    # decaying convention
    e = v[:, order].T.astype(complex).copy()       # (num_modes, ny), real evecs

    omega = C_0 * k0
    h = np.empty_like(e)
    for m in range(e.shape[0]):
        h[m] = beta[m] / (omega * MU_0) * e[m]
        norm = np.sum(e[m] * h[m]) * dy            # = beta/(omega mu0) |e|^2 dy
        s = 1.0 / np.sqrt(norm) if abs(norm) > 0 else 1.0
        e[m] *= s
        h[m] *= s
    return beta, e, h


def _overlap(a, b, dy):
    """Matrix ``O[i,k] = integral a_i(y) b_k(y) dy``."""
    return (a[:, None, :] * b[None, :, :]).sum(axis=2) * dy


def _interface_rt(eL, hL, eR, hR, dy):
    """Reflection/transmission blocks for left-incidence at an interface.

    Transverse-field continuity across the junction:

        Ez:  sum_i (a_i+b_i) e^L_i = sum_j (c_j+d_j) e^R_j
        Hy:  sum_i (a_i-b_i) h^L_i = sum_j (c_j-d_j) h^R_j

    Projecting both onto the left section's power-orthonormal modes (using
    ``<h^L_k, e^L_i> = <e^L_k, h^L_i> = delta_ki``) gives, for left incidence
    (d=0),

        a + b = B c,   a - b = C c,
        B[k,j] = <h^L_k, e^R_j>,   C[k,j] = <e^L_k, h^R_j>,

    so  c = 2 (B+C)^{-1} a  (transmission) and  b = (B-C)(B+C)^{-1} a
    (reflection). Two identical sections give B=C=I, hence r=0, t=I - exactly no
    reflection, and the form is power-conserving up to the truncated radiation
    continuum.
    """
    B = _overlap(hL, eR, dy)      # B[k,j] = <h^L_k, e^R_j>
    C = _overlap(eL, hR, dy)      # C[k,j] = <e^L_k, h^R_j>
    inv = np.linalg.inv(B + C)
    r = (B - C) @ inv
    t = 2.0 * inv
    return r, t


def _star(S_a, S_b, N):
    """Redheffer star product of two scattering matrices (N forward modes).

    Blocks: ``S11`` = left reflection, ``S21`` = forward transmission,
    ``S12`` = backward transmission, ``S22`` = right reflection.
    """
    a11, a12, a21, a22 = S_a[:N, :N], S_a[:N, N:], S_a[N:, :N], S_a[N:, N:]
    b11, b12, b21, b22 = S_b[:N, :N], S_b[:N, N:], S_b[N:, :N], S_b[N:, N:]
    I = np.eye(N)
    D1 = np.linalg.inv(I - b11 @ a22)
    D2 = np.linalg.inv(I - a22 @ b11)
    S11 = a11 + a12 @ D1 @ b11 @ a21
    S12 = a12 @ D1 @ b12
    S21 = b21 @ D2 @ a21
    S22 = b22 + b21 @ D2 @ a22 @ b12
    out = np.zeros((2 * N, 2 * N), dtype=complex)
    out[:N, :N] = S11
    out[:N, N:] = S12
    out[N:, :N] = S21
    out[N:, N:] = S22
    return out


def _prop_smatrix(beta, length, N):
    """Diagonal propagation S-matrix over a uniform section of given length."""
    ph = np.exp(1j * beta * length)
    P = np.diag(ph)
    S = np.zeros((2 * N, 2 * N), dtype=complex)
    S[N:, :N] = P            # forward transmit
    S[:N, N:] = P            # backward transmit
    return S


def _interface_full(eL, hL, eR, hR, dy, N):
    """Full interface S-matrix (both incidence directions)."""
    r, t = _interface_rt(eL, hL, eR, hR, dy)          # left incidence
    r2, t2 = _interface_rt(eR, hR, eL, hL, dy)        # right incidence
    S = np.zeros((2 * N, 2 * N), dtype=complex)
    S[:N, :N] = r
    S[N:, :N] = t
    S[N:, N:] = r2
    S[:N, N:] = t2
    return S


def eme_2d(
    sections: Sequence[Section],
    cell_size: float,
    wavelength: float,
    *,
    num_modes: int = 1,
    polarization: str = "Ez",
) -> EMEResult:
    """Scattering matrix of a piecewise-uniform 2-D device by EME.

    Parameters
    ----------
    sections : sequence of :class:`Section`
        The device as x-uniform slices, input first. All share one transverse
        grid (same ``eps.size``).
    cell_size : float
        Transverse grid step ``dy`` (m).
    wavelength : float
        Free-space wavelength (m).
    num_modes : int
        Modes kept per section (accuracy vs cost).
    polarization : {"Ez", "Hz"}
        2-D field polarisation (see module docstring).
    """
    if polarization == "Hz":
        raise NotImplementedError(
            "EME currently supports the 'Ez' polarisation (the out-of-plane-E "
            "2-D field that pairs with the effective-index reduction). 'Hz' is "
            "planned.")
    if polarization != "Ez":
        raise ValueError("polarization must be 'Ez'")
    if len(sections) < 1:
        raise ValueError("need at least one section")
    dy = float(cell_size)
    N = int(num_modes)

    modes = [_solve_modes(s.eps, dy, wavelength, N, polarization)
             for s in sections]

    # Cascade: [prop sec0] * [iface 0->1] * [prop sec1] * ...
    S = _prop_smatrix(modes[0][0], sections[0].length, N)
    for i in range(len(sections) - 1):
        _, eL, hL = modes[i]
        _, eR, hR = modes[i + 1]
        Sif = _interface_full(eL, hL, eR, hR, dy, N)
        S = _star(S, Sif, N)
        S = _star(S, _prop_smatrix(modes[i + 1][0], sections[i + 1].length, N), N)

    k0 = 2.0 * np.pi / wavelength
    return EMEResult(
        wavelength=float(wavelength), S=S,
        n_eff_in=np.real(modes[0][0]) / k0,
        n_eff_out=np.real(modes[-1][0]) / k0,
        num_modes=N,
    )


def sections_from_eps(
    eps_xy: np.ndarray,
    cell_size: float,
    *,
    group: int = 1,
) -> List[Section]:
    """Slice a 2-D ``eps_r(x, y)`` map into x-uniform EME sections.

    Each column of ``eps_xy`` (or each ``group`` columns, averaged) becomes one
    :class:`Section` of length ``group * cell_size``. A staircase of these
    approximates a taper or bend to the resolution of the grid.
    """
    nx = eps_xy.shape[0]
    dx = float(cell_size)
    out: List[Section] = []
    for x0 in range(0, nx, group):
        block = eps_xy[x0:x0 + group]
        out.append(Section(eps=block.mean(axis=0), length=block.shape[0] * dx))
    return out
