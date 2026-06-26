"""2D scalar (Helmholtz) waveguide mode solver.

Given a 2D refractive-index profile n(y, z) and a free-space wavelength
lambda0, the scalar Helmholtz mode equation is

    d2 psi / dy2  +  d2 psi / dz2  +  k0^2 n^2(y, z) psi  =  beta^2 psi

This is discretised with second-order central differences on a uniform grid
and recast as a sparse eigenvalue problem  A psi = beta^2 psi. The largest
real eigenvalues correspond to the most-confined guided modes.

The scalar approximation is exact for very-weakly-guiding waveguides and is
a reasonable first pass for any 2D cross-section. Full-vectorial mode
solving (which captures TE/TM polarisation effects and high-index-contrast
boundary corrections) is on the roadmap.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence, Tuple, Optional
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .constants import C_0
from .geometry import Box
from .materials import Medium


@dataclass
class ModeResult:
    wavelength: float
    n_eff: np.ndarray            # (num_modes,) real refractive index
    psi: np.ndarray              # (num_modes, ny, nz) eigenmode profiles
    y: np.ndarray
    z: np.ndarray


class ModeSolver:
    """2D scalar Helmholtz mode solver on a uniform Cartesian grid.

    Parameters
    ----------
    size : (Ly, Lz)
        Cross-section extent in metres.
    cell_size : float or (dy, dz)
        Discretisation step, metres.
    structures : sequence of Box
        Geometry in the cross-section. Boxes outside the slice are ignored.
    wavelength : float
        Free-space wavelength, metres.
    background_eps : float
        Relative permittivity of the background medium (default vacuum).
    num_modes : int
        Number of modes to return (sorted by descending n_eff).
    """

    def __init__(
        self,
        size: Tuple[float, float],
        cell_size,
        structures: Sequence[Box],
        wavelength: float,
        background_eps: float = 1.0,
        num_modes: int = 1,
    ) -> None:
        if isinstance(cell_size, (int, float)):
            dy = dz = float(cell_size)
        else:
            dy, dz = float(cell_size[0]), float(cell_size[1])
        Ly, Lz = float(size[0]), float(size[1])
        ny = int(round(Ly / dy))
        nz = int(round(Lz / dz))
        if ny < 4 or nz < 4:
            raise ValueError("Need at least 4 cells per axis")

        self.dy, self.dz, self.ny, self.nz = dy, dz, ny, nz
        self.y = (np.arange(ny) - (ny - 1) / 2) * dy
        self.z = (np.arange(nz) - (nz - 1) / 2) * dz
        self.wavelength = float(wavelength)
        self.num_modes = int(num_modes)

        # Build eps_r(y, z) by stamping the boxes.
        eps_r = np.full((ny, nz), float(background_eps))
        for s in structures:
            self._stamp_2d(s, eps_r)
        self.eps_r = eps_r

    def _stamp_2d(self, box: Box, eps_r: np.ndarray) -> None:
        c = list(box.center) + [0.0, 0.0]
        s = list(box.size) + [0.0, 0.0]
        # interpret as (y, z) if 2D centre / size; else (x, y, z) and use (y, z)
        if len(box.center) == 2:
            cy, cz = c[0], c[1]
            sy, sz = s[0], s[1]
        else:
            cy, cz = c[1], c[2]
            sy, sz = s[1], s[2]
        y_lo, y_hi = cy - sy / 2, cy + sy / 2
        z_lo, z_hi = cz - sz / 2, cz + sz / 2
        my = (self.y >= y_lo) & (self.y <= y_hi)
        mz = (self.z >= z_lo) & (self.z <= z_hi)
        if not my.any() or not mz.any():
            return
        idx_y = np.flatnonzero(my)
        idx_z = np.flatnonzero(mz)
        yy, zz = np.ix_(idx_y, idx_z)
        eps_r[yy, zz] = box.medium.eps_r

    # ------------------------------------------------------------------ #
    def solve(self) -> ModeResult:
        """Compute the `num_modes` highest-n_eff guided modes."""
        ny, nz = self.ny, self.nz
        dy, dz = self.dy, self.dz
        k0 = 2 * math.pi / self.wavelength

        N = ny * nz
        # Five-point Laplacian with Dirichlet boundaries.
        main = -2.0 * (1.0 / dy ** 2 + 1.0 / dz ** 2) * np.ones(N)
        off_y_pos = np.ones(N) / dy ** 2
        off_y_neg = np.ones(N) / dy ** 2
        off_z_pos = np.ones(N) / dz ** 2
        off_z_neg = np.ones(N) / dz ** 2

        # Boundaries: zero out shifts that wrap to the wrong row in y direction.
        # Indexing convention: flat = iy*nz + iz, so a shift of +nz moves +1 in y.
        # The +nz shift is invalid at the last y row (iy = ny-1); -nz invalid at iy=0.
        # The +1 shift is invalid at iz = nz-1; -1 invalid at iz = 0.
        for iy in range(ny):
            base = iy * nz
            off_z_pos[base + nz - 1] = 0.0
            off_z_neg[base] = 0.0
        off_y_pos[(ny - 1) * nz:] = 0.0
        off_y_neg[: nz] = 0.0

        # k0^2 * n^2  diagonal contribution
        diag_k = (k0 ** 2) * self.eps_r.reshape(N)

        data = [main + diag_k, off_y_pos[:-nz], off_y_neg[nz:], off_z_pos[:-1], off_z_neg[1:]]
        offsets = [0, nz, -nz, 1, -1]
        A = sp.diags(data, offsets, shape=(N, N), format="csr")

        # Largest real eigenvalues -> shift-invert near (k0 * n_max)^2
        n_max = float(np.sqrt(self.eps_r.max()))
        sigma = (k0 * n_max) ** 2 * 0.999

        try:
            beta2, vecs = spla.eigsh(A, k=self.num_modes, sigma=sigma, which="LM")
        except spla.ArpackNoConvergence as e:
            beta2 = e.eigenvalues
            vecs = e.eigenvectors

        # Order by descending eigenvalue (= descending n_eff)
        order = np.argsort(-beta2.real)
        beta2 = beta2[order]
        vecs = vecs[:, order]

        # n_eff = beta / k0 = sqrt(beta^2) / k0; eigenvalues here are beta^2
        n_eff = np.sqrt(np.clip(beta2.real, 0.0, None)) / k0

        psi = vecs.T.reshape(self.num_modes, ny, nz).real
        # Normalise each mode to unit max-abs amplitude
        for i in range(self.num_modes):
            mx = np.max(np.abs(psi[i]))
            if mx > 0:
                psi[i] /= mx

        return ModeResult(wavelength=self.wavelength,
                          n_eff=n_eff, psi=psi,
                          y=self.y, z=self.z)
