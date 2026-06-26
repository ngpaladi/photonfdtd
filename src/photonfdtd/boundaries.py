"""Convolutional Perfectly Matched Layer (CPML) boundary conditions.

Following Roden & Gedney (2000) with the kappa=1 simplification. Each PML
layer is parameterised by:

    sigma(u) = sigma_max * (u/d)**m       polynomial conductivity grade
    alpha(u) = alpha_max * (1 - u/d)      complex-frequency-shift term

where `u` is the distance into the PML measured in cells, `d` is the PML
thickness in cells, and `m` is the polynomial order (typically 3-4).

This module returns the per-axis update coefficients `b` and `c` that the
FDTD time loop uses to propagate the convolutional auxiliary variables.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .constants import EPS_0, ETA_0


@dataclass(frozen=True)
class CPMLParams:
    poly_order: int = 3
    sigma_factor: float = 0.8   # sigma_max relative to the optimal value
    alpha_max: float = 0.24
    R_target: float = 1e-7      # design-target reflection at normal incidence


def cpml_coeffs_1axis(
    n_cells: int,
    cell_size: float,
    n_pml: int,
    dt: float,
    params: CPMLParams = CPMLParams(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CPML update coefficients along one axis.

    Returns (b_e, c_e, b_h, c_h), each of shape (n_cells,). Entries outside
    the PML slabs are zero (b) and zero (c) so the update reduces to plain
    central-difference FDTD there.

    b_e / c_e apply on the *E-field* (integer-index) grid; b_h / c_h apply on
    the half-staggered H-field grid offset by +1/2 cell.
    """
    m = params.poly_order
    if n_pml == 0:
        z = np.zeros(n_cells)
        return z.copy(), z.copy(), z.copy(), z.copy()

    sigma_opt = -(m + 1) * np.log(params.R_target) / (2.0 * ETA_0 * n_pml * cell_size)
    sigma_max = params.sigma_factor * sigma_opt
    alpha_max = params.alpha_max

    def _profile(u_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # u_norm in [0,1]; 0 at the PML inner edge, 1 at the outer.
        sigma = sigma_max * u_norm ** m
        alpha = alpha_max * (1.0 - u_norm)
        return sigma, alpha

    def _bc(sigma: np.ndarray, alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        denom = (sigma + alpha)
        b = np.exp(-(sigma + alpha) * dt / EPS_0)
        c = np.zeros_like(sigma)
        # avoid division-by-zero where sigma==0 and alpha==0
        mask = denom > 0
        c[mask] = sigma[mask] / denom[mask] * (b[mask] - 1.0)
        return b, c

    # Cell-centre indices for E (integer) and H (integer + 1/2).
    i_e = np.arange(n_cells)
    i_h = np.arange(n_cells) + 0.5

    # Lower PML covers cells [0, n_pml-1]; upper covers [n_cells-n_pml, n_cells-1].
    b_e = np.zeros(n_cells)
    c_e = np.zeros(n_cells)
    b_h = np.zeros(n_cells)
    c_h = np.zeros(n_cells)

    # Lower
    u_e_lo = (n_pml - 1 - i_e[:n_pml]) / max(n_pml - 1, 1)
    u_h_lo = (n_pml - 1 - i_h[:n_pml]) / max(n_pml - 1, 1)
    u_e_lo = np.clip(u_e_lo, 0.0, 1.0)
    u_h_lo = np.clip(u_h_lo, 0.0, 1.0)
    s, a = _profile(u_e_lo); b_e[:n_pml], c_e[:n_pml] = _bc(s, a)
    s, a = _profile(u_h_lo); b_h[:n_pml], c_h[:n_pml] = _bc(s, a)

    # Upper
    u_e_hi = (i_e[-n_pml:] - (n_cells - n_pml)) / max(n_pml - 1, 1)
    u_h_hi = (i_h[-n_pml:] - (n_cells - n_pml)) / max(n_pml - 1, 1)
    u_e_hi = np.clip(u_e_hi, 0.0, 1.0)
    u_h_hi = np.clip(u_h_hi, 0.0, 1.0)
    s, a = _profile(u_e_hi); b_e[-n_pml:], c_e[-n_pml:] = _bc(s, a)
    s, a = _profile(u_h_hi); b_h[-n_pml:], c_h[-n_pml:] = _bc(s, a)

    return b_e, c_e, b_h, c_h
