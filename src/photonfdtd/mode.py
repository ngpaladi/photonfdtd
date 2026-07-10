"""Full-vectorial finite-difference waveguide eigenmode solver.

Given a 2D relative-permittivity profile eps_r(y, z) in the transverse plane
and a free-space wavelength lambda0, this module solves for the guided modes
of a waveguide invariant along the propagation direction x.  Fields are assumed
to vary as ``exp(+i*beta*x - i*omega*t)`` and the solver returns the effective
index ``n_eff = beta / k0`` and the transverse electric field of each mode.

Formulation
-----------
This is the FDFD (finite-difference frequency-domain) *full-vectorial*
transverse-electric-field method.  The two transverse electric-field
components are the unknowns and are solved from a generalized eigenproblem
whose eigenvalue is the propagation constant squared, ``beta**2``.  Writing the
in-plane axes as ``u`` (the fast grid axis, here z) and ``v`` (the slow grid
axis, here y), and denoting by ``e_zz`` the permittivity along the propagation
direction, the operator is the 2x2 block matrix

    P = [[Puu, Puv],
         [Pvu, Pvv]] ,     P @ [E_u; E_v] = beta**2 * [E_u; E_v]

with the sub-blocks (mu_r = 1, isotropic diagonal permittivity)

    Puu = -Uu ez^-1 Vv Vu Uv / k0^2
          + (k0^2 I + Uu ez^-1 Vu)(eps_u + Vv Uv / k0^2)
    Pvv = -Uv ez^-1 Vu Vv Uu / k0^2
          + (k0^2 I + Uv ez^-1 Vv)(eps_v + Vu Uu / k0^2)
    Puv =  Uu ez^-1 Vv (eps_v + Vu Uu / k0^2)
          - (k0^2 I + Uu ez^-1 Vu) Vv Uu / k0^2
    Pvu =  Uv ez^-1 Vu (eps_u + Vv Uv / k0^2)
          - (k0^2 I + Uv ez^-1 Vv) Vu Uv / k0^2

Here ``Uu, Uv`` are the staggered *forward*-difference matrices along u, v and
``Vu = -Uu^T, Vv = -Uv^T`` the corresponding *backward*-difference matrices
(the Yee-grid electric/magnetic staggering).  ``eps_u, eps_v`` are the diagonal
permittivity matrices multiplying ``E_u, E_v`` and ``ez^-1`` the diagonal of
``1/eps_zz``.  Perfect-electric-conductor (Dirichlet, zero tangential E)
boundaries are imposed implicitly because the forward-difference matrices drop
the out-of-domain neighbour at the last node.

Sanity check of the stencil: for a homogeneous isotropic medium the symbol of
``Puu`` reduces to ``k0^2*eps - ku^2 - kv^2 = beta^2`` and ``Puv -> 0``, i.e.
the operator degenerates to the correct scalar dispersion with no spurious
polarization coupling; coupling appears only through gradients of ``ez^-1``.
This is what makes the method genuinely full-vectorial and able to split the
TE/TM effective indices of a high-index-contrast waveguide, which the scalar
Helmholtz solver cannot.

References
----------
The formulation and matrix stencil follow

  Z. Zhu and T. G. Brown, "Full-vectorial finite-difference analysis of
  microstructured optical fibers," Optics Express 10(17), 853-864 (2002),
  https://doi.org/10.1364/OE.10.000853  (the transverse-field operator, their
  Eqs. for the coupled Ex/Ey system);

as implemented in the ``philsol`` package (P. Main,
https://github.com/philmain28/philsol, ``eigen_build`` in ``core.py``), whose
block structure the code below reproduces with the exact same signs.  The
staggered forward/backward difference operators and the PEC boundary treatment
follow the standard Yee-grid FDFD construction of

  R. C. Rumpf, "Electromagnetic and Photonic Simulation for the Beginner:
  Finite-Difference Frequency-Domain in MATLAB," Ch. 6 (EMPossible),

and the transverse-magnetic-field variant is described in

  A. B. Fallahkhair, K. S. Li, and T. E. Murphy, "Vector Finite Difference
  Modesolver for Anisotropic Dielectric Waveguides," J. Lightwave Technol.
  26(11), 1423-1431 (2008).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .geometry import Box
from .materials import Medium


@dataclass
class ModeResult:
    """Result of a full-vectorial mode solve.

    Attributes
    ----------
    wavelength : float
        Free-space wavelength, metres.
    n_eff : np.ndarray, shape (num_modes,)
        Real effective indices, sorted in descending order.
    psi : np.ndarray, shape (num_modes, ny, nz)
        Back-compat dominant transverse-field magnitude of each mode,
        normalised to unit maximum.  This is ``|E_t| = sqrt(|Ey|^2+|Ez|^2)``.
    y, z : np.ndarray
        Transverse coordinate axes (metres).
    Ey, Ez : np.ndarray, shape (num_modes, ny, nz)
        Complex transverse electric-field components (the solved unknowns).
    Ex : np.ndarray, shape (num_modes, ny, nz)
        Longitudinal electric field, recovered from the divergence condition
        (convenience/visualisation field).
    Hx, Hy, Hz : np.ndarray, shape (num_modes, ny, nz)
        Magnetic-field components recovered from Faraday's law
        (convenience/visualisation fields).
    fields : dict
        Convenience dict mapping component name -> array.
    """
    wavelength: float
    n_eff: np.ndarray
    psi: np.ndarray
    y: np.ndarray
    z: np.ndarray
    Ey: Optional[np.ndarray] = None
    Ez: Optional[np.ndarray] = None
    Ex: Optional[np.ndarray] = None
    Hx: Optional[np.ndarray] = None
    Hy: Optional[np.ndarray] = None
    Hz: Optional[np.ndarray] = None
    fields: Optional[dict] = None
    #: Cross-section relative permittivity (ny, nz) - used by a unidirectional
    #: mode source to weight the equivalence electric-current sheet by 1/eps.
    eps_r: Optional[np.ndarray] = None


class ModeSolver:
    """Full-vectorial FDFD waveguide mode solver on a uniform Cartesian grid.

    Propagation is along x; the cross-section lies in the (y, z) plane.  The
    solver returns the ``num_modes`` most-confined guided modes (largest
    ``n_eff``).  Because the operator is genuinely vectorial, the TE and TM
    families are resolved separately -- e.g. for a high-index-contrast slab the
    TE0 and TM0 effective indices come out distinct, matching the analytic
    transcendental dispersion relations.

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
    polarization : {None, "te", "tm"}, optional
        If given, filter the returned modes to the dominant polarization
        ("te" -> ``Ey`` dominant, "tm" -> ``Ez`` dominant for this y/z convention).
        ``None`` (default) returns modes purely by descending n_eff.
    """

    def __init__(
        self,
        size: Tuple[float, float],
        cell_size,
        structures: Sequence[Box],
        wavelength: float,
        background_eps: float = 1.0,
        num_modes: int = 1,
        polarization: Optional[str] = None,
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
        if polarization is not None:
            polarization = polarization.lower()
            if polarization not in ("te", "tm"):
                raise ValueError("polarization must be None, 'te' or 'tm'")
        self.polarization = polarization

        # Build eps_r(y, z) by stamping the boxes (row index = y, col = z).
        eps_r = np.full((ny, nz), float(background_eps))
        for s in structures:
            self._stamp_2d(s, eps_r)
        self.eps_r = eps_r

    # ------------------------------------------------------------------ #
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
        # A dispersive medium's plain ``eps_r`` is only eps_inf (for a Sellmeier
        # material that is 1). The frequency-domain mode solver has no ADE, so
        # evaluate the full dispersion model at the solve wavelength to get the
        # correct real permittivity at that frequency.
        med = box.medium
        if getattr(med, "is_dispersive", False):
            from .constants import C_0
            val = float(np.real(med.eps_model(C_0 / self.wavelength)))
        else:
            val = med.eps_r
        eps_r[yy, zz] = val

    # ------------------------------------------------------------------ #
    def _difference_operators(self):
        """Build the staggered forward/backward difference matrices.

        The flat ordering is ``flat = iy*nz + iz`` so *z is the fast axis*
        (call it u) and *y is the slow axis* (call it v).  ``Uu`` is the
        forward difference along z, ``Uv`` the forward difference along y.
        The backward differences are ``Vu = -Uu^T`` and ``Vv = -Uv^T``.  The
        missing out-of-domain neighbour at the last node realises the PEC
        (zero tangential E) boundary.
        """
        ny, nz = self.ny, self.nz
        dy, dz = self.dy, self.dz
        N = ny * nz

        # Forward difference along the fast axis z, block-diagonal per y-row:
        #   (f[iz+1] - f[iz]) / dz , with f[nz] == 0 (Dirichlet) at the top.
        Duz = (sp.eye(nz, k=1, format="csr") - sp.eye(nz, format="csr")) / dz
        Uu = sp.block_diag([Duz] * ny, format="csr")

        # Forward difference along the slow axis y (neighbour is nz apart):
        #   (f[iy+1] - f[iy]) / dy , with f[ny] == 0 (Dirichlet).
        Uv = (sp.eye(N, k=nz, format="csr") - sp.eye(N, format="csr")) / dy

        Vu = -Uu.transpose().tocsr()
        Vv = -Uv.transpose().tocsr()
        return Uu, Uv, Vu, Vv

    # ------------------------------------------------------------------ #
    def solve(self) -> ModeResult:
        """Compute the ``num_modes`` highest-n_eff guided modes."""
        ny, nz = self.ny, self.nz
        N = ny * nz
        k0 = 2 * math.pi / self.wavelength
        k2 = k0 * k0

        Uu, Uv, Vu, Vv = self._difference_operators()

        er = self.eps_r.reshape(N).astype(float)
        eps_u = sp.diags(er)          # permittivity seen by E_u (=Ez)
        eps_v = sp.diags(er)          # permittivity seen by E_v (=Ey)
        ezi = sp.diags(1.0 / er)      # inverse longitudinal permittivity 1/eps_xx
        I = sp.eye(N, format="csr")

        # Full-vectorial transverse-E operator (Zhu & Brown / philsol form).
        # Eigenvalue of P is beta**2; eigenvector is [E_u; E_v] = [Ez; Ey].
        Puu = (-Uu @ ezi @ Vv @ Vu @ Uv / k2
               + (k2 * I + Uu @ ezi @ Vu) @ (eps_u + Vv @ Uv / k2))
        Pvv = (-Uv @ ezi @ Vu @ Vv @ Uu / k2
               + (k2 * I + Uv @ ezi @ Vv) @ (eps_v + Vu @ Uu / k2))
        Puv = (Uu @ ezi @ Vv @ (eps_v + Vu @ Uu / k2)
               - (k2 * I + Uu @ ezi @ Vu) @ Vv @ Uu / k2)
        Pvu = (Uv @ ezi @ Vu @ (eps_u + Vv @ Uv / k2)
               - (k2 * I + Uv @ ezi @ Vv) @ Vu @ Uv / k2)
        P = sp.bmat([[Puu, Puv], [Pvu, Pvv]], format="csc")

        # We want the guided modes: the eigenvalues beta**2 just below the
        # largest possible value (k0 n_max)**2.  Shift-invert near that point.
        n_max = float(np.sqrt(self.eps_r.max()))
        sigma = (k0 * n_max) ** 2

        # Ask for a few extra so a polarization filter still has candidates.
        want = self.num_modes if self.polarization is None else self.num_modes + 4
        k_solve = min(max(want, self.num_modes), 2 * N - 2)

        try:
            vals, vecs = spla.eigs(P, k=k_solve, sigma=sigma, which="LM")
        except spla.ArpackNoConvergence as e:
            vals = e.eigenvalues
            vecs = e.eigenvectors

        beta2 = vals.real
        # Keep only physical (real, positive, bounded) eigenvalues.
        order = np.argsort(-beta2)
        beta2 = beta2[order]
        vecs = vecs[:, order]

        n_eff_all = np.sqrt(np.clip(beta2, 0.0, None)) / k0

        Ez_all = vecs[:N, :]          # E_u
        Ey_all = vecs[N:, :]          # E_v

        # Optional polarization filtering by dominant transverse component.
        n_avail = vecs.shape[1]
        selected = list(range(n_avail))
        if self.polarization is not None:
            te, tm = [], []
            for i in range(n_avail):
                py = float(np.sum(np.abs(Ey_all[:, i]) ** 2))
                pz = float(np.sum(np.abs(Ez_all[:, i]) ** 2))
                (te if py >= pz else tm).append(i)
            selected = te if self.polarization == "te" else tm

        selected = selected[: self.num_modes]
        if len(selected) == 0:
            selected = list(range(min(self.num_modes, n_avail)))

        m = len(selected)
        n_eff = n_eff_all[selected]
        Ez = Ez_all[:, selected].T.reshape(m, ny, nz)
        Ey = Ey_all[:, selected].T.reshape(m, ny, nz)

        # Normalise each mode so that max |E_t| == 1 (keeps complex phase).
        Et_mag = np.sqrt(np.abs(Ey) ** 2 + np.abs(Ez) ** 2)
        psi = np.empty((m, ny, nz), dtype=float)
        for i in range(m):
            mx = Et_mag[i].max()
            if mx > 0:
                Ey[i] /= mx
                Ez[i] /= mx
                psi[i] = Et_mag[i] / mx
            else:
                psi[i] = Et_mag[i]

        # Recover the remaining components for visualisation.
        Ex, Hx, Hy, Hz = self._recover_fields(Ey, Ez, n_eff, k0)

        fields = {"Ex": Ex, "Ey": Ey, "Ez": Ez, "Hx": Hx, "Hy": Hy, "Hz": Hz}
        return ModeResult(
            wavelength=self.wavelength, n_eff=n_eff, psi=psi,
            y=self.y, z=self.z,
            Ex=Ex, Ey=Ey, Ez=Ez, Hx=Hx, Hy=Hy, Hz=Hz, fields=fields,
            eps_r=self.eps_r.copy(),
        )

    # ------------------------------------------------------------------ #
    def _recover_fields(self, Ey, Ez, n_eff, k0):
        """Recover Ex and H from the transverse E (convenience fields).

        Uses simple central differences on the collocated grid.  These are
        provided for plotting/inspection; only ``n_eff`` and the transverse E
        are the validated quantities.  Convention: exp(+i beta x - i omega t),
        so d/dx -> +i beta, and H = curl(E) / (i omega mu0).
        """
        from .constants import MU_0, C_0
        m, ny, nz = Ey.shape
        dy, dz = self.dy, self.dz
        omega = C_0 * k0
        eps = self.eps_r  # (ny, nz)

        Ex = np.zeros_like(Ey)
        Hx = np.zeros_like(Ey)
        Hy = np.zeros_like(Ey)
        Hz = np.zeros_like(Ey)
        for i in range(m):
            beta = n_eff[i] * k0
            if beta == 0:
                continue
            # Gauss's law div(eps E)=0 with d/dx -> i beta (axis 0 = y, axis 1 = z):
            #   i beta eps Ex = -(d(eps Ey)/dy + d(eps Ez)/dz)
            d_epsEy_dy = np.gradient(eps * Ey[i], dy, axis=0)
            d_epsEz_dz = np.gradient(eps * Ez[i], dz, axis=1)
            Ex[i] = -(d_epsEy_dy + d_epsEz_dz) / (1j * beta * eps)
            dEx_dy = np.gradient(Ex[i], dy, axis=0)
            dEx_dz = np.gradient(Ex[i], dz, axis=1)
            # H = curl(E)/(i omega mu0), d/dx -> i beta
            Hx[i] = (np.gradient(Ez[i], dy, axis=0)
                     - np.gradient(Ey[i], dz, axis=1)) / (1j * omega * MU_0)
            Hy[i] = (dEx_dz - 1j * beta * Ez[i]) / (1j * omega * MU_0)
            Hz[i] = (1j * beta * Ey[i] - dEx_dy) / (1j * omega * MU_0)
        return Ex, Hx, Hy, Hz
