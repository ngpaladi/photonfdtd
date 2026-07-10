"""Differentiable etched-core geometry for photonic-integrated-circuit design.

A PIC device is a thin core layer (Si / SiN / LiNbO3) etched into an in-plane
pattern, sitting on a substrate (buried oxide) under a cladding. The *design
variable* is therefore a 2-D density field ``rho(x, y)`` in [0, 1] extruded
through the fixed core thickness - not a full 3-D grid. This module maps that
2-D density to the 3-D permittivity tensor the JAX stepper consumes, entirely
in JAX, so a mode-overlap / S-parameter loss is differentiable all the way back
to ``rho`` (topology optimization), with the adjoint memory bounded by the
gradient checkpointing in :mod:`photonfdtd.jaxbackend`.

Pipeline (all differentiable):

    rho  --conic filter (min feature size)-->  --tanh projection (toward binary)-->
    rho_phys  --> in-plane subpixel-smoothed permittivity tensor on the core layer

The etched sidewalls are vertical, so their material-boundary normal is
*in-plane* and the per-cell fill fraction is ``rho_phys`` itself: we feed those
directly to the anisotropic subpixel rule (harmonic mean along the normal,
arithmetic mean tangential; Farjadpour et al. 2006). The out-of-plane component
Ez is tangential to every sidewall, so it sees the arithmetic mean - the exact
2-D-TE subpixel rule.

References
----------
* O. Sigmund, "Morphology-based black and white filters," *Struct. Multidiscip.
  Optim.* 33, 401 (2007) - density filter.
* F. Wang, B. Lazarov, O. Sigmund, "On projection methods ... robust
  formulations," *Struct. Multidiscip. Optim.* 43, 767 (2011) - tanh projection.
* A. Farjadpour et al., *Opt. Lett.* 31, 2972 (2006) - subpixel smoothing.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .constants import EPS_0


def conic_filter(rho, radius_cells: float):
    """Conic (linear-decay) density filter of radius ``radius_cells`` cells.

    Enforces a minimum feature size and makes the geometry differentiable in the
    design. Implemented as a normalized convolution (edge-corrected so the
    boundary is not biased toward void).
    """
    import jax.numpy as jnp
    from jax.scipy.signal import convolve
    r = int(np.ceil(radius_cells))
    if r < 1:
        return rho
    ax = np.arange(-r, r + 1)
    xx, yy = np.meshgrid(ax, ax, indexing="ij")
    kern = np.maximum(0.0, 1.0 - np.sqrt(xx ** 2 + yy ** 2) / radius_cells)
    kern = jnp.asarray(kern)
    num = convolve(rho, kern, mode="same")
    den = convolve(jnp.ones_like(rho), kern, mode="same")
    return num / den


def tanh_projection(rho, beta: float = 8.0, eta: float = 0.5):
    """Smoothly threshold toward 0/1 about ``eta`` with sharpness ``beta``.

    ``beta -> inf`` gives a hard (binary) design; anneal it up during
    optimization. ``beta = 0`` is the identity.
    """
    import jax.numpy as jnp
    if beta <= 0:
        return rho
    num = jnp.tanh(beta * eta) + jnp.tanh(beta * (rho - eta))
    den = jnp.tanh(beta * eta) + jnp.tanh(beta * (1.0 - eta))
    return num / den


@dataclass
class EtchedCore:
    """A substrate / etched-core / cladding stack driven by a 2-D density.

    Parameters
    ----------
    eps_solid, eps_void : float
        Core permittivity where etched material is present / removed.
    eps_sub, eps_clad : float
        Substrate (below the core) and cladding (above) permittivity.
    core_z : (z0, z1)
        Core-layer vertical bounds (m); cells with ``z0 <= z < z1`` are the
        patterned core, below is substrate, above is cladding.
    filter_radius : float
        Conic-filter radius (m); 0 disables filtering.
    beta, eta : float
        tanh-projection sharpness / threshold.
    """
    eps_solid: float
    eps_void: float
    eps_sub: float
    eps_clad: float
    core_z: Tuple[float, float]
    filter_radius: float = 0.0
    beta: float = 0.0
    eta: float = 0.5

    def rho_phys(self, rho, dx: float):
        """Filtered + projected physical density from the raw design ``rho``."""
        if self.filter_radius > 0:
            rho = conic_filter(rho, self.filter_radius / dx)
        return tanh_projection(rho, self.beta, self.eta)

    def eps_components(self, grid, rho):
        """Anisotropic diagonal permittivity ``(eps_xx, eps_yy, eps_zz)`` (3-D
        JAX arrays of shape ``grid.shape``) for the raw design ``rho`` (nx, ny).

        Differentiable in ``rho``. Only the core layer depends on ``rho``; the
        substrate and cladding are constant.
        """
        import jax.numpy as jnp
        nx, ny, nz = grid.shape
        dx = grid.cell_size[0]
        dy = grid.cell_size[1] if grid.shape[1] > 1 else dx
        rp = self.rho_phys(rho, dx)                         # (nx, ny) in [0,1]

        de = self.eps_solid - self.eps_void
        eps_a = self.eps_void + rp * de                     # arithmetic mean
        eps_h = 1.0 / (rp / self.eps_solid + (1.0 - rp) / self.eps_void)  # harmonic

        # In-plane surface normal from the density gradient (vertical sidewalls).
        gx = jnp.gradient(rp, dx, axis=0)
        gy = jnp.gradient(rp, dy, axis=1)
        gmag = jnp.sqrt(gx ** 2 + gy ** 2)
        safe = gmag > (jnp.max(gmag) * 1e-9 + 1e-30)
        inv = jnp.where(safe, 1.0 / jnp.where(safe, gmag, 1.0), 0.0)
        nxc, nyc = gx * inv, gy * inv
        delta = eps_h - eps_a
        exx = eps_a + delta * nxc ** 2                      # Ex: in-plane
        eyy = eps_a + delta * nyc ** 2                      # Ey: in-plane
        ezz = eps_a                                         # Ez: vertical -> tangential

        # Extrude through the layer stack along z.
        zc = np.asarray(grid.coords[2])
        z0, z1 = self.core_z
        core = ((zc >= z0) & (zc < z1)).astype(float)       # (nz,)
        bg = np.where(zc < z0, self.eps_sub, self.eps_clad)  # (nz,) substrate/clad
        core_j = jnp.asarray(core).reshape(1, 1, nz)
        bg_j = jnp.asarray(bg).reshape(1, 1, nz)

        def extrude(eps_xy):
            return eps_xy[:, :, None] * core_j + bg_j * (1.0 - core_j)

        return extrude(exx), extrude(eyy), extrude(ezz)

    def ce_builder(self, sim):
        """Return ``rho -> {'Ex','Ey','Ez'}`` E-update coefficients for
        :func:`photonfdtd.jaxbackend.value_and_grad_params`."""
        dt = sim.dt
        grid = sim.grid

        def ce_of_rho(rho):
            exx, eyy, ezz = self.eps_components(grid, rho)
            return {"Ex": dt / (exx * EPS_0),
                    "Ey": dt / (eyy * EPS_0),
                    "Ez": dt / (ezz * EPS_0)}
        return ce_of_rho


def value_and_grad_density(sim, etched_core: EtchedCore, rho, loss,
                           remat: str = "nested", checkpoint_levels: int = 2):
    """``(value, d loss / d rho)`` for a 2-D etched-core design ``rho``.

    Convenience wrapper over
    :func:`photonfdtd.jaxbackend.value_and_grad_params`: maps the density
    through the differentiable etched-core pipeline and differentiates a
    monitor loss back to the 2-D design.
    """
    from .jaxbackend import value_and_grad_params
    return value_and_grad_params(sim, etched_core.ce_builder(sim), rho, loss,
                                 remat=remat, checkpoint_levels=checkpoint_levels)
