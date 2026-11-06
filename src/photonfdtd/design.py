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
    optimization. ``beta = 0`` is the identity. ``beta`` and ``eta`` may be
    traced JAX scalars, so a continuation schedule does not recompile.
    """
    import jax.numpy as jnp
    if isinstance(beta, (int, float)) and beta <= 0:
        return rho
    # Guard the beta -> 0 limit (0/0) so a traced beta of 0 is the identity.
    b = jnp.maximum(beta, 1e-6)
    num = jnp.tanh(b * eta) + jnp.tanh(b * (rho - eta))
    den = jnp.tanh(b * eta) + jnp.tanh(b * (1.0 - eta))
    return jnp.where(beta > 0, num / den, rho)


def _sidewall_eps(rp, eps_solid, eps_void, dx, dy):
    """In-plane anisotropic subpixel permittivity ``(exx, eyy, ezz)`` of a
    vertical-sidewall layer with fill fraction ``rp`` (2-D)."""
    import jax.numpy as jnp
    de = eps_solid - eps_void
    eps_a = eps_void + rp * de                          # arithmetic mean
    eps_h = 1.0 / (rp / eps_solid + (1.0 - rp) / eps_void)  # harmonic
    # In-plane surface normal from the density gradient (vertical sidewalls).
    gx = jnp.gradient(rp, dx, axis=0) if rp.shape[0] > 1 else jnp.zeros_like(rp)
    gy = jnp.gradient(rp, dy, axis=1) if rp.shape[1] > 1 else jnp.zeros_like(rp)
    # sqrt has an infinite derivative at 0, and 0 * inf = NaN in the backward
    # pass; uniform patches (gradient exactly 0) are common, so take the
    # square root only where it is positive. Same values either way.
    gsq = gx ** 2 + gy ** 2
    pos = gsq > 0
    gmag = jnp.where(pos, jnp.sqrt(jnp.where(pos, gsq, 1.0)), 0.0)
    safe = gmag > (jnp.max(gmag) * 1e-9 + 1e-30)
    inv = jnp.where(safe, 1.0 / jnp.where(safe, gmag, 1.0), 0.0)
    nxc, nyc = gx * inv, gy * inv
    delta = eps_h - eps_a
    exx = eps_a + delta * nxc ** 2                      # Ex: in-plane
    eyy = eps_a + delta * nyc ** 2                      # Ey: in-plane
    ezz = eps_a                                         # Ez: vertical -> tangential
    return exx, eyy, ezz


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

        exx, eyy, ezz = _sidewall_eps(rp, self.eps_solid, self.eps_void, dx, dy)

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


# --------------------------------------------------------------------------- #
# Design regions: optimize a box of the chip, keep the rest of the layout.
# --------------------------------------------------------------------------- #

def _axis_range(coords, lo, hi):
    idx = np.flatnonzero((coords >= lo) & (coords <= hi))
    if idx.size == 0:
        raise ValueError(f"design region [{lo:.4g}, {hi:.4g}] m contains no "
                         "grid cells")
    return int(idx[0]), int(idx[-1]) + 1


def _base_eps(sim):
    """The simulation's own per-component permittivity, as it would run."""
    if sim.subpixel:
        return tuple(np.asarray(e, float) for e in sim._smoothed_eps_components())
    e = np.asarray(sim.eps_r, float)
    return (e, e, e)


@dataclass
class DesignRegion:
    """A box of the core layer whose etch pattern is the design variable.

    Everything outside the box - input/output waveguides, substrate, cladding,
    the PML - keeps the permittivity the :class:`~photonfdtd.Simulation` was
    built with; inside it, the core layer ``core_z`` is replaced by the
    density ``rho`` (shape :attr:`shape`, one value per in-plane cell)
    mapped through the conic filter, the ``tanh`` projection and the
    anisotropic sidewall rule of :class:`EtchedCore`.

    Before filtering, ``rho`` is padded with the *fixed* core occupancy around
    the box (read from the simulation's permittivity), so a waveguide stub
    that ends at the box edge joins the design smoothly instead of being
    filtered toward void.

    Parameters
    ----------
    center, size : (x, y)
        In-plane box (m).
    core_z : (z0, z1)
        Vertical extent of the patterned layer (m). On a 2-D grid (one z
        cell) it must straddle 0.
    eps_solid, eps_void : float
        Permittivity of the core material and of the etched-out fill.
    filter_radius : float
        Conic-filter radius (m), i.e. the minimum feature size; 0 disables it.
    beta, eta : float
        Default projection sharpness / threshold; both can be overridden per
        call (``ce_of(rho, beta=..., eta=...)``) without recompiling.
    """
    center: Tuple[float, float]
    size: Tuple[float, float]
    core_z: Tuple[float, float]
    eps_solid: float
    eps_void: float
    filter_radius: float = 0.0
    beta: float = 0.0
    eta: float = 0.5

    # ------------------------------------------------------------------ #
    def cells(self, grid):
        """``(i0, i1, j0, j1, k_layers)``: the region's index ranges."""
        xc, yc, zc = (np.asarray(c) for c in grid.coords)
        (cx, cy), (sx, sy) = self.center, self.size
        i0, i1 = _axis_range(xc, cx - sx / 2, cx + sx / 2)
        j0, j1 = ((0, 1) if yc.size == 1
                  else _axis_range(yc, cy - sy / 2, cy + sy / 2))
        z0, z1 = self.core_z
        if zc.size == 1:
            if not z0 <= 0.0 <= z1:
                raise ValueError("on a 2-D grid core_z must straddle z = 0")
            ks = np.array([0])
        else:
            ks = np.flatnonzero((zc >= z0) & (zc < z1))
            if ks.size == 0:
                raise ValueError("core_z contains no grid cells")
        return i0, i1, j0, j1, ks

    def shape(self, grid):
        """Shape ``(nx_d, ny_d)`` of the design variable on ``grid``."""
        i0, i1, j0, j1, _ = self.cells(grid)
        return (i1 - i0, j1 - j0)

    def coords(self, grid):
        """Cell-centre coordinates ``(x, y)`` of the design cells (m)."""
        i0, i1, j0, j1, _ = self.cells(grid)
        return (np.asarray(grid.coords[0])[i0:i1],
                np.asarray(grid.coords[1])[j0:j1])

    def _geometry(self, sim):
        grid = sim.grid
        i0, i1, j0, j1, ks = self.cells(grid)
        nx, ny, _ = grid.shape
        dx = grid.cell_size[0]
        dy = grid.cell_size[1] if ny > 1 else dx
        for s in sim.structures:
            if getattr(getattr(s, "medium", None), "is_dispersive", False):
                m = s.region_mask(grid)
                if m[i0:i1, j0:j1][:, :, ks].any():
                    raise NotImplementedError(
                        "a dispersive structure overlaps the design region")
        # Filter padding (cells): clipped to the grid on each side.
        r = int(np.ceil(self.filter_radius / dx)) if self.filter_radius > 0 else 0
        pad = (min(r, i0), min(r, nx - i1),
               min(r, j0) if ny > 1 else 0, min(r, ny - j1) if ny > 1 else 0)
        # Fixed core occupancy on the padded window (mid core layer).
        km = int(ks[len(ks) // 2])
        win = np.asarray(sim.eps_r, float)[i0 - pad[0]:i1 + pad[1],
                                           j0 - pad[2]:j1 + pad[3], km]
        occ = np.clip((win - self.eps_void) / (self.eps_solid - self.eps_void),
                      0.0, 1.0)
        base = _base_eps(sim)
        return dict(i0=i0, i1=i1, j0=j0, j1=j1, ks=ks, pad=pad, occ=occ,
                    dx=dx, dy=dy, base=base, dt=sim.dt)

    def _phys_window(self, geo, rho, beta, eta):
        """Projected density on the padded window (fixed ring + design)."""
        import jax.numpy as jnp
        p = geo["pad"]
        nxd, nyd = geo["i1"] - geo["i0"], geo["j1"] - geo["j0"]
        win = jnp.asarray(geo["occ"])
        win = win.at[p[0]:p[0] + nxd, p[2]:p[2] + nyd].set(rho)
        if self.filter_radius > 0:
            win = conic_filter(win, self.filter_radius / geo["dx"])
        return tanh_projection(win, beta, eta)

    def _crop(self, geo, a):
        p = geo["pad"]
        nxd, nyd = geo["i1"] - geo["i0"], geo["j1"] - geo["j0"]
        return a[p[0]:p[0] + nxd, p[2]:p[2] + nyd]

    def rho_phys(self, sim, rho, beta=None, eta=None):
        """Filtered + projected physical density on the design cells."""
        geo = self._geometry(sim)
        beta = self.beta if beta is None else beta
        eta = self.eta if eta is None else eta
        return self._crop(geo, self._phys_window(geo, rho, beta, eta))

    def _ce_from_window(self, geo, rp_win):
        import jax.numpy as jnp
        exx, eyy, ezz = (self._crop(geo, e) for e in _sidewall_eps(
            rp_win, self.eps_solid, self.eps_void, geo["dx"], geo["dy"]))
        i0, i1, j0, j1, ks = (geo[k] for k in ("i0", "i1", "j0", "j1", "ks"))
        out = {}
        for c, base, e in zip(("Ex", "Ey", "Ez"), geo["base"], (exx, eyy, ezz)):
            ce = jnp.asarray(geo["dt"] / (base * EPS_0))
            val = geo["dt"] / (e * EPS_0)
            for k in ks:
                ce = ce.at[i0:i1, j0:j1, int(k)].set(val)
            out[c] = ce
        return out

    def ce_builder(self, sim):
        """``ce_of(rho, beta=None, eta=None) -> {'Ex','Ey','Ez'}``.

        The E-update coefficients of ``sim`` with the design region set from
        ``rho``; pass it to :func:`~photonfdtd.make_value_and_grad_params`.
        """
        geo = self._geometry(sim)

        def ce_of(rho, beta=None, eta=None):
            beta = self.beta if beta is None else beta
            eta = self.eta if eta is None else eta
            return self._ce_from_window(geo, self._phys_window(geo, rho, beta, eta))
        return ce_of


# --------------------------------------------------------------------------- #
# Parametric shapes: differentiable polygons rasterized to a fill fraction.
# --------------------------------------------------------------------------- #

def polygon_density(vertices, xs, ys, h=None):
    """Per-cell fill fraction of a polygon, differentiable in its vertices.

    ``vertices`` is an ``(n, 2)`` array (m, either orientation, simple
    polygon); ``xs``, ``ys`` the cell-centre coordinates. The fill is a
    one-cell linear ramp in the signed distance ``d`` to the boundary,
    ``clip(0.5 - d / h, 0, 1)``: exact for an axis-aligned edge crossing a
    cell, and first-order accurate otherwise. ``h`` defaults to the cell size.
    """
    import jax.numpy as jnp
    v = jnp.asarray(vertices)
    xs = jnp.asarray(xs)
    ys = jnp.asarray(ys)
    if h is None:
        h = float(np.asarray(xs[1] - xs[0])) if xs.size > 1 else \
            float(np.asarray(ys[1] - ys[0]))
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")
    P = jnp.stack([X, Y], axis=-1)[..., None, :]           # (nx, ny, 1, 2)
    a = v
    b = jnp.roll(v, -1, axis=0)
    ab = b - a                                             # (n, 2)
    ap = P - a                                             # (nx, ny, n, 2)
    t = jnp.clip(jnp.sum(ap * ab, -1) / jnp.sum(ab * ab, -1), 0.0, 1.0)
    diff = ap - t[..., None] * ab
    dist = jnp.sqrt(jnp.min(jnp.sum(diff ** 2, -1), axis=-1) + 1e-30 * h * h)
    # Even-odd crossing test for inside/outside (piecewise constant).
    ay, by = a[:, 1], b[:, 1]
    cond = (ay > Y[..., None]) != (by > Y[..., None])
    xint = a[:, 0] + (Y[..., None] - ay) * ab[:, 0] / jnp.where(
        ab[:, 1] == 0, 1.0, ab[:, 1])
    inside = jnp.sum(cond & (X[..., None] < xint), -1) % 2 == 1
    sd = jnp.where(inside, -dist, dist)
    return jnp.clip(0.5 - sd / h, 0.0, 1.0)


def rect(cx, cy, width_x, width_y):
    """Axis-aligned rectangle vertices (traceable)."""
    import jax.numpy as jnp
    hx, hy = width_x / 2, width_y / 2
    return jnp.stack([jnp.stack([cx - hx, cy - hy]), jnp.stack([cx + hx, cy - hy]),
                      jnp.stack([cx + hx, cy + hy]), jnp.stack([cx - hx, cy + hy])])


def taper(x0, x1, w0, w1, yc=0.0):
    """Linear taper along x from width ``w0`` at ``x0`` to ``w1`` at ``x1``."""
    import jax.numpy as jnp
    return jnp.stack([jnp.stack([x0, yc - w0 / 2]), jnp.stack([x1, yc - w1 / 2]),
                      jnp.stack([x1, yc + w1 / 2]), jnp.stack([x0, yc + w0 / 2])])


@dataclass
class ShapeDesign:
    """A parametric design: ``shape_fn(params)`` returns polygons in a region.

    ``shape_fn`` maps the parameter vector to a list of ``(n, 2)`` vertex
    arrays (built with JAX ops, e.g. :func:`rect` / :func:`taper`); their union
    is rasterized with :func:`polygon_density` on the region's cells and given
    the same anisotropic sidewall permittivity as a density design (the
    region's filter and projection are not applied). Gradients flow to the
    parameters through the boundary cells.
    """
    region: DesignRegion
    shape_fn: object

    def density(self, sim, params):
        import jax.numpy as jnp
        xs, ys = self.region.coords(sim.grid)
        h = sim.grid.cell_size[0]
        polys = self.shape_fn(params)
        return jnp.max(jnp.stack([polygon_density(p, xs, ys, h) for p in polys]),
                       axis=0)

    def polygons(self, params):
        """Final polygons as NumPy arrays (m), e.g. for GDS export."""
        return [np.asarray(p) for p in self.shape_fn(params)]

    def ce_builder(self, sim):
        """``ce_of(params) -> {'Ex','Ey','Ez'}`` for the parametric design."""
        import jax.numpy as jnp
        reg = self.region
        geo = reg._geometry(sim)
        xs, ys = reg.coords(sim.grid)
        h = sim.grid.cell_size[0]

        def ce_of(params):
            polys = self.shape_fn(params)
            rp = jnp.max(jnp.stack([polygon_density(p, xs, ys, h)
                                    for p in polys]), axis=0)
            # Fixed occupancy outside the box, shapes inside, no filtering.
            p = geo["pad"]
            win = jnp.asarray(geo["occ"])
            win = win.at[p[0]:p[0] + rp.shape[0], p[2]:p[2] + rp.shape[1]].set(rp)
            return reg._ce_from_window(geo, win)
        return ce_of
