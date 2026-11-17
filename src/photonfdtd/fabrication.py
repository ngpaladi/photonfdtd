"""Fabrication constraints for topology-optimized designs.

Tools to keep a 2-D density design manufacturable, alongside the conic filter
and tanh projection of :mod:`photonfdtd.design`:

* :func:`symmetrize` - impose mirror / diagonal / point symmetry on a density
  (differentiable; halves or quarters the free parameters in effect).
* :func:`lengthscale_constraints` - the geometric minimum-length-scale
  constraints of Zhou et al. (2015): two differentiable penalties, one for solid
  and one for void features, that are zero when every feature is wide enough to
  survive the eroded / dilated projection. Add them to the loss (or use them as
  inequality constraints) late in a beta continuation.
* :func:`brush_violations` / :func:`minimum_lengthscale` - check a binary
  design against a circular brush, as a fab rule check: a feature passes when
  a disk of that diameter can paint it (solid) or carve it (void) without
  touching the other phase (Hammond et al. 2021; Schubert et al. 2022).
* :func:`remove_floating` / :func:`fill_enclosed_voids` - clean-up of a binary
  design: drop solid islands not connected to the design-region edge (where
  the fixed waveguides attach) and fill voids that no etch could reach.

References
----------
* M. Zhou, B. S. Lazarov, F. Wang, O. Sigmund, "Minimum length scale in
  topology optimization by geometric constraints," *Comput. Methods Appl.
  Mech. Eng.* 293, 266 (2015).
* A. M. Hammond, A. Oskooi, S. G. Johnson, S. E. Ralph, "Photonic topology
  optimization with semiconductor-foundry design-rule constraints," *Opt.
  Express* 29, 23916 (2021).
* M. F. Schubert, A. K. C. Cheung, I. A. D. Williamson, A. Spyra, D. H. Alexander,
  "Inverse design of photonic devices with strict foundry fabrication
  constraints," *ACS Photonics* 9, 2327 (2022).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["symmetrize", "lengthscale_constraints", "brush_violations",
           "minimum_lengthscale", "remove_floating", "fill_enclosed_voids"]


def symmetrize(rho, mirror_x: bool = False, mirror_y: bool = False,
               diagonal: bool = False, rotate180: bool = False):
    """Average ``rho`` (2-D, axes x, y) over the requested symmetry operations.

    ``mirror_x`` mirrors across the plane normal to x (``x -> -x``), likewise
    ``mirror_y``; ``diagonal`` transposes (needs a square array); ``rotate180``
    is the point symmetry ``(x, y) -> (-x, -y)``. The result is invariant
    under each, and the map is linear, so it composes with the filter and
    projection in a differentiable pipeline (NumPy or JAX arrays).
    """
    try:
        import jax.numpy as xp
        if isinstance(rho, np.ndarray):
            xp = np
    except ImportError:                    # pragma: no cover
        xp = np
    out = rho
    if mirror_x:
        out = 0.5 * (out + xp.flip(out, axis=0))
    if mirror_y:
        out = 0.5 * (out + xp.flip(out, axis=1))
    if diagonal:
        if out.shape[0] != out.shape[1]:
            raise ValueError("diagonal symmetry needs a square design")
        out = 0.5 * (out + out.T)
    if rotate180:
        out = 0.5 * (out + xp.flip(out, axis=(0, 1)))
    return out


def lengthscale_constraints(rho_filtered, rho_projected, eta_e: float = 0.75,
                            eta_d: float = 0.25, c: Optional[float] = None,
                            radius_cells: Optional[float] = None):
    """Zhou et al. (2015) minimum-length-scale penalties ``(g_solid, g_void)``.

    ``rho_filtered`` is the filtered density (before projection) and
    ``rho_projected`` the projected one. In the *inflection region* of a solid
    feature (where the filtered field peaks, so its gradient vanishes), the
    filtered value must exceed the erosion threshold ``eta_e``; in that of a
    void feature it must stay below the dilation threshold ``eta_d``. Each
    penalty is the mean squared shortfall there, weighted by the indicator
    ``rho_projected * exp(-c |grad rho_filtered|^2)`` (and its void
    counterpart), so both are 0 for a design whose features all clear the
    filter's length scale, and differentiable. The gradient is taken per cell.
    ``c`` sets how narrowly the indicator picks out the inflection region;
    with ``radius_cells`` (the conic filter radius) given and ``c`` not, it
    defaults to ``radius_cells**4``, a heuristic choice. Tune it for your
    filter.
    """
    import jax.numpy as jnp
    if c is None:
        if radius_cells is None:
            raise ValueError("pass c or radius_cells")
        c = float(radius_cells) ** 4
    gx, gy = jnp.gradient(rho_filtered)
    g2 = gx ** 2 + gy ** 2
    w = jnp.exp(-c * g2)
    g_s = jnp.mean(rho_projected * w * jnp.minimum(rho_filtered - eta_e, 0.0) ** 2)
    g_v = jnp.mean((1.0 - rho_projected) * w
                   * jnp.minimum(eta_d - rho_filtered, 0.0) ** 2)
    return g_s, g_v


def _disk(diameter_cells: float) -> np.ndarray:
    d = float(diameter_cells)
    n = int(np.ceil(d))
    if n % 2 == 0:
        n += 1
    r = (n - 1) / 2.0
    ax = np.arange(n) - r
    xx, yy = np.meshgrid(ax, ax, indexing="ij")
    return (xx ** 2 + yy ** 2) <= (d / 2.0) ** 2 + 1e-9


def brush_violations(binary, diameter_cells: float,
                     pad_mode: str = "edge") -> Tuple[np.ndarray, np.ndarray]:
    """Cells where a binary design breaks a circular-brush rule.

    A solid cell passes if some disk of ``diameter_cells`` that lies entirely
    in the solid covers it (morphological opening); a void cell likewise with
    the void. Returns ``(solid_violations, void_violations)`` boolean arrays.
    The design is padded with its edge values (``pad_mode``), so features
    continuing into the fixed layout outside the region are not flagged at the
    boundary.
    """
    from scipy import ndimage
    b = np.asarray(binary) > 0.5
    k = _disk(diameter_cells)
    p = k.shape[0]
    bp = np.pad(b, p, mode=pad_mode)
    crop = (slice(p, -p), slice(p, -p))
    solid_ok = ndimage.binary_opening(bp, structure=k)[crop]
    void_ok = ndimage.binary_opening(~bp, structure=k)[crop]
    return b & ~solid_ok, ~b & ~void_ok


def minimum_lengthscale(binary, max_cells: Optional[int] = None) -> Tuple[float, float]:
    """Largest brush diameters (in cells) the solid and the void each pass.

    Scans integer diameters from 1 up to ``max_cells`` (default: the array
    size). Multiply by the cell size for a length.
    """
    b = np.asarray(binary) > 0.5
    max_cells = int(max_cells or max(b.shape))
    out = []
    for phase in (0, 1):
        best = 0
        for d in range(1, max_cells + 1):
            viol = brush_violations(b, d)[phase]
            if viol.any():
                break
            best = d
        out.append(float(best))
    return tuple(out)


def _edge_labels(labels: np.ndarray) -> set:
    edge = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    return set(np.unique(edge[edge > 0]).tolist())


def remove_floating(binary, anchors=None) -> np.ndarray:
    """Drop solid islands that touch neither the design-region edge nor
    ``anchors`` (an optional boolean mask of cells that are attached to
    something, e.g. the waveguide stubs). 4-connectivity."""
    from scipy import ndimage
    b = np.asarray(binary) > 0.5
    lab, _ = ndimage.label(b)
    keep = _edge_labels(lab)
    if anchors is not None:
        a = np.asarray(anchors, bool)
        keep |= set(np.unique(lab[a & b]).tolist()) - {0}
    return np.isin(lab, list(keep)) if keep else np.zeros_like(b)


def fill_enclosed_voids(binary) -> np.ndarray:
    """Fill void regions that do not reach the design-region edge (a closed
    hole in the solid that no etch or lift-off could clear). 4-connectivity."""
    from scipy import ndimage
    b = np.asarray(binary) > 0.5
    lab, _ = ndimage.label(~b)
    open_voids = _edge_labels(lab)
    return b | (~np.isin(lab, list(open_voids)) & ~b)
