"""Anisotropic subpixel (sub-cell) smoothing of material interfaces.

Plain FDTD stamps a single permittivity into each cell according to whether the
cell *centre* falls inside a structure (see :mod:`photonfdtd.geometry`). That
staircases every interface that does not land exactly on a cell boundary, which
degrades the solver's global convergence from second order to roughly first
order and makes derived quantities (resonant frequencies, effective indices,
transmission phases) jitter as a structure is moved by sub-cell amounts.

Subpixel smoothing removes the staircase by replacing the permittivity of a
partially filled cell with an *effective anisotropic tensor* that respects the
electromagnetic boundary conditions at the interface it contains:

* the field component **normal** to the interface sees the **harmonic mean**
  of the permittivity (because the normal component of D is continuous), and
* the components **tangential** to the interface see the **arithmetic mean**
  (because the tangential component of E is continuous).

In the interface's principal frame the effective tensor is therefore
``diag(eps_h, eps_a, eps_a)`` with the harmonic mean ``eps_h`` along the
surface normal ``n`` and the arithmetic mean ``eps_a`` in the two tangential
directions. Rotated into the lab frame this is

    eps_eff = eps_a * I + (eps_h - eps_a) * (n ⊗ n).

We keep only the diagonal of this tensor,

    eps_xx = eps_a + (eps_h - eps_a) * n_x**2
    eps_yy = eps_a + (eps_h - eps_a) * n_y**2
    eps_zz = eps_a + (eps_h - eps_a) * n_z**2,

and feed ``eps_xx / eps_yy / eps_zz`` to the ``Ex / Ey / Ez`` updates
respectively. For axis-aligned interfaces (rectangular ``Box`` structures and
Manhattan layouts, i.e. the overwhelming majority of integrated-photonics
geometry) the surface normal lies along a coordinate axis, the off-diagonal
tensor entries vanish identically, and this diagonal treatment is exact. For
obliquely oriented interfaces (e.g. angled ``PolySlab`` walls) dropping the
off-diagonal coupling is an approximation, but still a large improvement over
staircasing. This mirrors the "diagonal" subpixel-averaging option used by
several production FDTD codes; the full off-diagonal tensor (Meep's default)
would require interpolating field components between Yee locations and is left
for future work.

The per-cell arithmetic and harmonic means and the surface normal are obtained
by supersampling: each cell is subdivided into ``factor`` pieces per active
axis, the structures are rasterised onto that fine grid, and the means are
block-reduced back to the native resolution. The surface normal is estimated
from the gradient of the volume-averaged permittivity field, which points
across the interface and reduces to an exact axis-aligned unit vector whenever
the interface is axis-aligned.

References
----------
* A. Farjadpour, D. Roundy, A. Rundquist, M. Ibanescu, S. G. Johnson et al.,
  "Improving accuracy by subpixel smoothing in the finite-difference time
  domain," *Optics Letters* **31**(20), 2972-2974 (2006).
  https://doi.org/10.1364/OL.31.002972
* C. A. Kottke, A. Farjadpour, S. G. Johnson, "Perturbation theory for
  anisotropic dielectric interfaces, and application to subpixel smoothing of
  discretized numerical methods," *Physical Review E* **77**, 036611 (2008).
  https://doi.org/10.1103/PhysRevE.77.036611
* A. F. Oskooi, D. Roundy, M. Ibanescu, P. Bermel, J. D. Joannopoulos,
  S. G. Johnson, "MEEP: A flexible free-software package for electromagnetic
  simulations by the FDTD method," *Computer Physics Communications* **181**,
  687-702 (2010) — subpixel averaging in a production solver.
  https://doi.org/10.1016/j.cpc.2009.11.008
"""
from __future__ import annotations
from typing import Sequence, Tuple
import numpy as np

from .grid import Grid


def _fine_grid(grid: Grid, factor: int) -> Grid:
    """A copy of `grid` with `factor` cells per original cell on each active axis.

    Collapsed (size-0) axes stay collapsed. PML is irrelevant to rasterisation
    and set to zero. The domain is identically centred, so each native cell maps
    to a contiguous block of ``factor`` fine cells per active axis.
    """
    size = grid.size
    cs = tuple((c / factor) if c > 0 else 0.0 for c in grid.cell_size)
    return Grid(size=size, cell_size=cs, pml_layers=(0, 0, 0))


def _slab_means(grid: Grid, structures, background_eps: float, factor: int):
    """Arithmetic and harmonic per-cell permittivity means, built one native
    x-slab at a time so peak memory is a single refined slab rather than the
    whole ``factor**ndim``-times-larger fine grid.

    Bit-identical to refining the entire domain and block-reducing it: each
    native cell maps to a fixed contiguous block of fine cells, reduced here in
    the slab that contains it.
    """
    from types import SimpleNamespace
    nx, ny, nz = grid.shape
    fg = _fine_grid(grid, factor)                 # fine coords / spacing only
    xf, yf, zf = fg.coords
    fx = factor if nx > 1 else 1
    fy = factor if ny > 1 else 1
    fz = factor if nz > 1 else 1

    eps_a = np.empty(grid.shape, dtype=np.float64)
    eps_h = np.empty(grid.shape, dtype=np.float64)
    for i in range(nx):
        xs = xf[i * fx:(i + 1) * fx]
        duck = SimpleNamespace(shape=(xs.size, yf.size, zf.size),
                               coords=(xs, yf, zf), cell_size=fg.cell_size)
        sub = np.full(duck.shape, float(background_eps), dtype=np.float64)
        for s in structures:
            sub[s.region_mask(duck)] = s.medium.eps_r
        r = sub.reshape(1, fx, ny, fy, nz, fz)
        eps_a[i] = r.mean(axis=(1, 3, 5))[0]
        eps_h[i] = (1.0 / (1.0 / r).mean(axis=(1, 3, 5)))[0]
    return eps_a, eps_h


def smooth_permittivity(
    grid: Grid,
    structures: Sequence,
    background_eps: float = 1.0,
    factor: int = 3,
    dtype=np.float64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Anisotropic subpixel-smoothed diagonal permittivity tensor.

    Returns ``(eps_xx, eps_yy, eps_zz)``, each of shape ``grid.shape``, to be
    used by the ``Ex``, ``Ey`` and ``Ez`` updates respectively. In cells that
    lie entirely within one material all three equal that material's
    permittivity and the result is bit-identical to plain stamping; only cells
    straddling an interface differ.

    Parameters
    ----------
    factor : int
        Sub-cell supersampling factor per active axis (>= 1). The build cost
        scales as ``factor ** ndim`` but peak memory only as a single refined
        x-slab; 2-4 is plenty for axis-aligned geometry, more only sharpens
        oblique interfaces. ``1`` disables refinement and reproduces plain
        stamping.
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    shape = grid.shape

    eps_a, eps_h = _slab_means(grid, structures, background_eps, factor)

    # Surface normal from the gradient of the volume-averaged permittivity.
    # In uniform regions eps_a == eps_h so the normal is irrelevant (the tensor
    # is isotropic regardless); at an interface the gradient is large and, for
    # an axis-aligned interface, exactly along that axis.
    spacings = [grid.cell_size[a] if shape[a] > 1 else 1.0 for a in range(3)]
    grads = []
    for a in range(3):
        if shape[a] > 1:
            grads.append(np.gradient(eps_a, spacings[a], axis=a))
        else:
            grads.append(np.zeros_like(eps_a))
    gmag = np.sqrt(grads[0] ** 2 + grads[1] ** 2 + grads[2] ** 2)
    tiny = gmag.max() * 1e-9 if gmag.size else 0.0
    safe = gmag > max(tiny, np.finfo(np.float64).tiny)
    inv_g = np.where(safe, 1.0 / np.where(safe, gmag, 1.0), 0.0)
    n = [g * inv_g for g in grads]        # unit surface normal per cell (0 if flat)

    delta = eps_h - eps_a                  # <= 0; nonzero only at interfaces
    eps_xx = eps_a + delta * n[0] ** 2
    eps_yy = eps_a + delta * n[1] ** 2
    eps_zz = eps_a + delta * n[2] ** 2
    return (eps_xx.astype(dtype, copy=False),
            eps_yy.astype(dtype, copy=False),
            eps_zz.astype(dtype, copy=False))
