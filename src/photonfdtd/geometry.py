"""Geometric primitives.

Structures rasterise onto the Yee grid by writing a relative permittivity
into the eps_r array at cells whose centres fall inside the primitive. Later
structures overwrite earlier ones.

Currently shipped:

- :class:`Box`      - axis-aligned box.
- :class:`PolySlab` - polygon in the xy-plane, extruded between two z bounds.
  Used by the gdsfactory adapter to rasterise arbitrary planar layouts.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence, Tuple
import numpy as np

from .grid import Grid
from .materials import Medium


@dataclass(frozen=True)
class Box:
    center: Tuple[float, ...]
    size: Tuple[float, ...]
    medium: Medium

    def _bounds(self) -> Tuple[Tuple[float, float], ...]:
        c = _to3(self.center)
        s = _to3(self.size)
        return tuple((ci - si / 2, ci + si / 2) for ci, si in zip(c, s))

    def region_mask(self, grid: Grid) -> np.ndarray:
        """Boolean array (grid.shape) of cells whose centre falls inside the box."""
        bounds = self._bounds()
        mask = np.ones(grid.shape, dtype=bool)
        for axis in range(3):
            coord = grid.coords[axis]
            lo, hi = bounds[axis]
            if coord.size == 1:
                if not (lo <= 0.0 <= hi or (lo == hi == 0.0)):
                    return np.zeros(grid.shape, dtype=bool)
                axis_mask = np.array([True])
            else:
                axis_mask = (coord >= lo) & (coord <= hi)
            shape = [1, 1, 1]
            shape[axis] = axis_mask.size
            mask &= axis_mask.reshape(shape)
        return mask

    def stamp(self, grid: Grid, eps_r: np.ndarray) -> None:
        """Write self.medium.eps_r into the cells whose centres fall inside."""
        eps_r[self.region_mask(grid)] = self.medium.eps_r


@dataclass(frozen=True)
class PolySlab:
    """A polygon in the xy-plane extruded along z.

    Parameters
    ----------
    vertices : sequence of (x, y) pairs, in metres
        Vertices of the polygon, in order. Self-intersecting polygons are
        not supported. The polygon is closed automatically (last vertex
        joined to the first).
    z_bounds : (z_min, z_max), in metres
        Vertical extent of the slab. On a 2D simulation (z size = 0) the
        bounds must straddle 0 for the slab to be rasterised at all.
    medium : Medium
        Material filling the polygon.
    """
    vertices: Tuple[Tuple[float, float], ...]
    z_bounds: Tuple[float, float]
    medium: Medium

    def region_mask(self, grid: Grid) -> np.ndarray:
        """Boolean array (grid.shape) of cells whose centre falls inside the slab."""
        mask = np.zeros(grid.shape, dtype=bool)
        verts = np.asarray(self.vertices, dtype=float)
        if verts.ndim != 2 or verts.shape[1] != 2 or verts.shape[0] < 3:
            raise ValueError("PolySlab.vertices must be a (N>=3, 2) array")
        z_lo, z_hi = float(self.z_bounds[0]), float(self.z_bounds[1])
        if z_lo > z_hi:
            z_lo, z_hi = z_hi, z_lo
        zc = grid.coords[2]
        if zc.size == 1:
            if not (z_lo <= 0.0 <= z_hi):
                return mask
            k_idx = np.array([0])
        else:
            mz = (zc >= z_lo) & (zc <= z_hi)
            if not mz.any():
                return mask
            k_idx = np.flatnonzero(mz)
        xc, yc = grid.coords[0], grid.coords[1]
        nx, ny = xc.size, yc.size
        from matplotlib.path import Path
        XX, YY = np.meshgrid(xc, yc, indexing="ij")
        inside = Path(verts).contains_points(
            np.column_stack([XX.ravel(), YY.ravel()])).reshape(nx, ny)
        if not inside.any():
            return mask
        ii, jj = np.where(inside)
        for k in k_idx:
            mask[ii, jj, k] = True
        return mask

    def stamp(self, grid: Grid, eps_r: np.ndarray) -> None:
        verts = np.asarray(self.vertices, dtype=float)
        if verts.ndim != 2 or verts.shape[1] != 2 or verts.shape[0] < 3:
            raise ValueError("PolySlab.vertices must be a (N>=3, 2) array")
        z_lo, z_hi = float(self.z_bounds[0]), float(self.z_bounds[1])
        if z_lo > z_hi:
            z_lo, z_hi = z_hi, z_lo

        # Determine which cells lie inside the z extent. Use grid.coords[2].
        zc = grid.coords[2]
        if zc.size == 1:
            if not (z_lo <= 0.0 <= z_hi):
                return
            k_idx = np.array([0])
        else:
            mask_z = (zc >= z_lo) & (zc <= z_hi)
            if not mask_z.any():
                return
            k_idx = np.flatnonzero(mask_z)

        # In-plane point-in-polygon test for cell centres on the xy plane.
        xc, yc = grid.coords[0], grid.coords[1]
        nx, ny = xc.size, yc.size
        # Build (Npts, 2) point list lazily; rely on matplotlib.path for speed
        # without adding a hard dependency on Shapely.
        from matplotlib.path import Path
        XX, YY = np.meshgrid(xc, yc, indexing="ij")
        pts = np.column_stack([XX.ravel(), YY.ravel()])
        path = Path(verts)
        inside = path.contains_points(pts).reshape(nx, ny)
        if not inside.any():
            return
        ii, jj = np.where(inside)
        # Apply the medium to every selected (i, j) at every k_idx.
        for k in k_idx:
            eps_r[ii, jj, k] = self.medium.eps_r


def _to3(x: Sequence[float]) -> Tuple[float, float, float]:
    x = tuple(x)
    if len(x) == 1:
        return (float(x[0]), 0.0, 0.0)
    if len(x) == 2:
        return (float(x[0]), float(x[1]), 0.0)
    if len(x) == 3:
        return (float(x[0]), float(x[1]), float(x[2]))
    raise ValueError(f"Expected 1/2/3-tuple, got length {len(x)}")
