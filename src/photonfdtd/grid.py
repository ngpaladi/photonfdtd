"""Yee-grid abstraction.

A uniform Cartesian Yee grid in 1D, 2D, or 3D. The simulation extent is given
in metres; cell size is uniform per axis. Coordinates are stored on the
*primary* (E-field) grid; H-field samples sit at offsets of half a cell, the
standard Yee staggering.

The grid does not store fields itself - it just describes geometry. Field
arrays live on the Simulation object.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence, Tuple, Optional
import numpy as np


@dataclass
class Grid:
    size: Tuple[float, ...]          # physical extent of the simulation domain, m
    cell_size: Tuple[float, ...]     # uniform spacing per axis, m
    pml_layers: Tuple[int, ...] = (0, 0, 0)  # PML thickness per axis, in cells

    # Filled in __post_init__
    shape: Tuple[int, ...] = field(init=False)
    coords: Tuple[np.ndarray, ...] = field(init=False)
    ndim: int = field(init=False)

    def __post_init__(self) -> None:
        # Normalise inputs to fixed 3-tuples; None entries collapse a dimension.
        size = _to3(self.size)
        cs = _to3(self.cell_size)
        pml = _to3(self.pml_layers, default=0)

        ndim = sum(1 for s in size if s is not None and s > 0)
        if ndim == 0:
            raise ValueError("Grid must have at least one non-zero dimension")

        active = []
        shape = []
        coords = []
        pml_active = []
        for i, (s, d, p) in enumerate(zip(size, cs, pml)):
            if s is None or s == 0:
                shape.append(1)
                coords.append(np.array([0.0]))
                pml_active.append(0)
                continue
            if d is None or d <= 0:
                raise ValueError(f"axis {i}: cell_size must be > 0")
            n = int(round(s / d))
            if n < 4:
                raise ValueError(f"axis {i}: need at least 4 cells, got {n}")
            shape.append(n)
            # Coordinates of E-field samples at integer cell indices.
            coords.append((np.arange(n) - (n - 1) / 2) * d)
            pml_active.append(int(p))
            active.append(i)

        self.size = tuple(0.0 if s is None else float(s) for s in size)
        self.cell_size = tuple(0.0 if d is None else float(d) for d in cs)
        self.pml_layers = tuple(pml_active)
        self.shape = tuple(shape)
        self.coords = tuple(coords)
        self.ndim = ndim

    @property
    def dx(self) -> float:
        return float(self.cell_size[0])

    @property
    def dy(self) -> float:
        return float(self.cell_size[1])

    @property
    def dz(self) -> float:
        return float(self.cell_size[2])

    @property
    def min_cell(self) -> float:
        return min(d for d in self.cell_size if d > 0)

    def index_at(self, point: Sequence[float]) -> Tuple[int, int, int]:
        """Return the (i, j, k) index of the cell containing `point`.

        Accepts 1D, 2D, or 3D `point` tuples; missing coordinates default to 0.
        """
        pt = list(point) + [0.0] * (3 - len(point))
        idx = []
        for axis in range(3):
            c = self.coords[axis]
            if c.size == 1:
                idx.append(0)
            else:
                i = int(np.argmin(np.abs(c - pt[axis])))
                idx.append(i)
        return tuple(idx)


def _to3(x, default=None):
    if x is None:
        return (default, default, default)
    if isinstance(x, (int, float)):
        return (x, x, x)
    x = tuple(x)
    if len(x) == 1:
        return (x[0], default, default)
    if len(x) == 2:
        return (x[0], x[1], default)
    if len(x) == 3:
        return x
    raise ValueError(f"Expected 1/2/3-tuple or scalar, got length {len(x)}")
