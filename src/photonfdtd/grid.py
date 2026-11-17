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
    #: Periodic boundary per axis: the fields wrap around (cell n-1 neighbours
    #: cell 0), as for one unit cell of a grating, photonic crystal or
    #: metasurface at normal incidence (add ``bloch`` for oblique incidence).
    #: A periodic axis takes no PML. The
    #: default, non-periodic edge with no PML is a perfect electric conductor.
    periodic: Tuple[bool, ...] = (False, False, False)
    #: Mirror symmetry per axis about the plane through the domain centre
    #: (coordinate 0): ``0`` none, ``-1`` a PEC plane (tangential E odd, so
    #: zero on the plane), ``+1`` a PMC plane (tangential H zero on it). On a
    #: symmetric axis only the upper half (coordinate >= 0) is stepped, which
    #: halves the cells (and time) per symmetric axis; ``shape`` and ``coords``
    #: describe that half and ``full_shape`` the whole domain. The cell count
    #: of the full axis is made odd for PEC (an E node on the plane) and even
    #: for PMC (an H node on it), adding one cell if needed. PML sits on the
    #: high side only; a symmetric axis cannot be periodic.
    symmetry: Tuple[int, ...] = (0, 0, 0)
    #: Bloch wavevector (rad/m) per periodic axis: the fields obey
    #: ``F(x + L) = F(x) exp(i k L)`` across the cell of length ``L``. Any
    #: non-zero entry makes the run step complex fields (twice the memory),
    #: as an oblique plane wave on a periodic cell needs; see
    #: :meth:`photonfdtd.PlaneWave.bloch_vector`. Only periodic axes may
    #: have one.
    bloch: Tuple[float, ...] = (0.0, 0.0, 0.0)

    # Filled in __post_init__
    shape: Tuple[int, ...] = field(init=False)
    full_shape: Tuple[int, ...] = field(init=False)
    coords: Tuple[np.ndarray, ...] = field(init=False)
    ndim: int = field(init=False)

    def __post_init__(self) -> None:
        # Normalise inputs to fixed 3-tuples; None entries collapse a dimension.
        size = _to3(self.size)
        cs = _to3(self.cell_size)
        pml = _to3(self.pml_layers, default=0)
        per = _to3(self.periodic, default=False)
        sym = _to3(self.symmetry, default=0)

        ndim = sum(1 for s in size if s is not None and s > 0)
        if ndim == 0:
            raise ValueError("Grid must have at least one non-zero dimension")

        active = []
        shape = []
        full_shape = []
        coords = []
        pml_active = []
        per_active = []
        sym_active = []
        for i, (s, d, p) in enumerate(zip(size, cs, pml)):
            sy = 0 if sym[i] is None else int(sym[i])
            if sy not in (-1, 0, 1):
                raise ValueError(f"axis {i}: symmetry must be -1 (PEC), 0 or +1 (PMC)")
            if s is None or s == 0:
                if sy:
                    raise ValueError(f"axis {i}: a collapsed axis cannot be symmetric")
                shape.append(1)
                full_shape.append(1)
                coords.append(np.array([0.0]))
                pml_active.append(0)
                per_active.append(False)
                sym_active.append(0)
                continue
            if d is None or d <= 0:
                raise ValueError(f"axis {i}: cell_size must be > 0")
            n = int(round(s / d))
            if n < 4:
                raise ValueError(f"axis {i}: need at least 4 cells, got {n}")
            if sy and bool(per[i]):
                raise ValueError(f"axis {i}: a symmetric axis cannot be periodic")
            if sy == -1 and n % 2 == 0:
                n += 1                    # PEC: an E node on the plane
            elif sy == 1 and n % 2 == 1:
                n += 1                    # PMC: E nodes at +-d/2
            full_shape.append(n)
            # Coordinates of E-field samples at integer cell indices.
            c = (np.arange(n) - (n - 1) / 2) * d
            if sy:
                c = c[n // 2:]            # the upper half, from the plane up
                if c.size - int(p) < 2:
                    raise ValueError(f"axis {i}: the symmetric half is too "
                                     "small for its PML")
            shape.append(c.size)
            coords.append(c)
            pml_active.append(int(p))
            per_active.append(bool(per[i]))
            sym_active.append(sy)
            if per_active[-1] and int(p) > 0:
                raise ValueError(
                    f"axis {i}: a periodic axis cannot also have PML layers")
            active.append(i)

        self.size = tuple(0.0 if s is None else float(s) for s in size)
        self.cell_size = tuple(0.0 if d is None else float(d) for d in cs)
        self.pml_layers = tuple(pml_active)
        self.periodic = tuple(per_active)
        bl = tuple(0.0 if b is None else float(b) for b in _to3(self.bloch, default=0.0))
        for a in range(3):
            if bl[a] != 0.0 and not per_active[a]:
                raise ValueError(f"axis {a}: a Bloch wavevector needs a periodic axis")
        self.bloch = bl
        self.symmetry = tuple(sym_active)
        self.full_shape = tuple(full_shape)
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
    def is_complex(self) -> bool:
        """True when a Bloch wavevector makes the fields complex."""
        return any(b != 0.0 for b in self.bloch)

    def bloch_phase(self, axis: int) -> complex:
        """``exp(i k L)``: the field factor across one period of ``axis``."""
        return complex(np.exp(1j * self.bloch[axis]
                              * self.shape[axis] * self.cell_size[axis]))

    @property
    def has_symmetry(self) -> bool:
        return any(self.symmetry)

    @property
    def pml_low(self) -> Tuple[int, int, int]:
        """PML cells on the low side of each axis (none on a symmetry plane)."""
        return tuple(0 if s else p for p, s in zip(self.pml_layers, self.symmetry))

    def in_domain(self, point: Sequence[float]) -> bool:
        """False for a point in the mirrored-away half of a symmetric axis
        (more than a quarter cell below the plane); True otherwise."""
        pt = list(point) + [0.0] * (3 - len(point))
        return all(not s or pt[a] >= -0.25 * self.cell_size[a]
                   for a, s in enumerate(self.symmetry))

    def symmetry_factor(self, normal_axis: int) -> int:
        """How many copies of a plane normal to ``normal_axis`` the stepped
        half holds: 2 per symmetric axis in the plane. A plane integral over
        the stepped cells (a flux, a photon count) times this is the whole
        domain's."""
        return 2 ** sum(1 for a, s in enumerate(self.symmetry)
                        if s and a != normal_axis)

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
