"""Static field solvers: electrostatic potential and steady-state temperature.

These give the fields that drive the electro-optic and thermo-optic index
changes of :mod:`photonfdtd.active`. Both are the same finite-volume problem on
the cell-centred grid the optical solver uses,

    div(c grad u) = -q,

with ``c`` the DC permittivity and ``u`` the potential (``q = 0``), or ``c``
the thermal conductivity and ``u`` the temperature (``q`` the heater power
density). ``c`` may be a diagonal tensor (one array per axis, as for lithium
niobate's DC permittivity); the conductance across a face between two cells is
the harmonic mean of their ``c`` along that axis, which keeps the flux
continuous across material interfaces.

Electrodes are equipotential cells held at a fixed voltage; heaters deposit
their power uniformly over their cells. The outer boundary is insulating
(zero normal flux) unless a face is listed as a Dirichlet sink or a convective
(Robin) face.

The problems are linear, so a solve with several electrodes or heaters is a
sum of unit solves; :func:`solve_potential` and :func:`solve_thermal` return
those unit solutions (``basis``) so the optical index can be rebuilt for any
voltages or powers without solving again. The sparse system is factorized once
(SciPy's SuperLU) and reused for every right-hand side; above
``direct_limit`` unknowns it switches to preconditioned conjugate gradients
(with pyamg if it is installed).

Geometry
--------
``mesh`` is anything with ``coords`` (three 1-D arrays), ``cell_size`` and
``shape``: a :class:`~photonfdtd.grid.Grid`, or :func:`mesh_of` applied to a
:class:`~photonfdtd.mode.ModeSolver` (its y-z cross-section, as a mesh with one
cell along x). An axis with a single cell is collapsed. In a 2-D cross-section
the quantities are per unit length along the collapsed axis: a heater power is
then in W/m.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_FACES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


@dataclass
class Mesh:
    """A cell-centred mesh: ``coords`` (three 1-D arrays of cell centres),
    ``cell_size`` (three spacings, 0 on a collapsed axis) and ``shape``."""
    coords: Tuple[np.ndarray, np.ndarray, np.ndarray]
    cell_size: Tuple[float, float, float]
    shape: Tuple[int, int, int] = field(init=False)

    def __post_init__(self):
        self.coords = tuple(np.asarray(c, float) for c in self.coords)
        self.cell_size = tuple(float(d) for d in self.cell_size)
        self.shape = tuple(c.size for c in self.coords)


def mesh_of(target) -> Mesh:
    """The static-solve mesh of a Grid, a Simulation or a ModeSolver."""
    if hasattr(target, "grid"):                  # Simulation
        target = target.grid
    if hasattr(target, "coords") and hasattr(target, "cell_size"):
        return target if isinstance(target, Mesh) else Mesh(
            tuple(target.coords), tuple(target.cell_size))
    if hasattr(target, "y") and hasattr(target, "z") and hasattr(target, "dy"):
        return Mesh((np.array([0.0]), target.y, target.z), (0.0, target.dy, target.dz))
    raise TypeError(f"can't build a mesh from {type(target).__name__}")


def region_mask(geometry, mesh) -> np.ndarray:
    """Boolean mask (``mesh.shape``) of cells whose centre is inside ``geometry``
    (a :class:`~photonfdtd.geometry.Box` or :class:`~photonfdtd.geometry.PolySlab`).
    A 2-D box (centre and size of length 2) on a cross-section mesh is read as
    (y, z), the :class:`~photonfdtd.mode.ModeSolver` convention."""
    from .geometry import Box
    if isinstance(geometry, Box):
        c, sz = list(geometry.center), list(geometry.size)
        if len(c) == 2 and mesh.shape[0] == 1:
            c, sz = [0.0] + c, [1.0] + sz
        c += [0.0] * (3 - len(c)); sz += [0.0] * (3 - len(sz))
        mask = np.ones(mesh.shape, bool)
        for a in range(3):
            lo, hi = c[a] - sz[a] / 2, c[a] + sz[a] / 2
            x = mesh.coords[a]
            # A collapsed axis sits at its own coordinate (a cross-section
            # through a 3-D layout); a zero-size box extent on it always holds.
            m = (x >= lo) & (x <= hi) if not (x.size == 1 and lo == hi == 0.0) \
                else np.ones(1, bool)
            shp = [1, 1, 1]; shp[a] = x.size
            mask &= m.reshape(shp)
        return mask
    return np.asarray(geometry.region_mask(mesh), bool)


# --------------------------------------------------------------------------- #
# The finite-volume operator.
# --------------------------------------------------------------------------- #
def _axes(mesh):
    return [a for a in range(3) if mesh.shape[a] > 1]


def _cell_measure(mesh) -> float:
    """Cell volume (area in 2-D, length in 1-D); collapsed axes count as 1 m."""
    v = 1.0
    for a in range(3):
        if mesh.shape[a] > 1:
            v *= mesh.cell_size[a]
    return v


def _assemble(mesh, coef, dirichlet=None, robin=None):
    """Sparse ``A`` (N x N, symmetric positive semi-definite) and boundary
    right-hand side ``b`` for ``-div(c grad u)``, flattened in C order.

    ``coef``: three arrays (one per axis) of ``c``. ``dirichlet``: {face: value}
    (u fixed on the outer face, half a cell out). ``robin``: {face: (h, value)}
    (flux ``h (u - value)`` out through the face).
    """
    import scipy.sparse as sp
    shape = mesh.shape
    N = int(np.prod(shape))
    idx = np.arange(N).reshape(shape)
    rows, cols, vals = [], [], []
    diag = np.zeros(N)
    b = np.zeros(N)
    dirichlet = dict(dirichlet or {})
    robin = dict(robin or {})
    for a in _axes(mesh):
        d = mesh.cell_size[a]
        c = np.asarray(coef[a], float)
        lo = [slice(None)] * 3; hi = [slice(None)] * 3
        lo[a] = slice(0, -1); hi[a] = slice(1, None)
        c0, c1 = c[tuple(lo)], c[tuple(hi)]
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where((c0 > 0) & (c1 > 0), 2.0 * c0 * c1 / (c0 + c1), 0.0) / d ** 2
        i0, i1 = idx[tuple(lo)].ravel(), idx[tuple(hi)].ravel()
        g = g.ravel()
        rows += [i0, i1]; cols += [i1, i0]; vals += [-g, -g]
        np.add.at(diag, i0, g)
        np.add.at(diag, i1, g)
        for side, sl in (("min", 0), ("max", -1)):
            face = "xyz"[a] + side
            if face not in dirichlet and face not in robin:
                continue
            bs = [slice(None)] * 3; bs[a] = slice(sl, None) if sl == -1 else slice(0, 1)
            cb = c[tuple(bs)].ravel()
            ib = idx[tuple(bs)].ravel()
            if face in dirichlet:
                gb = 2.0 * cb / d ** 2                       # to the face, d/2 away
                val = float(dirichlet[face])
            else:
                h, val = robin[face]
                with np.errstate(divide="ignore"):
                    gb = np.where(cb > 0, 1.0 / (d / (2.0 * cb) + 1.0 / float(h)), 0.0) / d
                val = float(val)
            np.add.at(diag, ib, gb)
            np.add.at(b, ib, gb * val)
    rows.append(np.arange(N)); cols.append(np.arange(N)); vals.append(diag)
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(N, N))
    return A, b


class _Solver:
    """Factorize the free-cell block once; solve for many right-hand sides."""

    def __init__(self, A, free, direct_limit, tol):
        import scipy.sparse.linalg as spla
        self.free = free
        Aff = A[free][:, free].tocsc()
        self.Aff = Aff
        self.tol = tol
        self.direct = Aff.shape[0] <= direct_limit
        if self.direct:
            self.lu = spla.splu(Aff)
        else:
            try:
                import pyamg
                ml = pyamg.smoothed_aggregation_solver(Aff.tocsr())
                self.M = ml.aspreconditioner()
            except ImportError:
                d = Aff.diagonal()
                self.M = spla.LinearOperator(Aff.shape, matvec=lambda x: x / d)

    def __call__(self, rhs):
        if self.direct:
            return self.lu.solve(rhs)
        import scipy.sparse.linalg as spla
        x, info = spla.cg(self.Aff, rhs, M=self.M, rtol=self.tol, maxiter=20000)
        if info != 0:
            raise RuntimeError(f"conjugate gradients did not converge (info={info}); "
                               "install pyamg or lower direct_limit's threshold")
        return x


def _gradient(u, mesh):
    """``grad u`` at cell centres, three arrays (zero along collapsed axes)."""
    out = []
    for a in range(3):
        if mesh.shape[a] > 1:
            out.append(np.gradient(u, mesh.cell_size[a], axis=a))
        else:
            out.append(np.zeros_like(u))
    return tuple(out)


# --------------------------------------------------------------------------- #
# Electrostatics.
# --------------------------------------------------------------------------- #
@dataclass
class ElectrostaticResult:
    """Potential and field from :func:`solve_potential`.

    ``phi`` and ``E`` (three arrays ``(Ex, Ey, Ez)``, V/m, at cell centres) are
    for the requested ``voltages``; ``basis_phi[i]`` and ``basis_E[i]`` are the
    unit solutions with electrode ``i`` at 1 V and the others at 0 V, so any
    voltages ``V`` give ``phi = sum_i V_i basis_phi[i]``.
    """
    mesh: Mesh
    names: List[str]
    voltages: np.ndarray
    phi: np.ndarray
    E: Tuple[np.ndarray, np.ndarray, np.ndarray]
    basis_phi: np.ndarray
    basis_E: np.ndarray            # (n_electrodes, 3, *shape)

    def field(self, voltages) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(Ex, Ey, Ez)`` for other electrode voltages."""
        v = np.asarray(voltages, float)
        e = np.tensordot(v, self.basis_E, axes=1)
        return (e[0], e[1], e[2])


def solve_potential(mesh, eps_dc, electrodes: Sequence, voltages=None,
                    boundary: str = "neumann", direct_limit: int = 400_000,
                    tol: float = 1e-10) -> ElectrostaticResult:
    """Electrostatic potential for electrodes at given voltages.

    Parameters
    ----------
    mesh
        A :class:`Mesh`, :class:`~photonfdtd.grid.Grid`, Simulation or
        ModeSolver (see :func:`mesh_of`).
    eps_dc
        DC relative permittivity: one array (isotropic) or three (per axis).
    electrodes
        A sequence of electrodes: objects with a ``geometry`` (or geometries
        themselves), each an equipotential conductor.
    voltages
        Electrode voltages in V (default: 1 V on the first, 0 on the rest).
    boundary
        ``"neumann"`` (default): no field crosses the outer boundary, i.e. the
        domain is surrounded by an insulator; make it large enough that the
        field has decayed. ``"ground"``: the outer boundary is at 0 V.
    """
    mesh = mesh_of(mesh)
    n_el = len(electrodes)
    if n_el == 0:
        raise ValueError("need at least one electrode")
    coef = _components(eps_dc, mesh)
    masks = [region_mask(getattr(e, "geometry", e), mesh) for e in electrodes]
    names = [getattr(e, "name", "") or f"electrode{i}" for i, e in enumerate(electrodes)]
    for nm, m in zip(names, masks):
        if not m.any():
            raise ValueError(f"electrode {nm!r} covers no cells of the mesh")
    dirichlet = {f: 0.0 for f in _FACES} if boundary == "ground" else None
    if boundary not in ("neumann", "ground"):
        raise ValueError("boundary must be 'neumann' or 'ground'")
    fixed = np.zeros(mesh.shape, bool)
    for m in masks:
        fixed |= m
    # A conductor's own permittivity is irrelevant: make it very large so the
    # harmonic-mean face conductance is set by the dielectric neighbour, which
    # puts the electrode's potential on its surface (the cell face).
    coef = tuple(np.where(fixed, 1e30, c) for c in coef)
    A, _ = _assemble(mesh, coef, dirichlet=dirichlet)
    fixed = fixed.ravel()
    free = np.flatnonzero(~fixed)
    solve = _Solver(A, free, direct_limit, tol)
    A_fx = A[free][:, np.flatnonzero(fixed)]
    basis_phi = np.zeros((n_el,) + mesh.shape)
    basis_E = np.zeros((n_el, 3) + mesh.shape)
    for i, m in enumerate(masks):
        u = np.zeros(int(np.prod(mesh.shape)))
        u[m.ravel()] = 1.0
        u[free] = solve(-(A_fx @ u[fixed]))
        phi = u.reshape(mesh.shape)
        basis_phi[i] = phi
        basis_E[i] = -np.stack(_gradient(phi, mesh))
    v = np.zeros(n_el) if voltages is None else np.asarray(voltages, float)
    if voltages is None:
        v[0] = 1.0
    if v.shape != (n_el,):
        raise ValueError(f"need {n_el} voltages, got {v.shape}")
    phi = np.tensordot(v, basis_phi, axes=1)
    e = np.tensordot(v, basis_E, axes=1)
    return ElectrostaticResult(mesh, names, v, phi, (e[0], e[1], e[2]), basis_phi, basis_E)


# --------------------------------------------------------------------------- #
# Heat.
# --------------------------------------------------------------------------- #
@dataclass
class ThermalResult:
    """Temperature from :func:`solve_thermal`.

    ``T`` (K) is for the requested ``powers``; ``basis_dT[j]`` is the
    temperature rise for heater ``j`` alone at 1 W (1 W/m in a 2-D
    cross-section), so ``T = T0 + sum_j P_j basis_dT[j]``.
    """
    mesh: Mesh
    names: List[str]
    powers: np.ndarray
    T0: float
    T: np.ndarray
    basis_dT: np.ndarray

    @property
    def dT(self) -> np.ndarray:
        return self.T - self.T0

    def temperature(self, powers) -> np.ndarray:
        return self.T0 + np.tensordot(np.asarray(powers, float), self.basis_dT, axes=1)


def solve_thermal(mesh, conductivity, heaters: Sequence, powers=None,
                  T0: float = 293.15, sinks=("zmin",), convection=None,
                  direct_limit: int = 400_000, tol: float = 1e-10) -> ThermalResult:
    """Steady-state temperature for heaters at given powers.

    Parameters
    ----------
    conductivity
        Thermal conductivity in W/(m K): one array, or three (per axis).
    heaters
        Objects with a ``geometry`` (or geometries); each deposits its power
        uniformly over its cells.
    powers
        Heater powers in W (W/m in a 2-D cross-section). Default: 1 on each.
    sinks
        Outer faces held at ``T0`` (heat sinks), e.g. ``("zmin",)`` for a
        substrate on a temperature-controlled stage. Others are insulating
        unless listed in ``convection``.
    convection
        ``{face: h}`` with h in W/(m^2 K): heat leaves through that face at
        ``h (T - T0)``, e.g. ``{"zmax": 10.0}`` for still air.
    """
    mesh = mesh_of(mesh)
    coef = _components(conductivity, mesh)
    n_h = len(heaters)
    if n_h == 0:
        raise ValueError("need at least one heater")
    sinks = tuple(sinks or ())
    for f in sinks + tuple(convection or ()):
        if f not in _FACES:
            raise ValueError(f"unknown face {f!r}; use one of {_FACES}")
    if not sinks and not convection:
        raise ValueError("with no sink or convective face the temperature has no "
                         "steady state; give sinks= or convection=")
    # Unit solves are for the rise above T0, so boundaries sit at 0.
    A, _ = _assemble(mesh, coef, dirichlet={f: 0.0 for f in sinks},
                     robin={f: (h, 0.0) for f, h in (convection or {}).items()})
    N = int(np.prod(mesh.shape))
    solve = _Solver(A, np.arange(N), direct_limit, tol)
    masks = [region_mask(getattr(h, "geometry", h), mesh) for h in heaters]
    names = [getattr(h, "name", "") or f"heater{j}" for j, h in enumerate(heaters)]
    vol = _cell_measure(mesh)
    basis = np.zeros((n_h,) + mesh.shape)
    for j, m in enumerate(masks):
        n = int(m.sum())
        if n == 0:
            raise ValueError(f"heater {names[j]!r} covers no cells of the mesh")
        q = np.zeros(N)
        q[m.ravel()] = 1.0 / (n * vol)           # 1 W spread over the heater
        basis[j] = solve(q).reshape(mesh.shape)
    p = np.array([getattr(h, "power", 1.0) for h in heaters], float) \
        if powers is None else np.asarray(powers, float)
    T = T0 + np.tensordot(p, basis, axes=1)
    return ThermalResult(mesh, names, p, float(T0), T, basis)


def _components(c, mesh):
    if isinstance(c, (tuple, list)) and len(c) == 3:
        out = [np.broadcast_to(np.asarray(x, float), mesh.shape) for x in c]
    else:
        x = np.broadcast_to(np.asarray(c, float), mesh.shape)
        out = [x, x, x]
    return tuple(np.ascontiguousarray(x) for x in out)
