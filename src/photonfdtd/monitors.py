"""Monitors record field data during the time-stepping loop.

    FieldMonitor - take full-domain field snapshots at chosen times
    DFTMonitor   - accumulate a running Fourier transform at chosen frequencies
    FluxMonitor  - integrate Poynting flux through an axis-aligned plane

The FluxMonitor returns the time-integrated real Poynting flux over the
whole run. For frequency-resolved flux use a plane DFTMonitor with
:func:`photonfdtd.photon_spectrum`.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence, Tuple, Optional
import numpy as np


@dataclass
class FieldMonitor:
    """Snapshot the named field components at specified time indices.

    Parameters
    ----------
    components : tuple of str
        Any of 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'.
    interval : int, optional
        Record every Nth timestep. Mutually exclusive with `times`.
    times : sequence of float, optional
        Record at the timestep nearest each listed time (s).
    downsample : int, optional
        Spatial stride for stored snapshots: keep every ``downsample``-th cell
        on every axis (default 1 = full resolution). Cuts snapshot memory by
        ``downsample**ndim`` and the device-to-host transfer too.
        The stored array aligns to ``grid.coords[axis][::downsample]`` on each
        axis. Snapshots are stored in the simulation's ``'monitors'`` dtype
        (``precision={'monitors': 'float32'}`` halves snapshot memory even on a
        float64 run).
    plane_z : float, optional
        If given, record only the single z-plane nearest this z-coordinate (m)
        instead of the whole volume. The stored array keeps a size-1 z axis so
        downstream indexing is unchanged. For a top-down field movie this cuts
        snapshot memory — and the device-to-host transfer — by the full z-cell count
        (often 30-100x), since only the one displayed plane is kept.
    name : str
        Identifier used to retrieve results from sim.run().
    """
    name: str
    components: Tuple[str, ...] = ("Ez",)
    interval: Optional[int] = None
    times: Optional[Sequence[float]] = None
    downsample: int = 1
    plane_z: Optional[float] = None

    def __post_init__(self):
        if self.interval is None and self.times is None:
            self.interval = 1
        if self.interval is not None and self.times is not None:
            raise ValueError("Specify only one of interval or times")
        if int(self.downsample) != self.downsample or self.downsample < 1:
            raise ValueError("downsample must be a positive integer")
        self.downsample = int(self.downsample)


@dataclass
class DFTMonitor:
    """Accumulate a running discrete Fourier transform of field components.

    Instead of storing a snapshot at every recorded timestep (memory
    proportional to the number of time samples), a DFTMonitor keeps only the
    complex Fourier amplitude at a handful of chosen frequencies, updated in
    place each step::

        F_c(omega) = sum_n  f_c(t_n) * exp(+1j * omega * t_n) * dt

    Storage is therefore proportional to ``len(freqs)`` rather than the number
    of timesteps - for a steady-state / spectral result this is routinely a
    50-1000x reduction over an equivalent time-domain :class:`FieldMonitor`,
    and it is *exact* at the requested frequencies (no lossy quantisation). The
    accumulation runs in complex128 regardless of the field precision to avoid
    drift over long runs; ``result.dft[name][component]`` is a complex array of
    shape ``(len(freqs), *snapshot_shape)`` and ``result.dft_freqs[name]`` holds
    the frequencies.

    Each component is sampled at its own Yee half-step time (E at integer, H at
    half-integer steps) so the relative phase between E and H is correct.

    Parameters
    ----------
    name : str
        Identifier used to retrieve results from ``sim.run()``.
    components : tuple of str
        Any of 'Ex','Ey','Ez','Hx','Hy','Hz'.
    freqs : sequence of float
        Frequencies (Hz) at which to accumulate the transform.
    interval : int, optional
        Sample every ``interval``-th timestep (default 1 = every step, most
        accurate). Sub-sampling trades spectral accuracy for a little speed.
    downsample : int, optional
        Spatial stride, as in :class:`FieldMonitor`.
    plane_z : float, optional
        If given, accumulate only the single nearest z-plane, as in
        :class:`FieldMonitor`.
    """
    name: str
    components: Tuple[str, ...] = ("Ez",)
    freqs: Sequence[float] = ()
    interval: int = 1
    downsample: int = 1
    plane_z: Optional[float] = None
    # General single-plane restriction on any axis. ``plane_axis`` in
    # {'x','y','z'} with ``plane_position`` (m) records only the nearest plane
    # normal to that axis - essential for a memory-light port monitor on a
    # waveguide (a y-z plane at a given x is KB, not the GB a full-volume DFT
    # would cost). ``plane_z`` is the back-compatible shorthand for
    # ``plane_axis='z'``. On a plane, components staggered half a cell along
    # the normal (the normal E and the tangential H) are averaged over their
    # two neighbours, so every component is recorded on the plane itself and
    # flux / mode overlaps from it are second-order accurate.
    plane_axis: Optional[str] = None
    plane_position: Optional[float] = None

    def __post_init__(self):
        self.freqs = tuple(float(f) for f in self.freqs)
        if len(self.freqs) == 0:
            raise ValueError("DFTMonitor requires at least one frequency in `freqs`")
        if int(self.interval) != self.interval or self.interval < 1:
            raise ValueError("interval must be a positive integer")
        self.interval = int(self.interval)
        if int(self.downsample) != self.downsample or self.downsample < 1:
            raise ValueError("downsample must be a positive integer")
        self.downsample = int(self.downsample)
        # Normalise plane_z into the general (plane_axis, plane_position) form.
        if self.plane_z is not None and self.plane_axis is None:
            self.plane_axis = "z"
            self.plane_position = self.plane_z
        if self.plane_axis is not None:
            if self.plane_axis not in ("x", "y", "z"):
                raise ValueError("plane_axis must be 'x', 'y', or 'z'")
            if self.plane_position is None:
                raise ValueError("plane_axis requires plane_position")

    def plane(self):
        """Resolved ``(axis_index, position)`` for the plane restriction, or None."""
        if self.plane_axis is None:
            return None
        return ({"x": 0, "y": 1, "z": 2}[self.plane_axis], float(self.plane_position))


@dataclass
class FluxMonitor:
    """Time-integrated Poynting flux (J; J/m in 2-D) through an axis-aligned plane.

    The plane is snapped to the nearest tangential-E grid plane. The two
    tangential H components sit half a cell off it, so each is averaged over
    its two half-offset neighbours before forming ``E x H``. This makes the
    one-sided flux second-order accurate in the cell size; pairing the
    samples without averaging biased it by O(dx), e.g. about 5% at 20 cells
    per wavelength.

    H is also half a step off E in time, so each step contributes
    ``E^n x (H^{n-1/2} + H^{n+1/2}) / 2``: the Yee discrete energy flux. The
    band integral of :func:`photonfdtd.photon_spectrum` on the same plane
    reproduces it to round-off.

    Parameters
    ----------
    name : str
        Identifier.
    plane_axis : str
        'x', 'y', or 'z' - the axis normal to the integration plane.
    plane_position : float
        Coordinate along plane_axis where the plane sits (m).
    bounds : optional
        ``(lo, hi)`` (m) per axis, ``None`` for the whole axis, restricting
        the plane to a rectangle (the entry on ``plane_axis`` is ignored).
        Each bound snaps to the nearest node, and the cells on it count half,
        so faces sharing an edge (see :func:`closed_surface`) close exactly.
    """
    name: str
    plane_axis: str
    plane_position: float
    bounds: Optional[Sequence[Optional[Tuple[float, float]]]] = None


@dataclass
class EnergyMonitor:
    """Time series of the electromagnetic energy in a box (J; J/m in 2-D).

    ``U = 1/2 sum (eps |E|^2 + mu |H|^2) dV`` over the cells inside ``bounds``
    (``(lo, hi)`` per axis, ``None`` for the whole axis; default the whole
    grid, PML included), every ``interval`` steps. On a grid with symmetry
    planes it is the whole-domain energy. ``result.energy[name]`` holds the
    values and ``result.monitor_times[name]`` their times. In a lossless,
    source-free box the energy is conserved, so its decay measures loss and
    outflow; a Q factor follows from the decay rate.
    """
    name: str
    interval: int = 1
    bounds: Optional[Sequence[Optional[Tuple[float, float]]]] = None

    def __post_init__(self):
        if int(self.interval) != self.interval or self.interval < 1:
            raise ValueError("interval must be a positive integer")
        self.interval = int(self.interval)


_FACES = ("-x", "+x", "-y", "+y", "-z", "+z")


def closed_surface(name: str, center, size, grid=None):
    """Flux monitors on the faces of a box, for :func:`closed_surface_flux`.

    Returns one bounded :class:`FluxMonitor` per face, named
    ``f"{name}{face}"`` with ``face`` in ``-x, +x, -y, +y, -z, +z``; faces on
    an axis collapsed in ``grid`` (or of zero ``size``) are left out, so in
    2-D a box is four lines. On a grid with symmetry planes pass ``grid``: the
    box must be centred on each plane, its ``-`` face there is the mirror
    image of the ``+`` face and is left out, and :func:`closed_surface_flux`
    counts the ``+`` face twice.
    """
    c = list(center) + [0.0] * (3 - len(center))
    sz = list(size) + [0.0] * (3 - len(size))
    lim = [(c[a] - sz[a] / 2.0, c[a] + sz[a] / 2.0) for a in range(3)]
    sym = getattr(grid, "symmetry", (0, 0, 0)) if grid is not None else (0, 0, 0)
    for a in range(3):
        if sym[a] and abs(c[a]) > 1e-12 * max(abs(sz[a]), 1e-30):
            raise ValueError(f"the box must be centred on the symmetry plane of "
                             f"axis {'xyz'[a]}")
    mons = []
    for face in _FACES:
        a = "xyz".index(face[1])
        if sz[a] == 0 or (grid is not None and grid.shape[a] == 1):
            continue
        if sym[a] and face[0] == "-":
            continue
        pos = lim[a][0] if face[0] == "-" else lim[a][1]
        b = [None if (sz[k] == 0 or (grid is not None and grid.shape[k] == 1))
             else lim[k] for k in range(3)]
        mons.append(FluxMonitor(name=name + face, plane_axis=face[1],
                                plane_position=pos, bounds=b))
    return mons


def closed_surface_flux(result, name: str) -> float:
    """Net outward energy through the faces made by :func:`closed_surface`."""
    total, found = 0.0, False
    for face in _FACES:
        key = name + face
        if key in result.flux:
            found = True
            sign = 1.0 if face[0] == "+" else -1.0
            # A + face without its - partner sits on a symmetric axis, where
            # the - face is its mirror image: count it twice.
            twice = face[0] == "+" and (name + "-" + face[1]) not in result.flux
            total += sign * (2.0 if twice else 1.0) * float(result.flux[key])
    if not found:
        raise KeyError(f"no closed_surface monitors named {name!r} in the result")
    return total
