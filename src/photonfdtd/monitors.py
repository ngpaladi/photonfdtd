"""Monitors record field data during the time-stepping loop.

v0.1 ships two:
    FieldMonitor - take full-domain field snapshots at chosen times
    FluxMonitor  - integrate Poynting flux through an axis-aligned plane

The FluxMonitor in this release evaluates the time-averaged real Poynting
flux over the whole run. For frequency-resolved analysis run a discrete
Fourier transform over the recorded field samples in post-processing.
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
        ``downsample**ndim`` and, on the GPU backend, the host transfer too.
        The stored array aligns to ``grid.coords[axis][::downsample]`` on each
        axis. Snapshots are stored in the simulation's working dtype.
    plane_z : float, optional
        If given, record only the single z-plane nearest this z-coordinate (m)
        instead of the whole volume. The stored array keeps a size-1 z axis so
        downstream indexing is unchanged. For a top-down field movie this cuts
        snapshot memory — and the GPU host transfer — by the full z-cell count
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
class FluxMonitor:
    """Integrate time-averaged Poynting flux through an axis-aligned plane.

    Parameters
    ----------
    name : str
        Identifier.
    plane_axis : str
        'x', 'y', or 'z' - the axis normal to the integration plane.
    plane_position : float
        Coordinate along plane_axis where the plane sits (m).
    """
    name: str
    plane_axis: str
    plane_position: float
