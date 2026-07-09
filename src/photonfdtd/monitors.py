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
        axis. Snapshots are stored in the simulation's ``'monitors'`` dtype
        (``precision={'monitors': 'float32'}`` halves snapshot memory even on a
        float64 run).
    plane_z : float, optional
        If given, record only the single z-plane nearest this z-coordinate (m)
        instead of the whole volume. The stored array keeps a size-1 z axis so
        downstream indexing is unchanged. For a top-down field movie this cuts
        snapshot memory — and the GPU host transfer — by the full z-cell count
        (often 30-100x), since only the one displayed plane is kept.
    compression : {None, True, 'zstd', 'zlib'}, optional
        If set, stream snapshots to a temporary file as lossily-compressed
        frames instead of keeping them all in RAM. Each frame is stored as a
        per-frame float16-scaled, then compressed, block (see
        :class:`photonfdtd.storage.CompressedFieldSeries`), which keeps the
        run's memory footprint flat for arbitrarily long/large recordings and
        typically shrinks stored data ~10-20x versus an uncompressed float64
        monitor. ``True`` selects zstd if the ``zstandard`` package is present
        and falls back to stdlib zlib; ``'zstd'`` requires that package;
        ``'zlib'`` is always available. The result
        (``result.fields[name][component]``) is a lazily-decompressing,
        array-like series rather than a plain ``ndarray``: ``series[i]`` reads
        one frame, ``np.asarray(series)`` materialises all of them. This mode is
        lossy; use ``compression=None`` for exact snapshots.
    compression_bits : {8, 16}, optional
        Quantisation depth for the compressed mode (default 8). ``8`` (int8)
        gives ~10-30x reduction versus an uncompressed float64 monitor at
        ~4e-3 relative error - it comfortably meets a 10x target and suits
        field movies / visualisation. ``16`` (float16) trades ratio for
        fidelity (~5-7x, ~2e-4 relative error). Ignored when
        ``compression`` is None.
    name : str
        Identifier used to retrieve results from sim.run().
    """
    name: str
    components: Tuple[str, ...] = ("Ez",)
    interval: Optional[int] = None
    times: Optional[Sequence[float]] = None
    downsample: int = 1
    plane_z: Optional[float] = None
    compression: object = None
    compression_bits: int = 8

    def __post_init__(self):
        if self.interval is None and self.times is None:
            self.interval = 1
        if self.interval is not None and self.times is not None:
            raise ValueError("Specify only one of interval or times")
        if int(self.downsample) != self.downsample or self.downsample < 1:
            raise ValueError("downsample must be a positive integer")
        self.downsample = int(self.downsample)
        if self.compression is not None and \
                self.compression not in (True, "auto", "zstd", "zlib"):
            raise ValueError(
                "compression must be None, True, 'zstd', or 'zlib', got "
                f"{self.compression!r}"
            )
        if self.compression_bits not in (8, 16):
            raise ValueError("compression_bits must be 8 or 16")


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
