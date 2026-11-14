"""Photon-count estimates.

Two routes to "how many photons", which can be checked against each other:

- **From a simulation** (:func:`photon_spectrum`): the energy spectrum carried
  through a plane, taken from a plane-restricted :class:`~photonfdtd.DFTMonitor`
  recording the two tangential E and two tangential H components. For real
  fields, Parseval's theorem turns the time-integrated Poynting flux into a
  spectral density::

      U = integral S . n dA dt  =  integral_0^inf  dU/df  df,
      dU/df = 2 Re integral_plane (E(f) x H(f)*) . n dA  *  cos(pi f dt)

  where ``E(f) = sum_n E(t_n) exp(+i 2 pi f t_n) dt`` is exactly what a
  DFTMonitor accumulates. The ``cos(pi f dt)`` factor (1 - O((f dt)^2))
  converts the half-step-timed H to the time-centred pairing of the Yee
  energy flux, so the band integral reproduces a
  :class:`~photonfdtd.FluxMonitor` on the same plane to round-off. Dividing by the photon energy ``h f`` gives the
  photon spectral density ``dN/df``; integrating over a band gives a count.
  Accumulate enough frequencies to resolve the spectrum, since the band
  integral is a trapezoid rule over the monitor's ``freqs``.

- **Analytic Cherenkov references** (:func:`frank_tamm_spectrum`): the
  Frank-Tamm yield of a point charge in 3-D, or the equivalent for a line
  charge in a 2-D simulation (derived below). Both accept the Gaussian cloud
  width of a :class:`~photonfdtd.ChargedParticle`, whose form factor
  band-limits the simulated emission.

Charge and dimensionality
-------------------------
Energy scales with the charge squared, so a run made with an exaggerated
``charge`` for visible fields converts to a physical count with
:meth:`PhotonSpectrum.rescale_charge`. In a 2-D simulation every field is per
metre of the collapsed axis and ``ChargedParticle.charge`` is a line charge
(C/m): a 2-D spectrum is per metre of out-of-plane extent, and there is no
one-electron 2-D count. Use a 3-D run for the yield of a single particle.

2-D line-charge Cherenkov spectrum
----------------------------------
A line charge ``lambda`` (C/m, uniform along z) moving along x at ``v`` in a
medium of index ``n`` drives ``J_x = lambda v delta(x - v t) delta(y)``. Each
temporal frequency ``omega`` has ``k_x = omega / v`` and, above threshold
(``n beta > 1``), a real transverse wavenumber
``k_y = (omega / v) sqrt(n^2 beta^2 - 1)``. Solving for ``H_z`` and taking the
flux through a line parallel to the trajectory gives, per metre of trajectory,
per metre of z, and per side::

    dU / (dx df) = lambda^2 sqrt(n^2 beta^2 - 1) / (2 eps0 n^2 beta c)

This is flat in frequency, whereas the 3-D Frank-Tamm energy spectrum grows
as ``f``. The photon density ``dN / (dx df)`` therefore falls as ``1/f``.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union
import math
import numpy as np

from .constants import C_0, EPS_0, MU_0, Q_E
from .sources import PLANCK_H
from .monitors import DFTMonitor

HBAR = PLANCK_H / (2.0 * math.pi)
ALPHA = Q_E ** 2 / (4.0 * math.pi * EPS_0 * HBAR * C_0)   # fine-structure constant

# Tangential (E1, E2, H1, H2) for a plane normal to each axis, ordered so that
# (E x H) . n_hat = E1 * H2 - E2 * H1.
_TANGENTIAL = {0: ("Ey", "Ez", "Hy", "Hz"),
               1: ("Ez", "Ex", "Hz", "Hx"),
               2: ("Ex", "Ey", "Hx", "Hy")}
_AXIS = {"x": 0, "y": 1, "z": 2}


@dataclass
class PhotonSpectrum:
    """An energy spectrum and the photon counts that follow from it.

    Attributes
    ----------
    freqs : ndarray
        Frequencies (Hz), ascending.
    energy_density : ndarray
        ``dU/df`` in J/Hz (per metre of out-of-plane extent in 2-D; per metre
        of trajectory as well for the analytic per-length references).
    note : str
        What the numbers are per, for the record.
    """
    freqs: np.ndarray
    energy_density: np.ndarray
    note: str = ""

    def __post_init__(self):
        self.freqs = np.asarray(self.freqs, dtype=float)
        self.energy_density = np.asarray(self.energy_density, dtype=float)
        if self.freqs.shape != self.energy_density.shape:
            raise ValueError("freqs and energy_density must have the same shape")
        order = np.argsort(self.freqs)
        self.freqs = self.freqs[order]
        self.energy_density = self.energy_density[order]

    @property
    def wavelengths(self) -> np.ndarray:
        """Vacuum wavelengths (m) of :attr:`freqs`."""
        return C_0 / self.freqs

    @property
    def photon_density(self) -> np.ndarray:
        """``dN/df`` (photons per Hz): ``energy_density / (h f)``."""
        return self.energy_density / (PLANCK_H * self.freqs)

    @property
    def photon_density_per_wavelength(self) -> np.ndarray:
        """``dN/dlambda`` (photons per metre of vacuum wavelength)."""
        return self.photon_density * self.freqs ** 2 / C_0

    def _band(self, fmin, fmax):
        lo = self.freqs[0] if fmin is None else float(fmin)
        hi = self.freqs[-1] if fmax is None else float(fmax)
        if lo < self.freqs[0] or hi > self.freqs[-1] or lo >= hi:
            raise ValueError(
                f"band [{lo:.4g}, {hi:.4g}] Hz must lie inside the sampled "
                f"range [{self.freqs[0]:.4g}, {self.freqs[-1]:.4g}] Hz")
        inner = self.freqs[(self.freqs > lo) & (self.freqs < hi)]
        return np.concatenate(([lo], inner, [hi]))

    def total_energy(self, fmin: Optional[float] = None,
                     fmax: Optional[float] = None) -> float:
        """Energy (J) in ``[fmin, fmax]`` (default: the sampled range),
        by the trapezoid rule on the sampled spectrum."""
        f = self._band(fmin, fmax)
        return float(np.trapezoid(np.interp(f, self.freqs, self.energy_density), f))

    def total_photons(self, fmin: Optional[float] = None,
                      fmax: Optional[float] = None) -> float:
        """Photon count in ``[fmin, fmax]`` (default: the sampled range)."""
        f = self._band(fmin, fmax)
        return float(np.trapezoid(np.interp(f, self.freqs, self.photon_density), f))

    def total_photons_wavelength(self, lam_min: float, lam_max: float) -> float:
        """Photon count between two vacuum wavelengths (m)."""
        return self.total_photons(C_0 / lam_max, C_0 / lam_min)

    def scaled(self, factor: float) -> "PhotonSpectrum":
        """A copy with the energy (and so the photon count) multiplied by ``factor``."""
        return PhotonSpectrum(self.freqs, self.energy_density * float(factor), self.note)

    def rescale_charge(self, simulated_charge: float,
                       physical_charge: float = Q_E) -> "PhotonSpectrum":
        """Convert a run made with ``simulated_charge`` to ``physical_charge``.

        Maxwell's equations are linear in the source, so radiated energy (and
        the photon count) scales as ``(physical_charge / simulated_charge)**2``.
        """
        return self.scaled((physical_charge / simulated_charge) ** 2)

    def __add__(self, other: "PhotonSpectrum") -> "PhotonSpectrum":
        """Sum two spectra on the same frequencies (e.g. the faces of a box)."""
        if not np.array_equal(self.freqs, other.freqs):
            raise ValueError("spectra must share the same frequencies")
        return PhotonSpectrum(self.freqs, self.energy_density + other.energy_density,
                              self.note)


def photon_count(energy: float, freq: float) -> float:
    """Photons in a narrow-band energy ``energy`` (J) at ``freq`` (Hz): ``U / (h f)``.

    Pairs with a :class:`~photonfdtd.FluxMonitor`, whose result is the
    time-integrated flux (J), for a source whose bandwidth is small compared
    with ``freq``. For a broadband source use :func:`photon_spectrum`.
    """
    return float(energy) / (PLANCK_H * float(freq))


def photon_spectrum(result, sim, monitor: str, direction: int = +1,
                    bounds: Optional[Sequence[Optional[Tuple[float, float]]]] = None,
                    ) -> PhotonSpectrum:
    """Energy and photon spectrum through the plane of a DFT monitor.

    Parameters
    ----------
    result : Result
        Output of ``sim.run()``.
    sim : Simulation
        The simulation that produced it (supplies the grid and the monitor).
    monitor : str
        Name of a :class:`~photonfdtd.DFTMonitor` with ``plane_axis`` set,
        recording the plane's two tangential E and two tangential H components
        (``('Ey','Ez','Hy','Hz')`` for an x-normal plane, ``('Ez','Ex','Hz','Hx')``
        for y, ``('Ex','Ey','Hx','Hy')`` for z). On a 2-D grid record them all;
        the out-of-plane pair is simply zero.
    direction : +1 or -1
        Count flux along the positive (``+1``) or negative (``-1``) normal.
    bounds : optional
        ``(lo, hi)`` (m) per axis (``None`` for no limit) restricting the
        integration to part of the plane, e.g. ``[(-5e-6, 5e-6), None, None]``.

    Returns
    -------
    PhotonSpectrum
        ``dU/df`` at the monitor frequencies. Negative values mean net flux
        against ``direction``. The monitor records tangential H averaged onto
        the E plane (see :class:`~photonfdtd.DFTMonitor`), as
        :class:`~photonfdtd.FluxMonitor` does.

    Notes
    -----
    On a grid with symmetry planes the result is the whole plane's (the
    stepped part times two per symmetric in-plane axis). For the complex
    fields of a Bloch-periodic run the absolute scale assumes real fields, so
    use ratios (transmission against an incident run) there.
    """
    m = next((mm for mm in sim.monitors if mm.name == monitor), None)
    if not isinstance(m, DFTMonitor):
        raise ValueError(f"{monitor!r} is not a DFTMonitor of this simulation")
    pl = m.plane()
    if pl is None:
        raise ValueError(f"DFTMonitor {monitor!r} needs plane_axis/plane_position")
    if direction not in (1, -1):
        raise ValueError("direction must be +1 or -1")
    axis = pl[0]
    names = _TANGENTIAL[axis]
    missing = [c for c in names if c not in m.components]
    if missing:
        raise ValueError(f"DFTMonitor {monitor!r} must record {names}; "
                         f"missing {tuple(missing)}")
    data = result.dft[monitor]
    E1, E2, H1, H2 = (np.asarray(data[c]) for c in names)
    flux = (E1 * H2.conj() - E2 * H1.conj()).real        # (n_freq, *snap)

    grid = sim.grid
    ds = m.downsample
    dA = 1.0
    mask = np.ones(flux.shape[1:], dtype=bool)
    for a in range(3):
        if a == axis or grid.shape[a] == 1:
            continue
        dA *= grid.cell_size[a] * ds
        if bounds is not None and bounds[a] is not None:
            lo, hi = bounds[a]
            c = np.asarray(grid.coords[a])[::ds]
            sel = (c >= lo) & (c <= hi)
            shp = [1, 1, 1]
            shp[a] = -1
            mask = mask & sel.reshape(shp)
    sym = getattr(grid, "symmetry", (0, 0, 0))
    for a in range(3):
        if sym[a] and a != axis:
            if bounds is not None and bounds[a] is not None:
                raise ValueError(
                    f"bounds on symmetric axis {a} are ambiguous (the stepped "
                    "half is mirrored); leave that axis unbounded")
            dA *= 2.0                             # the mirrored half too
    dUdf = 2.0 * direction * (flux * mask).reshape(flux.shape[0], -1).sum(axis=1) * dA
    # The DFT samples H at its own half-step times. The Yee discrete energy
    # flux pairs E^n with (H^{n-1/2} + H^{n+1/2}) / 2, whose transform is that
    # H times cos(pi f dt); with it the band integral equals the time-centred
    # FluxMonitor energy to round-off. The factor -> 1 as dt -> 0.
    freqs = np.asarray(result.dft_freqs[monitor], dtype=float)
    dUdf = dUdf * np.cos(np.pi * freqs * sim.dt)

    ndim = sum(s > 1 for s in grid.shape)
    note = "per metre of out-of-plane extent" if ndim == 2 else (
        "per square metre of transverse area" if ndim == 1 else "")
    return PhotonSpectrum(freqs, dUdf, note)


def frank_tamm_spectrum(freqs: Sequence[float],
                        n: Union[float, Callable[[np.ndarray], np.ndarray]],
                        beta: float, length: float = 1.0,
                        charge: float = Q_E, ndim: int = 3,
                        cloud_sigma: float = 0.0, sides: int = 2,
                        ) -> PhotonSpectrum:
    r"""Analytic Cherenkov energy spectrum of a uniformly moving charge.

    Parameters
    ----------
    freqs : array_like
        Frequencies (Hz).
    n : float or callable
        Refractive index, or a function ``n(freqs)`` for a dispersive medium.
        Frequencies below threshold (``n beta <= 1``) contribute zero.
    beta : float
        Particle speed over ``c``.
    length : float
        Path length (m) in the radiator.
    charge : float
        Point charge (C) for ``ndim=3``; line charge (C/m) for ``ndim=2``.
    ndim : 3 or 2
        3: Frank-Tamm, ``d^2N/(dx df) = (2 pi alpha / c) (q/e)^2 (1 - 1/(n beta)^2)``.
        2: line charge, energy per metre of out-of-plane extent (module docstring).
    cloud_sigma : float
        Standard deviation (m) of the Gaussian charge cloud
        (``ChargedParticle.cloud_cells * cell_size``). The cloud's form factor
        scales the energy by ``exp(-(n omega sigma / c)^2)``. 0 for a point charge.
    sides : 1 or 2
        2-D only: count both half-planes (2) or one (1). In 3-D the full cone
        is counted.
    """
    f = np.asarray(freqs, dtype=float)
    nn = np.asarray(n(f) if callable(n) else np.full_like(f, float(n)), dtype=float)
    nb = nn * float(beta)
    above = nb > 1.0
    if ndim == 3:
        x = np.where(above, 1.0 - 1.0 / np.where(above, nb, 1.0) ** 2, 0.0)
        dNdf = 2.0 * math.pi * ALPHA / C_0 * (charge / Q_E) ** 2 * x * length
        dUdf = dNdf * PLANCK_H * f
        note = f"over {length:g} m of path"
    elif ndim == 2:
        if sides not in (1, 2):
            raise ValueError("sides must be 1 or 2")
        root = np.sqrt(np.where(above, nb ** 2 - 1.0, 0.0))
        dUdf = (sides * charge ** 2 * root / (2.0 * EPS_0 * nn ** 2 * beta * C_0)
                * length)
        note = (f"over {length:g} m of path, per metre of out-of-plane extent, "
                f"{sides} side(s)")
    else:
        raise ValueError("ndim must be 2 or 3")
    if cloud_sigma > 0.0:
        dUdf = dUdf * np.exp(-(nn * 2.0 * math.pi * f * cloud_sigma / C_0) ** 2)
    return PhotonSpectrum(f, dUdf, note)


def frank_tamm_photons(n, beta: float, length: float,
                       wavelength_range: Tuple[float, float],
                       charge: float = Q_E, num: int = 2001) -> float:
    """Frank-Tamm photon count of a point charge over a vacuum-wavelength band.

    For constant ``n`` this is the textbook
    ``N = 2 pi alpha z^2 L (1 - 1/(n beta)^2) (1/lam_min - 1/lam_max)``;
    a callable ``n(freqs)`` is integrated numerically on ``num`` points.
    """
    lam_min, lam_max = sorted(float(v) for v in wavelength_range)
    f = np.linspace(C_0 / lam_max, C_0 / lam_min, num)
    return frank_tamm_spectrum(f, n, beta, length=length,
                               charge=charge).total_photons()
