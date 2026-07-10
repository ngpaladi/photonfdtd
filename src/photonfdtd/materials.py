"""Material definitions.

v0.1 supports isotropic, non-dispersive dielectrics. Each Medium is reduced to
a single relative permittivity that the grid rasteriser stamps into a per-cell
epsilon_r array. Magnetic permeability is fixed at mu_r = 1.

Dispersive media (Lorentz, Drude, Debye) are planned for a future release via
the auxiliary-differential-equation method; the API is left open so a Lorentz
class can slot in without breaking callers.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar, Sequence, Tuple
import math


@dataclass(frozen=True)
class Medium:
    """Isotropic, non-dispersive dielectric.

    Parameters
    ----------
    permittivity : float
        Relative permittivity eps_r at the simulation's centre frequency.
        Pass eps_r = n^2 for a refractive index n.
    name : str, optional
        Identifier, used only for logging.
    """
    permittivity: float = 1.0
    name: str = ""

    #: Marks this as a plain non-dispersive dielectric (see DispersiveMedium).
    is_dispersive: ClassVar[bool] = False

    @property
    def eps_r(self) -> float:
        return float(self.permittivity)

    @classmethod
    def from_index(cls, n: float, name: str = "") -> "Medium":
        return cls(permittivity=float(n) ** 2, name=name)


VACUUM = Medium(permittivity=1.0, name="vacuum")


# --------------------------------------------------------------------------- #
# Dispersive media.
#
# A dispersive material is written as a background permittivity eps_inf plus a
# sum of poles, each contributing a susceptibility
#
#     chi_p(omega) = strength / (omega0**2 - omega**2 - 2j*gamma*omega)
#
# so that  eps_r(omega) = eps_inf + sum_p chi_p(omega).  This single "pole"
# form spans every model this package ships:
#
#   * Lorentz resonance:  strength = delta_eps * omega0**2,  gamma = half-width.
#   * Drude free-electron: omega0 = 0,  strength = omega_p**2,  gamma = Gamma/2.
#   * Lossless Sellmeier term  B*lam**2/(lam**2 - C):  it is exactly a lossless
#     Lorentz pole with delta_eps = B, omega0 = 2*pi*c/sqrt(C), gamma = 0,
#     because  lam**2/(lam**2 - C) = omega0**2/(omega0**2 - omega**2).
#
# Each pole is advanced during time stepping by the auxiliary-differential-
# equation (ADE) method (Taflove & Hagness, *Computational Electrodynamics*,
# 3rd ed., ch. 9): the polarization P_p obeys
#     d2P/dt2 + 2*gamma dP/dt + omega0**2 P = eps_0 * strength * E,
# central-differenced to the explicit recursion built in Simulation.
# --------------------------------------------------------------------------- #

# Photon energy (eV) -> angular frequency (rad/s):  omega = (e/hbar) * E_eV.
_EV_TO_RAD_S = 1.519_267_447e15
# Speed of light, m/s (kept local so materials.py has no cross-import).
_C0 = 299_792_458.0


@dataclass(frozen=True)
class Pole:
    """One dispersion pole  strength / (omega0**2 - omega**2 - 2j*gamma*omega).

    strength, omega0, gamma are all in SI angular-frequency units:
    ``strength`` has units of rad**2/s**2, ``omega0`` and ``gamma`` rad/s.
    A Drude pole has ``omega0 == 0``; a lossless (Sellmeier) pole ``gamma == 0``.
    """
    strength: float
    omega0: float
    gamma: float = 0.0


@dataclass(frozen=True)
class DispersiveMedium:
    """Isotropic dispersive medium: eps_inf plus a sum of poles.

    Stamps ``eps_inf`` into the permittivity grid (that sets the instantaneous
    E-update coefficient); the poles are stepped by the ADE recursion inside
    :class:`~photonfdtd.simulation.Simulation`, which activates automatically
    when any structure uses a dispersive medium.
    """
    eps_inf: float
    poles: Tuple[Pole, ...] = ()
    name: str = ""

    is_dispersive: ClassVar[bool] = True

    @property
    def eps_r(self) -> float:
        # Instantaneous (high-frequency) permittivity used to stamp the grid.
        return float(self.eps_inf)

    def eps_model(self, freq_hz):
        """Complex relative permittivity at frequency ``freq_hz`` (Hz).

        Uses the e^{-i*omega*t} convention (loss => positive imaginary part),
        matching the DFT monitor. Accepts scalars or arrays.
        """
        import numpy as _np
        w = 2.0 * _np.pi * _np.asarray(freq_hz, dtype=float)
        eps = _np.full(w.shape, complex(self.eps_inf)) if w.ndim else complex(self.eps_inf)
        for p in self.poles:
            eps = eps + p.strength / (p.omega0 ** 2 - w ** 2 - 2j * p.gamma * w)
        return eps

    def index(self, freq_hz):
        """Complex refractive index n = sqrt(eps_model)."""
        import numpy as _np
        return _np.sqrt(self.eps_model(freq_hz))

    def at_wavelength(self, wavelength_m: float, name: str = "") -> "Medium":
        """A plain non-dispersive :class:`Medium` matching this model's real
        permittivity at one wavelength.

        Use this when a full dispersive time-stepping run is not warranted or
        not stable at the chosen resolution (see the pole-stability note in
        Simulation): it captures the correct index at ``wavelength_m`` with no
        poles to resolve, which is exactly what a mode solve or a narrowband
        simulation needs.
        """
        import numpy as _np
        eps = float(_np.real(self.eps_model(_C0 / float(wavelength_m))))
        return Medium(permittivity=eps, name=name or f"{self.name}@{wavelength_m*1e9:.0f}nm")

    def max_pole_omega(self) -> float:
        """Largest pole resonance frequency omega0 (rad/s); sets the ADE dt limit."""
        return max((p.omega0 for p in self.poles), default=0.0)

    # ---- constructors --------------------------------------------------- #
    @classmethod
    def lorentz(cls, eps_inf: float,
                resonances: Sequence[Tuple[float, float, float]],
                name: str = "") -> "DispersiveMedium":
        """Build from Lorentz resonances ``(delta_eps, f0_Hz, delta_f_Hz)``.

        ``f0_Hz`` is the resonance frequency, ``delta_f_Hz`` the (full) linewidth
        in Hz (gamma = pi*delta_f). Set ``delta_f_Hz = 0`` for a lossless pole.
        """
        poles = []
        for delta_eps, f0, df in resonances:
            w0 = 2.0 * math.pi * f0
            gamma = math.pi * df                     # 2*gamma = 2*pi*df
            poles.append(Pole(strength=delta_eps * w0 ** 2, omega0=w0, gamma=gamma))
        return cls(eps_inf=float(eps_inf), poles=tuple(poles), name=name)

    @classmethod
    def drude(cls, eps_inf: float, f_plasma_Hz: float, f_collision_Hz: float,
              name: str = "") -> "DispersiveMedium":
        """Build a Drude metal from plasma and collision frequencies (Hz)."""
        wp = 2.0 * math.pi * f_plasma_Hz
        gamma = math.pi * f_collision_Hz             # 2*gamma = 2*pi*f_collision
        return cls(eps_inf=float(eps_inf),
                   poles=(Pole(strength=wp ** 2, omega0=0.0, gamma=gamma),),
                   name=name)

    @classmethod
    def sellmeier(cls, terms: Sequence[Tuple[float, float]],
                  name: str = "") -> "DispersiveMedium":
        """Build from Sellmeier terms ``(B_i, C_i)`` of n^2 - 1 = sum B_i lam^2/(lam^2 - C_i).

        ``C_i`` is in micrometres squared (the standard tabulated form); each
        term becomes a lossless Lorentz pole. eps_inf = 1.
        """
        poles = []
        for B, C_um2 in terms:
            lam0 = math.sqrt(C_um2) * 1e-6           # resonance wavelength, m
            w0 = 2.0 * math.pi * _C0 / lam0
            poles.append(Pole(strength=B * w0 ** 2, omega0=w0, gamma=0.0))
        return cls(eps_inf=1.0, poles=tuple(poles), name=name)


def _rakic_ld(eps_inf: float, wp_eV: float,
              table: Sequence[Tuple[float, float, float]],
              name: str) -> DispersiveMedium:
    """Rakic et al. (1998) Lorentz-Drude model from (f_j, Gamma_j, omega_j) in eV.

    Row 0 is the Drude term (omega_j == 0). See docs/source/material_data.md.
    """
    wp = wp_eV * _EV_TO_RAD_S
    poles = []
    for f_j, Gamma_j, w_j in table:
        strength = f_j * wp ** 2
        omega0 = w_j * _EV_TO_RAD_S
        gamma = 0.5 * Gamma_j * _EV_TO_RAD_S         # 2*gamma = Gamma_j
        poles.append(Pole(strength=strength, omega0=omega0, gamma=gamma))
    return DispersiveMedium(eps_inf=float(eps_inf), poles=tuple(poles), name=name)


# --------------------------------------------------------------------------- #
# Cited material library. Full sources and parameters: docs/source/material_data.md
# --------------------------------------------------------------------------- #
def silica() -> DispersiveMedium:
    """Fused silica SiO2 (Malitson 1965 Sellmeier). Valid 0.21-6.7 um."""
    return DispersiveMedium.sellmeier(
        [(0.6961663, 0.0684043 ** 2),
         (0.4079426, 0.1162414 ** 2),
         (0.8974794, 9.896161 ** 2)],
        name="SiO2 (Malitson 1965)")


def silicon() -> DispersiveMedium:
    """Crystalline silicon (Salzberg & Villa 1957 Sellmeier). Valid 1.36-11 um."""
    return DispersiveMedium.sellmeier(
        [(10.6684293, 0.301516485 ** 2),
         (0.0030434748, 1.13475115 ** 2),
         (1.54133408, 1104.0 ** 2)],
        name="Si (Salzberg-Villa 1957)")


def silicon_nitride() -> DispersiveMedium:
    """Stoichiometric LPCVD Si3N4 (Luke et al. 2015 Sellmeier). Valid 0.31-5.5 um."""
    return DispersiveMedium.sellmeier(
        [(3.0249, 0.1353406 ** 2),
         (40314.0, 1239.842 ** 2)],
        name="Si3N4 (Luke 2015)")


def lithium_niobate(axis: str = "o") -> DispersiveMedium:
    """Congruent undoped LiNbO3 (Zelmon et al. 1997). axis='o' or 'e'. 0.4-5 um."""
    if axis == "o":
        terms = [(2.6734, 0.01764), (1.2290, 0.05914), (12.614, 474.60)]
    elif axis == "e":
        terms = [(2.9804, 0.02047), (0.5981, 0.0666), (8.9543, 416.08)]
    else:
        raise ValueError("axis must be 'o' (ordinary) or 'e' (extraordinary)")
    return DispersiveMedium.sellmeier(terms, name=f"LiNbO3-{axis} (Zelmon 1997)")


def gold() -> DispersiveMedium:
    """Gold, Rakic et al. (1998) Lorentz-Drude model."""
    # (f_j, Gamma_j [eV], omega_j [eV]); row 0 is the Drude term.
    return _rakic_ld(1.0, 9.03, [
        (0.760, 0.053, 0.000),
        (0.024, 0.241, 0.415),
        (0.010, 0.345, 0.830),
        (0.071, 0.870, 2.969),
        (0.601, 2.494, 4.304),
        (4.384, 2.214, 13.32),
    ], name="Au (Rakic 1998 LD)")


def silver() -> DispersiveMedium:
    """Silver, Rakic et al. (1998) Lorentz-Drude model."""
    return _rakic_ld(1.0, 9.01, [
        (0.845, 0.048, 0.000),
        (0.065, 3.886, 0.816),
        (0.124, 0.452, 4.481),
        (0.011, 0.065, 8.185),
        (0.840, 0.916, 9.083),
        (5.646, 2.419, 20.29),
    ], name="Ag (Rakic 1998 LD)")
