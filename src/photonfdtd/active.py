"""Active devices: index changes from voltages, heaters and carriers.

This module turns the electrical and thermal state of a layout into the
optical permittivity the mode solver and the FDTD engine see, so a voltage or
heater sweep can be simulated end to end:

1. **Materials** (:class:`ActiveMedium`) carry, next to their optical
   permittivity, the DC permittivity, thermal conductivity, thermo-optic
   coefficient and (for crystals) the Pockels tensor and crystal orientation.
   The library has thin-film lithium niobate (x- and z-cut), lithium
   tantalate, barium titanate, an electro-optic polymer, Si, SiN, SiO2, air
   and heater metals.
2. **Drives**: :class:`Electrode` (an equipotential conductor at a voltage),
   :class:`Heater` (a resistor dissipating a power) and :class:`PNJunction` (a
   reverse-biased silicon junction whose depletion width changes the free
   carrier density).
3. **Statics**: :class:`ActiveRegion` solves the electrostatic potential
   and the steady-state temperature (:mod:`photonfdtd.statics`) once per
   electrode and heater, on the cross-section of a
   :class:`~photonfdtd.mode.ModeSolver` or the grid of a
   :class:`~photonfdtd.simulation.Simulation`.
4. **Optics**: the prepared region (:class:`PreparedActive`) gives the
   permittivity for any voltages and powers, as a mode solver, as E-update
   coefficients for :meth:`Simulation.run(ce=...) <photonfdtd.simulation.Simulation.run>`,
   as a JAX-differentiable coefficient builder for the design-gradient tools,
   or as a time-varying drive (RF modulation).

Physics and limits
------------------
* **Pockels effect.** In the crystal frame the index ellipsoid changes by
  ``d(1/n^2)_i = sum_j r_ij E_j`` (Voigt notation, i = 1..6). Only the
  diagonal terms (i = 1, 2, 3) are applied, as ``d eps_ii = -eps_ii^2
  d(1/n^2)_i``: the solvers use a diagonal permittivity tensor, so the crystal
  axes must lie along grid axes. The off-diagonal terms are measured and
  reported (:attr:`PreparedActive.offdiagonal_ratio`); a warning is raised when
  they are not small. Barium titanate's large r42 is off-diagonal, so it is
  not captured.
* **Linearity.** The potential is linear in the electrode voltages, so the
  Pockels change is too: one unit solve per electrode gives the permittivity
  for any voltages, exactly. The thermal change is linear in the heater powers
  in the same way. The free-carrier change of a PN junction is not (the
  depletion width grows as the square root of the bias), so it is evaluated
  directly at each bias.
* **Thermo-optic effect.** ``d eps = 2 n (dn/dT) dT`` per axis.
* **Free-carrier (plasma) dispersion in silicon.** Soref and Bennett (1987)
  fits at 1310 and 1550 nm give the index change and the absorption from the
  electron and hole densities. The absorption enters FDTD as a conductivity.
  Carriers come from an abrupt-junction depletion model, not a drift-diffusion
  solve.
* **Time-varying drives** (:meth:`PreparedActive.modulation`) use an
  instantaneous permittivity, valid when the drive changes slowly compared
  with the optical period. Note the cost: a GHz drive has a period of
  millions of FDTD steps, so time-domain modulation is practical for short
  pulses or for fast (THz-scale) demonstrations. For switching at DC use
  voltage sweeps.
* **The 2-D (effective-index) chip plane** can't see the vertical structure
  that sets a modulator's efficiency (film thickness, electrode height, mode
  overlap). For a 2-D FDTD run, prepare the region on the waveguide's
  cross-section with a ModeSolver and transfer the effective-index change to
  the chip plane with :meth:`PreparedActive.eim`.

Units are SI throughout: metres, volts, watts, kelvin, and m/V for the
electro-optic coefficients (1 pm/V = 1e-12 m/V).
"""
from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .constants import C_0, EPS_0, Q_E
from .materials import AnisotropicMedium, eps_components_of
from . import statics

_PM = 1e-12          # pm/V in m/V
_K_B = 1.380649e-23  # J/K

# --------------------------------------------------------------------------- #
# Materials.
# --------------------------------------------------------------------------- #
_AXES = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class ActiveMedium(AnisotropicMedium):
    """A medium with electrical and thermal properties.

    It is an :class:`~photonfdtd.materials.AnisotropicMedium`, so it goes
    straight into ``structures`` for the mode solver and FDTD. Tensor
    quantities are diagonal and given in the *grid* frame (x, y, z).

    Parameters
    ----------
    permittivity : (eps_xx, eps_yy, eps_zz)
        Optical relative permittivity at the design wavelength.
    eps_dc : (3,) or None
        Low-frequency relative permittivity for the electrostatic solve
        (default: the optical permittivity).
    thermal_conductivity : float or (3,), W/(m K), or None
        None means "not a thermal material": a thermal solve raises.
    dn_dT : (3,), 1/K
        Thermo-optic coefficient of the index along each axis.
    r : 6x3 nested tuple (m/V) or None
        Pockels tensor in the crystal frame, Voigt rows (11, 22, 33, 23, 13, 12)
        and columns E1..E3.
    crystal_axes : (3,) ints
        The grid axis each crystal axis 1, 2, 3 lies along; e.g. ``(2, 0, 1)``
        puts crystal X along grid z, Y along x and Z along y (x-cut, light
        along crystal Y = grid x).
    free_carriers : bool
        Silicon free-carrier dispersion applies (see :class:`PNJunction`).
    conductivity : float
        Optical-frequency conductivity (S/m); for metals the DC value makes them
        near-perfect conductors in FDTD.
    """
    eps_dc: Optional[Tuple[float, float, float]] = None
    thermal_conductivity: Optional[object] = None
    dn_dT: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    r: Optional[Tuple[Tuple[float, float, float], ...]] = None
    crystal_axes: Tuple[int, int, int] = (0, 1, 2)
    free_carriers: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.eps_dc is not None:
            object.__setattr__(self, "eps_dc", _three(self.eps_dc))
        object.__setattr__(self, "dn_dT", _three(self.dn_dT))
        ax = tuple(int(a) for a in self.crystal_axes)
        if sorted(ax) != [0, 1, 2]:
            raise ValueError("crystal_axes must be a permutation of (0, 1, 2)")
        object.__setattr__(self, "crystal_axes", ax)
        if self.r is not None:
            r = np.asarray(self.r, float)
            if r.shape != (6, 3):
                raise ValueError("r must be a 6x3 Pockels tensor (Voigt rows)")
            object.__setattr__(self, "r", tuple(tuple(row) for row in r))

    @property
    def dc_permittivity(self) -> Tuple[float, float, float]:
        return self.eps_dc if self.eps_dc is not None else self.permittivity

    @property
    def thermal(self) -> Optional[Tuple[float, float, float]]:
        k = self.thermal_conductivity
        return None if k is None else _three(k)


def _three(v) -> Tuple[float, float, float]:
    a = np.broadcast_to(np.asarray(v, float), (3,))
    return tuple(float(x) for x in a)


def _optical_eps(medium, wavelength) -> Tuple[float, float, float]:
    if getattr(medium, "is_dispersive", False):
        e = float(np.real(medium.eps_model(C_0 / wavelength)))
        return (e, e, e)
    return eps_components_of(medium)


def with_response(medium, wavelength: Optional[float] = None, *, eps_dc=None,
                  thermal_conductivity=None, dn_dT=0.0, conductivity=None,
                  name: str = "") -> ActiveMedium:
    """An :class:`ActiveMedium` with ``medium``'s optical permittivity (a
    dispersive medium is evaluated at ``wavelength``) and the given electrical
    and thermal properties."""
    if getattr(medium, "is_dispersive", False) and wavelength is None:
        raise ValueError("a dispersive medium needs a wavelength")
    return ActiveMedium(
        permittivity=_optical_eps(medium, wavelength), name=name or medium.name,
        conductivity=float(conductivity if conductivity is not None
                           else getattr(medium, "conductivity", 0.0)),
        eps_dc=eps_dc, thermal_conductivity=thermal_conductivity, dn_dT=dn_dT)


def _cut_axes(cut: str, propagation: str, vertical: str = "z") -> Tuple[int, int, int]:
    """Crystal axes (X, Y, Z) -> grid axes for a wafer cut.

    ``cut`` is the crystal axis normal to the wafer (grid ``vertical``). For an
    x-cut the light travels along crystal Y (grid ``propagation``) and the optic
    axis Z is in-plane and transverse; for a z-cut the optic axis is vertical
    and the light travels along crystal X."""
    p, v = _AXES[propagation], _AXES[vertical]
    if p == v:
        raise ValueError("propagation and vertical axes must differ")
    t = 3 - p - v                                       # in-plane transverse
    if cut == "x":
        return (v, p, t)
    if cut == "y":
        return (p, v, t)
    if cut == "z":
        return (p, t, v)
    raise ValueError("cut must be 'x', 'y' or 'z'")


def _uniaxial(n_o, n_e, axes, r, eps_dc_c, dn_dT_c, k, name) -> ActiveMedium:
    """Grid-frame ActiveMedium from crystal-frame uniaxial data (Z = optic axis)."""
    eps_c = (n_o ** 2, n_o ** 2, n_e ** 2)
    perm = [0.0] * 3; dc = [0.0] * 3; dndt = [0.0] * 3
    for i, g in enumerate(axes):
        perm[g] = eps_c[i]; dc[g] = eps_dc_c[i]; dndt[g] = dn_dT_c[i]
    return ActiveMedium(permittivity=tuple(perm), name=name, eps_dc=tuple(dc),
                        thermal_conductivity=k, dn_dT=tuple(dndt), r=r,
                        crystal_axes=axes)


def _r_3m(r13, r22, r33, r51):
    """Pockels tensor of point group 3m (LiNbO3, LiTaO3), Voigt rows."""
    return ((0.0, -r22, r13), (0.0, r22, r13), (0.0, 0.0, r33),
            (0.0, r51, 0.0), (r51, 0.0, 0.0), (-r22, 0.0, 0.0))


def _r_4mm(r13, r33, r42):
    """Pockels tensor of point group 4mm (tetragonal BaTiO3), Voigt rows."""
    return ((0.0, 0.0, r13), (0.0, 0.0, r13), (0.0, 0.0, r33),
            (0.0, r42, 0.0), (r42, 0.0, 0.0), (0.0, 0.0, 0.0))


def lithium_niobate_eo(wavelength: float, cut: str = "x", propagation: str = "x",
                       vertical: str = "z", clamped: bool = True) -> ActiveMedium:
    """Congruent LiNbO3 (thin film or bulk), electro-optic and thermo-optic.

    Indices from the Zelmon (1997) Sellmeier at ``wavelength``. Pockels
    coefficients and DC permittivity from Weis & Gaylord (1985): ``clamped``
    (the default; above ~10 MHz, what a modulator sees) r13, r22, r33, r51 =
    8.6, 3.4, 30.8, 28 pm/V and eps_11, eps_33 = 44, 27.9; unclamped (DC)
    10, 6.8, 32.2, 32 pm/V and 85, 28.7. dn_e/dT = 3.34e-5 /K, dn_o/dT ~ 0 at
    room temperature in the near infrared; thermal conductivity 4.6 W/(m K).
    See ``docs/material_data.md``.

    ``cut`` / ``propagation`` / ``vertical`` place the crystal on the grid; an
    x-cut film with light along grid x has its optic axis (extraordinary
    index, r33) along grid y, so a TE mode (Ey) between two lateral electrodes
    uses r33.
    """
    from .materials import lithium_niobate
    f = C_0 / wavelength
    n_o = float(np.sqrt(np.real(lithium_niobate("o").eps_model(f))))
    n_e = float(np.sqrt(np.real(lithium_niobate("e").eps_model(f))))
    if clamped:
        r = _r_3m(8.6 * _PM, 3.4 * _PM, 30.8 * _PM, 28.0 * _PM)
        e11, e33 = 44.0, 27.9
    else:
        r = _r_3m(10.0 * _PM, 6.8 * _PM, 32.2 * _PM, 32.0 * _PM)
        e11, e33 = 85.0, 28.7
    return _uniaxial(n_o, n_e, _cut_axes(cut, propagation, vertical), r,
                     (e11, e11, e33), (0.0, 0.0, 3.34e-5), 4.6,
                     f"LiNbO3 {cut}-cut")


def lithium_tantalate_eo(wavelength: float = 1.55e-6, cut: str = "x",
                         propagation: str = "x", vertical: str = "z") -> ActiveMedium:
    """LiTaO3: n_o = 2.119, n_e = 2.123 near 1.55 um, r13 = 8.4, r33 = 30.5
    pm/V (clamped), eps_11, eps_33 = 41, 43, thermal conductivity 4.6 W/(m K).
    Indices are fixed values for the telecom band (``wavelength`` is recorded
    in the name only); dn/dT is not included."""
    return _uniaxial(2.119, 2.123, _cut_axes(cut, propagation, vertical),
                     _r_3m(8.4 * _PM, 0.0, 30.5 * _PM, 20.0 * _PM),
                     (41.0, 41.0, 43.0), (0.0, 0.0, 0.0), 4.6,
                     f"LiTaO3 {cut}-cut @{wavelength * 1e9:.0f}nm")


def barium_titanate_eo(wavelength: float = 1.55e-6, cut: str = "z",
                       propagation: str = "x", vertical: str = "z") -> ActiveMedium:
    """Tetragonal BaTiO3 (bulk, clamped): n_o = 2.286, n_e = 2.263 near
    1.55 um, r13 = 8, r33 = 28, r42 = 1300 pm/V, eps_11, eps_33 = 2200, 56.

    Only r13 and r33 act here: r42, BTO's large coefficient, is off-diagonal
    and needs a rotated crystal, which the diagonal-tensor solvers can't
    represent (a warning reports its size). Thin-film BTO values differ
    strongly with growth; treat these as a starting point."""
    return _uniaxial(2.286, 2.263, _cut_axes(cut, propagation, vertical),
                     _r_4mm(8.0 * _PM, 28.0 * _PM, 1300.0 * _PM),
                     (2200.0, 2200.0, 56.0), (0.0, 0.0, 0.0), 6.0,
                     f"BaTiO3 {cut}-cut @{wavelength * 1e9:.0f}nm")


def eo_polymer(n: float = 1.7, r33: float = 100e-12, poling_axis: str = "y",
               eps_dc: float = 4.0, thermal_conductivity: float = 0.2) -> ActiveMedium:
    """A poled electro-optic polymer: isotropic index ``n``, ``r33`` (m/V)
    along the poling axis and r13 = r33/3 (the usual poled-polymer ratio)."""
    g = _AXES[poling_axis]
    others = [a for a in range(3) if a != g]
    axes = (others[0], others[1], g)
    r = ((0.0, 0.0, r33 / 3), (0.0, 0.0, r33 / 3), (0.0, 0.0, r33),
         (0.0, r33 / 3, 0.0), (r33 / 3, 0.0, 0.0), (0.0, 0.0, 0.0))
    return ActiveMedium(permittivity=(n * n,) * 3, name="EO polymer",
                        eps_dc=(eps_dc,) * 3, thermal_conductivity=thermal_conductivity,
                        r=r, crystal_axes=axes)


def silicon_active(wavelength: float = 1.55e-6) -> ActiveMedium:
    """Crystalline Si: Salzberg-Villa index, eps_dc 11.7, k = 148 W/(m K),
    dn/dT = 1.86e-4 /K, with free-carrier dispersion (see :class:`PNJunction`)."""
    from .materials import silicon
    return replace(with_response(silicon(), wavelength, eps_dc=11.7,
                                 thermal_conductivity=148.0, dn_dT=1.86e-4,
                                 name="Si"), free_carriers=True)


def silica_active(wavelength: float = 1.55e-6) -> ActiveMedium:
    """SiO2: Malitson index, eps_dc 3.9, k = 1.38 W/(m K), dn/dT = 0.95e-5 /K."""
    from .materials import silica
    return with_response(silica(), wavelength, eps_dc=3.9, thermal_conductivity=1.38,
                         dn_dT=0.95e-5, name="SiO2")


def silicon_nitride_active(wavelength: float = 1.55e-6) -> ActiveMedium:
    """LPCVD Si3N4: Luke index, eps_dc 7.5, k = 20 W/(m K) (films vary,
    roughly 10-30), dn/dT = 2.45e-5 /K."""
    from .materials import silicon_nitride
    return with_response(silicon_nitride(), wavelength, eps_dc=7.5,
                         thermal_conductivity=20.0, dn_dT=2.45e-5, name="SiN")


def air_active() -> ActiveMedium:
    """Air: n = 1, eps_dc = 1, k = 0.026 W/(m K)."""
    return ActiveMedium(permittivity=(1.0, 1.0, 1.0), name="air", eps_dc=(1.0,) * 3,
                        thermal_conductivity=0.026)


def metal_active(name: str = "Au") -> ActiveMedium:
    """A heater or electrode metal: optical permittivity 1 with its DC
    conductivity (so FDTD treats it as a near-perfect conductor) and its
    thermal conductivity. ``name`` is one of Au, Al, TiN, NiCr, Ti, W."""
    table = {  # (electrical S/m, thermal W/(m K))
        "Au": (4.1e7, 317.0), "Al": (3.5e7, 237.0), "TiN": (4.0e6, 29.0),
        "NiCr": (9.1e5, 11.3), "Ti": (2.4e6, 21.9), "W": (1.8e7, 173.0),
    }
    if name not in table:
        raise ValueError(f"unknown metal {name!r}; one of {sorted(table)}")
    sigma, k = table[name]
    return ActiveMedium(permittivity=(1.0, 1.0, 1.0), name=name, conductivity=sigma,
                        eps_dc=(1.0,) * 3, thermal_conductivity=k)


# --------------------------------------------------------------------------- #
# Free-carrier dispersion in silicon.
# --------------------------------------------------------------------------- #
# Soref & Bennett (1987), as tabulated by Reed et al., Nat. Photon. 4, 518
# (2010). Densities in cm^-3, absorption in cm^-1:
#   dn = -(a_n dN + a_p dP**0.8),  dalpha = b_n dN + b_p dP
_SOREF = {1.31e-6: (6.2e-22, 6.0e-18, 6.0e-18, 4.0e-18),
          1.55e-6: (8.8e-22, 8.5e-18, 8.5e-18, 6.0e-18)}


def plasma_dispersion(N_e, N_h, wavelength: float = 1.55e-6, coefficients=None):
    """Index change and intensity absorption (1/m) of silicon from electron and
    hole densities ``N_e``, ``N_h`` (m^-3), by the Soref-Bennett fits.

    ``coefficients`` = ``(a_n, a_p, b_n, b_p)`` in the cm units of the module
    note overrides the table (e.g. with the Nedeljkovic 2011 fits); otherwise
    the nearer of 1310 and 1550 nm is used (a warning is raised if
    ``wavelength`` is more than 50 nm from both)."""
    if coefficients is None:
        lam = min(_SOREF, key=lambda l: abs(l - wavelength))
        if abs(lam - wavelength) > 50e-9:
            warnings.warn(f"free-carrier coefficients exist for 1310 and 1550 nm; "
                          f"using {lam * 1e9:.0f} nm for {wavelength * 1e9:.0f} nm",
                          RuntimeWarning, stacklevel=2)
        coefficients = _SOREF[lam]
    a_n, a_p, b_n, b_p = coefficients
    ne = np.asarray(N_e, float) * 1e-6           # m^-3 -> cm^-3
    nh = np.asarray(N_h, float) * 1e-6
    dn = -(a_n * ne + a_p * np.power(nh, 0.8))
    alpha = (b_n * ne + b_p * nh) * 100.0        # cm^-1 -> m^-1
    return dn, alpha


@dataclass(frozen=True)
class PNJunction:
    """An abrupt lateral PN junction in silicon, reverse biased.

    ``region`` (a Box or PolySlab) is the doped silicon; the junction is the
    plane ``axis = position``, with the p side below it (``p_side="lower"``) or
    above it. Doping ``N_a`` (p) and ``N_d`` (n) in m^-3. At reverse bias V the
    depletion width is ``w = sqrt(2 eps_Si (V_bi + V) (N_a + N_d) / (q N_a N_d))``,
    split in proportion ``N_d : N_a`` between the p and n sides; outside it the
    p side holds holes at ``N_a`` and the n side electrons at ``N_d``.
    """
    region: object
    axis: str = "y"
    position: float = 0.0
    N_a: float = 5e23
    N_d: float = 5e23
    p_side: str = "lower"
    name: str = "pn"
    temperature: float = 300.0
    n_i: float = 1.0e16

    def built_in(self) -> float:
        return _K_B * self.temperature / Q_E * math.log(self.N_a * self.N_d / self.n_i ** 2)

    def depletion(self, V: float) -> Tuple[float, float]:
        """``(x_p, x_n)``: depletion depth into the p and n sides at reverse bias V."""
        Vt = self.built_in() + float(V)
        if Vt <= 0:
            raise ValueError(f"forward bias beyond the built-in voltage ({self.built_in():.2f} V)")
        eps_si = 11.7 * EPS_0
        w = math.sqrt(2 * eps_si * Vt * (self.N_a + self.N_d) / (Q_E * self.N_a * self.N_d))
        return w * self.N_d / (self.N_a + self.N_d), w * self.N_a / (self.N_a + self.N_d)

    def carriers(self, mesh, V: float):
        """Electron and hole densities (m^-3) on ``mesh`` at reverse bias V."""
        mesh = statics.mesh_of(mesh)
        inside = statics.region_mask(self.region, mesh)
        a = _AXES[self.axis]
        c = mesh.coords[a].reshape([-1 if i == a else 1 for i in range(3)])
        s = (c - self.position) * (1.0 if self.p_side == "lower" else -1.0)
        xp, xn = self.depletion(V)
        p = inside & (s < -xp)
        n = inside & (s > xn)
        N_e = np.where(n, self.N_d, 0.0)
        N_h = np.where(p, self.N_a, 0.0)
        return np.broadcast_to(N_e, mesh.shape).copy(), np.broadcast_to(N_h, mesh.shape).copy()


# --------------------------------------------------------------------------- #
# Drives.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Electrode:
    """An equipotential conductor (``geometry`` a Box or PolySlab). Include
    its metal in ``structures`` too if the optics should see it."""
    geometry: object
    name: str = ""


@dataclass(frozen=True)
class Heater:
    """A resistive heater dissipating ``power`` (W; W/m in a 2-D
    cross-section) uniformly over ``geometry``."""
    geometry: object
    power: float = 1.0
    name: str = ""


# Waveforms for time-varying drives (callables t -> V).
@dataclass(frozen=True)
class Sinusoid:
    """``offset + amplitude sin(2 pi frequency t + phase)``."""
    amplitude: float
    frequency: float
    phase: float = 0.0
    offset: float = 0.0

    def __call__(self, t):
        return self.offset + self.amplitude * np.sin(2 * np.pi * self.frequency * np.asarray(t) + self.phase)


@dataclass(frozen=True)
class Ramp:
    """Linear from ``v0`` at ``t0`` to ``v1`` at ``t1``, constant outside."""
    v0: float
    v1: float
    t0: float
    t1: float

    def __call__(self, t):
        t = np.asarray(t, float)
        f = np.clip((t - self.t0) / max(self.t1 - self.t0, 1e-300), 0.0, 1.0)
        return self.v0 + (self.v1 - self.v0) * f


@dataclass(frozen=True)
class Step:
    """``v0`` before ``t0``, ``v1`` after (a smoothed edge of width ``rise``)."""
    v0: float
    v1: float
    t0: float
    rise: float = 0.0

    def __call__(self, t):
        t = np.asarray(t, float)
        if self.rise <= 0:
            return np.where(t < self.t0, self.v0, self.v1)
        return self.v0 + (self.v1 - self.v0) * 0.5 * (1 + np.tanh((t - self.t0) / self.rise))


# --------------------------------------------------------------------------- #
# The region and its prepared state.
# --------------------------------------------------------------------------- #
@dataclass
class ActiveRegion:
    """Electrodes, heaters and junctions acting on a layout.

    Parameters
    ----------
    structures
        The layout (the same list the optical solver uses). Media that aren't
        :class:`ActiveMedium` take their optical permittivity as DC
        permittivity and have no thermal conductivity or electro-optic effect.
    electrodes, heaters, junctions
        The drives; each needs a unique ``name`` (defaults are generated).
    wavelength
        Design wavelength (for dispersive media and free-carrier data).
    background
        Medium filling cells no structure covers (default air).
    boundary
        Electrostatic outer boundary: ``"neumann"`` (insulating) or ``"ground"``.
    T0, sinks, convection
        Thermal boundary (see :func:`photonfdtd.statics.solve_thermal`).
    """
    structures: Sequence
    electrodes: Sequence[Electrode] = ()
    heaters: Sequence[Heater] = ()
    junctions: Sequence[PNJunction] = ()
    wavelength: float = 1.55e-6
    background: ActiveMedium = field(default_factory=air_active)
    boundary: str = "neumann"
    T0: float = 293.15
    sinks: Tuple[str, ...] = ("zmin",)
    convection: Optional[Dict[str, float]] = None
    free_carrier_coefficients: Optional[Tuple[float, float, float, float]] = None

    def __post_init__(self):
        self.electrodes = tuple(self.electrodes)
        self.heaters = tuple(self.heaters)
        self.junctions = tuple(self.junctions)
        self.electrode_names = [e.name or f"electrode{i}" for i, e in enumerate(self.electrodes)]
        self.heater_names = [h.name or f"heater{j}" for j, h in enumerate(self.heaters)]
        self.junction_names = [j.name or f"pn{i}" for i, j in enumerate(self.junctions)]
        for names in (self.electrode_names, self.heater_names, self.junction_names):
            if len(set(names)) != len(names):
                raise ValueError(f"drive names must be unique: {names}")

    def prepare(self, target, extrude: Optional[str] = None,
                section: Optional[float] = None) -> "PreparedActive":
        """Solve the statics for ``target`` and return a :class:`PreparedActive`.

        ``target`` is a :class:`~photonfdtd.mode.ModeSolver` (solves on its
        y-z cross-section) or a :class:`~photonfdtd.simulation.Simulation`
        (solves on its grid). For a 3-D simulation with electrodes and heaters
        that are straight along one axis, pass ``extrude`` (the axis, e.g.
        ``"x"``): the statics are solved once on the cross-section at
        ``section`` (default: the middle of the drives along that axis) and
        extruded over the drives' extent, which is far cheaper than a 3-D
        solve and exact for straight electrodes away from their ends.
        """
        return PreparedActive(self, target, extrude=extrude, section=section)


class PreparedActive:
    """The statics of an :class:`ActiveRegion` solved on one target.

    Voltages ``V``, powers ``P`` and junction biases ``bias`` are dicts keyed
    by drive name (missing drives are 0) or, for ``V`` and ``P``, arrays in
    drive order.
    """

    #: First-order mode-weighted size of the neglected off-diagonal Pockels
    #: terms (set by the first :meth:`dneff` on a ModeSolver target).
    offdiagonal_ratio: Optional[float] = None

    def __init__(self, region: ActiveRegion, target, extrude=None, section=None):
        from .mode import ModeSolver
        self.region = region
        self.target = target
        self.is_mode_solver = isinstance(target, ModeSolver)
        lam = region.wavelength
        self.wavelength = lam

        # Optical permittivity the target runs with (grid frame).
        if self.is_mode_solver:
            comps = target.eps_components
            base = tuple(np.asarray(c, float)[None] for c in
                         (comps if comps is not None else (target.eps_r,) * 3))
            mesh = statics.mesh_of(target)
            self.full_shape = mesh.shape
        else:
            base = tuple(np.asarray(e, float) for e in target.eps_components())
            mesh = statics.mesh_of(target.grid)
            self.full_shape = mesh.shape
        self.base_eps = base

        # Cross-section extrusion for straight 3-D drives.
        self.extrude_axis = None
        self.extent = None
        if extrude is not None:
            if self.is_mode_solver:
                raise ValueError("extrude applies to a 3-D Simulation, not a ModeSolver")
            a = _AXES[extrude]
            lo, hi = self._drive_extent(mesh, a)
            if section is None:
                section = 0.5 * (lo + hi)
            coords = list(mesh.coords)
            coords[a] = np.array([float(section)])
            cs = list(mesh.cell_size); cs[a] = 0.0
            ax_c = mesh.coords[a]
            sel = (ax_c >= lo) & (ax_c <= hi)
            if not sel.any():
                raise ValueError("the drives cover no cells along the extrusion axis")
            self.extrude_axis = a
            self.extent = (int(np.flatnonzero(sel)[0]), int(np.flatnonzero(sel)[-1]) + 1)
            # The mesh's own coordinate along the collapsed axis must be 0 for
            # the geometry masks; shift structures by evaluating at ``section``
            # through an offset mesh.
            mesh = _OffsetMesh(coords, cs, a, float(section))
        self.mesh = mesh

        props = self._material_maps(mesh)
        self._props = props

        # Unit solves.
        self.electrostatics = None
        self.thermal = None
        n_el, n_h = len(region.electrodes), len(region.heaters)
        self.deps_V = np.zeros((n_el, 3) + mesh.shape)
        self.deps_P = np.zeros((n_h, 3) + mesh.shape)
        # Off-diagonal permittivity changes (eps_yz, eps_xz, eps_xy) per unit
        # voltage, kept only to check that neglecting them is safe.
        self.deps_V_off = np.zeros((n_el, 3) + mesh.shape)
        self._offdiag_checked = False
        if n_el:
            self.electrostatics = statics.solve_potential(
                mesh, props["eps_dc"], region.electrodes, boundary=region.boundary)
            for i in range(n_el):
                self.deps_V[i], self.deps_V_off[i] = self._pockels(
                    self.electrostatics.basis_E[i], props)
        if n_h:
            k = props["k"]
            if any(np.isnan(c).any() for c in k):
                bad = sorted(props["no_k"])
                raise ValueError(
                    "a thermal solve needs the thermal conductivity of every "
                    f"material; missing for {bad}. Use active.with_response(...) "
                    "or the library media (silicon_active, silica_active, ...).")
            self.thermal = statics.solve_thermal(
                mesh, k, region.heaters, T0=region.T0, sinks=region.sinks,
                convection=region.convection)
            n = props["n"]
            for j in range(n_h):
                dT = self.thermal.basis_dT[j]
                self.deps_P[j] = np.stack([2.0 * n[c] * props["dn_dT"][c] * dT
                                           for c in range(3)])

    # -------------------------------------------------------------- #
    def _drive_extent(self, mesh, a):
        lo, hi = np.inf, -np.inf
        drives = [e.geometry for e in self.region.electrodes] + \
                 [h.geometry for h in self.region.heaters] + \
                 [j.region for j in self.region.junctions]
        for g in drives:
            m = statics.region_mask(g, mesh)
            if not m.any():
                continue
            idx = np.flatnonzero(m.any(axis=tuple(i for i in range(3) if i != a)))
            c = mesh.coords[a]
            lo, hi = min(lo, c[idx[0]]), max(hi, c[idx[-1]])
        if not np.isfinite(lo):
            raise ValueError("no drive covers any cell of the grid")
        return lo, hi

    def _material_maps(self, mesh):
        reg = self.region
        shape = mesh.shape
        bg = reg.background
        eps_dc = [np.full(shape, v) for v in bg.dc_permittivity]
        bk = bg.thermal
        k = [np.full(shape, np.nan if bk is None else v) for v in (bk or (0, 0, 0))]
        dndt = [np.full(shape, v) for v in bg.dn_dT]
        med_id = np.full(shape, -1, np.int32)
        media, no_k = [], set()
        for st in reg.structures:
            med = st.medium
            m = statics.region_mask(st, mesh)
            if not m.any():
                continue
            if isinstance(med, ActiveMedium):
                dc = med.dc_permittivity
                kk = med.thermal
                dt = med.dn_dT
            else:
                dc = _optical_eps(med, self.wavelength)
                kk, dt = None, (0.0, 0.0, 0.0)
            if kk is None:
                no_k.add(med.name or type(med).__name__)
            for c in range(3):
                eps_dc[c][m] = dc[c]
                k[c][m] = np.nan if kk is None else kk[c]
                dndt[c][m] = dt[c]
            if med not in media:
                media.append(med)
            med_id[m] = media.index(med)
        # Optical index at the static mesh, from the target's own permittivity.
        n = [np.sqrt(np.maximum(self._on_mesh(b), 1e-12)) for b in self.base_eps]
        return dict(eps_dc=eps_dc, k=k, dn_dT=dndt, med_id=med_id, media=media,
                    n=n, no_k=no_k)

    def _on_mesh(self, full):
        """Restrict a full-target array to the static mesh (one section when
        extruding)."""
        if self.extrude_axis is None:
            return full
        a = self.extrude_axis
        i = int(np.argmin(np.abs(self._full_coords(a) - self.mesh.section)))
        sl = [slice(None)] * 3; sl[a] = slice(i, i + 1)
        return full[tuple(sl)]

    def _full_coords(self, a):
        return statics.mesh_of(self.target.grid).coords[a]

    def _pockels(self, E, props):
        """Diagonal (3, *shape) and off-diagonal (yz, xz, xy; 3, *shape)
        permittivity changes per unit drive from the field ``E`` (3, *shape),
        in the grid frame."""
        d = np.zeros((3,) + self.mesh.shape)
        off = np.zeros((3,) + self.mesh.shape)
        for idx, med in enumerate(props["media"]):
            r = getattr(med, "r", None)
            if r is None:
                continue
            m = props["med_id"] == idx
            if not m.any():
                continue
            ax = med.crystal_axes
            r = np.asarray(r, float)
            Ec = np.stack([E[ax[j]][m] for j in range(3)])          # crystal frame
            eta = r @ Ec                                            # (6, n)
            eps_c = np.array([med.permittivity[ax[i]] for i in range(3)])
            for i in range(3):
                d[ax[i]][m] = -eps_c[i] ** 2 * eta[i]
            # Voigt 4, 5, 6 = crystal (23), (13), (12): d eps_ij = -eps_i eps_j d eta_ij,
            # stored under the grid pair they map to (yz, xz, xy).
            for v, (i, j) in zip(range(3, 6), ((1, 2), (0, 2), (0, 1))):
                gi, gj = sorted((ax[i], ax[j]))
                slot = {(1, 2): 0, (0, 2): 1, (0, 1): 2}[(gi, gj)]
                off[slot][m] = -eps_c[i] * eps_c[j] * eta[v]
        return d, off

    def offdiagonal_check(self, V=None, mode: int = 0) -> float:
        """How much the neglected off-diagonal Pockels terms would move n_eff,
        relative to the modelled diagonal ones, to first order in the mode's
        fields: ``|sum 2 Re(E_i* d eps_ij E_j)| / |sum |E_i|^2 d eps_ii|``.
        Needs a ModeSolver target. ``V`` gives the voltages; by default each
        electrode is driven alone at 1 V and the largest ratio is returned."""
        if not self.is_mode_solver:
            raise TypeError("the check needs the mode fields; prepare on a ModeSolver")
        n_el = len(self.region.electrodes)
        if not n_el:
            return 0.0
        res = self.target.solve()
        E = [res.Ex[mode], res.Ey[mode], res.Ez[mode]]
        drives = [self._vec(V, self.region.electrode_names)] if V is not None \
            else list(np.eye(n_el))
        worst = 0.0
        for v in drives:
            dd = np.tensordot(v, self.deps_V, axes=1)[:, 0]
            do = np.tensordot(v, self.deps_V_off, axes=1)[:, 0]
            diag = sum(np.sum(np.abs(E[c]) ** 2 * dd[c]) for c in range(3))
            off = sum(2 * np.real(np.sum(np.conj(E[i]) * do[k] * E[j]))
                      for k, (i, j) in enumerate(((1, 2), (0, 2), (0, 1))))
            if abs(diag) > 0:
                worst = max(worst, float(abs(off) / abs(diag)))
        return worst

    def _warn_offdiagonal(self, mode):
        if self._offdiag_checked or not self.is_mode_solver or not np.any(self.deps_V_off):
            return
        self._offdiag_checked = True
        ratio = self.offdiagonal_check(mode=mode)
        self.offdiagonal_ratio = ratio
        if ratio > 0.05:
            warnings.warn(
                f"the off-diagonal Pockels terms, which are not modelled, would "
                f"shift n_eff by about {ratio:.0%} of the modelled change for this "
                "mode (first order); see photonfdtd.active",
                RuntimeWarning, stacklevel=3)

    # -------------------------------------------------------------- #
    def _vec(self, values, names):
        out = np.zeros(len(names))
        if values is None:
            return out
        if isinstance(values, dict):
            for k, v in values.items():
                if k not in names:
                    raise KeyError(f"unknown drive {k!r}; have {names}")
                out[names.index(k)] = float(v)
            return out
        a = np.asarray(values, float).reshape(-1)
        if a.size != len(names):
            raise ValueError(f"need {len(names)} values for {names}, got {a.size}")
        return a

    def _plasma(self, bias):
        """(d eps (3, *mesh), sigma (*mesh)) from the junctions at ``bias``."""
        reg = self.region
        if not reg.junctions:
            return None, None
        b = self._vec(bias, reg.junction_names)
        N_e = np.zeros(self.mesh.shape); N_h = np.zeros(self.mesh.shape)
        for j, v in zip(reg.junctions, b):
            ne, nh = j.carriers(self.mesh, v)
            N_e += ne; N_h += nh
        dn, alpha = plasma_dispersion(N_e, N_h, self.wavelength,
                                      reg.free_carrier_coefficients)
        n = self._props["n"]
        deps = np.stack([2.0 * n[c] * dn + dn ** 2 for c in range(3)])
        sigma = alpha * n[1] * EPS_0 * C_0          # alpha = sigma / (n eps0 c)
        return deps, sigma

    def delta_eps(self, V=None, P=None, bias=None):
        """Permittivity change ``(3, *mesh.shape)`` for voltages ``V``, powers
        ``P`` and junction biases ``bias``, plus the free-carrier conductivity
        (or None)."""
        v = self._vec(V, self.region.electrode_names)
        p = self._vec(P, self.region.heater_names)
        d = np.tensordot(v, self.deps_V, axes=1) if v.size else np.zeros((3,) + self.mesh.shape)
        if p.size:
            d = d + np.tensordot(p, self.deps_P, axes=1)
        dp, sigma = self._plasma(bias)
        if dp is not None:
            d = d + dp
        return d, sigma

    def _expand(self, small):
        """A static-mesh array (..., *mesh) on the full target shape (zero
        outside the extruded extent)."""
        if self.extrude_axis is None:
            return small
        a = self.extrude_axis
        lead = small.shape[:-3]
        out = np.zeros(lead + self.full_shape)
        i0, i1 = self.extent
        sl = [slice(None)] * (len(lead) + 3)
        sl[len(lead) + a] = slice(i0, i1)
        out[tuple(sl)] = small                      # broadcast along the axis
        return out

    def eps_components(self, V=None, P=None, bias=None):
        """``(eps_xx, eps_yy, eps_zz)`` of the target under the drives."""
        d, _ = self.delta_eps(V, P, bias)
        d = self._expand(d)
        return tuple(b + d[c] for c, b in enumerate(self.base_eps))

    def conductivity(self, bias=None):
        """Full-target conductivity: the target's own plus the free-carrier part."""
        _, sig = self._plasma(bias)
        own = None if self.is_mode_solver else self.target.conductivity_map()
        if sig is None:
            return own
        sig = self._expand(sig)
        return sig if own is None else own + sig

    # -------------------------------------------------------------- #
    # Mode solver.
    # -------------------------------------------------------------- #
    def mode_solver(self, V=None, P=None, bias=None):
        """A copy of the target ModeSolver with the driven permittivity."""
        if not self.is_mode_solver:
            raise TypeError("prepared on a Simulation; prepare on a ModeSolver")
        ms = copy.copy(self.target)
        comps = tuple(c[0] for c in self.eps_components(V, P, bias))
        ms.eps_components = comps
        ms.eps_r = sum(comps) / 3.0
        return ms

    def n_eff(self, V=None, P=None, bias=None, mode: int = 0) -> float:
        return float(np.real(self.mode_solver(V, P, bias).solve().n_eff[mode]))

    def neff_sweep(self, drive: str, values, mode: int = 0, **fixed):
        """``n_eff`` at each value of one drive (an electrode voltage, a heater
        power or a junction bias), the other drives at ``fixed`` (``V=``,
        ``P=``, ``bias=`` dicts)."""
        out = []
        for x in values:
            kw = {k: dict(v or {}) for k, v in (("V", fixed.get("V")), ("P", fixed.get("P")),
                                                ("bias", fixed.get("bias")))}
            kw[self._kind(drive)][drive] = x
            out.append(self.n_eff(mode=mode, **kw))
        return np.array(out)

    def _kind(self, drive):
        if drive in self.region.electrode_names:
            return "V"
        if drive in self.region.heater_names:
            return "P"
        if drive in self.region.junction_names:
            return "bias"
        raise KeyError(f"unknown drive {drive!r}")

    def dneff(self, drive, step: Optional[float] = None, mode: int = 0, **fixed) -> float:
        """d n_eff / d drive by a central difference about 0 (about ``step``
        for a junction bias, which can't go far forward). The permittivity is
        linear in voltage and power, so this is exact to first order. Steps
        default to 1 V, 1 W/m (the cross-section's heater power per length)
        and 0.5 V of junction bias."""
        kind = self._kind(drive)
        if kind == "V":
            self._warn_offdiagonal(mode)
        if step is None:
            step = {"V": 1.0, "P": 1.0, "bias": 0.5}[kind]
        pts = [-step, step] if kind != "bias" else [0.0, 2 * step]
        lo, hi = self.neff_sweep(drive, pts, mode, **fixed)
        return float((hi - lo) / (2 * step))

    def vpi_l(self, electrode: str, mode: int = 0, **fixed) -> float:
        """Half-wave voltage-length product (V m) of driving ``electrode``,
        ``lambda / (2 |d n_eff / dV|)``. Divide by 1e-2 for V cm. For push-pull
        on two arms, drive both and use the phase difference instead."""
        return self.wavelength / (2.0 * abs(self.dneff(electrode, mode=mode, **fixed)))

    def p_pi(self, heater: str, mode: int = 0, **fixed) -> float:
        """Heater power for a pi phase shift, ``lambda / (2 |d n_eff / dP'|)``.

        On a 2-D cross-section the heater power is per unit length (W/m); the
        phase is ``k0 L dn_eff`` with ``dn_eff`` proportional to ``P/L``, so the
        result is the total power in W, independent of the heater length."""
        return self.wavelength / (2.0 * abs(self.dneff(heater, mode=mode, **fixed)))

    # -------------------------------------------------------------- #
    # FDTD.
    # -------------------------------------------------------------- #
    def _sim(self, sim):
        if self.is_mode_solver:
            raise TypeError("prepared on a ModeSolver; use eim() to carry the "
                            "result to a 2-D simulation, or prepare on the Simulation")
        sim = self.target if sim is None else sim
        if tuple(sim.grid.shape) != tuple(self.full_shape):
            raise ValueError("simulation grid differs from the one prepared on")
        return sim

    def coefficients(self, sim=None, V=None, P=None, bias=None):
        """``(ce, ca)`` for :meth:`Simulation.run(ce=..., ca=...)`."""
        from .jaxbackend import update_coefficients
        sim = self._sim(sim)
        return update_coefficients(sim, self.eps_components(V, P, bias),
                                   self.conductivity(bias))

    def run(self, sim=None, V=None, P=None, bias=None, **kw):
        """Run the simulation with the drives applied."""
        sim = self._sim(sim)
        ce, ca = self.coefficients(sim, V, P, bias)
        return sim.run(ce=ce, ca=ca, **kw)

    def ce_builder(self, sim=None, P_fixed=None, bias=None):
        """``ce_of_params({'V': (n_el,), 'P': (n_h,)}) -> ce`` in JAX, for
        :func:`~photonfdtd.make_value_and_grad_params` and friends: optimize
        drive voltages and powers by gradient descent. Free-carrier changes
        (``bias``) are held fixed. Missing keys mean zero."""
        import jax.numpy as jnp
        sim = self._sim(sim)
        dt = sim.dt
        cdt = np.dtype(sim.dtypes["Ex"])        # coefficients in the stepping dtype
        base = [jnp.asarray(b) for b in self.base_eps]
        dV = jnp.asarray(self._expand(self.deps_V)) if self.deps_V.size else None
        dP = jnp.asarray(self._expand(self.deps_P)) if self.deps_P.size else None
        dpl, sig = self._plasma(bias)
        if dpl is not None:
            dpl = jnp.asarray(self._expand(dpl))
        sigma = self.conductivity(bias)
        sigma = None if sigma is None else jnp.asarray(sigma)

        def ce_of_params(params, **_):
            eps = list(base)
            for c in range(3):
                if dV is not None and "V" in params:
                    eps[c] = eps[c] + jnp.tensordot(params["V"], dV[:, c], axes=1)
                if dP is not None and "P" in params:
                    eps[c] = eps[c] + jnp.tensordot(params["P"], dP[:, c], axes=1)
                if dpl is not None:
                    eps[c] = eps[c] + dpl[c]
            out = {}
            ca = {}
            for c, name in enumerate(("Ex", "Ey", "Ez")):
                b = dt / (eps[c] * EPS_0)
                if sigma is None:
                    out[name] = b.astype(cdt)
                else:
                    s = sigma * dt / (2.0 * eps[c] * EPS_0)
                    out[name] = (b / (1.0 + s)).astype(cdt)
                    ca[name] = ((1.0 - s) / (1.0 + s)).astype(cdt)
            if sigma is not None:
                out["ca"] = ca
            return out
        return ce_of_params

    def modulation(self, drives: Dict[str, Callable], sim=None, P=None, bias=None):
        """A time-varying drive for :meth:`Simulation.run(modulation=...)`.

        ``drives`` maps electrode names to waveforms (callables ``t -> V``,
        e.g. :class:`Sinusoid`, :class:`Ramp`, :class:`Step`); other electrodes
        stay at 0 V, and heater powers ``P`` and junction biases are static.
        Returns ``(modulation, ce, ca)``: pass all three to ``run``."""
        sim = self._sim(sim)
        names = list(drives)
        idx = [self.region.electrode_names.index(n) for n in names]
        deps = self._expand(self.deps_V[idx])                       # (n, 3, *full)
        live = np.any(deps != 0, axis=(0, 1))
        if not live.any():
            raise ValueError("the driven electrodes change no permittivity")
        box = []
        for a in range(3):
            ax = np.flatnonzero(live.any(axis=tuple(i for i in range(3) if i != a)))
            box.append(slice(int(ax[0]), int(ax[-1]) + 1))
        box = tuple(box)
        if getattr(sim, "_has_dispersion", False):
            from .jaxbackend import _build_static, _build_ade_jax
            ade = _build_ade_jax(sim, _build_static(sim))
            if ade is not None:
                inside = np.zeros(sim.grid.shape, bool)
                inside[box] = True
                if inside[tuple(ade["idx"])].any():
                    raise ValueError("the modulated region contains dispersive "
                                     "cells; use non-dispersive media there")
        sigma = self.conductivity(bias)
        if sigma is not None and np.any(sigma[box] != 0):
            raise ValueError("the modulated region contains conductive cells")
        eps0 = self.eps_components(None, P, bias)
        eps_b = {c: np.asarray(e[box]) for c, e in zip(("Ex", "Ey", "Ez"), eps0)}
        dbox = {c: deps[(slice(None), k) + box] for k, c in enumerate(("Ex", "Ey", "Ez"))}
        t = (np.arange(sim.n_steps) + 0.5) * sim.dt
        v = np.stack([np.broadcast_to(np.asarray(drives[n](t), float), t.shape)
                      for n in names], axis=1)
        ce, ca = self.coefficients(sim, None, P, bias)
        return dict(box=box, eps_b=eps_b, deps=dbox, v=v), ce, ca

    # -------------------------------------------------------------- #
    # Effective-index transfer to a 2-D chip-plane simulation.
    # -------------------------------------------------------------- #
    def eim(self, footprint: Sequence, drives: Optional[Sequence[str]] = None,
            sign: float = 1.0, mode: int = 0) -> "EIMActive":
        """Carry the cross-section's ``d n_eff / d drive`` to a 2-D chip-plane
        simulation.

        The returned :class:`EIMActive` raises the index of the cells inside
        ``footprint`` (geometries in the chip plane: the waveguide core under
        the electrodes or heater) by ``sign * sum(d n_eff / d drive * value)``.
        ``sign = -1`` describes the other arm of a push-pull pair, where the
        same electrodes give the opposite field; add the arms with ``+``. The
        in-plane mode is not fully confined to the footprint, so the shift it
        sees is a little below the cross-section value (by roughly its in-core
        power fraction): draw ``footprint`` over the whole core."""
        if not self.is_mode_solver:
            raise TypeError("eim() takes the result of preparing on a ModeSolver")
        names = list(drives) if drives is not None else (
            self.region.electrode_names + self.region.heater_names)
        slopes = {n: float(sign) * self.dneff(n, mode=mode) for n in names}
        return EIMActive([(list(footprint), slopes)])

    def sweep(self, values: Sequence[dict], sim=None):
        """Run the simulation at each drive setting in ``values`` (dicts with
        ``V``, ``P`` keys, each a dict or array as for :meth:`eps_components`),
        compiling once. Returns the monitor outputs, one dict per setting:
        ``{'dft': {name: {comp: array}}, 'flux': {...}, 'fields': {...}}``."""
        import jax.numpy as jnp
        sim = self._sim(sim)
        build = self.ce_builder(sim)
        nv = len(self.region.electrode_names); nh = len(self.region.heater_names)

        def params(v):
            return {"V": jnp.asarray(self._vec(v.get("V"), self.region.electrode_names)) if nv else jnp.zeros(0),
                    "P": jnp.asarray(self._vec(v.get("P"), self.region.heater_names)) if nh else jnp.zeros(0)}
        return _sweep(sim, build, [params(v) for v in values])


def _sweep(sim, build, params_list):
    """Forward runs of ``sim`` with coefficients ``build(p)``, compiled once."""
    import jax
    from .jaxbackend import _run_fn
    run = jax.jit(lambda p: _run_fn(sim)(build(p)))
    outs = []
    for p in params_list:
        out = run(p)
        outs.append(jax.tree_util.tree_map(np.asarray, out))
    return outs


class _OffsetMesh(statics.Mesh):
    """A cross-section mesh at ``section`` along a collapsed axis.

    Geometry masks test the collapsed coordinate against the shape's extent,
    so the coordinate is the real section position rather than 0."""

    def __init__(self, coords, cell_size, axis, section):
        super().__init__(tuple(coords), tuple(cell_size))
        self.axis = axis
        self.section = section


@dataclass
class EIMActive:
    """Drive response of waveguides transferred to a 2-D chip-plane run (see
    :meth:`PreparedActive.eim`).

    ``regions`` is a list of ``(footprint geometries, {drive: d n / d drive})``;
    ``a + b`` combines two (e.g. the two arms of a push-pull modulator).
    Drive values are passed as one dict keyed by drive name (voltages and
    heater powers alike, in the units of the cross-section they came from)."""
    regions: list

    def __add__(self, other: "EIMActive") -> "EIMActive":
        return EIMActive(list(self.regions) + list(other.regions))

    @property
    def drives(self):
        return sorted({k for _, sl in self.regions for k in sl})

    def _masks(self, sim):
        out = []
        for geoms, _ in self.regions:
            m = np.zeros(sim.grid.shape, bool)
            for g in geoms:
                m |= np.asarray(g.region_mask(sim.grid), bool)
            if not m.any():
                raise ValueError("a footprint covers no cells of the simulation")
            out.append(m)
        return out

    def _dn(self, slopes, drives):
        dn = 0.0
        for k, v in (drives or {}).items():
            if k not in self.drives:
                raise KeyError(f"unknown drive {k!r}; have {self.drives}")
            dn = dn + slopes.get(k, 0.0) * v
        return dn

    def eps_components(self, sim, drives=None):
        """``(eps_xx, eps_yy, eps_zz)`` of ``sim`` with the drives applied."""
        out = [np.array(e, float) for e in sim.eps_components()]
        for m, (_, slopes) in zip(self._masks(sim), self.regions):
            dn = float(self._dn(slopes, drives))
            for e in out:
                e[m] = (np.sqrt(e[m]) + dn) ** 2
        return tuple(out)

    def coefficients(self, sim, drives=None):
        from .jaxbackend import update_coefficients
        return update_coefficients(sim, self.eps_components(sim, drives))

    def run(self, sim, drives=None, **kw):
        ce, ca = self.coefficients(sim, drives)
        return sim.run(ce=ce, ca=ca, **kw)

    def ce_builder(self, sim):
        """``ce_of_params({drive: value, ...}) -> ce`` in JAX, for the
        design-gradient tools (e.g. find the voltage for a target split)."""
        import jax.numpy as jnp
        masks = [jnp.asarray(m) for m in self._masks(sim)]
        base = [jnp.asarray(np.asarray(e, float)) for e in sim.eps_components()]
        dt = sim.dt
        cdt = np.dtype(sim.dtypes["Ex"])
        regions = [dict(sl) for _, sl in self.regions]

        def ce_of_params(params, **_):
            eps = list(base)
            for m, slopes in zip(masks, regions):
                dn = self._dn(slopes, params)
                eps = [jnp.where(m, (jnp.sqrt(e) + dn) ** 2, e) for e in eps]
            return {name: (dt / (eps[c] * EPS_0)).astype(cdt)
                    for c, name in enumerate(("Ex", "Ey", "Ez"))}
        return ce_of_params

    def sweep(self, sim, values: Sequence[dict]):
        """Run ``sim`` at each drive setting (dicts keyed by drive name),
        compiling once; returns the monitor outputs (see
        :meth:`PreparedActive.sweep`)."""
        import jax.numpy as jnp
        names = self.drives
        build = self.ce_builder(sim)
        plist = [{k: jnp.asarray(float(v.get(k, 0.0))) for k in names} for v in values]
        return _sweep(sim, build, plist)
