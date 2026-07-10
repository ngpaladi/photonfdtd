"""Sources.

Three primitive source types:

- :class:`PointDipole`     - soft additive source on a single E or H cell.
- :class:`ModeSource`      - distributed soft source on an injection plane,
  weighted by a user-supplied transverse profile.
- :class:`ChargedParticle` - a point charge moving in a straight line,
  injected as a moving current source; radiates Cherenkov light when it
  outruns the local phase velocity.

And one factory built on top:

- :class:`SinglePhotonSource` - a :class:`ModeSource` whose overall amplitude
  is set so that the time-integrated power carried by the launched wavepacket
  equals one photon's energy, h * freq0. Useful for end-to-end semi-classical
  single-photon experiments where what matters is photon count, not the
  absolute field magnitude.

All sources here are *soft additive*: each timestep they add to one field
component at chosen cells. A `Simulation` expands distributed sources into a
list of `PointDipole` instances before running.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Tuple, List, TYPE_CHECKING
import math
import numpy as np

from .constants import C_0, EPS_0, MU_0

if TYPE_CHECKING:
    from .grid import Grid


PLANCK_H = 6.62607015e-34   # J*s, exact (SI redefinition 2019)


@dataclass(frozen=True)
class GaussianPulse:
    """Gaussian-modulated continuous wave.

    waveform(t) = exp(-((t - t0)/sigma)^2) * sin(2*pi*f0*(t - t0))

    where sigma is derived from `fwhm` (intensity FWHM in time).
    """
    freq0: float           # carrier frequency, Hz
    fwhm: float            # intensity FWHM of the envelope, s
    delay: float = 0.0     # extra offset applied on top of 4*sigma startup

    @property
    def sigma(self) -> float:
        return float(self.fwhm) / (2.0 * math.sqrt(2.0 * math.log(2.0)))

    @property
    def t0(self) -> float:
        # start with 4 sigma of ramp-up so the pulse begins essentially zero.
        return 4.0 * self.sigma + float(self.delay)

    @property
    def envelope_l2(self) -> float:
        """Time integral of envelope-squared, int exp(-2 (t/sigma)^2) dt."""
        return float(self.sigma) * math.sqrt(math.pi / 2.0)

    def __call__(self, t):
        t = np.asarray(t)
        envelope = np.exp(-((t - self.t0) / self.sigma) ** 2)
        carrier = np.sin(2.0 * math.pi * self.freq0 * (t - self.t0))
        return envelope * carrier


@dataclass(frozen=True)
class PointDipole:
    """Soft additive point source on a single E or H component."""
    position: Tuple[float, ...]
    component: str                # 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'
    waveform: GaussianPulse
    amplitude: float = 1.0


def _to3vec(x) -> Tuple[float, float, float]:
    """Pad a scalar / 1-/2-/3-sequence into a (x, y, z) float tuple (zeros fill)."""
    if isinstance(x, (int, float)):
        return (float(x), 0.0, 0.0)
    vals = [float(v) for v in x]
    vals += [0.0] * (3 - len(vals))
    if len(vals) > 3:
        raise ValueError("expected a scalar or a 1-/2-/3-component vector")
    return (vals[0], vals[1], vals[2])


@dataclass(frozen=True)
class ChargedParticle:
    """A point charge moving in a straight line at constant velocity.

    The charge contributes a current density ``J = q v delta(r - r0(t))`` to
    Ampere's law, so each FDTD step it adds ``-dt/eps * J`` to the E-field
    component(s) along its velocity. When the speed exceeds the local phase
    velocity ``c/n`` (i.e. ``n * beta > 1`` with ``beta = v/c``) the charge
    emits **Cherenkov radiation**: a shock cone trailing the particle, with the
    radiation propagating at the Cherenkov angle ``theta_c = arccos(1/(n beta))``
    to the velocity, and the visible wavefront (Mach) cone making a half-angle
    ``mu = arcsin(1/(n beta)) = pi/2 - theta_c`` with the trajectory.

    To keep the radiated spectrum band-limited (an ideal point charge in a
    non-dispersive medium radiates a UV-divergent Frank-Tamm spectrum that the
    grid cannot represent), the charge is deposited as a small Gaussian "cloud"
    of width ``cloud_cells`` cells rather than onto a single cell.

    Notes
    -----
    * In a 2-D simulation the out-of-plane axis is treated as 1 m thick, so
      ``charge`` is effectively a charge *per unit length* (C/m) and the field
      is per unit length. The cone geometry is unaffected.
    * Velocity components on collapsed (size-1) axes are rejected by the
      :class:`~photonfdtd.Simulation`, since they have nowhere to radiate.
    * The deposition ignores the half-cell Yee stagger (a constant sub-cell
      offset that does not change the cone geometry).

    Parameters
    ----------
    charge : float
        Particle charge in coulombs (C/m in 2-D). Sets only the amplitude;
        Maxwell's equations are linear, so the cone geometry is independent
        of it. Use :data:`~photonfdtd.Q_E` for an electron, or a much larger
        value to get O(1) fields for visualisation.
    velocity : tuple of float
        Velocity vector in m/s. Its direction is the trajectory; its magnitude
        sets ``beta`` and hence whether (and at what angle) the charge radiates.
    start : tuple of float
        Position (m) at ``t_start``. Defaults to the origin.
    t_start, t_stop : float
        The charge is injected only while ``t_start <= t <= t_stop`` (s).
        Defaults: from ``t = 0`` until it leaves the domain.
    cloud_cells : float
        Standard deviation of the Gaussian charge cloud, in grid cells. Larger
        values give a smoother, more band-limited (lower-frequency) cone;
        smaller values approach a point charge and grid noise. Default 1.5.
    """
    charge: float
    velocity: Tuple[float, ...]
    start: Tuple[float, ...] = (0.0, 0.0, 0.0)
    t_start: float = 0.0
    t_stop: float = math.inf
    cloud_cells: float = 1.5

    @property
    def velocity3(self) -> Tuple[float, float, float]:
        return _to3vec(self.velocity)

    @property
    def start3(self) -> Tuple[float, float, float]:
        return _to3vec(self.start)

    @property
    def speed(self) -> float:
        vx, vy, vz = self.velocity3
        return math.sqrt(vx * vx + vy * vy + vz * vz)

    @property
    def beta(self) -> float:
        """Speed as a fraction of the vacuum speed of light."""
        return self.speed / C_0

    def position_at(self, t: float) -> Tuple[float, float, float]:
        """Particle position (m) at time ``t``."""
        s = self.start3
        v = self.velocity3
        dt = t - self.t_start
        return (s[0] + v[0] * dt, s[1] + v[1] * dt, s[2] + v[2] * dt)

    def cherenkov_angle(self, n: float) -> float:
        """Cherenkov emission angle ``arccos(1/(n*beta))`` (rad) between the
        velocity and the radiated wavevector.

        Raises ``ValueError`` below the Cherenkov threshold (``n*beta <= 1``),
        where no coherent radiation is emitted.
        """
        x = 1.0 / (n * self.beta)
        if x >= 1.0:
            raise ValueError(
                f"below Cherenkov threshold: n*beta = {n * self.beta:.3f} <= 1"
            )
        return math.acos(x)

    def mach_angle(self, n: float) -> float:
        """Half-angle (rad) of the visible shock (Mach) cone with the
        trajectory: ``arcsin(1/(n*beta)) = pi/2 - cherenkov_angle(n)``."""
        return 0.5 * math.pi - self.cherenkov_angle(n)


@dataclass(frozen=True)
class ModeSource:
    """Distributed soft additive source on an injection plane.

    The injection plane is the planar slice through ``center`` whose extent
    matches ``size``; the axis along which ``size`` is zero (or smallest) is
    the propagation axis. Inside the plane the source is rasterised into one
    `PointDipole` per cell, with the amplitude at each cell proportional to
    ``amplitude * profile[..]``.

    Parameters
    ----------
    center, size : 3-tuples (m)
        Geometric extent. One component of ``size`` must be 0 - that axis is
        the propagation direction.
    component : str
        'Ex', 'Ey' or 'Ez'. For a 2D TE-like simulation in xy this is
        usually 'Ez'.
    waveform : GaussianPulse
        Temporal pulse (carrier + envelope).
    profile : 1-D or 2-D array
        Transverse mode profile. Length matches the in-plane axis (or axes)
        of the injection plane sampled at ``profile_coords``.
    profile_coords : tuple of arrays
        Coordinates (m) at which ``profile`` is sampled, *relative to the
        source centre*. One array for a line-shaped plane, two for a
        rectangular plane. So a mode profile solved on a centred
        cross-section can be dropped at any source position on the layout
        without re-sampling.
    amplitude : float
        Overall scalar prefactor applied on top of the profile.
    """
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    component: str
    waveform: GaussianPulse
    profile: np.ndarray
    profile_coords: Tuple[np.ndarray, ...]
    amplitude: float = 1.0

    def _propagation_axis(self) -> int:
        sizes = list(self.size) + [0.0] * (3 - len(self.size))
        zeros = [i for i, s in enumerate(sizes) if s == 0]
        if not zeros:
            return int(np.argmin(sizes))
        return zeros[0]

    def expand(self, grid: "Grid") -> List[PointDipole]:
        """Sample the profile onto grid cells and return a list of dipoles."""
        center = list(self.center) + [0.0] * (3 - len(self.center))
        size = list(self.size) + [0.0] * (3 - len(self.size))
        prop_ax = self._propagation_axis()
        tan_axes = [a for a in range(3) if a != prop_ax]

        # In-plane axis sampling.
        plane_idx_per_axis = []
        plane_coords = []
        for ax in tan_axes:
            c = grid.coords[ax]
            if c.size == 1:
                # collapsed axis - the plane is effectively a line; include the
                # one cell as-is. We require profile_coords to match.
                plane_idx_per_axis.append(np.array([0]))
                plane_coords.append(np.array([0.0]))
                continue
            lo = center[ax] - size[ax] / 2.0
            hi = center[ax] + size[ax] / 2.0
            mask = (c >= lo) & (c <= hi)
            if not mask.any():
                raise ValueError(
                    f"ModeSource injection plane has zero overlap on axis {ax}"
                )
            plane_idx_per_axis.append(np.flatnonzero(mask))
            plane_coords.append(c[mask])

        # Propagation-axis index: nearest cell to centre[prop_ax]
        c_prop = grid.coords[prop_ax]
        if c_prop.size == 1:
            i_prop = 0
        else:
            i_prop = int(np.argmin(np.abs(c_prop - center[prop_ax])))

        # Interpolate the user-supplied profile onto plane_coords. The
        # profile_coords are interpreted *relative to the source centre*, so
        # we shift plane_coords into the profile's frame before interpolating.
        # This lets the user mode-solve once on a centred cross-section and
        # then drop the same profile at any source location on the layout.
        prof = np.asarray(self.profile)
        rel_plane_coords = [
            plane_coords[0] - center[tan_axes[0]],
            (plane_coords[1] - center[tan_axes[1]]) if len(tan_axes) > 1
                else plane_coords[1],
        ]
        if prof.ndim == 1:
            if len(self.profile_coords) != 1:
                raise ValueError("1-D profile requires one profile_coords array")
            interp_vals = np.interp(rel_plane_coords[0], self.profile_coords[0],
                                    prof, left=0.0, right=0.0)
            if len(plane_coords) == 2 and plane_coords[1].size == 1:
                interp = interp_vals[:, None]
            else:
                interp = interp_vals
        elif prof.ndim == 2:
            if len(self.profile_coords) != 2:
                raise ValueError("2-D profile requires two profile_coords arrays")
            # Bilinear interpolation via np.interp twice (separable on a grid).
            row = np.empty((prof.shape[0], plane_coords[1].size))
            for ir in range(prof.shape[0]):
                row[ir] = np.interp(rel_plane_coords[1], self.profile_coords[1],
                                    prof[ir], left=0.0, right=0.0)
            interp = np.empty((plane_coords[0].size, plane_coords[1].size))
            for ic in range(plane_coords[1].size):
                interp[:, ic] = np.interp(rel_plane_coords[0], self.profile_coords[0],
                                          row[:, ic], left=0.0, right=0.0)
        else:
            raise ValueError("profile must be 1-D or 2-D")

        # Build dipoles. Note that interp shape matches len(plane_coords[0]) x
        # len(plane_coords[1]) - even for collapsed axes we add a singleton.
        if interp.ndim == 1:
            interp = interp[:, None]

        dipoles: List[PointDipole] = []
        for ia, i_t1 in enumerate(plane_idx_per_axis[0]):
            for ib, i_t2 in enumerate(plane_idx_per_axis[1] if len(plane_idx_per_axis) > 1
                                       else [0]):
                w = float(interp[ia, ib]) * float(self.amplitude)
                if w == 0.0:
                    continue
                pos = [0.0, 0.0, 0.0]
                pos[prop_ax] = float(c_prop[i_prop] if c_prop.size > 1
                                      else center[prop_ax])
                pos[tan_axes[0]] = float(grid.coords[tan_axes[0]][i_t1]
                                          if grid.coords[tan_axes[0]].size > 1
                                          else center[tan_axes[0]])
                if len(tan_axes) > 1:
                    pos[tan_axes[1]] = float(grid.coords[tan_axes[1]][i_t2]
                                              if grid.coords[tan_axes[1]].size > 1
                                              else center[tan_axes[1]])
                dipoles.append(PointDipole(
                    position=tuple(pos),
                    component=self.component,
                    waveform=self.waveform,
                    amplitude=w,
                ))
        return dipoles


@dataclass
class UniModeSource:
    """Unidirectional waveguide-mode source (equivalence-principle current sheets).

    Launches a solved waveguide mode in ONE direction by co-locating an electric
    and a magnetic surface-current sheet on the injection plane (Love / Huygens
    equivalence): with electric current ``J = n_hat x H_m`` and magnetic current
    ``M = -n_hat x E_m`` the forward radiation adds and the backward radiation
    cancels, unlike the bidirectional soft :class:`ModeSource`. Clean one-way
    launch is what makes a reflection (S11) measurement well posed.

    The sheets are emitted as additive soft sources (electric on the transverse
    E components, magnetic on the transverse H components a half Yee-cell / half
    time-step downstream, which the stepper already staggers). For propagation
    along x (n_hat = x_hat):

        A(Ey) =  C/(eps0*eps_r) * H_mz     A(Ez) = -C/(eps0*eps_r) * H_my
        A(Hy) = -s * C/mu0 * E_mz          A(Hz) =  s * C/mu0 * E_my

    with ``s=+1`` for +x and ``s=-1`` for -x. The eps0/mu0 factors set the
    electric-vs-magnetic ratio that produces the cancellation; ``C`` is an
    overall scale (S-parameters normalise it out by also measuring the input).

    Parameters
    ----------
    center, size : 3-tuples (m)
        Injection-plane geometry; the zero-size axis is the propagation axis.
    waveform : GaussianPulse
        Temporal pulse.
    mode : ModeResult
        A full-vectorial mode solved on the SAME transverse (y, z) grid as the
        plane (same cell size and extent, centred). Provides Ey/Ez/Hy/Hz and the
        cross-section eps_r.
    mode_index : int
        Which solved mode to launch.
    direction : str
        '+x' or '-x' (only x propagation is supported).
    amplitude : float
        Overall scale ``C``.
    """
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    waveform: GaussianPulse
    mode: "object"
    mode_index: int = 0
    direction: str = "+x"
    amplitude: float = 1.0

    def expand(self, grid: "Grid") -> List[PointDipole]:
        if self.direction not in ("+x", "-x"):
            raise ValueError("UniModeSource supports direction '+x' or '-x'")
        s = 1.0 if self.direction == "+x" else -1.0
        center = list(self.center) + [0.0] * (3 - len(self.center))
        m = self.mode
        i = self.mode_index

        # Transverse mode fields, phase-aligned so the transverse components are
        # (essentially) real and in phase - as they are for a lossless
        # propagating mode - then taken real for a real-valued time source.
        Ey, Ez = np.asarray(m.Ey[i]), np.asarray(m.Ez[i])
        Hy, Hz = np.asarray(m.Hy[i]), np.asarray(m.Hz[i])
        k = int(np.argmax(np.abs(Ey) + np.abs(Ez)))
        phase = np.exp(-1j * np.angle((Ey.ravel() + Ez.ravel())[k]))
        Ey, Ez, Hy, Hz = (np.real(f * phase) for f in (Ey, Ez, Hy, Hz))
        eps = np.asarray(m.eps_r)
        my, mz = np.asarray(m.y), np.asarray(m.z)          # mode transverse coords

        # Map each mode cell to the nearest sim plane cell (grids expected to
        # match; nearest-cell tolerates small offsets).
        yc, zc = grid.coords[1], grid.coords[2]
        xc = grid.coords[0]
        i_x = 0 if xc.size == 1 else int(np.argmin(np.abs(xc - center[0])))

        C = float(self.amplitude)
        specs = [
            ("Ey", C / (EPS_0 * eps) * Hz),
            ("Ez", -C / (EPS_0 * eps) * Hy),
            ("Hy", -s * C / MU_0 * Ez),
            ("Hz", s * C / MU_0 * Ey),
        ]
        dipoles: List[PointDipole] = []
        for jy in range(my.size):
            iy = 0 if yc.size == 1 else int(np.argmin(np.abs(yc - (center[1] + my[jy]))))
            for jz in range(mz.size):
                iz = 0 if zc.size == 1 else int(np.argmin(np.abs(zc - (center[2] + mz[jz]))))
                pos = (float(xc[i_x] if xc.size > 1 else center[0]),
                       float(yc[iy] if yc.size > 1 else center[1]),
                       float(zc[iz] if zc.size > 1 else center[2]))
                for comp, amp in specs:
                    a = float(amp[jy, jz])
                    if a != 0.0:
                        dipoles.append(PointDipole(position=pos, component=comp,
                                                   waveform=self.waveform, amplitude=a))
        return dipoles


def _profile_l2(profile: np.ndarray,
                profile_coords: Tuple[np.ndarray, ...]) -> float:
    r"""Integral of \|profile\|^2 over the spatial coords.

    For a 1-D profile (used in 2-D simulations) this has units of metres
    and is the effective mode width. For a 2-D profile (used in 3-D
    simulations) it has units of square metres and is the effective mode
    area.
    """
    prof = np.asarray(profile)
    if prof.ndim == 1:
        return float(np.trapezoid(np.abs(prof) ** 2, profile_coords[0]))
    if prof.ndim == 2:
        inner = np.trapezoid(np.abs(prof) ** 2, profile_coords[1], axis=1)
        return float(np.trapezoid(inner, profile_coords[0]))
    raise ValueError("profile must be 1-D or 2-D")


def single_photon_field_amplitude(
    waveform: GaussianPulse,
    n_eff: float,
    mode_area: float,
) -> float:
    r"""Peak E-field of a guided-mode wavepacket that carries h*freq0 of energy.

    Derivation (slowly-varying envelope, time-averaged Poynting flux). With
    E(r, t) = E0 * envelope(t) * cos(omega t) * psi(x, y) and psi(x, y)
    normalised so that its peak is 1, the integrated guided-mode power is

        P(t) = (n_eff / (2 Z0)) * \|E0\|^2 * envelope(t)^2 * A_mode

    where A_mode = integral \|psi\|^2 dA. Demanding the wavepacket carry
    exactly h*freq0 of energy gives

        \|E0\| = sqrt( 2 * Z0 * h * freq0 / (n_eff * A_mode * T_eff) )

    with T_eff = integral envelope(t)^2 dt and Z0 = sqrt(mu0/eps0).

    Parameters
    ----------
    waveform : GaussianPulse
        Temporal pulse driving the source.
    n_eff : float
        Phase index of the guided mode at the carrier frequency.
    mode_area : float
        integral \|psi\|^2 dA, in m^2 (3D) or m (2D, treated as effective
        mode width times unit out-of-plane thickness).
    """
    Z0 = math.sqrt(MU_0 / EPS_0)
    return math.sqrt(2.0 * Z0 * PLANCK_H * waveform.freq0
                     / (float(n_eff) * float(mode_area) * waveform.envelope_l2))


@dataclass(frozen=True)
class SinglePhotonSource:
    r"""A `ModeSource` whose amplitude is set so the launched wavepacket carries
    exactly one photon worth of energy (h * freq0).

    The amplitude is computed as

        amp = 2 * \|E0\|       (for soft additive sources that radiate in both
                              directions; the factor of 2 places one photon
                              of energy on each side of the injection plane
                              when the source is at a symmetric location).

    For a true unidirectional photon source, gate the simulation with a TFSF
    contour, or place the source on the boundary of the structure and absorb
    the back-going wavepacket with the CPML.

    The single-photon amplitude is *approximate* - it assumes the user-provided
    `profile` is an L^2-normalised transverse mode profile of an actual guided
    mode at ``waveform.freq0``. To verify the photon count, place a
    `FluxMonitor` downstream of the source and compare its integrated energy
    to ``h * freq0``.
    """
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    component: str
    waveform: GaussianPulse
    profile: np.ndarray
    profile_coords: Tuple[np.ndarray, ...]
    n_eff: float
    mode_area: float = 0.0       # m^2 (3D) or m (2D); if 0, computed from profile
    bidirectional: bool = True   # if True multiply amplitude by 2

    @property
    def effective_mode_area(self) -> float:
        if self.mode_area > 0:
            return float(self.mode_area)
        return _profile_l2(self.profile, self.profile_coords)

    @property
    def peak_field(self) -> float:
        """Peak E-field of one-photon-equivalent wavepacket (V/m)."""
        return single_photon_field_amplitude(
            self.waveform, self.n_eff, self.effective_mode_area,
        )

    def as_mode_source(self) -> ModeSource:
        amp = self.peak_field * (2.0 if self.bidirectional else 1.0)
        return ModeSource(
            center=self.center, size=self.size,
            component=self.component, waveform=self.waveform,
            profile=self.profile, profile_coords=self.profile_coords,
            amplitude=amp,
        )

    def expand(self, grid: "Grid") -> List[PointDipole]:
        return self.as_mode_source().expand(grid)
