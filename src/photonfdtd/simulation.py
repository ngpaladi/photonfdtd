"""Simulation - the FDTD time-stepping engine (JAX/XLA).

Workflow

    sim = Simulation(grid, structures, sources, monitors, run_time)
    result = sim.run()
    snapshot = result.fields["my_mon"]["Ez"][time_index]

The solver operates on a uniform Yee grid and supports 1D, 2D, and 3D
simulations. The dimensionality is determined by which axes of `grid.shape`
exceed 1: collapsed axes naturally drop out of the curl operations.

Numerical details
-----------------
- The time step is set to 0.99 of the Courant limit by default.
- Absorbing boundaries are CPML with kappa=1 (Roden & Gedney 2000).
- Sources are soft additive: they add to one field component at one cell.
- Materials are isotropic dielectrics: non-dispersive (eps_r per cell) or
  dispersive (Lorentz/Drude/Sellmeier poles advanced by the ADE method; see
  photonfdtd.jaxbackend / photonfdtd.materials). Anisotropic subpixel smoothing of
  interfaces is available via Simulation(subpixel=True) (photonfdtd.smoothing).

Yee staggering (all six components live on different sub-grids):

    Ex[i,j,k] at (i+1/2,  j,      k     )
    Ey[i,j,k] at (i,      j+1/2,  k     )
    Ez[i,j,k] at (i,      j,      k+1/2 )
    Hx[i,j,k] at (i,      j+1/2,  k+1/2 )
    Hy[i,j,k] at (i+1/2,  j,      k+1/2 )
    Hz[i,j,k] at (i+1/2,  j+1/2,  k     )

H is updated at half-integer times from forward differences of E; E is
updated at integer times from backward differences of H.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import math
import numpy as np

from .constants import C_0
from .grid import Grid
from .geometry import Box
from .sources import PointDipole, ChargedParticle

from .boundaries import cpml_coeffs_1axis, CPMLParams


# ---------------------------------------------------------------------------- #
# Per-array precision control.
#
# `precision` may be a single dtype string applied to everything (the common
# case, fully back-compatible), or a dict that sets the dtype of individual
# arrays. Addressable array keys:
#
#     'Ex','Ey','Ez','Hx','Hy','Hz'  - the six field components
#     'eps_r'                        - per-cell permittivity / E-update coeff
#     'psi'                          - CPML convolutional state + coefficients
#     'monitors'                     - stored monitor snapshots
#
# plus group aliases that fan out to several arrays at once (most specific key
# wins: individual > e_fields/h_fields > fields > compute > default):
#
#     'default'  - fallback for any key not otherwise set (default 'float64')
#     'compute'  - every stepping array (all fields + eps_r + psi); the natural
#                  "compute precision" knob, paired with 'monitors' for storage
#     'fields'   - all six field components
#     'e_fields' / 'h_fields' - the E or H components only
#
# The stepping arrays (fields + eps_r + psi) must resolve to one dtype; only
# 'monitors' may differ. Example - step in double precision but halve monitor
# memory:
#     precision={'compute': 'float64', 'monitors': 'float32'}
# ---------------------------------------------------------------------------- #
_FIELD_KEYS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_E_KEYS = ("Ex", "Ey", "Ez")
_H_KEYS = ("Hx", "Hy", "Hz")
_ARRAY_KEYS = _FIELD_KEYS + ("eps_r", "psi", "monitors")
_COMPUTE_KEYS = _FIELD_KEYS + ("eps_r", "psi")   # all arrays except monitor storage
_GROUP_ALIASES = ("default", "compute", "fields", "e_fields", "h_fields")
_VALID_DTYPES = {"float32": np.float32, "float64": np.float64}


def _resolve_precision(precision) -> Dict[str, np.dtype]:
    """Resolve the `precision` argument to a {array_key: numpy dtype} map.

    `precision` is either a dtype string (applied to every array) or a dict
    keyed by individual array names and/or group aliases (see module notes).
    """
    if isinstance(precision, str):
        if precision not in _VALID_DTYPES:
            raise ValueError("precision must be 'float32' or 'float64'")
        d = _VALID_DTYPES[precision]
        return {k: d for k in _ARRAY_KEYS}

    if not isinstance(precision, dict):
        raise TypeError(
            "precision must be a str ('float32'/'float64') or a dict mapping "
            "array keys to dtype strings"
        )

    allowed = set(_ARRAY_KEYS) | set(_GROUP_ALIASES)
    unknown = set(precision) - allowed
    if unknown:
        raise ValueError(
            f"unknown precision key(s) {sorted(unknown)}; "
            f"allowed keys: {sorted(allowed)}"
        )
    for k, v in precision.items():
        if v not in _VALID_DTYPES:
            raise ValueError(
                f"precision[{k!r}] must be 'float32' or 'float64', got {v!r}"
            )

    def pick(*keys, fallback):
        for k in keys:                       # first listed key present wins
            if k in precision:
                return _VALID_DTYPES[precision[k]]
        return fallback

    default = _VALID_DTYPES[precision.get("default", "float64")]
    out: Dict[str, np.dtype] = {}
    for k in _FIELD_KEYS:
        grp = "e_fields" if k in _E_KEYS else "h_fields"
        out[k] = pick(k, grp, "fields", "compute", fallback=default)
    out["eps_r"] = pick("eps_r", "compute", fallback=default)
    out["psi"] = pick("psi", "compute", fallback=default)
    out["monitors"] = pick("monitors", fallback=default)
    return out


@dataclass
class Result:
    times: np.ndarray
    fields: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    monitor_times: Dict[str, np.ndarray] = field(default_factory=dict)
    flux: Dict[str, float] = field(default_factory=dict)
    # DFTMonitor output: dft[name][component] is a complex array of shape
    # (n_freq, *snapshot_shape); dft_freqs[name] holds the frequencies (Hz).
    dft: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    dft_freqs: Dict[str, np.ndarray] = field(default_factory=dict)



class Simulation:
    """An FDTD run: a grid, the structures on it, sources, monitors, and the
    JAX/XLA engine that steps them (see :mod:`photonfdtd.jaxbackend`).

    Parameters
    ----------
    grid : Grid
        The Yee grid (size, cell size, PML thickness per axis).
    structures : sequence of Box / PolySlab
        Geometry, stamped in order (later structures overwrite earlier ones).
    sources : sequence
        Point / soft sources (``PointDipole``, ``ModeSource``,
        ``UniModeSource``, ``SinglePhotonSource``) and moving charges
        (``ChargedParticle``).
    monitors : sequence
        ``FieldMonitor``, ``FluxMonitor`` and/or ``DFTMonitor`` instances.
    run_time : float
        Physical duration to simulate (s); the number of steps follows from
        the Courant-stable ``dt``.
    courant : float
        Fraction of the Courant limit used for ``dt`` (default 0.99).
    cpml_params : CPMLParams
        CPML grading parameters (see :mod:`photonfdtd.boundaries`).
    verbose : bool
        Print the JAX device the run is placed on.
    precision : str or dict
        ``"float64"`` (default) or ``"float32"`` for every array, or a dict
        keyed by array name (``Ex`` ... ``Hz``, ``eps_r``, ``psi``,
        ``monitors``) or group alias (``default``, ``compute``, ``fields``,
        ``e_fields``, ``h_fields``). The stepping arrays (fields, ``eps_r``,
        ``psi``) must share one dtype; only ``monitors`` may differ.
    subpixel, subpixel_factor : bool, int
        Anisotropic subpixel smoothing of material interfaces and its
        supersampling factor (see :mod:`photonfdtd.smoothing`).
    """

    def __init__(
        self,
        grid: Grid,
        structures: Sequence[Box] = (),
        sources: Sequence[PointDipole] = (),
        monitors: Sequence = (),
        run_time: float = 1e-12,
        courant: float = 0.99,
        cpml_params: CPMLParams = CPMLParams(),
        verbose: bool = False,
        precision="float64",
        subpixel: bool = False,
        subpixel_factor: int = 3,
    ) -> None:
        # Working dtype(s) for fields, CPML state, coefficients and monitor
        # storage. float32 halves memory and roughly doubles throughput (FDTD is
        # memory-bandwidth bound). `precision` may be a single dtype string or a
        # per-array dict - see _resolve_precision / module notes above.
        self.precision = precision
        self.dtypes = _resolve_precision(precision)
        if len({self.dtypes[k] for k in _COMPUTE_KEYS}) > 1:
            raise ValueError(
                "The stepping arrays (fields, eps_r, psi) must share one "
                "precision; only 'monitors' may differ. Use a uniform "
                "'compute' precision."
            )
        # Representative compute dtype (== every stepping array's dtype).
        self.dtype = self.dtypes["Ex"]
        # Anisotropic subpixel smoothing of material interfaces (see
        # photonfdtd.smoothing). Off by default. When on, the E-update uses a
        # per-component (diagonal permittivity tensor) coefficient instead of
        # one shared scalar.
        self.subpixel = bool(subpixel)
        self.subpixel_factor = int(subpixel_factor)
        self.grid = grid
        self.structures = list(structures)
        # Dispersive media: any structure whose medium carries poles activates
        # the auxiliary-differential-equation (ADE) polarization stepping.
        self._has_dispersion = any(
            getattr(getattr(s, "medium", None), "is_dispersive", False)
            for s in self.structures
        )
        if self._has_dispersion and self.subpixel:
            raise NotImplementedError(
                "Subpixel smoothing combined with dispersive media is not "
                "yet supported. Use one or the other."
            )
        # Moving current sources (e.g. ChargedParticle) are deposited along
        # their trajectory each step rather than expanded into fixed-cell soft
        # dipoles (see jaxbackend._particle_plan).
        self.particle_sources: List[ChargedParticle] = []
        expanded: List[PointDipole] = []
        for s in sources:
            if isinstance(s, PointDipole):
                expanded.append(s)
            elif isinstance(s, ChargedParticle):
                self._register_particle(s)
            elif hasattr(s, "expand"):
                expanded.extend(s.expand(grid))
            else:
                raise TypeError(
                    f"Unsupported source type: {type(s).__name__}. "
                    "Pass a PointDipole, ModeSource, SinglePhotonSource, "
                    "or ChargedParticle."
                )
        self.sources = expanded
        self.monitors = list(monitors)
        self.run_time = float(run_time)
        self.courant = float(courant)
        self.cpml_params = cpml_params
        self.verbose = verbose
        # Optional progress hook, called as progress_callback(n_steps, n_steps)
        # once run() completes (the whole time loop is one compiled lax.scan, so
        # there are no intermediate steps to report). Left as a settable
        # attribute so callers can attach it to an already-built Simulation
        # (e.g. one made by from_gdsfactory).
        self.progress_callback: Optional[Callable[[int, int], None]] = None

        # Per-cell relative permittivity (background = 1), stamped on host in
        # the 'eps_r' working dtype from grid + structures.
        self._eps_r: Optional[np.ndarray] = None
        # True once a caller assigns a custom eps_r grid that cannot be
        # regenerated from `structures` (and so cannot be subpixel-smoothed).
        self._eps_r_custom = False
        self._materialize_eps_r()

        # Time step from the Courant condition.
        inv_dl2 = 0.0
        for d in grid.cell_size:
            if d > 0:
                inv_dl2 += 1.0 / d ** 2
        self.dt = self.courant / (C_0 * math.sqrt(inv_dl2))
        self.n_steps = int(math.ceil(self.run_time / self.dt))

        self._build_cpml()

    # ------------------------------------------------------------------ #
    # Relative permittivity grid.
    # ------------------------------------------------------------------ #
    def _materialize_eps_r(self) -> np.ndarray:
        eps = np.ones(self.grid.shape, dtype=self.dtypes["eps_r"])
        for s in self.structures:
            s.stamp(self.grid, eps)
        self._eps_r = eps
        return eps

    def _smoothed_eps_components(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Anisotropic subpixel-smoothed diagonal permittivity (eps_xx, eps_yy,
        eps_zz), one array per E-field component. Requires structures; a
        caller-supplied custom scalar eps_r cannot be smoothed (there is no
        geometry to super-sample) and falls back to that scalar for all three.
        """
        from .smoothing import smooth_permittivity
        if self._eps_r_custom:
            eps = self.eps_r
            return (eps, eps, eps)
        return smooth_permittivity(
            self.grid, self.structures, background_eps=1.0,
            factor=self.subpixel_factor, dtype=self.dtypes["eps_r"],
        )

    @property
    def eps_r(self) -> np.ndarray:
        if self._eps_r is None:
            self._materialize_eps_r()
        return self._eps_r

    @eps_r.setter
    def eps_r(self, value) -> None:
        # Allow callers to stamp a custom permittivity grid. Once set
        # explicitly it is no longer regenerated from `structures`.
        self._eps_r = value
        self._eps_r_custom = True

    # ------------------------------------------------------------------ #
    # Attach sources / monitors after construction. Useful when the
    # Simulation was produced by an adapter (e.g. from_gdsfactory) and the
    # user adds the excitation later.
    # ------------------------------------------------------------------ #
    def add_source(self, source) -> None:
        if isinstance(source, PointDipole):
            self.sources.append(source)
        elif isinstance(source, ChargedParticle):
            self._register_particle(source)
        elif hasattr(source, "expand"):
            self.sources.extend(source.expand(self.grid))
        else:
            raise TypeError(
                f"Unsupported source type: {type(source).__name__}"
            )

    def _register_particle(self, p: ChargedParticle) -> None:
        """Validate and store a moving charged-particle current source."""
        for axis, v in enumerate(p.velocity3):
            if v != 0.0 and self.grid.shape[axis] == 1:
                raise ValueError(
                    f"ChargedParticle has a velocity component on axis {axis}, "
                    "which is collapsed (size 1) in this grid - it has nowhere "
                    "to move or radiate. Drop that component or add that axis."
                )
        if p.speed == 0.0:
            raise ValueError("ChargedParticle velocity must be non-zero.")
        self.particle_sources.append(p)

    def add_monitor(self, monitor) -> None:
        self.monitors.append(monitor)

    def _build_cpml(self) -> None:
        be, ce, bh, ch = [], [], [], []
        for ax in range(3):
            n_cells = self.grid.shape[ax]
            dl = self.grid.cell_size[ax]
            n_pml = self.grid.pml_layers[ax] if n_cells > 1 else 0
            be_ax, ce_ax, bh_ax, ch_ax = cpml_coeffs_1axis(
                n_cells, dl if n_cells > 1 else 1.0, n_pml, self.dt, self.cpml_params
            )
            be.append(be_ax); ce.append(ce_ax)
            bh.append(bh_ax); ch.append(ch_ax)
        self._b_e = be; self._c_e = ce
        self._b_h = bh; self._c_h = ch

    # ------------------------------------------------------------------ #
    def run(self) -> Result:
        """Time-step the simulation on JAX/XLA and return the recorded monitor
        data (see :func:`photonfdtd.jaxbackend.run_jax`)."""
        from .jaxbackend import run_jax
        if self.verbose:
            import jax
            print(f"[photonfdtd] JAX {jax.__version__} on "
                  f"{jax.devices()[0]}, grid {self.grid.shape}, "
                  f"{self.n_steps} steps")
        result = run_jax(self)
        if self.progress_callback is not None:
            self.progress_callback(self.n_steps, self.n_steps)
        return result
