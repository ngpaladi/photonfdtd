"""photonfdtd - local FDTD + waveguide mode solver for integrated photonics.

Capabilities:

- 1D / 2D / 3D Yee-grid FDTD with CPML absorbing boundaries
- Isotropic dielectric media, non-dispersive or dispersive (Lorentz / Drude /
  Sellmeier poles stepped by the ADE method), with a cited material library
  (SiO2, Si, Si3N4, LiNbO3, Au, Ag)
- Anisotropic subpixel (sub-cell) smoothing of material interfaces for
  second-order accuracy at boundaries (``Simulation(subpixel=True)``)
- Axis-aligned Box and arbitrary-polygon PolySlab geometry primitives
- Point-dipole, distributed mode, and energy-normalised single-photon sources
- Field snapshot monitors and a flux monitor through a coordinate plane
- Full-vectorial (FDFD) waveguide mode solver
- :func:`from_gdsfactory` adapter that turns a gdsfactory Component into a
  fully-built Simulation

A NumPy backend is used throughout. It is correct but not fast; for production
runs of large 3D problems a compiled or GPU backend is on the roadmap.
"""
from .constants import C_0, EPS_0, MU_0, ETA_0, Q_E
from .grid import Grid
from .materials import (
    Medium, DispersiveMedium, Pole,
    silica, silicon, silicon_nitride, lithium_niobate, gold, silver,
)
from . import materials
from .geometry import Box, PolySlab
from .sources import (
    GaussianPulse, PointDipole, ModeSource, UniModeSource, SinglePhotonSource,
    ChargedParticle, single_photon_field_amplitude, PLANCK_H,
)
from .monitors import FieldMonitor, FluxMonitor, DFTMonitor
from .storage import CompressedFieldSeries
from .simulation import Simulation
# Differentiable JAX entry point (import is lazy - jax is an optional dep;
# accessing the name only fails if jax is missing when actually called).
from .jaxbackend import value_and_grad_eps as jax_value_and_grad_eps
from .reversible import (
    value_and_grad_eps_reversible as jax_value_and_grad_eps_reversible,
)
from .reversible_pml import (
    value_and_grad_eps_reversible_pml as jax_value_and_grad_eps_reversible_pml,
)
from .mode import ModeSolver
from .smatrix import mode_amplitudes, port_fields, s_parameters
from .eim import (
    Layer, SlabModes, slab_modes, effective_index, EffectiveIndex2D,
)
from .eme import Section, EMEResult, eme_2d, sections_from_eps
from .adi import ADISimulation2D, ADISource, ADIResult
from .design import (
    EtchedCore, value_and_grad_density, conic_filter, tanh_projection,
)
from .memory import (
    estimate_memory, recommend_mode, format_report, available_memory,
)
from . import adapters
from .adapters import from_gdsfactory

__all__ = [
    "C_0", "EPS_0", "MU_0", "ETA_0", "Q_E", "PLANCK_H",
    "Grid", "Medium", "DispersiveMedium", "Pole", "materials",
    "silica", "silicon", "silicon_nitride", "lithium_niobate", "gold", "silver",
    "Box", "PolySlab",
    "GaussianPulse", "PointDipole", "ModeSource", "UniModeSource",
    "SinglePhotonSource",
    "ChargedParticle", "single_photon_field_amplitude",
    "FieldMonitor", "FluxMonitor", "DFTMonitor",
    "CompressedFieldSeries",
    "Simulation", "ModeSolver", "jax_value_and_grad_eps",
    "jax_value_and_grad_eps_reversible",
    "jax_value_and_grad_eps_reversible_pml",
    "mode_amplitudes", "port_fields", "s_parameters",
    # 2.5-D effective-index reduction, eigenmode expansion, ADI-FDTD
    "Layer", "SlabModes", "slab_modes", "effective_index", "EffectiveIndex2D",
    "Section", "EMEResult", "eme_2d", "sections_from_eps",
    "ADISimulation2D", "ADISource", "ADIResult",
    "EtchedCore", "value_and_grad_density", "conic_filter", "tanh_projection",
    "estimate_memory", "recommend_mode", "format_report", "available_memory",
    "adapters", "from_gdsfactory",
]

__version__ = "0.12.0"
