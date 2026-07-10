"""photonfdtd - local FDTD + waveguide mode solver for integrated photonics.

Capabilities:

- 1D / 2D / 3D Yee-grid FDTD with CPML absorbing boundaries
- Isotropic non-dispersive dielectric media (per-cell epsilon_r)
- Axis-aligned Box and arbitrary-polygon PolySlab geometry primitives
- Point-dipole, distributed mode, and energy-normalised single-photon sources
- Field snapshot monitors and a flux monitor through a coordinate plane
- 2D scalar (Helmholtz) waveguide mode solver
- :func:`from_gdsfactory` adapter that turns a gdsfactory Component into a
  fully-built Simulation

A NumPy backend is used throughout. It is correct but not fast; for production
runs of large 3D problems a compiled or GPU backend is on the roadmap.
"""
from .constants import C_0, EPS_0, MU_0, ETA_0, Q_E
from .grid import Grid
from .materials import Medium
from .geometry import Box, PolySlab
from .sources import (
    GaussianPulse, PointDipole, ModeSource, SinglePhotonSource,
    ChargedParticle, single_photon_field_amplitude, PLANCK_H,
)
from .monitors import FieldMonitor, FluxMonitor, DFTMonitor
from .storage import CompressedFieldSeries
from .simulation import Simulation
# Differentiable JAX entry point (import is lazy - jax is an optional dep;
# accessing the name only fails if jax is missing when actually called).
from .jaxbackend import value_and_grad_eps as jax_value_and_grad_eps
from .mode import ModeSolver
from . import adapters
from .adapters import from_gdsfactory

__all__ = [
    "C_0", "EPS_0", "MU_0", "ETA_0", "Q_E", "PLANCK_H",
    "Grid", "Medium", "Box", "PolySlab",
    "GaussianPulse", "PointDipole", "ModeSource", "SinglePhotonSource",
    "ChargedParticle", "single_photon_field_amplitude",
    "FieldMonitor", "FluxMonitor", "DFTMonitor",
    "CompressedFieldSeries",
    "Simulation", "ModeSolver", "jax_value_and_grad_eps",
    "adapters", "from_gdsfactory",
]

__version__ = "0.4.0"
