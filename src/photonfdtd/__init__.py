"""photonfdtd - local FDTD + waveguide mode solver for integrated photonics.

Capabilities:

- 1D / 2D / 3D Yee-grid FDTD with CPML absorbing boundaries
- Isotropic dielectric media, non-dispersive or dispersive (Lorentz / Drude /
  Sellmeier poles stepped by the ADE method), with a cited material library
  (SiO2, Si, Si3N4, LiNbO3, Au, Ag)
- Anisotropic subpixel (sub-cell) smoothing of material interfaces for
  second-order accuracy at boundaries (``Simulation(subpixel=True)``)
- Axis-aligned Box and arbitrary-polygon PolySlab geometry primitives
- Point-dipole, distributed mode (bidirectional and one-way), energy-normalised
  single-photon, and moving-charge (Cherenkov) sources
- Field-snapshot, flux, and running-DFT monitors; mode-decomposition
  S-parameters from a port plane (:func:`s_parameters`)
- Photon-count estimates: photon spectra through a DFT plane
  (:func:`photon_spectrum`) and analytic Frank-Tamm Cherenkov yields
  (:mod:`photonfdtd.photons`)
- Full-vectorial (FDFD) waveguide mode solver
- 2.5-D effective-index reduction (:mod:`photonfdtd.eim`), eigenmode
  expansion (:mod:`photonfdtd.eme`), and CFL-free ADI-FDTD
  (:mod:`photonfdtd.adi`) for layout-scale devices
- JAX/XLA time stepping (CPU / GPU / TPU from one code path), differentiable
  with checkpointed and reversible time-domain adjoints, and a 2-D etch-density inverse-design pipeline
  (:mod:`photonfdtd.design`)
- Gradient-descent geometry optimization (:mod:`photonfdtd.optimize`): design
  regions, parametric shapes, Adam / optax, beta continuation, robust
  (eroded/dilated) objectives, GDS export (:mod:`photonfdtd.gds`), and
  training a chip like a neural network on a dataset (:mod:`photonfdtd.train`)
- :func:`from_gdsfactory` adapter that turns a gdsfactory Component into a
  fully-built Simulation
- Active devices (:mod:`photonfdtd.active`): electro-optic (lithium niobate and
  other crystals), thermo-optic (heaters) and free-carrier (silicon PN) index
  changes, from electrostatic and thermal solves (:mod:`photonfdtd.statics`),
  for voltage and heater sweeps in the mode solver and FDTD, and
  time-varying (RF) drives

Every time loop runs on JAX (:mod:`photonfdtd.jaxbackend`, and the ADI-FDTD
stepper in :mod:`photonfdtd.adi`); :mod:`photonfdtd.memory` sizes a run against
the device before it starts, and a run too big for the device, forward or
adjoint, streams from disk in x-slabs (:mod:`photonfdtd.tiled`).
"""
from .constants import C_0, EPS_0, MU_0, ETA_0, Q_E
from .grid import Grid
from .materials import (
    Medium, DispersiveMedium, Pole, AnisotropicMedium,
    silica, silicon, silicon_nitride, lithium_niobate, gold, silver,
)
from . import materials
from .geometry import Box, PolySlab
from .sources import (
    GaussianPulse, PointDipole, ModeSource, UniModeSource, SinglePhotonSource,
    PlaneWave, GaussianBeam,
    ChargedParticle, single_photon_field_amplitude, PLANCK_H,
)
from .monitors import FieldMonitor, FluxMonitor, DFTMonitor
from .farfield import (FarField, project_far_field, spherical_directions,
                       polar_directions)
from .simulation import Simulation
# Differentiable JAX entry points.
from .jaxbackend import value_and_grad_eps as jax_value_and_grad_eps
from .reversible import (
    value_and_grad_eps_reversible as jax_value_and_grad_eps_reversible,
)
from .reversible_pml import (
    value_and_grad_eps_reversible_pml as jax_value_and_grad_eps_reversible_pml,
)
from .tiled import value_and_grad_eps_tiled as jax_value_and_grad_eps_tiled
from .mode import ModeSolver
from .smatrix import mode_amplitudes, port_fields, s_parameters
from .photons import (
    PhotonSpectrum, photon_spectrum, photon_count, frank_tamm_spectrum,
    frank_tamm_photons,
)
from .eim import (
    Layer, SlabModes, slab_modes, effective_index, EffectiveIndex2D,
)
from .eme import Section, EMEResult, eme_2d, sections_from_eps
from .adi import ADISimulation2D, ADISource, ADIResult
from .design import (
    EtchedCore, value_and_grad_density, conic_filter, tanh_projection,
    DesignRegion, ShapeDesign, polygon_density,
)
from .jaxbackend import make_value_and_grad_params, value_and_grad_params
from .optimize import (
    Adam, minimize, OptResult, beta_continuation, binarization, robust,
    ModeTransmission, FieldProjection, combine,
)
from .train import PhotonicModel, train, fit_transfer, accuracy
from .gds import export_gds
from .memory import (
    estimate_memory, recommend_mode, format_report, available_memory,
)
from . import adapters
from .adapters import from_gdsfactory
from . import statics, active
from .statics import solve_potential, solve_thermal
from .active import (
    ActiveMedium, ActiveRegion, Electrode, Heater, PNJunction,
    Sinusoid, Ramp, Step,
)

__all__ = [
    "C_0", "EPS_0", "MU_0", "ETA_0", "Q_E", "PLANCK_H",
    "Grid", "Medium", "DispersiveMedium", "Pole", "AnisotropicMedium", "materials",
    "silica", "silicon", "silicon_nitride", "lithium_niobate", "gold", "silver",
    "Box", "PolySlab",
    "GaussianPulse", "PointDipole", "ModeSource", "UniModeSource",
    "PlaneWave", "GaussianBeam",
    "FarField", "project_far_field", "spherical_directions", "polar_directions",
    "SinglePhotonSource",
    "ChargedParticle", "single_photon_field_amplitude",
    "FieldMonitor", "FluxMonitor", "DFTMonitor",
    "Simulation", "ModeSolver", "jax_value_and_grad_eps",
    "jax_value_and_grad_eps_reversible",
    "jax_value_and_grad_eps_reversible_pml", "jax_value_and_grad_eps_tiled",
    "mode_amplitudes", "port_fields", "s_parameters",
    "PhotonSpectrum", "photon_spectrum", "photon_count", "frank_tamm_spectrum",
    "frank_tamm_photons",
    # 2.5-D effective-index reduction, eigenmode expansion, ADI-FDTD
    "Layer", "SlabModes", "slab_modes", "effective_index", "EffectiveIndex2D",
    "Section", "EMEResult", "eme_2d", "sections_from_eps",
    "ADISimulation2D", "ADISource", "ADIResult",
    "EtchedCore", "value_and_grad_density", "conic_filter", "tanh_projection",
    # geometry optimization and chip training
    "DesignRegion", "ShapeDesign", "polygon_density",
    "make_value_and_grad_params", "value_and_grad_params",
    "Adam", "minimize", "OptResult", "beta_continuation", "binarization",
    "robust", "ModeTransmission", "FieldProjection", "combine",
    "PhotonicModel", "train", "fit_transfer", "accuracy", "export_gds",
    "estimate_memory", "recommend_mode", "format_report", "available_memory",
    "adapters", "from_gdsfactory",
    # active devices: electro-optic, thermo-optic and free-carrier tuning
    "statics", "active", "solve_potential", "solve_thermal",
    "ActiveMedium", "ActiveRegion", "Electrode", "Heater", "PNJunction",
    "Sinusoid", "Ramp", "Step",
]

__version__ = "0.16.0"
