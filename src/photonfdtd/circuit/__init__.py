"""Chip-scale simulation: a whole layout as a circuit of physics-based models.

A full chip is far too large for FDTD, so :class:`Chip` treats it as a
circuit: each component has a model (an S-matrix that depends on wavelength
and on its electrical drive), computed by photonfdtd where it matters (the
:mod:`photonfdtd.lnoi` switches from their cross-section electrostatics and
modes) or taken from the library in :mod:`~photonfdtd.circuit.models`. The
netlist comes from the layout (:meth:`Chip.from_gdsfactory`), the optics are
solved with every wave in the circuit (:mod:`~photonfdtd.circuit.solver`),
and the electrical side (pads, electrodes, heaters, photodetectors, load
resistors) is a nodal network on the same netlist
(:mod:`~photonfdtd.circuit.electrical`).

- :meth:`Chip.solve`: steady state over wavelength with DC pad voltages.
- :meth:`Chip.s_matrix`: the chip's optical S-matrix at given drives.
- :meth:`Chip.transient`: square waves, sines, PRBS on the pads, with each
  modulator's bandwidth, each heater's time constant and each detector's
  bandwidth.
- :func:`~photonfdtd.circuit.sax.to_sax`: the same circuit in sax.
- :mod:`~photonfdtd.circuit.library`: component files, one JSON file per
  part with its layout, ports, pins and model, to share and import like a
  KiCad library part.
"""
from .chip import Chip, ChipState, Transient
from .electrical import DC, PRBS, PWL, Pulse, Sine, Square
from .models import (Heater, Model, Modulator, Photodetector, Resistor, conductor,
                     directional_coupler, from_lnoi, heater, mmi1x2, mmi2x2,
                     photodetector, reciprocal, resistor, terminator, waveguide)
from .solver import Network, Waves
from .sax import to_sax
from .library import ComponentFile, export_component, load_component, sample_sparams

__all__ = ["Chip", "ChipState", "Transient", "DC", "Square", "Sine", "Pulse", "PRBS", "PWL",
           "Model", "Modulator", "Heater", "Photodetector", "Resistor", "conductor",
           "waveguide", "directional_coupler", "mmi1x2", "mmi2x2", "terminator", "heater",
           "photodetector", "resistor", "from_lnoi", "reciprocal", "Network", "Waves", "to_sax",
           "ComponentFile", "export_component", "load_component", "sample_sparams"]
