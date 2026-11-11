Chip simulation: a layout as a circuit
======================================

A whole chip is millimetres to centimetres across, far beyond any FDTD run.
:mod:`photonfdtd.circuit` simulates it as a circuit instead: every component
gets a model (its S-matrix as a function of wavelength and of its electrical
drive) and the layout's netlist joins them. The models come from photonfdtd
where the physics matters: the :mod:`photonfdtd.lnoi` switches from the
electrostatics and modes of their own cross-section. The electrical side (bond
pads, metal routes, electrodes, heaters, photodetectors, load resistors) is a
nodal network on the same netlist. A chip of tens of components solves in
milliseconds per wavelength.

.. code-block:: python

   import gdsfactory as gf
   from photonfdtd import circuit, lnoi

   chip = circuit.Chip.from_gdsfactory(layout)          # any gdsfactory Component
   st = chip.solve(1.55, drives={"pad": 4.4, "gnd": 0.0}, inputs={"laser": 1e-3})
   st.detector_current, st.power, st.voltage

   tr = chip.transient(t, drives={"pad": circuit.PRBS(0, 4.4, rate=10e9), "gnd": 0.0},
                       inputs={"laser": 1e-3})

Example 12 (:doc:`examples`) builds a lithium niobate chip (an MZI switch
wired to a pad, photodetectors on both outputs) and runs all three analyses.

From layout to circuit
----------------------

:meth:`~photonfdtd.circuit.Chip.from_gdsfactory` reads
``component.get_netlist()``, sorts the nets into optical and electrical by
their port types, and gives each instance a model:

.. list-table::
   :widths: 40 60

   * - :mod:`photonfdtd.lnoi` cells
     - their voltage models (:func:`~photonfdtd.lnoi.model`), solved once per
       design and cached on disk
   * - :func:`photonfdtd.circuit.cells.photodetector`
     - a photodetector (responsivity, dark current, bandwidth from the cell)
   * - ``straight_heater_metal*``
     - a heater (resistor plus thermo-optic phase)
   * - cells with only electrical ports (pads, metal routes, vias)
     - one conductor
   * - two-port optical cells (straights, bends, tapers)
     - a waveguide of their length, with ``waveguide=dict(neff, ng, loss_db_cm)``
   * - ``mmi1x2``, ``mmi2x2``
     - ideal splitters

Pass ``models={"cell or instance name": model}`` for anything else (an
unknown cell raises and names itself). The library in
:mod:`photonfdtd.circuit.models` has waveguides, couplers, MMIs, heaters,
detectors and resistors; a model is any :class:`~photonfdtd.circuit.Model`, a
function ``fn(wl, **settings) -> {(p_in, p_out): S}`` in sax's convention
(``wl`` in um). A :class:`~photonfdtd.circuit.Chip` can also be built by hand
from models and nets.

Drives and results
------------------

``drives`` hold nets at voltages: keys are the layout's electrical port names,
pad instance names, ``"instance,port"`` or ``(instance, conductor)``. Every
electrode, heater terminal and detector terminal is a conductor; metal
connected through routes is one net. Heaters dissipate ``(V_a - V_b)^2 / R``;
photodetectors drive ``responsivity * P + dark_current`` from cathode to
anode, which can feed back through load resistors into the optics (the solve
iterates to self-consistency). A net carrying an electrode or heater with no
drive and no resistive path to one floats; it is taken as 0 V with a warning.

:meth:`~photonfdtd.circuit.Chip.solve` returns a
:class:`~photonfdtd.circuit.ChipState` over wavelength: the power at every
external optical port, the light and photocurrent at every detector, every net
voltage, the current each drive supplies and every heater's power.
:meth:`~photonfdtd.circuit.Chip.s_matrix` gives the chip's own S-matrix, and
:func:`~photonfdtd.circuit.to_sax` the same circuit for ``sax.circuit`` (the
two agree to round-off).

Component files
---------------

A component file (``*.pcomp.json``, plain-text JSON) is a part you can share
and import, like a KiCad library part: one file with

- the **layout**: fixed polygons per GDS layer;
- the **ports** (optical and electrical) and the **pins**: which electrical
  ports are one piece of metal;
- the **model**, in up to three forms: ``physics`` (solved parameters: an LN
  switch's ``d n_eff / dV`` and coupling, a detector's responsivity, a
  heater's ``P_pi``, a waveguide's indices, or a whole sub-circuit of such
  parts), ``sparams`` (S-matrices sampled over wavelength and drive voltages,
  from FDTD, a measurement or the physics), and ``recipe`` (what the physics
  was solved from, to re-solve at another wavelength or mesh);
- optionally a **generator** to rebuild the part with other parameters: a
  cell function of photonfdtd or gdsfactory (a registered PDK cell), or other
  code (another module, or Python source carried in the file), which runs
  only when loaded with ``trust=True``. A rebuilt part that photonfdtd cannot
  model from its geometry keeps the file's model, solved for the file's own
  parameters.

.. code-block:: python

   from photonfdtd.circuit import export_component, load_component, sample_sparams

   export_component(lnoi.mzi_switch(arm_length=5000), "mzi_5mm", description="5 mm MZI switch")
   part = load_component("mzi_5mm.pcomp.json")
   cell = part.cell()                 # place it in any gdsfactory layout
   part.model()                       # its circuit model (no re-solve)
   part.resolve(wavelength=1.31)      # re-solve from the recipe
   part.model(wavelength=1.31)        # the same, cached, as a circuit model
   part.regenerate(arm_length=8000)   # rebuild through the generator

A component holding several instances (a sub-layout) exports as a
``circuit``: its parts' physics, the connections between them, and the ports
and pads it exposes as its own pins. Circuits built from a layout that uses a
loaded part's cell find its model by themselves. Built with
``Chip.from_gdsfactory(..., wavelength=1.31)``, an LN part whose file carries
a recipe is re-solved at that wavelength, like the LN cells placed directly.

A sub-circuit is one optical part with electrical pins in the outer circuit:
it drives its internal electrodes through its pins, but its heaters,
resistors and detectors are not part of the outer electrical network. Their
currents are not drawn from the outer drives, and a path through one of its
resistors does not count as a return path for an outer photodetector.

Time domain
-----------

:meth:`~photonfdtd.circuit.Chip.transient` steps the drives through time with
the waveforms in :mod:`photonfdtd.circuit.electrical`
(:class:`~photonfdtd.circuit.DC`, :class:`~photonfdtd.circuit.Square`,
:class:`~photonfdtd.circuit.Sine`, :class:`~photonfdtd.circuit.Pulse`,
:class:`~photonfdtd.circuit.PRBS`, :class:`~photonfdtd.circuit.PWL`, or any
function of ``t``). Each modulator's ``bandwidth``, each heater's thermal
``tau`` and each detector's ``bandwidth`` act as first-order low-passes.

The run is quasi-static: at each time the optics are in steady state for the
(filtered) drives. That holds while the drives change slowly next to the time
light takes to cross the chip, about 10 ps per mm of LN waveguide, so a GHz
drive on a centimetre circuit is at the limit, and resonators with photon
lifetimes near the bit period are outside it. The electrode bandwidth is a
parameter, not derived from the electrode's RF design.

Approximations
--------------

- The LN models are solved at one wavelength; elsewhere their phases scale as
  1/λ with the coupling and ``d n_eff / dV`` held. A balanced MZI therefore
  comes out achromatic at 0 V, where a real directional coupler's split drifts
  with wavelength.
- Waveguides take ``n_eff`` and ``n_g`` you give (from a mode solve, say), with
  first-order dispersion.
- Reflections are zero in the built-in models; the solver handles them if a
  model has them.
