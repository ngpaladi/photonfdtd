photonfdtd
==========

A small, fully-local, open-source **FDTD engine and waveguide mode solver** for
integrated photonics: a Yee-grid time-stepper with CPML absorbing boundaries and
a full-vectorial mode solver, with an API borrowed in spirit from Tidy3D. The
time loop runs on JAX/XLA, one differentiable code path for CPU, GPU and TPU.
The whole thing runs on your own machine. No cloud, no license
server, no account.

This is alpha-stage software, and I'd rather be upfront about which parts are
which: the pieces that exist are tested and correct; the pieces that don't,
don't. See :doc:`api` for the full reference.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   backends
   accuracy
   optimization
   training_tutorial
   active_devices
   layout_scale
   examples
   api

Capabilities
------------

- 1D / 2D / 3D Yee-grid FDTD with the Courant-stable time step selected
  automatically.
- **CPML** absorbing boundaries (Roden & Gedney 2000, kappa = 1) on any
  number of axes, and **periodic** boundaries per axis
  (``Grid(periodic=...)``) for unit cells of gratings, photonic crystals and
  metasurfaces.
- Isotropic dielectric media, non-dispersive or **dispersive** (Lorentz /
  Drude / Sellmeier poles via the ADE method) with a cited material library
  (SiO2, Si, Si3N4, LiNbO3, Au, Ag). See :doc:`accuracy`.
- **Anisotropic subpixel smoothing** of material interfaces for second-order
  boundary accuracy (``Simulation(subpixel=True)``). See :doc:`accuracy`.
- Geometry primitives: axis-aligned :class:`~photonfdtd.Box` and
  arbitrary-polygon :class:`~photonfdtd.PolySlab`.
- Sources: soft point dipole (:class:`~photonfdtd.PointDipole`), distributed
  mode injection (:class:`~photonfdtd.ModeSource`, and the one-way
  :class:`~photonfdtd.UniModeSource`), one-way normal-incidence
  :class:`~photonfdtd.PlaneWave` and :class:`~photonfdtd.GaussianBeam`
  sources from a Huygens current sheet, an energy-normalised
  :class:`~photonfdtd.SinglePhotonSource`, and a moving-charge
  :class:`~photonfdtd.ChargedParticle` that emits Cherenkov radiation.
- Monitors: time-domain snapshots (:class:`~photonfdtd.FieldMonitor`, with
  plane and stride selection), Poynting flux
  (:class:`~photonfdtd.FluxMonitor`), and a running frequency-domain
  :class:`~photonfdtd.DFTMonitor` whose storage scales with the number of
  frequencies rather than timesteps.
- **Near-to-far-field projection** (:func:`~photonfdtd.project_far_field`)
  from plane DFT monitors on a closed box or an open plane, in 2-D and 3-D:
  radiation patterns, radiant intensity, and Cherenkov angular spectra.
- **Mode-decomposition S-parameters** (:func:`~photonfdtd.s_parameters`) from a
  memory-light port plane. See :doc:`accuracy`.
- **Photon-count estimates**: photon spectra and band counts through a DFT
  plane (:func:`~photonfdtd.photon_spectrum`), with analytic Frank-Tamm
  references (:func:`~photonfdtd.frank_tamm_spectrum`) and charge rescaling.
- A **full-vectorial** finite-difference :class:`~photonfdtd.ModeSolver`
  (distinguishes TE/TM; validated against analytic slab dispersion).
- **Layout-scale methods**: the 2.5-D effective-index reduction
  (:class:`~photonfdtd.EffectiveIndex2D`), eigenmode expansion
  (:func:`~photonfdtd.eme_2d`), and unconditionally-stable ADI-FDTD
  (:class:`~photonfdtd.ADISimulation2D`) that steps past the CFL limit. See
  :doc:`layout_scale`.
- **JAX engine**: a JIT-compiled, **differentiable** ``lax.scan`` time loop
  (CPU / GPU / TPU) with checkpointed, O(1)-memory reversible, and disk-tiled
  adjoints, all of which handle dispersive media. See
  :doc:`backends`.
- **Inverse design**: a differentiable 2-D etch-density pipeline
  (:class:`~photonfdtd.EtchedCore`, :func:`~photonfdtd.value_and_grad_density`)
  for topology optimization of PIC devices. See :doc:`accuracy`.
- **Geometry optimization and chip training**: design regions inside a real
  layout, parametric shapes, Adam / optax loops with beta continuation, robust
  (eroded / dilated) objectives, GDS export, and training a chip on a dataset
  like a neural network (:func:`~photonfdtd.train`). See :doc:`optimization`.
- **Active devices**: electro-optic (lithium niobate and other crystals),
  thermo-optic (heaters) and free-carrier (silicon PN) index changes from
  electrostatic and thermal solves, for ``V_pi * L`` and ``P_pi`` in the mode
  solver, voltage sweeps of switches in FDTD, drive optimization by gradient
  descent, and RF drives. Anisotropic (diagonal-tensor) and conductive media.
  See :doc:`active_devices`.
- **Big runs**: a memory planner (:func:`~photonfdtd.estimate_memory`,
  :func:`~photonfdtd.recommend_mode`) that sizes a run against the device
  before it starts, and disk-tiled forward runs and adjoints for grids that
  don't fit (:mod:`photonfdtd.tiled`).
- A :func:`~photonfdtd.from_gdsfactory` adapter that turns a gdsfactory
  ``Component`` into a fully-built :class:`~photonfdtd.Simulation`.

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
