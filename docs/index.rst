photonfdtd
==========

A small, fully-local, open-source **FDTD engine and waveguide mode solver**,
written in Python and NumPy. Under the hood it's a Yee-grid time-stepper with
CPML absorbing boundaries and a full-vectorial mode solver, with an API borrowed
in spirit from Tidy3D — but the whole thing runs on your own machine. No cloud,
no license server, no account.

This is alpha-stage software, and I'd rather be upfront about which parts are
which: the pieces that exist are tested and correct; the pieces that don't,
don't. See :doc:`api` for the full reference.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   accuracy
   examples
   api

Capabilities
------------

- 1D / 2D / 3D Yee-grid FDTD with the Courant-stable time step selected
  automatically.
- **CPML** absorbing boundaries (Roden & Gedney 2000, kappa = 1) on any
  number of axes.
- Isotropic dielectric media, non-dispersive or **dispersive** (Lorentz /
  Drude / Sellmeier poles via the ADE method) with a cited material library
  (SiO2, Si, Si3N4, LiNbO3, Au, Ag). See :doc:`accuracy`.
- **Anisotropic subpixel smoothing** of material interfaces for second-order
  boundary accuracy (``Simulation(subpixel=True)``). See :doc:`accuracy`.
- Geometry primitives: axis-aligned :class:`~photonfdtd.Box` and
  arbitrary-polygon :class:`~photonfdtd.PolySlab`.
- Sources: soft point dipole (:class:`~photonfdtd.PointDipole`), distributed
  mode injection (:class:`~photonfdtd.ModeSource`), an energy-normalised
  :class:`~photonfdtd.SinglePhotonSource`, and a moving-charge
  :class:`~photonfdtd.ChargedParticle` that emits Cherenkov radiation.
- Field-snapshot and flux monitors.
- A **full-vectorial** finite-difference :class:`~photonfdtd.ModeSolver`
  (distinguishes TE/TM; validated against analytic slab dispersion).
- A :func:`~photonfdtd.from_gdsfactory` adapter that turns a gdsfactory
  ``Component`` into a fully-built :class:`~photonfdtd.Simulation`.

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
