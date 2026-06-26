photonfdtd
==========

A small, fully-local open-source **FDTD + waveguide mode solver** written in
Python/NumPy. It implements a Yee-grid FDTD time-stepper with CPML absorbing
boundaries and a 2D scalar Helmholtz mode solver, with an API intentionally
similar in spirit to Tidy3D.

This is alpha-stage software. The pieces that exist are tested and correct;
the pieces that don't, don't. See :doc:`api` for the full reference.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   examples
   api

Capabilities
------------

- 1D / 2D / 3D Yee-grid FDTD with the Courant-stable time step selected
  automatically.
- **CPML** absorbing boundaries (Roden & Gedney 2000, kappa = 1) on any
  number of axes.
- Isotropic non-dispersive dielectric media stamped per cell.
- Geometry primitives: axis-aligned :class:`~photonfdtd.Box` and
  arbitrary-polygon :class:`~photonfdtd.PolySlab`.
- Sources: soft point dipole (:class:`~photonfdtd.PointDipole`), distributed
  mode injection (:class:`~photonfdtd.ModeSource`), an energy-normalised
  :class:`~photonfdtd.SinglePhotonSource`, and a moving-charge
  :class:`~photonfdtd.ChargedParticle` that emits Cherenkov radiation.
- Field-snapshot and flux monitors.
- A 2D scalar Helmholtz :class:`~photonfdtd.ModeSolver`.
- A :func:`~photonfdtd.from_gdsfactory` adapter that turns a gdsfactory
  ``Component`` into a fully-built :class:`~photonfdtd.Simulation`.

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
