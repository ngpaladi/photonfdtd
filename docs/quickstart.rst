Quick start
===========

Two small programs to get the feel of it: one that time-steps a field, and one
that solves for a waveguide mode. Both run in a second or two on a laptop.

A point dipole radiating in 2D vacuum
-------------------------------------

Drop a source in an empty box, wrap it in absorbing boundary, and watch it
radiate:

.. code-block:: python

   import photonfdtd as pf

   lam0 = 1.0e-6
   freq0 = pf.C_0 / lam0
   dx = lam0 / 20

   grid = pf.Grid(size=(4e-6, 4e-6), cell_size=dx, pml_layers=(12, 12, 0))
   src = pf.PointDipole(
       position=(0.0, 0.0),
       component="Ez",
       waveform=pf.GaussianPulse(freq0=freq0, fwhm=10e-15),
   )
   mon = pf.FieldMonitor(name="snap", components=("Ez",), interval=20)

   sim = pf.Simulation(grid, sources=[src], monitors=[mon], run_time=200e-15)
   result = sim.run()

``result.fields["snap"]["Ez"]`` comes back as a ``(n_frames, ny, nz)`` array of
snapshots — one frame per recorded step, ready to plot or animate.

Solving a slab waveguide mode
-----------------------------

No time-stepping this time. Hand the solver a cross-section and it returns the
guided modes directly, as an eigenvalue problem:

.. code-block:: python

   import photonfdtd as pf

   clad = pf.Medium.from_index(1.0)
   core = pf.Medium.from_index(2.0)
   slab = pf.Box(center=(0.0, 0.0), size=(0.5e-6, 0.3e-6), medium=core)

   ms = pf.ModeSolver(
       size=(4e-6, 3e-6),
       cell_size=20e-9,
       structures=[slab],
       wavelength=1.55e-6,
       num_modes=2,
   )
   result = ms.solve()
   print(result.n_eff)            # array of effective indices

From here, the :doc:`examples` walk through runnable scripts that produce
figures, and the :doc:`accuracy` guide covers the parts that make the answers
trustworthy — subpixel smoothing, dispersion, full-vectorial modes, and the
differentiable inverse-design stack.
