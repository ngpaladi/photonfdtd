The JAX engine
==============

Every time loop in photonfdtd runs on `JAX <https://jax.readthedocs.io>`_:
:meth:`Simulation.run() <photonfdtd.Simulation.run>`, the differentiable and
reversible adjoints, and :class:`~photonfdtd.ADISimulation2D`. There is no
backend switch to set; ``sim.run()`` builds the update coefficients on the
host, hands them to a JIT-compiled ``jax.lax.scan`` over the timesteps, and
pulls the monitor outputs back as NumPy arrays.

.. code-block:: python

   sim = pf.Simulation(grid, structures, sources, monitors, run_time=200e-15)
   result = sim.run()          # XLA, on whatever device JAX picks

What runs where
---------------

The device is whatever JAX defaults to: the GPU if a CUDA (or ROCm, or TPU)
``jaxlib`` is installed, the CPU otherwise. Pass ``verbose=True`` and the
simulation prints the device it landed on. Because it's one functional code
path, everything works on every device: dispersive media, subpixel smoothing,
moving charges, and all three monitor types.

The setup work stays on the host in NumPy/SciPy: stamping geometry, building the
CPML coefficient tables, the subpixel smoothing, the mode solver, the
effective-index and EME solvers, and the S-parameter overlaps. None of that is
time-stepping, and none of it is where the time goes.

The first ``run()`` of a given configuration pays the XLA compile, typically a
few seconds. For a run that finishes in milliseconds, the compile is most of the
wall time. For anything substantial, fusion (and the GPU, if you have one) more
than pays for it.

GPU setup
---------

.. code-block:: bash

   pip install "jax[cuda12]"   # NVIDIA / CUDA 12 GPU

Two things I've had to set on my own machine:

- If JAX quietly falls back to CPU with a "cuSPARSE not found"-style error, the
  CUDA libraries aren't on the loader path. Point ``LD_LIBRARY_PATH`` at the
  ``nvidia/*/lib`` directories the ``jax[cuda12]`` install dropped in
  site-packages.
- ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` makes JAX allocate VRAM on demand
  instead of grabbing most of the card up front. Set it when you share the GPU
  or want honest memory numbers.

Precision
---------

``precision="float32"`` halves memory and, because FDTD is bandwidth-bound,
roughly doubles throughput. The default ``"float64"`` switches on
``jax_enable_x64`` for you. You can also pass a dict
(see :class:`~photonfdtd.Simulation`). The stepping arrays (fields, ``eps_r``,
CPML state) must share one dtype, but monitor storage may differ, e.g.
``precision={"compute": "float64", "monitors": "float32"}`` to step in double
and store snapshots in single. Single precision drifts from double by round-off
accumulated over the run. That's plenty for field movies and most
transmission numbers, and not what you want for a resonance you're trying to
pin to five digits.

Memory
------

The whole domain lives on the device: the six fields, the update coefficients,
a dense CPML state, and every recorded monitor frame. There's no disk-tiled
fallback, so size a big run before you start it.
:func:`~photonfdtd.estimate_memory`, :func:`~photonfdtd.recommend_mode` and
:func:`~photonfdtd.format_report` give the peak bytes of a forward run and of
each adjoint mode. :func:`~photonfdtd.available_memory` reports what the
machine actually has (VRAM if a GPU is present, else host RAM), which is also
what ``recommend_mode`` budgets against by default. It returns
``"infeasible"`` when nothing fits.

To keep the monitor side small:

- record a single plane (``FieldMonitor(plane_z=...)``)
- stride it (``downsample=``)
- or use a :class:`~photonfdtd.DFTMonitor`, whose storage scales with the
  number of frequencies rather than timesteps

The memory notes in :doc:`accuracy` go deeper on the adjoint side.

Gradients
---------

Because the stepper is a pure function of the permittivity-derived
coefficients, ``jax.grad`` of any scalar built from the monitor outputs is a
time-domain adjoint. Start from :func:`~photonfdtd.jax_value_and_grad_eps`
(checkpointed), :func:`~photonfdtd.jax_value_and_grad_eps_reversible`
(O(1) memory in timesteps for closed domains), or the etch-density pipeline
:func:`~photonfdtd.value_and_grad_density`; :doc:`accuracy` walks through them.
