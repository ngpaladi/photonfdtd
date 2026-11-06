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

By default the whole domain lives on the device: the six fields, the update
coefficients, a dense CPML state, and every recorded monitor frame.
:func:`~photonfdtd.estimate_memory`, :func:`~photonfdtd.recommend_mode` and
:func:`~photonfdtd.format_report` give the peak bytes of a forward run and of
each adjoint mode. :func:`~photonfdtd.available_memory` reports what the
machine actually has (VRAM if a GPU is present, else host RAM), which is also
what ``recommend_mode`` budgets against by default.

When a forward run doesn't fit, it can run **disk-tiled**
(:mod:`photonfdtd.tiled`). The fields and CPML state live in memory-mapped
files, and the domain streams through the device in slabs along x. Each slab
is loaded with a halo and advanced ``steps_per_tile`` steps per load, so disk
traffic is one pass over the grid per block of steps rather than per step.
The OS page cache turns host RAM into a middle tier for grids that fit in RAM
but not in VRAM. Results match the in-memory engine to floating-point
reassociation (~1e-14 relative)::

    result = sim.run(disk_tiled=True, tile_cells=64, steps_per_tile=16,
                     workdir="/scratch/run1")

``sim.run()`` falls back to it on its own, with a ``RuntimeWarning``, when the
forward estimate exceeds the available device memory; ``disk_tiled=False``
turns that off. ``tile_cells`` defaults to the largest slab that fits half the
device budget. ``recommend_mode(differentiable=False)`` returns
``"disk_tiled"`` when a forward run doesn't fit but a slab does, and
``"infeasible"`` when nothing does. Dispersive media, moving charges and
subpixel smoothing all work tiled; monitor outputs stay in RAM. It is a few
times slower than an in-memory run: 2.3-5.6x on a 240x120x120 float32 grid,
less with wider slabs and more steps per load.

Gradients tile the same way. :func:`~photonfdtd.jax_value_and_grad_eps_tiled`
has the contract of :func:`~photonfdtd.jax_value_and_grad_eps` and returns
``d loss / d eps_r`` as a host array::

    value, grad = pf.jax_value_and_grad_eps_tiled(sim, loss, tile_cells=32,
                                                  workdir="/scratch/adj")

Each block of ``steps_per_tile`` steps is pulled back slab by slab with
``jax.vjp`` of the same compiled slab update. The block inputs come from
checkpoints on disk: every ``checkpoint_every`` blocks (default
``sqrt`` of the block count) the state is kept, and the blocks in between are
recomputed during the backward pass, so disk holds about
``2 * sqrt(n_steps / steps_per_tile)`` copies of the state and the compute is
about twice a forward run plus the VJPs. ``checkpoint_every=1`` keeps every
block and skips the recomputation. The gradient matches the in-memory reverse
pass to ~1e-15 (``tests/test_tiled.py``). ``recommend_mode`` returns
``"adjoint_disk_tiled"`` when no in-memory adjoint fits but a slab and its VJP
do. Subpixel smoothing is not supported here, as in the in-memory
``eps_r`` gradient.

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
(O(1) memory in timesteps for closed domains),
:func:`~photonfdtd.jax_value_and_grad_eps_tiled` (disk-tiled, for grids too
big for the device), or the etch-density pipeline
:func:`~photonfdtd.value_and_grad_density`; :doc:`accuracy` walks through them.
All of them handle dispersive media, with the poles held fixed (the reversible
ones need lossless poles).
