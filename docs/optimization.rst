Geometry optimization and chip training
=======================================

:doc:`accuracy` shows how to get the gradient of a figure of merit with respect
to the etch pattern. This page covers what you do with that gradient. You
optimize a region of a real layout with gradient descent, keep the design
fabricable, write it out as GDS, and, if you want, train a chip on a dataset
the way you would train a neural network.

Everything here sits on one primitive. That primitive is
:func:`~photonfdtd.make_value_and_grad_params`: it takes a coefficient builder
``ce_of(params, **aux)`` and a loss over the monitor outputs, and it returns
``vg(params, **aux) -> (value, grad)``. Each call to ``vg`` is one forward and
one adjoint FDTD run, compiled once. The ``aux`` keyword arguments, such as
the projection sharpness ``beta``, are traced, so changing them between
iterations does not recompile. On a small 2-D test, the first call takes about
0.6 s and every later call about 0.02 s.

Design regions
--------------

A :class:`~photonfdtd.DesignRegion` is a box of the core layer whose etch is
free. Everything outside the box keeps the permittivity the
:class:`~photonfdtd.Simulation` was built with: the input and output
waveguides, cladding and PML are unchanged bit for bit. Inside the box, the
density ``rho`` goes through the same pipeline as
:class:`~photonfdtd.EtchedCore`:

* a conic filter, which sets the minimum feature size;
* a ``tanh`` projection, which pushes the design toward binary;
* the anisotropic sidewall smoothing.

Before filtering, the density is padded with the *fixed* layout around the
box. A waveguide stub that ends at the box edge is therefore part of what the
filter sees, and joins the design instead of being filtered away.

.. code-block:: python

   region = pf.DesignRegion(center=(0, 0), size=(1.6e-6, 1.6e-6),
                            core_z=(-0.11e-6, 0.11e-6),
                            eps_solid=3.48**2, eps_void=1.44**2,
                            filter_radius=120e-9)
   loss = lambda out: -fom(out)                    # maximize a figure of merit
   vg = pf.make_value_and_grad_params(sim, region.ce_builder(sim), loss)
   res = pf.minimize(vg, jnp.full(region.shape(sim.grid), 0.5),
                     pf.Adam(0.05), n_iter=60,
                     schedule=pf.beta_continuation(2, 2, every=15, beta_max=32))
   rho_phys = region.rho_phys(sim, res.x, beta=32)
   print(pf.binarization(rho_phys))               # 1.0 = fully binary

:func:`~photonfdtd.minimize` runs projected gradient descent. Each iteration it
takes a step, clips the parameters to ``[0, 1]``, and records the value,
gradient norm and schedule values in ``res.history``.
:func:`~photonfdtd.beta_continuation` doubles the projection sharpness on a
fixed schedule, so material moves freely early on and the design binarizes
later. Pass ``checkpoint="run.npz"`` to save the state after every iteration;
rerunning the same call resumes from the file and gives the same result as an
uninterrupted run.

**Optimizers.** :class:`~photonfdtd.Adam` is built in. It follows optax's
``init`` / ``update`` protocol, so any optax transformation can be used
instead, e.g. ``optax.chain(optax.clip_by_global_norm(1.0), optax.adam(0.05))``.
Scale the loss to order one: like any Adam, a loss of order ``1e-30`` (raw
FDTD field units) leaves the gradient below Adam's ``eps`` and the design does
not move.

Objectives
----------

:class:`~photonfdtd.ModeTransmission` gives a port's modal amplitude, and its
power ``|alpha|**2``, inside the loss. It projects a port-plane
:class:`~photonfdtd.DFTMonitor` onto a solved mode (see
:func:`~photonfdtd.mode_amplitudes`); with a unit-power one-way source this is
``|S|**2``. :class:`~photonfdtd.FieldProjection` is a cheaper linear readout
onto any weight profile, which is useful on 2-D grids.
:func:`~photonfdtd.combine` merges several objectives (ports or wavelengths)
by a weighted mean, or by the worst one (``reduce="min"``) for a balanced
device.

Robust designs
--------------

A design that only works when it is etched exactly as drawn is not much use.
:func:`~photonfdtd.robust` evaluates the eroded, nominal and dilated designs.
These come from shifting the projection threshold ``eta`` to, for example,
0.55 / 0.5 / 0.45 (Wang, Lazarov & Sigmund 2011). It then optimizes the worst
case (``reduce="max"``) or the mean. ``eta`` is a traced argument, so the three
variants share one compiled program; each iteration costs three simulations.

.. code-block:: python

   vg_robust = pf.robust(vg, etas=(0.45, 0.5, 0.55), reduce="max")
   res = pf.minimize(vg_robust, rho0, pf.Adam(0.05), n_iter=60, schedule=...)

Parametric shapes
-----------------

When the device is a known topology with a few dimensions to tune (a taper's
widths, a coupler's gap, a ring's waveguide width), optimize those numbers
directly. :class:`~photonfdtd.ShapeDesign` takes a function from parameters to
polygons built from JAX operations. Two helpers are provided:
:func:`photonfdtd.design.rect` and :func:`photonfdtd.design.taper`.
:func:`~photonfdtd.polygon_density` rasterizes the polygons to a per-cell fill
fraction that is differentiable in the vertex coordinates, and the sidewalls
get the same anisotropic subpixel rule as density designs:

.. code-block:: python

   from photonfdtd.design import taper
   shape = pf.ShapeDesign(region, lambda p: [taper(-0.8e-6, 0.8e-6, p[0], p[1])])
   vg = pf.make_value_and_grad_params(sim, shape.ce_builder(sim), loss)
   res = pf.minimize(vg, jnp.array([0.4e-6, 0.8e-6]), pf.Adam(5e-9),
                     n_iter=30, bounds=(0.2e-6, 1.4e-6))

The shape gradient agrees with Richardson-extrapolated finite differences to
1e-5 (``tests/test_optimize.py``).

GDS export
----------

:func:`~photonfdtd.export_gds` writes the result for fabrication. It needs
``gdstk``: ``pip install photonfdtd[gds]``. A density design is binarized at
0.5 and contoured. Nested contours become holes, and the result is clipped to
the design box. A parametric design is written from its polygons directly:

.. code-block:: python

   pf.export_gds("device.gds", region=region, rho=rho_phys, grid=sim.grid)
   pf.export_gds("taper.gds", polygons=shape.polygons(res.x))

Training a chip like a neural network
-------------------------------------

Here the etch pattern plays the role of the network's weights.

* **Input.** A sample is the vector of complex amplitudes driving the chip's
  ``N`` input ports, optionally one vector per wavelength channel.
* **Output.** The model output is what detectors read at its ``M`` output
  ports: the powers ``|a|**2``, which are the network's nonlinearity, or, with
  ``detection="coherent"``, the complex amplitudes.
* **Training.** :func:`~photonfdtd.train` fits the weights to a dataset
  ``(X, Y)`` by minibatch gradient descent. It uses a cross-entropy loss over
  the output-power fractions for classification, a mean-squared error, or a
  loss you supply.

The trick that makes this affordable is linearity. For fixed weights, Maxwell's
equations make the chip an exact complex transfer matrix ``T`` of shape
``(n_freq, M, N)``. :class:`~photonfdtd.PhotonicModel` runs one simulation per
input port to get ``T``, and every sample in every minibatch is then
``a = T @ x``, a matrix product rather than an FDTD run. Differentiating the
batch loss pulls ``dL/dT`` back through the same ``N`` simulations. So **one
optimizer step costs N forward and N adjoint simulations, however large the
batch or the dataset**.

.. code-block:: python

   sims = [chip(port=0), chip(port=1)]       # identical except the driven input
   model = pf.PhotonicModel(sims, region, readouts)   # readouts: one per output
   res = pf.train(model, rho0, X, Y, epochs=8, batch_size=32,
                  optimizer=pf.Adam(0.04),
                  schedule=pf.beta_continuation(2, 2, 16, 16),
                  validation=(Xv, Yv))
   print(pf.accuracy(model.predict(res.x, Xv, beta=16), Yv))

:doc:`training_tutorial` goes through a complete run step by step.

The tests check the shortcut against brute force:

* ``T @ x`` matches a direct run with all inputs driven together to 1e-10;
* the training gradient through ``T`` equals the gradient through one direct
  simulation per sample to 1e-10.

For a target linear map instead of a dataset, such as a splitter, a mode router
or a DFT, :func:`~photonfdtd.fit_transfer` minimizes the relative mismatch
``||T - T_target||**2``, with the arbitrary global phase factored out.

Training memory scales with the number of input ports, because the ``N``
adjoints are taken together inside one ``jax.value_and_grad``. With
``method="tiled"`` the port simulations are disk-backed instead (see below).

.. _choosing-an-adjoint:

Choosing an adjoint
-------------------

Every design gradient here (:func:`~photonfdtd.make_value_and_grad_params`,
and through it :func:`~photonfdtd.minimize` and :func:`~photonfdtd.robust`,
plus :class:`~photonfdtd.PhotonicModel`, :func:`~photonfdtd.train` and
:func:`~photonfdtd.fit_transfer`) takes a ``method``. All four give the same
gradient to rounding error (``tests/test_optimize.py`` checks each against the
default to 1e-10), so the choice is purely about memory:

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - ``method``
     - Memory
     - Restrictions
   * - ``"checkpoint"`` (default)
     - about ``sqrt(n_steps)`` field states on the device
     - none
   * - ``"reversible"``
     - independent of ``n_steps``: the backward pass steps the physics
       backward
     - no PML; lossless dispersion poles only; no moving charges; DFT and
       flux monitors only
   * - ``"reversible_pml"``
     - the reversible pass plus a stored PML-shell tape, currently held as
       a full grid array per step
     - as ``"reversible"``, but with PML
   * - ``"tiled"``
     - the device holds one x-slab; field states and checkpoints live on
       disk
     - none; ``tiled_options`` sets the slab size, steps per slab, scratch
       directory and checkpoint spacing

With ``"tiled"`` the coefficient builder runs on the CPU, so the three
coefficient arrays and their cotangents sit in host RAM, and each iteration is
a host loop over compiled slab updates rather than one compiled program. Because
the design hands the tiled run its own per-component coefficients, a
subpixel-smoothed layout works too, which the scalar
:func:`~photonfdtd.jax_value_and_grad_eps_tiled` cannot do.

.. code-block:: python

   vg = pf.make_value_and_grad_params(
       sim, region.ce_builder(sim), loss, method="tiled",
       tiled_options=dict(steps_per_tile=8, workdir="/scratch/run"))

Dispersive layouts
------------------

Dispersive materials (metals, or the frequency-dependent glass and
semiconductor models) can sit anywhere on the chip around a design region, and
are differentiated with their poles held fixed. The design region itself must
be a plain dielectric, and building one that overlaps a dispersive structure
raises an error. The reversible methods need lossless poles; lossy ones such as
metals need ``"checkpoint"`` or ``"tiled"``.

Limitations
-----------

* :class:`~photonfdtd.ModeTransmission` ports must be normal to ``x``, like
  the mode solver.
* The design region cannot contain dispersive material (see above).

The worked examples 6 (a 1x2 splitter) and 7 (a phase classifier trained on a
dataset) in :doc:`examples` run all of this end to end.
:doc:`inverse_design_tutorial` and :doc:`training_tutorial` go through them
step by step.
