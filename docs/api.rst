API reference
=============

The full public API, one section per submodule. If you can reach it as
``pf.something`` from the top-level :mod:`photonfdtd` namespace, it's documented
somewhere below.

Package overview
----------------

.. automodule:: photonfdtd
   :no-members:

Constants
---------

.. automodule:: photonfdtd.constants

Grid
----

.. automodule:: photonfdtd.grid

Materials
---------

.. automodule:: photonfdtd.materials

Geometry
--------

.. automodule:: photonfdtd.geometry

Subpixel smoothing
------------------

.. automodule:: photonfdtd.smoothing

Sources
-------

.. automodule:: photonfdtd.sources

Monitors
--------

.. automodule:: photonfdtd.monitors

Boundaries
----------

.. automodule:: photonfdtd.boundaries

Simulation
----------

.. automodule:: photonfdtd.simulation

JAX engine
----------

The differentiable time loop behind :meth:`Simulation.run
<photonfdtd.Simulation.run>`, its adjoints, and memory planning; see
:doc:`backends`.

Time stepping (differentiable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: photonfdtd.jaxbackend

Disk-tiled stepping
~~~~~~~~~~~~~~~~~~~

.. automodule:: photonfdtd.tiled

Reversible adjoint
~~~~~~~~~~~~~~~~~~

.. automodule:: photonfdtd.reversible

Reversible adjoint with CPML
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: photonfdtd.reversible_pml

Memory planning
~~~~~~~~~~~~~~~

.. automodule:: photonfdtd.memory

Mode solver
-----------

.. automodule:: photonfdtd.mode

S-parameters
------------

.. automodule:: photonfdtd.smatrix

Photon counts
-------------

.. automodule:: photonfdtd.photons

Inverse design
--------------

.. automodule:: photonfdtd.design

Optimization loops
------------------

.. automodule:: photonfdtd.optimize

Training a chip
---------------

.. automodule:: photonfdtd.train

GDS export
----------

.. automodule:: photonfdtd.gds

Active devices
--------------

See :doc:`active_devices` for the guide.

.. automodule:: photonfdtd.active

Static solvers
~~~~~~~~~~~~~~

.. automodule:: photonfdtd.statics

Layout-scale methods
--------------------

See :doc:`layout_scale` for the guide.

Effective-index method (2.5-D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: photonfdtd.eim

Eigenmode expansion
~~~~~~~~~~~~~~~~~~~

.. automodule:: photonfdtd.eme

ADI-FDTD (CFL-free)
~~~~~~~~~~~~~~~~~~~

.. automodule:: photonfdtd.adi

Adapters
--------

.. automodule:: photonfdtd.adapters
