Accuracy features
=================

Three features move ``photonfdtd`` toward the accuracy class of production
FDTD tools: anisotropic subpixel smoothing, dispersive materials, and a
full-vectorial mode solver. Each is validated against an analytic benchmark in
the test suite.

Subpixel smoothing
------------------

Plain FDTD assigns one permittivity per cell by testing whether the cell centre
lies inside a structure. That staircases every interface not aligned to the
grid, dropping the global convergence rate from second to roughly first order.
:class:`~photonfdtd.Simulation` with ``subpixel=True`` replaces the permittivity
of a partially filled cell with an effective **anisotropic tensor**: the field
component *normal* to the interface sees the harmonic mean of permittivity
(because normal :math:`D` is continuous) and the *tangential* components see the
arithmetic mean (because tangential :math:`E` is continuous). In the interface
frame the tensor is :math:`\mathrm{diag}(\varepsilon_h, \varepsilon_a,
\varepsilon_a)`; only the diagonal is kept, which is exact for axis-aligned
interfaces (rectangular ``Box`` / Manhattan geometry) and a large improvement
over staircasing otherwise. The per-cell means and surface normal are obtained
by supersampling (``subpixel_factor``).

.. code-block:: python

   sim = pf.Simulation(grid, structures=[...], sources=[...],
                       run_time=..., subpixel=True, subpixel_factor=3)

Supported on the NumPy, CuPy (GPU), and JAX backends (the JAX path uses a
per-component coefficient, so the smoothed result is JIT-compiled and
differentiable; it matches the NumPy reference to floating-point reordering).
Validated in ``tests/test_smoothing.py`` and ``tests/test_jax_accuracy.py``: as
an interface is swept across one cell, a
transmitted-phase observable staircases (one abrupt jump at the cell crossing)
without smoothing but ramps smoothly with it, both capturing the same net
physical change — the Farjadpour convergence signature.

References:

* A. Farjadpour *et al.*, "Improving accuracy by subpixel smoothing in the
  finite-difference time domain," *Opt. Lett.* **31**(20), 2972 (2006).
  DOI: 10.1364/OL.31.002972
* C. A. Kottke, A. Farjadpour, S. G. Johnson, *Phys. Rev. E* **77**, 036611
  (2008). DOI: 10.1103/PhysRevE.77.036611
* A. F. Oskooi *et al.* (MEEP), *Comput. Phys. Commun.* **181**, 687 (2010).
  DOI: 10.1016/j.cpc.2009.11.008

Dispersive materials
--------------------

A dispersive medium is written as a background permittivity plus a sum of
poles,

.. math::

   \varepsilon_r(\omega) = \varepsilon_\infty +
     \sum_p \frac{S_p}{\omega_{0p}^2 - \omega^2 - 2i\gamma_p\omega},

which spans Lorentz resonances (:math:`S = \Delta\varepsilon\,\omega_0^2`),
Drude free-electron response (:math:`\omega_0 = 0`), and lossless Sellmeier
terms (a term :math:`B\lambda^2/(\lambda^2-C)` is exactly a lossless Lorentz
pole with :math:`\Delta\varepsilon = B`, :math:`\omega_0 = 2\pi c/\sqrt{C}`).
Each pole is advanced by the auxiliary-differential-equation (ADE) method
(Taflove & Hagness, *Computational Electrodynamics*, 3rd ed., ch. 9): the
polarization obeys :math:`\ddot P + 2\gamma\dot P + \omega_0^2 P =
\varepsilon_0 S\,E`, central-differenced to an explicit recursion applied only
in dispersive cells. Dispersion activates automatically when any structure uses
a dispersive medium.

.. code-block:: python

   au = pf.gold()                       # Rakic 1998 Lorentz-Drude
   n_si = pf.silicon().index(pf.C_0 / 1.55e-6)   # complex index at 1.55 um
   custom = pf.DispersiveMedium.lorentz(2.25, [(1.2, 500e12, 0.0)])

The cited library — :func:`~photonfdtd.silica`, :func:`~photonfdtd.silicon`,
:func:`~photonfdtd.silicon_nitride`, :func:`~photonfdtd.lithium_niobate`,
:func:`~photonfdtd.gold`, :func:`~photonfdtd.silver` — reproduces each source's
published refractive index (see ``docs/material_data.md`` for full parameters
and citations).

**Pole stability.** The explicit ADE update is stable only while
:math:`\omega_0\,\Delta t < 2`. Deep-UV Sellmeier poles and high-energy Lorentz
poles violate this on a grid tuned for near-IR/optical work, so the library
presets are directly time-steppable only on fine grids that resolve their
poles. For a narrowband or mode-solving use case call
``medium.at_wavelength(lambda)`` to get a fixed-index :class:`~photonfdtd.Medium`
with the correct index; otherwise restrict the medium to in-band poles. (Fitting
a stable reduced pole model over a chosen band — what Tidy3D's dispersion
fitter does — is the natural next step.) The ADE stepper itself is validated in
``tests/test_dispersion.py`` against the analytic phase index of a Lorentz
medium to within ~1%.

Supported on the NumPy, CuPy, and JAX backends. On the JAX path each pole's
polarization is threaded through the ``lax.scan`` carry, so a dispersive run is
JIT-compiled and **differentiable** — ``jax.grad`` of a monitor-derived loss
flows back to the pole strengths (``tests/test_jax_accuracy.py`` checks this
against finite differences), enabling inverse design over material dispersion.

Full-vectorial mode solver
--------------------------

:class:`~photonfdtd.ModeSolver` solves the full-vectorial finite-difference
eigenproblem for the transverse fields on the Yee grid, returning the effective
indices and the full complex field components. Unlike a scalar Helmholtz solver
it distinguishes TE from TM and captures high-index-contrast boundary effects.
Validated in ``tests/test_mode_vectorial.py`` against the exact symmetric-slab
transcendental dispersion: for a high-contrast slab (n=3.0 core, n=1.0 clad,
0.30 um) it reproduces both TE0 (analytic 2.5397) and TM0 (analytic 1.8988) to
better than 0.01% — a split a scalar solver cannot represent.

Memory notes
------------

The accuracy features are built to keep peak memory near the problem's
intrinsic size:

* **Dispersion** stores the auxiliary polarization only on the dispersive cells
  (both the NumPy and JAX paths), so a metal nanostructure in a large domain
  costs its own footprint, not the whole grid.
* **Subpixel smoothing** supersamples one native x-slab at a time, so the
  transient build memory is a single refined slab rather than the full
  ``factor**ndim``-times-larger fine grid.
* **Differentiable runs (JAX).** Reverse-mode AD through the ``lax.scan`` time
  loop would otherwise store the field state at every step - O(n_steps) memory,
  the wall for gradient-based inverse design. :func:`value_and_grad_eps`
  defaults to two-level gradient checkpointing (``remat="nested"``): the
  backward pass keeps only ~sqrt(n_steps) segment-boundary states and
  recomputes each segment, cutting adjoint peak memory to O(sqrt(n_steps)) for
  ~2x compute (measured ~3-4x lower peak on a few-thousand-step run), at a
  bit-for-bit identical gradient.

References:

* Z. Zhu & T. G. Brown, "Full-vectorial finite-difference analysis of
  microstructured optical fibers," *Opt. Express* **10**(17), 853 (2002).
  DOI: 10.1364/OE.10.000853
* A. B. Fallahkhair, K. S. Li, T. E. Murphy, "Vector Finite Difference
  Modesolver for Anisotropic Dielectric Waveguides," *J. Lightwave Technol.*
  **26**(11), 1423 (2008). DOI: 10.1109/JLT.2008.923643
* R. C. Rumpf, *Electromagnetic and Photonic Simulation for the Beginner:
  FDFD in MATLAB* (Artech House, 2022).
