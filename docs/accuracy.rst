Accuracy features
=================

FDTD is a wonderfully simple idea — leapfrog Maxwell's two curl equations on a
staggered grid and let the fields march forward in time — and a naive
implementation quietly throws away a lot of accuracy in a few very predictable
places. Three of them dominate for real integrated photonics: how the grid sees
a material interface, whether a material is even allowed to be dispersive, and
whether the mode solver is honest about polarization. ``photonfdtd`` handles all
three, and every number below is checked against an analytic benchmark in the
test suite rather than asserted.

Subpixel smoothing
------------------

Ask a plain FDTD grid what material is in a cell and it does the laziest thing
possible: it checks whether the cell *centre* lands inside your structure and
stamps that one material across the whole cell. Every interface that doesn't
happen to fall on a cell boundary turns into a jagged staircase — and that
staircase quietly knocks your global convergence from second order down to
roughly first, right at the interfaces, which is exactly where the physics you
care about lives.

:class:`~photonfdtd.Simulation` with ``subpixel=True`` does the honest thing
instead. A partially filled cell gets an effective **anisotropic tensor** that
respects the boundary conditions the staircase was ignoring: the field component
*normal* to the interface sees the harmonic mean of permittivity (because normal
:math:`D` is continuous), while the *tangential* components see the arithmetic
mean (because tangential :math:`E` is continuous). In the interface's own frame
that's :math:`\mathrm{diag}(\varepsilon_h, \varepsilon_a, \varepsilon_a)`. We
keep only the diagonal, which is *exact* for axis-aligned interfaces (rectangular
``Box`` / Manhattan geometry — i.e. most of what you'll ever draw) and a large
improvement everywhere else. The per-cell means and the surface normal come from
supersampling (``subpixel_factor``).

.. code-block:: python

   sim = pf.Simulation(grid, structures=[...], sources=[...],
                       run_time=..., subpixel=True, subpixel_factor=3)

Runs on the NumPy and JAX backends (the JAX path carries a per-component
coefficient, so the smoothed result is JIT-compiled *and* differentiable, and
matches the NumPy reference to floating-point reordering). The payoff is easy to
see and hard to argue with (``tests/test_smoothing.py``,
``tests/test_jax_accuracy.py``): slide an interface across a single cell and
watch a transmitted-phase observable. Without smoothing it sits perfectly still
and then lurches in one abrupt jump when the cell finally flips; with smoothing
it moves smoothly the whole way — same net physics, none of the staircase. That
is the Farjadpour convergence signature.

References:

* A. Farjadpour *et al.*, "Improving accuracy by subpixel smoothing in the
  finite-difference time domain," *Opt. Lett.* **31**\ (20), 2972 (2006).
  DOI: 10.1364/OL.31.002972
* C. A. Kottke, A. Farjadpour, S. G. Johnson, *Phys. Rev. E* **77**, 036611
  (2008). DOI: 10.1103/PhysRevE.77.036611
* A. F. Oskooi *et al.* (MEEP), *Comput. Phys. Commun.* **181**, 687 (2010).
  DOI: 10.1016/j.cpc.2009.11.008

Dispersive materials
--------------------

Real materials don't have a refractive index; they have a curve. We write a
dispersive medium as a background permittivity plus a sum of poles,

.. math::

   \varepsilon_r(\omega) = \varepsilon_\infty +
     \sum_p \frac{S_p}{\omega_{0p}^2 - \omega^2 - 2i\gamma_p\omega},

which is more general than it looks. One pole form covers Lorentz resonances
(:math:`S = \Delta\varepsilon\,\omega_0^2`), Drude free-electron metals
(:math:`\omega_0 = 0`), and — the pleasant surprise — lossless Sellmeier terms,
because a Sellmeier term :math:`B\lambda^2/(\lambda^2-C)` *is* exactly a lossless
Lorentz pole (:math:`\Delta\varepsilon = B`, :math:`\omega_0 = 2\pi c/\sqrt{C}`).
So silica, silicon and gold all fall out of the same engine. Each pole is marched
by the auxiliary-differential-equation (ADE) method (Taflove & Hagness,
*Computational Electrodynamics*, 3rd ed., ch. 9): its polarization obeys
:math:`\ddot P + 2\gamma\dot P + \omega_0^2 P = \varepsilon_0 S\,E`,
central-differenced to a cheap recursion that only runs in the cells that are
actually dispersive. It switches on by itself the moment a structure uses a
dispersive medium.

.. code-block:: python

   au = pf.gold()                       # Rakic 1998 Lorentz-Drude
   n_si = pf.silicon().index(pf.C_0 / 1.55e-6)   # complex index at 1.55 um
   custom = pf.DispersiveMedium.lorentz(2.25, [(1.2, 500e12, 0.0)])

The cited library — :func:`~photonfdtd.silica`, :func:`~photonfdtd.silicon`,
:func:`~photonfdtd.silicon_nitride`, :func:`~photonfdtd.lithium_niobate`,
:func:`~photonfdtd.gold`, :func:`~photonfdtd.silver` — reproduces each source's
published index (``docs/material_data.md`` has the parameters and the papers).

**One honest catch.** The explicit ADE update is only stable while
:math:`\omega_0\,\Delta t < 2`, and the deep-UV poles buried in a Sellmeier fit
sail right past that on a grid sized for telecom wavelengths. So the library
presets are directly time-steppable only on grids fine enough to resolve those
poles. If you just want the correct index at a single wavelength — a mode solve,
a narrowband run — call ``medium.at_wavelength(lambda)`` and get a plain
fixed-index :class:`~photonfdtd.Medium`; otherwise keep the medium to in-band
poles. (Fitting a stable low-order model over your band, which is what Tidy3D's
dispersion fitter does, is the obvious next step.) The stepper itself matches the
analytic phase index of a Lorentz medium to ~1% (``tests/test_dispersion.py``).

Runs on NumPy and JAX. On the JAX path each pole's polarization rides along in
the ``lax.scan`` carry, so a dispersive run is JIT-compiled and
**differentiable**: ``jax.grad`` of a monitor loss flows all the way back to the
pole strengths (checked against finite differences in
``tests/test_jax_accuracy.py``), which is what you'd need to inverse-design over
material dispersion.

Full-vectorial mode solver
--------------------------

A scalar mode solver quietly assumes your waveguide is weakly guiding and then
hands you a single effective index regardless of polarization. For a
high-contrast silicon wire that is simply the wrong answer.
:class:`~photonfdtd.ModeSolver` solves the full-vectorial finite-difference
eigenproblem for the transverse fields on the Yee grid and returns the effective
indices *and* the full complex field components, so TE and TM come out as the
genuinely different modes they are. The proof is a slab, where the answer is
known exactly (``tests/test_mode_vectorial.py``): for a high-contrast slab
(n=3.0 core, n=1.0 clad, 0.30 um) it lands both TE0 (analytic 2.5397) and TM0
(analytic 1.8988) to better than 0.01% — a TE/TM split a scalar solver cannot
see at all.

S-parameters (mode decomposition)
---------------------------------

S-parameters are what you actually report for a photonic device, and you get
them by asking one simple question at a port plane: how much of *this* mode is
going *that* way? Record the four tangential components on a **port plane** with
a plane-restricted :class:`~photonfdtd.DFTMonitor` (any axis, via
``plane_axis``/``plane_position``) — keeping only that 2-D slice means a port
costs kilobytes instead of the gigabytes a full-volume DFT would burn — and
project onto the solved mode:

.. code-block:: python

   port = pf.DFTMonitor(name="out", components=("Ey", "Ez", "Hy", "Hz"),
                        freqs=freqs, plane_axis="x", plane_position=x_out)
   # ... run ...
   alpha_plus, alpha_minus = pf.s_parameters(result, "out", mode_result, mode_index=0,
                                             dA=dy * dz)

The forward and backward amplitudes come from the unconjugated/conjugated
cross-product overlap, and the reason it works is that oppositely-propagating
modes are orthogonal under it — so transmission and reflection separate cleanly
even though they are superimposed on the same plane. Launch a mode down a
straight lossless guide and you get ``|S21| ~= 1`` back, exactly as you should
(``tests/test_smatrix.py``). The overlap is plain array math, so it runs on
NumPy or JAX, and evaluated on JAX DFT outputs it makes ``|S_ij|**2``
differentiable through the adjoint — which means it *is* a ready-made
inverse-design objective.

Inverse design (differentiable etched-core PIC)
-----------------------------------------------

Put the last three pieces together and you can optimize a real device. A PIC is
a substrate / etched-core / cladding stack, and the thing you are actually free
to change is the etch — a 2-D pattern, not a 3-D blob. So the design variable is
a 2-D density ``rho(x, y)`` extruded through the fixed core thickness
(:class:`~photonfdtd.EtchedCore`): ``rho`` runs through a conic density filter
(to enforce a minimum feature size), a ``tanh`` projection (to push it toward a
binary, fabricable pattern), and anisotropic subpixel-smoothed sidewalls, ending
as a 3-D permittivity tensor — all in JAX.

Feed the port with a one-way :class:`~photonfdtd.UniModeSource`
(equivalence-principle electric + magnetic current sheets; validated backward
extinction well below the bidirectional soft source), make the objective a
mode-overlap S-parameter, and — because every stage is differentiable —
:func:`~photonfdtd.value_and_grad_density` returns ``d(loss)/d(rho)`` through the
entire time evolution: a topology-optimization gradient checked against finite
differences to ~1e-8 (``tests/test_design.py``), with adjoint memory kept in
line by the checkpointing below. And since the design lives in 2-D, the gradient
reduces over the core z-layers to ``(nx, ny)`` — orders of magnitude smaller than
a 3-D design grid, which is often the difference between "fits" and "doesn't."

.. code-block:: python

   ec = pf.EtchedCore(eps_solid=3.48**2, eps_void=1.44**2,
                      eps_sub=1.44**2, eps_clad=1.0, core_z=(-0.11e-6, 0.11e-6),
                      filter_radius=140e-9, beta=8.0)
   loss = lambda out: -abs(mode_overlap(out))**2          # e.g. -|S21|^2
   value, grad = pf.value_and_grad_density(sim, ec, rho, loss)  # grad is (nx, ny)

Memory notes
------------

Everything here is built so that peak memory tracks the problem's real size, not
its bounding box:

* **Dispersion** keeps the auxiliary polarization only on the dispersive cells
  (NumPy and JAX both), so a metal nanostructure in a large domain costs what the
  nanostructure costs, not what the domain costs.
* **Subpixel smoothing** supersamples one x-slab at a time, so the transient
  build memory is a single refined slab rather than the whole
  ``factor**ndim``-times-bigger fine grid.
* **Differentiable runs (JAX).** This is the one that bites. Reverse-mode AD
  through the ``lax.scan`` time loop wants to stash the field state at *every*
  step — O(n_steps), and that is the wall you hit doing gradient-based inverse
  design. :func:`value_and_grad_eps` defaults to two-level gradient checkpointing
  (``remat="nested"``): keep only ~sqrt(n_steps) segment-boundary states,
  recompute the rest, and pay ~2x compute for O(sqrt(n_steps)) memory (measured
  ~3-4x lower peak on a few-thousand-step run) — same gradient, to the bit.
* **Reversible adjoint (O(1) in timesteps).** Here is the nicer trick, when you
  can use it. A lossless, non-dispersive run with **no absorbing boundaries**
  (``pml_layers`` all zero — closed or periodic domains: resonators, photonic
  crystals, periodic metasurfaces) is *exactly* time-reversible, so
  :func:`~photonfdtd.jax_value_and_grad_eps_reversible` rebuilds the whole forward
  field history by stepping backward instead of storing it — adjoint memory that
  no longer cares how many steps you took. Forward-then-reverse returns to
  machine precision, the gradient matches the plain reverse pass to ~1e-15, and
  peak memory came in **14x below the plain adjoint and ~4x below checkpointing**
  on a ~1800-step 3-D run (``tests/test_reversible.py``). There is a PML variant
  too (:func:`~photonfdtd.jax_value_and_grad_eps_reversible_pml`) that
  reconstructs the interior and replays a per-step PML-shell tape, and it gives
  the exact gradient with CPML — but I'll be straight with you: the tape is
  O(shell * n_steps), so it only wins when the PML shell is a small slice of the
  volume (under ~5%), and real 3-D geometries (~14-60% shell) almost never are.
  For an ordinary PML PIC run, ``remat="nested"`` checkpointing is still the tool
  to reach for.
* **Planning a big run.** :func:`~photonfdtd.estimate_memory` /
  :func:`~photonfdtd.recommend_mode` / :func:`~photonfdtd.format_report` estimate
  the peak bytes of each mode (forward, the three adjoints, disk-tiled
  out-of-core) and pick one that fits your budget — so you find out a run won't
  fit *before* it OOMs, not after. For a volume that doesn't fit at all, the
  out-of-core stepper (``Simulation.run(out_of_core=True)``) memory-maps the
  fields to disk and marches in tiles bounded by ``tile_cells`` planes. Turn on
  the legacy CuPy backend (``use_gpu=True``) and it becomes a full
  **GPU/host/disk hierarchy**: each disk-backed tile is worked on the GPU and
  sent back, so the GPU only ever holds one tile while the full arrays sit on
  disk — a volume bigger than your VRAM still runs. Validated on an RTX 4080
  (bit-identical to the in-core CPU result; peak GPU memory scales with
  ``tile_cells``, not the grid).

References:

* Z. Zhu & T. G. Brown, "Full-vectorial finite-difference analysis of
  microstructured optical fibers," *Opt. Express* **10**\ (17), 853 (2002).
  DOI: 10.1364/OE.10.000853
* A. B. Fallahkhair, K. S. Li, T. E. Murphy, "Vector Finite Difference
  Modesolver for Anisotropic Dielectric Waveguides," *J. Lightwave Technol.*
  **26**\ (11), 1423 (2008). DOI: 10.1109/JLT.2008.923643
* R. C. Rumpf, *Electromagnetic and Photonic Simulation for the Beginner:
  FDFD in MATLAB* (Artech House, 2022).
