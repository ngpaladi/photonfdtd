Active devices: voltages, heaters and carriers
==============================================

:mod:`photonfdtd.active` simulates light switched by a voltage or a heater.
It takes a layout with electrodes, heaters and PN junctions, solves for the
static electric field or temperature they produce
(:mod:`photonfdtd.statics`), turns that into a change of the optical
permittivity, and hands the result to the mode solver or the FDTD engine. The
same few objects answer the usual questions:

- the half-wave voltage-length product ``V_pi * L`` of an electro-optic phase
  shifter, and the ``P_pi`` of a heater (mode solver, seconds);
- a voltage or heater sweep of a switch, run in FDTD;
- the drive voltages that give a target output, found by gradient descent;
- a time-varying (RF) drive, stepped in FDTD.

Examples 8, 9 and 10 (:doc:`examples`) cover the first three in full.

The workflow
------------

.. code-block:: python

   import photonfdtd as pf
   from photonfdtd import active

   lam = 1.55e-6
   ln = active.lithium_niobate_eo(lam, cut="x")      # optic axis along grid y
   au = active.metal_active("Au")
   left = pf.Box((-4.5e-6, 0.55e-6), (4e-6, 0.5e-6), au)
   right = pf.Box((4.5e-6, 0.55e-6), (4e-6, 0.5e-6), au)
   layout = [pf.Box((0, -1e-6), (20e-6, 2e-6), active.silica_active(lam)),
             pf.Box((0, 0.15e-6), (20e-6, 0.3e-6), ln),
             pf.Box((0, 0.45e-6), (1.2e-6, 0.3e-6), ln), left, right]

   ms = pf.ModeSolver(size=(12e-6, 3.6e-6), cell_size=0.04e-6,
                      structures=layout, wavelength=lam, polarization="te")
   region = active.ActiveRegion(
       layout, electrodes=[active.Electrode(right, "signal"),
                           active.Electrode(left, "ground")], wavelength=lam)
   mod = region.prepare(ms)           # one electrostatic solve per electrode

   mod.vpi_l("signal") / 1e-2         # V cm  (about 4.2)
   mod.n_eff(V={"signal": 5.0})       # the mode index at 5 V

1. **Materials.** Use :class:`~photonfdtd.active.ActiveMedium` for anything
   the drives act on. Next to the optical permittivity it carries the DC
   permittivity, the thermal conductivity, the thermo-optic coefficient and,
   for crystals, the Pockels tensor and the crystal orientation. The library:

   .. list-table::
      :widths: 45 55

      * - :func:`~photonfdtd.active.lithium_niobate_eo`
        - LiNbO3, x/y/z cut, clamped or not
      * - :func:`~photonfdtd.active.lithium_tantalate_eo`
        - LiTaO3
      * - :func:`~photonfdtd.active.barium_titanate_eo`
        - BaTiO3 (r13, r33 only; see below)
      * - :func:`~photonfdtd.active.eo_polymer`
        - poled polymer, your r33
      * - :func:`~photonfdtd.active.silicon_active`
        - Si, with free-carrier dispersion
      * - :func:`~photonfdtd.active.silicon_nitride_active`
        - SiN
      * - :func:`~photonfdtd.active.silica_active`
        - SiO2
      * - :func:`~photonfdtd.active.air_active`
        - air (the default background)
      * - :func:`~photonfdtd.active.metal_active`
        - Au, Al, TiN, NiCr, Ti, W

   :func:`~photonfdtd.active.with_response` adds properties to any other
   medium. A plain :class:`~photonfdtd.Medium` works in an electrostatic solve
   (its permittivity is used at DC), but a thermal solve needs a thermal
   conductivity for every material and says which are missing.

2. **Drives.** :class:`~photonfdtd.active.Electrode` (an equipotential
   conductor), :class:`~photonfdtd.active.Heater` (power in W, or W/m on a
   cross-section) and :class:`~photonfdtd.active.PNJunction` (a reverse-biased
   lateral junction in silicon). Each has a name; drive values are dicts keyed
   by name, ``V={"signal": 5.0}``, ``P={"heater": 0.02}``,
   ``bias={"pn": 2.0}``.

3. **Prepare** on a target. ``region.prepare(mode_solver)`` solves on the
   cross-section; ``region.prepare(simulation)`` solves on the FDTD grid, and
   ``region.prepare(simulation, extrude="x")`` solves once on the y-z
   cross-section and extrudes it over the drives' length, which is exact for
   straight electrodes away from their ends and far cheaper than a 3-D solve.

4. **Use** the prepared region (:class:`~photonfdtd.active.PreparedActive`):

   - mode solver: :meth:`~photonfdtd.active.PreparedActive.n_eff`,
     :meth:`~photonfdtd.active.PreparedActive.neff_sweep`,
     :meth:`~photonfdtd.active.PreparedActive.dneff`,
     :meth:`~photonfdtd.active.PreparedActive.vpi_l`,
     :meth:`~photonfdtd.active.PreparedActive.p_pi`, or
     :meth:`~photonfdtd.active.PreparedActive.mode_solver` for the driven
     ModeSolver itself;
   - FDTD: :meth:`~photonfdtd.active.PreparedActive.run` (one setting),
     :meth:`~photonfdtd.active.PreparedActive.sweep` (many settings, compiled
     once), :meth:`~photonfdtd.active.PreparedActive.coefficients` (for
     ``Simulation.run(ce=..., ca=...)``);
   - gradients: :meth:`~photonfdtd.active.PreparedActive.ce_builder` plugs
     into :func:`~photonfdtd.make_value_and_grad_params`, so drive voltages and
     powers can be optimized like any other design parameter;
   - RF: :meth:`~photonfdtd.active.PreparedActive.modulation`.

Switching in a 2-D chip-plane FDTD run
--------------------------------------

The 2.5-D effective-index reduction flattens the chip to its plane, and with
it the things that set a modulator's efficiency: the film thickness, the
electrode height, and how much of the field lands inside the mode. An
electrostatic solve in the chip plane would overestimate the index change.
Instead, prepare the region on the waveguide's **cross-section** (a
ModeSolver) and carry the result to the chip plane:

.. code-block:: python

   arm = region.prepare(cross_section_mode_solver)
   drive = (arm.eim([upper_core_box], drives=["signal"])
            + arm.eim([lower_core_box], drives=["signal"], sign=-1))   # push-pull
   outs = drive.sweep(sim2d, [{"signal": v} for v in voltages])

:meth:`~photonfdtd.active.PreparedActive.eim` measures ``d n_eff / dV`` on the
cross-section (which includes the mode-field overlap) and raises the index of
the footprint cells by that much per volt. The in-plane mode isn't entirely
inside the footprint, so it sees a slightly smaller shift (about its in-core
power fraction); draw the footprint over the whole core.

FDTD domains are tens of micrometres long, and the voltage a switch needs
scales as 1 / length. Example 9's coupler switches at about 1 kV over 41 um;
the same design 5 mm long would switch at about 8 V. The shape of the
switching curve and the extinction carry over unchanged.

The physics, and what it leaves out
-----------------------------------

**Electrostatics.** ``div(eps_dc grad phi) = 0`` by finite volumes on the
optical grid, with a diagonal DC permittivity (lithium niobate's is strongly
anisotropic: 28 along the optic axis and 44 (clamped) or 85 across it).
Electrodes are equipotential conductors whose potential sits on their
surface. The outer boundary is insulating by default (``boundary="ground"``
grounds it), so give the field room to decay. One unit solve per electrode is
kept, and any voltages follow by superposition: the potential is linear in the
voltages, and so is the Pockels change. The voltage dependence is therefore
exact, cheap and differentiable.

**Pockels effect.** In the crystal frame the index ellipsoid changes by
``d(1/n^2)_i = sum_j r_ij E_j`` (Voigt notation). The solvers here use a
diagonal permittivity tensor, so

- the crystal's principal axes must lie along grid axes (x-, y- and z-cut
  with light along a grid axis are all fine);
- only the diagonal changes ``d eps_ii = -eps_ii^2 d(1/n^2)_i`` are applied.

The off-diagonal changes (from r51, r42, ...) are computed anyway: the first
:meth:`~photonfdtd.active.PreparedActive.dneff` on a ModeSolver estimates how
much they would move ``n_eff`` relative to the modelled change, to first order
in the mode fields, stores it in ``offdiagonal_ratio``, and warns above 5%.
For x-cut thin-film lithium niobate with lateral electrodes it is about 1.5%.
Barium titanate's large r42 is off-diagonal, so it is outside this model
(only r13 and r33 act).

**Thermal.** Steady state ``div(k grad T) = -q``; heaters spread their power
uniformly over their cells. Faces listed in ``sinks`` are held at ``T0`` (for
example the bottom of the buried oxide, standing for the silicon substrate),
faces in ``convection`` lose heat at ``h (T - T0)``, and the rest are
insulating. The index change is ``d eps = 2 n (dn/dT) dT`` per axis. On a
2-D cross-section, heater power is per unit length; ``P_pi`` comes out as a
total power, independent of the heater's length. ``P_pi`` depends on where the
heat can go, so check its sensitivity to the domain and the sink: example 10
moves the insulating sides from 8 to 15 um and ``P_pi`` changes by 1%.

**Free carriers (silicon).** The Soref-Bennett fits at 1310 and 1550 nm give
the index change and absorption from the electron and hole densities
(:func:`~photonfdtd.active.plasma_dispersion`; pass other coefficients, such
as the Nedeljkovic 2011 fits, through ``free_carrier_coefficients``). Carriers
come from an abrupt-junction depletion model
(:class:`~photonfdtd.active.PNJunction`), not a drift-diffusion solve; the
absorption enters FDTD as a conductivity. The carrier response is not linear in
the bias, so it is evaluated at each bias and held fixed in
:meth:`~photonfdtd.active.PreparedActive.ce_builder`.

**Time-varying drives.** :meth:`~photonfdtd.active.PreparedActive.modulation`
rebuilds the permittivity in the driven region every time step from the
waveforms you give (:class:`~photonfdtd.Sinusoid`, :class:`~photonfdtd.Ramp`,
:class:`~photonfdtd.Step`, or any ``t -> V`` callable). It is an
instantaneous-permittivity model, accurate while the drive changes slowly
next to the optical period. Mind the cost: a 1 GHz period is about ten million
FDTD steps at telecom resolution, so time-domain modulation suits short pulses
and fast (THz-scale) demonstrations, and DC switching is better done as a
voltage sweep. The modulated region may not contain dispersive or conductive
cells. Disk-tiled runs and the reversible adjoints don't support modulation or
conductive media.

Lower-level pieces
------------------

- :class:`~photonfdtd.AnisotropicMedium`: a diagonal-tensor dielectric, usable
  in any simulation and the mode solver (not with ``subpixel=True``).
- ``Medium(conductivity=...)``: a lossy dielectric (S/m), stepped with the
  standard lossy E update.
- ``Simulation.run(ce=..., ca=..., modulation=...)``: run with your own E-update
  coefficients (:func:`photonfdtd.jaxbackend.update_coefficients` builds them
  from per-component permittivities and a conductivity map).
- :func:`~photonfdtd.solve_potential`, :func:`~photonfdtd.solve_thermal`: the
  static solvers on their own.
- :func:`~photonfdtd.from_gdsfactory` with ``electrodes=`` / ``heaters=``
  picks metal polygons from layout layers (by a point inside each) and returns
  the :class:`~photonfdtd.ActiveRegion` alongside the Simulation.
