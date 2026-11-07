"""Example 10: a thermo-optic phase shifter (silicon waveguide with a heater).

A 500 x 220 nm silicon strip sits on 2.2 um of buried oxide under 2 um of
oxide cladding, with a 2 um wide, 100 nm thick TiN heater 1 um above the core.
photonfdtd.active solves the steady-state temperature on the cross-section
(the substrate under the oxide is held at room temperature; the other sides
are insulating), turns it into the thermo-optic index change
(dn/dT = 1.86e-4 /K for Si, 0.95e-5 /K for SiO2) and re-solves the mode.

On a 2-D cross-section the heater power is per unit length (W/m). The phase is
k0 * L * dn_eff and dn_eff grows as P/L, so P_pi, the total power for a pi
phase shift, doesn't depend on the heater's length.

Expect P_pi around 17 mW, close to the ~20 mW typical of measured heaters on
this stack. It depends on where heat can go: here the substrate under the
buried oxide is a perfect sink and the sides, 8 um out, are insulating
(moving them to 15 um changes P_pi by about 1%). Undercut trenches, which
stop heat leaking into the substrate, bring P_pi down several-fold.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import photonfdtd as pf
from photonfdtd import active

um = 1e-6
lam = 1.55e-6

si = active.silicon_active(lam)
sio2 = active.silica_active(lam)
tin = active.metal_active("TiN")

heater = pf.Box((0, 1.16 * um), (2 * um, 0.1 * um), tin)
layout = [
    pf.Box((0, 0.1 * um), (40 * um, 4.6 * um), sio2),     # BOX + cladding
    pf.Box((0, 0.11 * um), (0.5 * um, 0.22 * um), si),    # the waveguide
    heater,
]
# The cross-section's bottom edge is the bottom of the buried oxide (z = -2.2 um),
# held at room temperature by the silicon substrate below.
ms = pf.ModeSolver(size=(16 * um, 4.4 * um), cell_size=0.05 * um,
                   structures=layout, wavelength=lam, polarization="te")
region = active.ActiveRegion(layout, heaters=[active.Heater(heater, name="heater")],
                             wavelength=lam, sinks=("zmin",))
ps = region.prepare(ms)

p_pi = ps.p_pi("heater")
dT_per = ps.thermal.basis_dT[0][0]
j, k = np.argmin(np.abs(ms.y)), np.argmin(np.abs(ms.z - 0.11 * um))
P_line = np.linspace(0, 400, 3)                 # W/m
n_eff = ps.neff_sweep("heater", P_line)
print(f"core temperature rise: {dT_per[j, k]:.3f} K per W/m")
print(f"d n_eff / dP':         {ps.dneff('heater'):.3e} per W/m")
print(f"P_pi:                  {p_pi * 1e3:.1f} mW (any heater length)")
L = 100 * um
print(f"  for a {L / um:.0f} um heater: {p_pi / L:.0f} W/m, core "
      f"{dT_per[j, k] * p_pi / L:.0f} K above ambient")

y, z = ms.y / um, ms.z / um
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
im = ax[0].imshow((dT_per * p_pi / L).T, origin="lower", aspect="auto", cmap="inferno",
                  extent=[y.min(), y.max(), z.min(), z.max()])
ax[0].set(xlim=(-8, 8), title=f"Temperature rise at P_pi over a {L / um:.0f} um heater",
          xlabel="y (um)", ylabel="z (um)")
fig.colorbar(im, ax=ax[0], label="K")
ax[1].plot(P_line * L * 1e3, (n_eff - n_eff[0]) * 1e3, "o-")
ax[1].axhline(lam / (2 * L) * 1e3, color="gray", ls="--", label="pi phase")
ax[1].set(xlabel=f"heater power for L = {L / um:.0f} um (mW)", ylabel="dn_eff x 1e3",
          title=f"P_pi = {p_pi * 1e3:.1f} mW")
ax[1].legend(); ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig("10_thermo_optic_heater.png", dpi=130)
print("saved 10_thermo_optic_heater.png")
