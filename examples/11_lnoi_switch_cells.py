"""Example 11: lithium niobate switch cells for gdsfactory, and their models.

photonfdtd.lnoi has gdsfactory cells for thin-film LN switches, with a model
computed by photonfdtd from each cell's own cross-section. This script draws
an x-cut Mach-Zehnder switch and a z-cut delta-beta coupler, writes
them to GDS, and plots bar and cross against voltage.

- ``mzi_switch``: 5 mm push-pull arms between ground-signal-ground
  electrodes, and 50:50 directional couplers. Cross at 0 V, bar at V_pi
  (about 4.4 V).
- ``dbeta_coupler_switch``: three coupling lengths (4 mm) of z-cut coupler
  with electrodes on top of both waveguides. Cross at 0 V, bar at about 38 V;
  ``sections=2`` splits the electrodes for reversed delta-beta operation.

Needs gdsfactory. Each model takes about a minute (electrostatics and mode
solves on the cross-section at 50 nm).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gdsfactory as gf
from photonfdtd import lnoi

gf.gpdk.PDK.activate()

mzi = lnoi.mzi_switch()
dbeta = lnoi.dbeta_coupler_switch()

top = gf.Component("lnoi_switches")
top << mzi
d = top << dbeta
d.move((0, -150))
top.write_gds("11_lnoi_switch_cells.gds")
print("wrote 11_lnoi_switch_cells.gds")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for a, (cell, title, vmax) in zip(ax, ((mzi, "MZI switch (x-cut)", 12),
                                       (dbeta, "delta-beta coupler switch (z-cut)", 50))):
    m = lnoi.model(cell)
    v = np.linspace(-vmax, vmax, 241)
    bar, cross = m.sweep(v)
    v_bar = m.switching_voltage("bar", v_max=vmax)
    print(f"{title}: coupling length {m.info['coupling_length'] * 1e6:.0f} um, "
          f"cross {m.cross(0):.3f} at 0 V, bar {m.bar(v_bar):.3f} at {v_bar:+.1f} V")
    a.plot(v, bar, label="bar (o1 -> o4)")
    a.plot(v, cross, label="cross (o1 -> o3)")
    a.axvline(v_bar, color="gray", ls=":", lw=0.8)
    a.set(xlabel="signal voltage (V)", ylabel="output fraction", title=title)
    a.legend(); a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("11_lnoi_switch_cells.png", dpi=130)
print("saved 11_lnoi_switch_cells.png")
