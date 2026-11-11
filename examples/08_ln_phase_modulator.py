"""Example 8: a thin-film lithium niobate phase modulator.

An x-cut LN rib waveguide (600 nm film, 300 nm etch, 1.2 um wide) on silica,
with gold electrodes on the slab 5 um apart. The optic axis lies across the
waveguide (grid y), so the TE mode's Ey and the electrodes' lateral field both
use the large r33 coefficient. photonfdtd.active solves the electrostatic
potential on the mode solver's cross-section, turns the field into the Pockels
index change, and re-solves the mode at each voltage.

Expect V_pi * L around 4.5 V cm for one arm: a push-pull Mach-Zehnder on this
geometry (each arm between its own pair of electrodes, driven oppositely) gets
about half that, matching the ~2.2 V cm reported for such devices.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import photonfdtd as pf
from photonfdtd import active

um = 1e-6
lam = 1.55e-6

ln = active.lithium_niobate_eo(lam, cut="x")      # optic axis along grid y
sio2 = active.silica_active(lam)
au = active.metal_active("Au")

gap = 5 * um
left = pf.Box((-(gap / 2 + 2 * um), 0.55 * um), (4 * um, 0.5 * um), au)
right = pf.Box((gap / 2 + 2 * um, 0.55 * um), (4 * um, 0.5 * um), au)
layout = [
    pf.Box((0, -1.0 * um), (20 * um, 2 * um), sio2),      # buried oxide
    pf.Box((0, 0.15 * um), (20 * um, 0.3 * um), ln),      # LN slab
    pf.Box((0, 0.45 * um), (1.2 * um, 0.3 * um), ln),     # LN rib
    left, right,
]
ms = pf.ModeSolver(size=(12 * um, 3.6 * um), cell_size=0.04 * um,
                   structures=layout, wavelength=lam, polarization="te")

region = active.ActiveRegion(
    layout,
    electrodes=[active.Electrode(right, "signal"), active.Electrode(left, "ground")],
    wavelength=lam,
)
mod = region.prepare(ms)

V = np.linspace(-10, 10, 5)
n_eff = mod.neff_sweep("signal", V)
vpl = mod.vpi_l("signal")
print(f"n_eff at 0 V:            {n_eff[len(V) // 2]:.5f}")
print(f"d n_eff / dV:            {mod.dneff('signal'):.3e} /V")
print(f"V_pi * L (one arm):      {vpl / 1e-2:.2f} V cm")
print(f"  -> 1 cm long arm:      V_pi = {vpl / 1e-2:.2f} V")
print(f"off-diagonal terms:      {mod.offdiagonal_ratio:.1%} of the modelled shift")

# Plot: potential, the index change at 10 V, n_eff vs V.
res = ms.solve()
y, z = ms.y / um, ms.z / um
ext = [y.min(), y.max(), z.min(), z.max()]
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
pot = mod.electrostatics.basis_phi[0][0]
im = ax[0].imshow(pot.T, origin="lower", extent=ext, cmap="viridis", aspect="auto")
ax[0].contour(y, z, np.abs(res.Ey[0]).T, levels=4, colors="w", linewidths=0.7)
ax[0].set(title="Potential (signal at 1 V), mode |Ey| contours", xlabel="y (um)", ylabel="z (um)")
fig.colorbar(im, ax=ax[0], label="V")
d, _ = mod.delta_eps(V={"signal": 10.0})
dn = d[1][0] / (2 * np.sqrt(mod.base_eps[1][0]))
im = ax[1].imshow(dn.T * 1e4, origin="lower", extent=ext, cmap="RdBu_r", aspect="auto",
                  vmin=-np.abs(dn).max() * 1e4, vmax=np.abs(dn).max() * 1e4)
ax[1].set(title="Index change n_yy at 10 V", xlabel="y (um)")
fig.colorbar(im, ax=ax[1], label="dn x 1e4")
ax[2].plot(V, (n_eff - n_eff[len(V) // 2]) * 1e5, "o-")
ax[2].set(title=f"V_pi L = {vpl / 1e-2:.2f} V cm", xlabel="voltage (V)",
          ylabel="dn_eff x 1e5")
ax[2].grid(alpha=0.3)
fig.tight_layout()
fig.savefig("08_ln_phase_modulator.png", dpi=130)
print("saved 08_ln_phase_modulator.png")
