"""Example 9: switching light between two waveguides with a voltage (FDTD).

Two thin-film lithium niobate waveguides run side by side, close enough to
couple: light launched in the lower one crosses to the upper one over the
coupling length L_c. Push-pull electrodes raise the index of one waveguide and
lower the other's. The detuning spoils the phase matching, and once
``delta_beta * L_c = sqrt(3) pi`` the light comes back out of the waveguide it
entered (the "bar" state). This is the classic delta-beta coupler switch.

The steps:

1. The modulator cross-section from example 8 gives d n_eff / dV for one arm
   (photonfdtd.active, on a ModeSolver).
2. A 2-D chip-plane FDTD model of the coupler uses effective indices for the
   rib (core) and the slab around it (2.5-D reduction).
3. ``PreparedActive.eim`` carries d n_eff / dV onto the two cores, with
   opposite signs for push-pull. The sweep over voltages compiles once.

**Scale.** The FDTD coupler is only tens of micrometres long, so it switches
at kilovolts: the voltage needed scales as 1/length, and practical devices are
millimetres long. The plot's top axis converts to a 5 mm device. The physics
(the shape of the switching curve and the bar/cross extinction) carries over;
the absolute voltage scales.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import photonfdtd as pf
from photonfdtd import active

um = 1e-6
lam = 1.55e-6
f0 = pf.C_0 / lam

# ---- 1. The arm's voltage response, from the cross-section (as in example 8).
ln = active.lithium_niobate_eo(lam, cut="x")
sio2 = active.silica_active(lam)
au = active.metal_active("Au")
gap_e = 5 * um
left = pf.Box((-(gap_e / 2 + 2 * um), 0.55 * um), (4 * um, 0.5 * um), au)
right = pf.Box((gap_e / 2 + 2 * um, 0.55 * um), (4 * um, 0.5 * um), au)
section = [pf.Box((0, -1.0 * um), (20 * um, 2 * um), sio2),
           pf.Box((0, 0.15 * um), (20 * um, 0.3 * um), ln),
           pf.Box((0, 0.45 * um), (1.2 * um, 0.3 * um), ln), left, right]
ms = pf.ModeSolver(size=(12 * um, 3.6 * um), cell_size=0.06 * um,
                   structures=section, wavelength=lam, polarization="te")
arm = active.ActiveRegion(section, electrodes=[active.Electrode(right, "V"),
                                               active.Electrode(left, "gnd")],
                          wavelength=lam).prepare(ms)
print(f"d n_eff / dV per arm: {arm.dneff('V'):.3e} /V "
      f"(V_pi L = {arm.vpi_l('V') / 1e-2:.2f} V cm)")

# ---- 2. The coupler in the chip plane (effective indices of rib and slab).
n_e = np.sqrt(ln.permittivity[1])
n_core = pf.effective_index([pf.Layer(0.6 * um, n_e)], lam, substrate_index=1.444)
n_slab = pf.effective_index([pf.Layer(0.3 * um, n_e)], lam, substrate_index=1.444)
print(f"effective indices: rib {n_core:.3f}, slab {n_slab:.3f}")
w, g = 1.0 * um, 0.45 * um
yc = (w + g) / 2
Lx, Ly, dx = 60 * um, 8 * um, 0.04 * um
core = pf.Medium(permittivity=n_core ** 2)
lower = pf.Box((0, -yc), (2 * Lx, w), core)
upper = pf.Box((0, +yc), (2 * Lx, w), core)
grid = pf.Grid(size=(Lx, Ly), cell_size=dx, pml_layers=(15, 15, 0))
slab = pf.Box((0, 0), (2 * Lx, 2 * Ly), pf.Medium(permittivity=n_slab ** 2))
x0 = -Lx / 2 + 2 * um
srcs = [pf.PointDipole(position=(x0, -yc + dy), component="Ey",
                       waveform=pf.GaussianPulse(freq0=f0, fwhm=30e-15),
                       amplitude=float(np.cos(np.pi * dy / w)))
        for dy in np.arange(-w / 2 + dx, w / 2, dx)]
mon = pf.DFTMonitor(name="plane", components=("Ey",), freqs=np.array([f0]))
sim = pf.Simulation(grid, structures=[slab, lower, upper], sources=srcs,
                    monitors=[mon], run_time=1.2e-12)

# ---- 3. Push-pull drive on the electrode-covered length.
x_e = (x0 + 3 * um, Lx / 2 - 2 * um)
fp = lambda y: pf.Box(((x_e[0] + x_e[1]) / 2, y), (x_e[1] - x_e[0], w), core)
drive = arm.eim([fp(-yc)], drives=["V"]) + arm.eim([fp(+yc)], drives=["V"], sign=-1)

voltages = np.linspace(0, 1500, 16)
outs = drive.sweep(sim, [{"V": v} for v in voltages])

y = grid.coords[1]
low = (np.abs(y + yc) < w / 2)
up = (np.abs(y - yc) < w / 2)


def split(out):
    """Power in each waveguide along x (|Ey|^2 summed over its width)."""
    E = np.abs(out["dft"]["plane"]["Ey"][0, :, :, 0]) ** 2
    return E[:, low].sum(1) + 1e-300, E[:, up].sum(1)


lo0, up0 = split(outs[0])
x = grid.coords[0]
# Read out where the light has fully crossed at 0 V (the coupling length).
window = (x > x0 + 5 * um) & (x < x_e[1] - 1 * um)
i_out = np.flatnonzero(window)[np.argmax((up0 / (lo0 + up0))[window])]
print(f"coupling length L_c ~ {(x[i_out] - x0) / um:.1f} um "
      f"(read out at x = {x[i_out] / um:.1f} um)")
cross = []
for o in outs:
    lo, upp = split(o)
    cross.append(upp[i_out] / (lo[i_out] + upp[i_out]))
cross = np.array(cross)
i_bar = int(np.argmin(cross))
L_drive = x[i_out] - x_e[0]
print(f"cross fraction: {cross[0]:.3f} at 0 V, {cross[i_bar]:.3f} at "
      f"{voltages[i_bar]:.0f} V")
print(f"a 5 mm long device would switch at {voltages[i_bar] * L_drive / 5e-3:.1f} V")

fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
for k, i in enumerate((0, i_bar)):
    E = np.abs(outs[i]["dft"]["plane"]["Ey"][0, :, :, 0]).T
    ax[k].imshow(E, origin="lower", aspect="auto", cmap="magma",
                 extent=[x[0] / um, x[-1] / um, y[0] / um, y[-1] / um])
    ax[k].axvline(x[i_out] / um, color="c", lw=0.8)
    ax[k].set(title=f"|Ey| at {voltages[i]:.0f} V", xlabel="x (um)", ylabel="y (um)")
ax[2].plot(voltages, cross, "o-", label="cross (upper)")
ax[2].plot(voltages, 1 - cross, "s-", label="bar (lower)")
ax[2].set(xlabel=f"voltage (V) for this {L_drive / um:.0f} um electrode",
          ylabel="output fraction", title="delta-beta switching")
sec = ax[2].secondary_xaxis("top", functions=(lambda v: v * L_drive / 5e-3,
                                              lambda v: v * 5e-3 / L_drive))
sec.set_xlabel("equivalent voltage for a 5 mm device (V)")
ax[2].legend(); ax[2].grid(alpha=0.3)
fig.tight_layout()
fig.savefig("09_eo_coupler_switch.png", dpi=130)
print("saved 09_eo_coupler_switch.png")
