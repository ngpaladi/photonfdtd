"""Example 1: a Gaussian pulse propagating through 1D vacuum.

Launches an Ey dipole pulse and watches it propagate to the right. CPML
boundaries on both ends absorb the outgoing wave with very small reflection.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import photonfdtd as pf

lam0 = 1.0e-6
freq0 = pf.C_0 / lam0
dx = lam0 / 30
L = 8e-6

grid = pf.Grid(size=(L,), cell_size=dx, pml_layers=(12,))
src = pf.PointDipole(
    position=(-2.5e-6,), component="Ey",
    waveform=pf.GaussianPulse(freq0=freq0, fwhm=6e-15),
)
mon = pf.FieldMonitor(name="snap", components=("Ey",), interval=20)
sim = pf.Simulation(grid, sources=[src], monitors=[mon], run_time=60e-15)
res = sim.run()

Ey = res.fields["snap"]["Ey"][..., 0, 0]
times_fs = res.monitor_times["snap"] * 1e15
x_um = grid.coords[0] * 1e6

fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(
    Ey, aspect="auto", origin="lower", cmap="RdBu_r",
    extent=[x_um.min(), x_um.max(), times_fs.min(), times_fs.max()],
    vmin=-np.abs(Ey).max(), vmax=np.abs(Ey).max(),
)
ax.set_xlabel("x (um)")
ax.set_ylabel("time (fs)")
ax.set_title("1D Ey pulse propagating in vacuum")
fig.colorbar(im, ax=ax, label="Ey (V/m)")
fig.tight_layout()
out = "01_1d_vacuum_pulse.png"
fig.savefig(out, dpi=140)
print(f"saved {out}")
