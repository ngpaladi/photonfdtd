"""Example 2: 2D point dipole radiating in vacuum.

A z-polarised electric dipole in the centre of a 2D xy domain emits cylindrical
waves. CPML on all four sides absorbs them with negligible reflection.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import photonfdtd as pf

lam0 = 1.0e-6
freq0 = pf.C_0 / lam0
dx = lam0 / 20
L = 6e-6

grid = pf.Grid(size=(L, L), cell_size=dx, pml_layers=(12, 12, 0))
src = pf.PointDipole(
    position=(0.0, 0.0), component="Ez",
    waveform=pf.GaussianPulse(freq0=freq0, fwhm=8e-15),
)
mon = pf.FieldMonitor(
    name="snap", components=("Ez",),
    times=np.linspace(10e-15, 35e-15, 4).tolist(),
)
sim = pf.Simulation(grid, sources=[src], monitors=[mon], run_time=40e-15)
res = sim.run()

Ez = res.fields["snap"]["Ez"][..., 0]                    # (n_frames, nx, ny)
times_fs = res.monitor_times["snap"] * 1e15
x_um = grid.coords[0] * 1e6
y_um = grid.coords[1] * 1e6

vmax = np.abs(Ez).max()
fig, axes = plt.subplots(1, len(times_fs), figsize=(3.0 * len(times_fs), 3.2))
for ax, frame, t in zip(axes, Ez, times_fs):
    im = ax.pcolormesh(
        x_um, y_um, frame.T,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto",
    )
    ax.set_title(f"t = {t:.1f} fs")
    ax.set_aspect("equal")
    ax.set_xlabel("x (um)")
axes[0].set_ylabel("y (um)")
fig.suptitle("Ez radiated by a 2D point dipole", y=1.02)
fig.tight_layout()
out = "02_2d_dipole.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved {out}")
