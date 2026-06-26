"""Example 4: Cherenkov radiation from a fast charged particle.

A point charge is fired through a uniform dielectric (refractive index n = 2)
at v = 0.9 c. Because the particle outruns the local phase velocity c/n
(n*beta = 1.8 > 1), it radiates a Cherenkov shock cone: the electromagnetic
analogue of a sonic boom.

The radiation propagates at the Cherenkov angle theta_c = arccos(1/(n*beta))
to the trajectory, so the visible wavefront (Mach) cone makes a half-angle
mu = arcsin(1/(n*beta)) with the path. We overlay that predicted cone on the
simulated out-of-plane magnetic field Hz; the two coincide.

An in-plane velocity drives the in-plane electric field, so the radiation lives
in the (Ex, Ey, Hz) polarisation - we visualise Hz.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import photonfdtd as pf

n = 2.0                       # dielectric index (phase velocity c/n)
beta = 0.9                    # particle speed in units of c  ->  n*beta = 1.8
Lx, Ly = 10e-6, 7e-6
dx = 40e-9

grid = pf.Grid(size=(Lx, Ly), cell_size=dx, pml_layers=(14, 14, 0))

# Fill the whole domain with the dielectric.
medium = pf.Box(center=(0.0, 0.0), size=(3 * Lx, 3 * Ly),
                medium=pf.Medium.from_index(n))

# A charge launched from the left, travelling in +x. The charge value only sets
# the field amplitude (Maxwell is linear); we pick a large value for O(1) fields.
particle = pf.ChargedParticle(
    charge=2e-9,
    velocity=(beta * pf.C_0, 0.0),
    start=(-4e-6, 0.0),
)

t_snap = 26e-15
mon = pf.FieldMonitor(name="snap", components=("Hz",), times=[t_snap])

sim = pf.Simulation(grid, structures=[medium], sources=[particle],
                    monitors=[mon], run_time=30e-15)
res = sim.run()

Hz = res.fields["snap"]["Hz"][0, :, :, 0]               # (nx, ny)
x_um = grid.coords[0] * 1e6
y_um = grid.coords[1] * 1e6

# Particle position at the snapshot, and the predicted Mach cone.
x_p = particle.position_at(res.monitor_times["snap"][0])[0]
mu = particle.mach_angle(n)
theta_c = particle.cherenkov_angle(n)

vmax = np.percentile(np.abs(Hz), 99.5)
fig, ax = plt.subplots(figsize=(8.0, 5.6))
im = ax.pcolormesh(x_um, y_um, Hz.T, cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, shading="auto")
# Predicted Mach cone, trailing behind the particle.
x_line = np.array([x_p, x_p - 6e-6])
for sgn in (+1.0, -1.0):
    ax.plot(x_line * 1e6, sgn * (x_p - x_line) * np.tan(mu) * 1e6,
            "k--", lw=1.2, label="predicted cone" if sgn > 0 else None)
ax.plot(x_p * 1e6, 0.0, "ko", ms=5, label="particle")
ax.set_aspect("equal")
ax.set_xlabel("x (um)")
ax.set_ylabel("y (um)")
ax.set_title(f"Cherenkov radiation (Hz):  n={n}, beta={beta},  "
             f"theta_c={np.degrees(theta_c):.1f} deg, mach={np.degrees(mu):.1f} deg")
ax.legend(loc="lower left", fontsize=8)
fig.colorbar(im, ax=ax, shrink=0.8, label="Hz (a.u.)")
fig.tight_layout()
out = "04_cherenkov.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved {out}")
