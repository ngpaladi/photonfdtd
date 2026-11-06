"""Example 6: inverse-design a 2D 1x2 power splitter by gradient descent.

One input waveguide on the left, two output waveguides on the right, and a
design box in between whose etch pattern is free. The figure of merit is the
*weaker* of the two output-port overlaps (so the optimizer must split
evenly), normalized by what a straight waveguide delivers. In 2-D the port
"mode" here is a Gaussian projection - a cheap proxy for modal power; a 3-D
device would use ``pf.ModeTransmission`` with a solved mode instead. The density is filtered
to a minimum feature size, projected toward binary with a sharpness ``beta``
that doubles every 10 iterations, and stepped with Adam; every gradient is
one forward + one adjoint FDTD run, compiled once for the whole loop.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import photonfdtd as pf

jax.config.update("jax_enable_x64", True)

lam = 1.55e-6
f0 = pf.C_0 / lam
dx = 50e-9
core = pf.Medium.from_index(2.8)
w = 0.4e-6                                  # waveguide width
y_out = 0.6e-6                              # output-arm offset
box = 1.6e-6                                # design box side


def chip(straight=False):
    grid = pf.Grid(size=(4.4e-6, 3.0e-6), cell_size=dx, pml_layers=(10, 10, 0))
    wgs = [pf.Box(center=(-1.6e-6, 0), size=(1.6e-6, w), medium=core)]
    if straight:                             # reference: one straight guide
        wgs = [pf.Box(center=(0, 0), size=(5e-6, w), medium=core)]
    else:
        wgs += [pf.Box(center=(1.6e-6, s * y_out), size=(1.6e-6, w), medium=core)
                for s in (-1, 1)]
    src = pf.PointDipole(position=(-1.6e-6, 0), component="Ez",
                         waveform=pf.GaussianPulse(freq0=f0, fwhm=8e-15))
    mon = pf.DFTMonitor(name="out", components=("Ez",), freqs=[f0],
                        plane_axis="x", plane_position=1.6e-6)
    return pf.Simulation(grid, structures=wgs, sources=[src], monitors=[mon],
                         run_time=90e-15)


sim = chip()
y = np.asarray(sim.grid.coords[1])
profile = lambda yc: np.exp(-((y - yc) / (0.6 * w)) ** 2)[None, :, None]

# Normalization: the power a straight guide delivers to one centred port.
ref = chip(straight=True).run()
p_ref = np.abs(np.sum(ref.dft["out"]["Ez"] * profile(0.0))) ** 2

ports = [pf.FieldProjection("out", "Ez", profile(s * y_out)) for s in (-1, 1)]
fom = pf.combine(ports, reduce="min")        # the weaker arm
loss = lambda out: -fom(out) / p_ref

region = pf.DesignRegion(center=(0, 0), size=(box, box), core_z=(-1, 1),
                         eps_solid=core.eps_r, eps_void=1.0,
                         filter_radius=120e-9)
vg = pf.make_value_and_grad_params(sim, region.ce_builder(sim), loss)
rho0 = jnp.full(region.shape(sim.grid), 0.5)
res = pf.minimize(vg, rho0, pf.Adam(0.05), n_iter=40,
                  schedule=pf.beta_continuation(2.0, 2.0, every=10, beta_max=16.0),
                  verbose=True)
rho_phys = np.asarray(region.rho_phys(sim, res.x, beta=16.0))
print(f"weaker arm: {-res.history[-1]['value']:.3f} x the straight-guide overlap; "
      f"binarization {pf.binarization(rho_phys):.2f}")

try:
    pf.export_gds("06_splitter.gds", region=region, rho=rho_phys, grid=sim.grid)
    print("wrote 06_splitter.gds")
except ImportError:
    print("gdstk not installed; skipping GDS export (pip install photonfdtd[gds])")

# Final design in context + its field. For the picture the run uses the
# isotropic (Ez) permittivity, without the in-plane sidewall anisotropy.
final = region.ce_builder(sim)(res.x, beta=16.0)
eps = sim.dt / (np.asarray(final["Ez"])[..., 0] * pf.EPS_0)
snap = pf.FieldMonitor(name="f", components=("Ez",), times=[70e-15])
sim_f = chip()
sim_f.eps_r = eps[..., None]
sim_f.monitors.append(snap)
Ez = sim_f.run().fields["f"]["Ez"][0, ..., 0]

x_um, y_um = sim.grid.coords[0] * 1e6, y * 1e6
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
it = [h["it"] for h in res.history]
ax[0].plot(it, [-h["value"] for h in res.history], "k-")
ax[0].set_xlabel("iteration")
ax[0].set_ylabel("weaker-arm overlap / straight guide")
for b in range(10, 40, 10):
    ax[0].axvline(b, color="0.8", lw=0.8)
ax[0].set_title("convergence (grey: beta doubles)")
ax[1].pcolormesh(x_um, y_um, eps.T, cmap="Greys", shading="auto")
ax[1].set_title("optimized permittivity")
vmax = np.abs(Ez).max()
ax[2].pcolormesh(x_um, y_um, Ez.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                 shading="auto")
ax[2].set_title("Ez at 70 fs")
for a in ax[1:]:
    a.set_aspect("equal")
    a.set_xlabel("x (um)")
    a.set_ylabel("y (um)")
fig.tight_layout()
fig.savefig("06_inverse_design_splitter.png", dpi=120)
print("saved 06_inverse_design_splitter.png")
