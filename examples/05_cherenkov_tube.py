"""Example 5: Cherenkov radiator in a tube - driven by an evanescent field.

A charged particle flies down the vacuum channel of a dielectric tube (here
the 2-D version: a channel of half-gap ``a`` between two n = 2 walls). In the
channel ``n * beta = 0.9 < 1``, so the particle radiates nothing there; what
it carries is a *near* field. Decompose that field into its spatial
frequencies ``k_x`` along the trajectory (equivalently temporal frequencies
``omega = k_x v``) and every component is evanescent in the transverse
direction, falling as

    E(k_x, y)  ~  exp(-k_x |y| / gamma)  =  exp(-omega |y| / (gamma v)).

It is the tail of this evanescent field that reaches the wall, where
``n * beta = 1.8 > 1`` and the wall emits Cherenkov radiation. The wall
excitation at each frequency is therefore set by ``exp(-k_x a / gamma)``: a
wider channel cuts the radiator's spectrum off above ``omega ~ gamma v / a``,
which is why the gap is the design knob of a dielectric Cherenkov radiator (or
wakefield structure). Getting any of this right requires the solver to
reproduce the evanescent field itself, slope and amplitude, so the script
checks exactly that:

  (a) Hz of the particle in the tube: a clean channel, Cherenkov cones in the
      walls at the predicted Mach angle.
  (b) In vacuum (no walls): |Ey(k_x, y)| against |y| for several k_x, on a log
      axis, against the analytic law of a moving 2-D line charge.
  (c) In the tube: the wall excitation at fixed k_x against the half-gap a,
      against exp(-k_x a / gamma).

Six 2-D runs on the JAX engine.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import photonfdtd as pf

n_wall = 2.0                  # tube material (phase velocity c/2)
beta = 0.9                    # in vacuum n*beta = 0.9 (no radiation),
gamma = 1.0 / np.sqrt(1.0 - beta ** 2)   # in the wall n*beta = 1.8 (radiates)
q = 2e-9                      # C/m - amplitude only, Maxwell is linear
Lx, Ly, dx = 34e-6, 7.2e-6, 40e-9
x_start = -15.5e-6
W = 3.2e-6                    # +-W analysis window around the particle

# The particle appears at t = 0, which launches a light shell at c while the
# particle moves at 0.9 c. Snapshot once the shell has pulled W ahead of the
# particle, so the window around it holds only the steady field.
t_snap = W / (pf.C_0 * (1.0 - beta))


def run(half_gap):
    """One 2-D run; ``half_gap=None`` is vacuum. Returns fields on the grid."""
    grid = pf.Grid(size=(Lx, Ly), cell_size=dx, pml_layers=(14, 14, 0))
    structures = []
    if half_gap is not None:
        H = 10e-6              # walls extend through the PML
        for sgn in (+1.0, -1.0):
            structures.append(pf.Box(center=(0.0, sgn * (half_gap + H / 2)),
                                     size=(3 * Lx, H),
                                     medium=pf.Medium.from_index(n_wall)))
    particle = pf.ChargedParticle(charge=q, velocity=(beta * pf.C_0, 0.0),
                                  start=(x_start, 0.0))
    mon = pf.FieldMonitor(name="snap", components=("Ey", "Hz"), times=[t_snap])
    sim = pf.Simulation(grid, structures=structures, sources=[particle],
                        monitors=[mon], run_time=t_snap + 1e-16)
    res = sim.run()
    t = res.monitor_times["snap"][0]
    return dict(
        Ey=res.fields["snap"]["Ey"][0, :, :, 0],
        Hz=res.fields["snap"]["Hz"][0, :, :, 0],
        x=grid.coords[0], y=grid.coords[1],
        x_p=particle.position_at(t)[0], particle=particle,
    )


def spectrum_along_x(field_row, x, x_p):
    """|F(k_x)| of a row of the field over the +-W window around the particle."""
    ix = np.flatnonzero((x > x_p - W) & (x < x_p + W))
    k = 2 * np.pi * np.fft.rfftfreq(ix.size, dx)
    return k, np.abs(np.fft.rfft(field_row[ix])) * dx


gaps = (0.8e-6, 1.0e-6, 1.2e-6, 1.4e-6, 1.6e-6)
print("vacuum run ...", flush=True)
vac = run(None)
tube = {}
for a in gaps:
    print(f"tube half-gap {a*1e6:.1f} um ...", flush=True)
    tube[a] = run(a)

# ---------------------------------------------------------------- figure --- #
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.8),
                       gridspec_kw=dict(width_ratios=[1.5, 1, 1]))

# (a) the tube: Hz snapshot, half-gap 1.2 um
a_show = 1.2e-6
d = tube[a_show]
x_um, y_um, x_p = d["x"] * 1e6, d["y"] * 1e6, d["x_p"]
Hz = d["Hz"]
vmax = np.percentile(np.abs(Hz), 99.7)
sel = (x_um > x_p * 1e6 - 9.0) & (x_um < x_p * 1e6 + 2.0)
ax[0].pcolormesh(x_um[sel], y_um, Hz[sel].T, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                 shading="auto")
for sgn in (+1.0, -1.0):
    ax[0].axhline(sgn * a_show * 1e6, color="k", lw=0.8)
mu = d["particle"].mach_angle(n_wall)          # cone half-angle in the wall
x_line = np.array([x_p, x_p - (Ly / 2 - a_show) / np.tan(mu)])   # to the domain edge
for sgn in (+1.0, -1.0):
    ax[0].plot(x_line * 1e6, sgn * (a_show + (x_p - x_line) * np.tan(mu)) * 1e6,
               "k--", lw=1.0, label="predicted cone" if sgn > 0 else None)
ax[0].plot(x_p * 1e6, 0.0, "ko", ms=4, label="particle")
ax[0].set_aspect("equal")
ax[0].set_xlabel("x (um)")
ax[0].set_ylabel("y (um)")
ax[0].set_title(f"(a) Hz: beta={beta}, n_wall={n_wall}, half-gap {a_show*1e6:.1f} um\n"
                f"no radiation in the channel, Cherenkov in the walls "
                f"(mach {np.degrees(mu):.0f} deg)", fontsize=10, loc="left")
ax[0].legend(loc="lower left", fontsize=8)

# (b) vacuum: per-k_x transverse decay of the near field
rows = [0.3e-6, 0.45e-6, 0.6e-6, 0.8e-6, 1.0e-6, 1.2e-6]
bins = [2, 3, 4, 5]                              # lam_x = 3.2, 2.13, 1.6, 1.28 um
sigma = vac["particle"].cloud_cells * dx         # deposition cloud width
ys, amps = [], []
for y_t in rows:
    j = int(np.argmin(np.abs(vac["y"] - y_t)))
    k, amp = spectrum_along_x(vac["Ey"][:, j], vac["x"], vac["x_p"])
    ys.append(vac["y"][j] + 0.5 * dx)            # Ey sits on the half-cell in y
    amps.append(amp[bins])
ys, amps = np.asarray(ys), np.asarray(amps)
y_fine = np.linspace(0.2e-6, 1.3e-6, 100)
E0 = q / (2 * pf.EPS_0)                          # on-axis spectral amplitude
colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(bins)))
for c, (col, b) in zip(colors, enumerate(bins)):
    lam_x = 2 * np.pi / k[b]
    cloud = np.exp(-0.5 * (k[b] * sigma) ** 2)
    ax[1].semilogy(y_fine * 1e6, E0 * cloud * np.exp(-k[b] * y_fine / gamma) / E0,
                   "-", color=c, lw=1.2)
    ax[1].semilogy(ys * 1e6, amps[:, col] / E0, "o", color=c, ms=5,
                   label=f"lam_x = {lam_x*1e6:.2f} um  (k_x/gamma = {k[b]/gamma*1e-6:.2f}/um)")
ax[1].set_xlabel("|y| from trajectory (um)")
ax[1].set_ylabel("|Ey(k_x, y)|  /  (q / 2 eps0)")
ax[1].set_title("(b) vacuum: each k_x decays as exp(-k_x |y| / gamma)\n"
                "points FDTD, lines analytic (with cloud factor)", fontsize=10, loc="left")
ax[1].legend(fontsize=7.5, frameon=False)

# (c) tube: wall excitation vs half-gap, at fixed depth inside the wall
depth = 0.4e-6
bins_c = [3, 4, 5]                               # lam_x = 2.13, 1.6, 1.28 um
a_arr = np.array(gaps)
wall = np.empty((len(gaps), len(bins_c)))
for i, a in enumerate(gaps):
    d = tube[a]
    j = int(np.argmin(np.abs(d["y"] - (-(a + depth)))))
    k, amp = spectrum_along_x(d["Hz"][:, j], d["x"], d["x_p"])
    wall[i] = amp[bins_c]
colors = plt.cm.viridis(np.linspace(0.3, 0.85, len(bins_c)))
a_fine = np.linspace(a_arr[0], a_arr[-1], 50)
for c, (col, b) in zip(colors, enumerate(bins_c)):
    ax[2].semilogy(a_fine * 1e6, np.exp(-k[b] * (a_fine - a_arr[0]) / gamma),
                   "-", color=c, lw=1.2)
    ax[2].semilogy(a_arr * 1e6, wall[:, col] / wall[0, col], "s", color=c, ms=5,
                   label=f"lam_x = {2*np.pi/k[b]*1e6:.2f} um")
ax[2].set_xlabel("channel half-gap a (um)")
ax[2].set_ylabel(f"|Hz(k_x)| in the wall, {depth*1e9:.0f} nm deep  (norm.)")
ax[2].set_title("(c) wall excitation vs gap: exp(-k_x a / gamma)\n"
                "the evanescent tail sets the radiator's spectrum", fontsize=10, loc="left")
ax[2].legend(fontsize=8, frameon=False)

fig.tight_layout()
out = "05_cherenkov_tube.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved {out}")

# A number to read off: measured vs predicted decay rate per k_x in vacuum.
for col, b in enumerate(bins):
    slope = -np.polyfit(ys, np.log(amps[:, col]), 1)[0]
    print(f"lam_x = {2*np.pi/k[b]*1e6:.2f} um: decay {slope*1e-6:.2f}/um, "
          f"k_x/gamma = {k[b]/gamma*1e-6:.2f}/um")
