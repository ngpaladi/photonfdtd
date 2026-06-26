"""Example 3: fundamental TE-like mode of an LNOI strip waveguide.

This is the same waveguide as in the photonfdtd/FastTiming/ModeSolving demo
(LiNbO3 strip on SiO2 BOX on Si). The scalar mode solver under-estimates
n_eff relative to a full-vectorial solve (high-index contrast amplifies the
neglected boundary terms), but the mode profile and the qualitative answer
are correct, and it runs in seconds with no API key or remote service.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import photonfdtd as pf

lam0 = 0.6e-6                              # 600 nm

ln = pf.Medium.from_index(2.30, name="LiNbO3")
sio2 = pf.Medium.from_index(1.46, name="SiO2")

# Stack: SiO2 box below the LN strip, air above, Si substrate further below
ln_strip = pf.Box(center=(0.0, 0.15e-6), size=(0.8e-6, 0.3e-6), medium=ln)
box = pf.Box(center=(0.0, -0.75e-6), size=(20e-6, 1.5e-6), medium=sio2)

ms = pf.ModeSolver(
    size=(3e-6, 3.0e-6),
    cell_size=lam0 / 30,
    structures=[box, ln_strip],
    wavelength=lam0,
    background_eps=1.0,
    num_modes=2,
)
res = ms.solve()
print("Mode  n_eff")
for i, n in enumerate(res.n_eff):
    print(f"  {i}   {n:.4f}")

# Plot the fundamental mode profile
y_um = res.y * 1e6
z_um = res.z * 1e6
psi = res.psi[0]
fig, ax = plt.subplots(figsize=(6, 4))
pcm = ax.pcolormesh(y_um, z_um, (psi ** 2).T, shading="auto", cmap="inferno")
# overlay the LN strip outline
import matplotlib.patches as mpatches
ax.add_patch(mpatches.Rectangle(
    (-0.4, 0.0), 0.8, 0.3,
    fill=False, edgecolor="white", lw=1.0, ls="--",
))
ax.set_aspect("equal")
ax.set_xlabel("y (um)")
ax.set_ylabel("z (um)")
ax.set_title(f"LNOI TE0 mode (scalar solver)  n_eff = {res.n_eff[0]:.3f}")
fig.colorbar(pcm, ax=ax, label="|psi|^2 (norm.)")
fig.tight_layout()
out = "03_lnoi_mode.png"
fig.savefig(out, dpi=140)
print(f"saved {out}")
