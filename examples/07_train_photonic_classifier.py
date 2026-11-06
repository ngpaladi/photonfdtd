"""Example 7: train a chip like a neural network.

Two input waveguides carry coherent light; a sample is the pair of complex
amplitudes driving them, and its class is their relative phase (in phase ->
class 0, anti-phase -> class 1, with noise and random overall brightness).
Detectors at the two output ports read the powers, and the fraction of power
in port ``m`` is the score for class ``m``. The etch pattern of the design box
is fitted to the training set by minibatch Adam on a cross-entropy loss.

The chip is linear in the fields, so ``PhotonicModel`` simulates each input
port once to get the 2x2 transfer matrix, and every sample of a batch is then
just ``T @ x``: a training step costs 2 forward + 2 adjoint FDTD runs no
matter how big the batch or the dataset.
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
dx = 60e-9
core = pf.Medium.from_index(2.8)
w = 0.36e-6
ys = (-0.5e-6, 0.5e-6)                       # port positions


def chip(port):
    """The chip with only input ``port`` driven."""
    grid = pf.Grid(size=(4.0e-6, 3.0e-6), cell_size=dx, pml_layers=(8, 8, 0))
    wgs = [pf.Box(center=(s * 1.5e-6, y), size=(1.4e-6, w), medium=core)
           for s in (-1, 1) for y in ys]
    src = pf.PointDipole(position=(-1.6e-6, ys[port]), component="Ez",
                         waveform=pf.GaussianPulse(freq0=f0, fwhm=8e-15))
    mon = pf.DFTMonitor(name="out", components=("Ez",), freqs=[f0],
                        plane_axis="x", plane_position=1.5e-6)
    return pf.Simulation(grid, structures=wgs, sources=[src], monitors=[mon],
                         run_time=70e-15)


sims = [chip(0), chip(1)]
y = np.asarray(sims[0].grid.coords[1])
readouts = [pf.FieldProjection("out", "Ez",
                               np.exp(-((y - yc) / (0.6 * w)) ** 2)[None, :, None])
            for yc in ys]
region = pf.DesignRegion(center=(0, 0), size=(1.6e-6, 2.0e-6), core_z=(-1, 1),
                         eps_solid=core.eps_r, eps_void=1.0,
                         filter_radius=120e-9)
model = pf.PhotonicModel(sims, region, readouts)


def dataset(n, seed):
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, n)
    phase = np.where(label == 0, 0.0, np.pi) + rng.normal(0, 0.5, n)
    X = np.stack([np.ones(n), np.exp(1j * phase)], 1) * rng.uniform(0.5, 1.5, (n, 1))
    return X, label


X, Y = dataset(256, 0)
Xv, Yv = dataset(128, 1)
rho0 = jnp.full(region.shape(sims[0].grid), 0.5)
print("untrained accuracy:", pf.accuracy(model.predict(rho0, Xv, beta=2.0), Yv))
res = pf.train(model, rho0, X, Y, optimizer=pf.Adam(0.04), epochs=8,
               batch_size=32, schedule=pf.beta_continuation(2.0, 2.0, 16, 16.0),
               validation=(Xv, Yv), verbose=True)
P = model.predict(res.x, Xv, beta=16.0)
print("validation accuracy:", pf.accuracy(P, Yv))

h = res.history
step = [r["it"] for r in h]
ep = [r for r in h if "val_accuracy" in r]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].plot(step, [r["value"] for r in h], color="0.6", label="batch loss")
ax[0].plot([r["it"] for r in ep], [r["val_loss"] for r in ep], "ko-",
           label="validation loss")
ax[0].set_xlabel("optimizer step")
ax[0].set_ylabel("cross-entropy")
ax[0].legend()
ax[1].plot([r["it"] for r in ep], [r["val_accuracy"] for r in ep], "ko-")
ax[1].set_ylim(0, 1.05)
ax[1].set_xlabel("optimizer step")
ax[1].set_ylabel("validation accuracy")
frac = np.asarray(P[:, 1] / P.sum(1))
phase = np.angle(Xv[:, 1] / Xv[:, 0])
ax[2].scatter(phase, frac, c=Yv, cmap="coolwarm", s=10)
ax[2].set_xlabel("input relative phase (rad)")
ax[2].set_ylabel("power fraction in port 1")
ax[2].set_title("trained chip on validation set")
fig.tight_layout()
fig.savefig("07_train_photonic_classifier.png", dpi=120)
print("saved 07_train_photonic_classifier.png")
