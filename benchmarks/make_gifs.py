"""Render the grating-coupler benchmark movies (one GIF per backend).

    python benchmarks/make_gifs.py <datadir> <backend> [<backend> ...]

Reads <datadir>/gc_<backend>.npz (from bench_grating_coupler.py) and writes
<datadir>/photonfdtd_gc_<backend>.gif. Field polarity is a diverging encoding:
RdBu, white at zero, symmetric robust limits; the structure outline is a
recessive single-hue overlay.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

INK = "#3d3d3d"
MUTED = "#8a8a8a"


def make_gif(datadir, backend):
    d = np.load(f"{datadir}/gc_{backend}.npz")
    frames, eps = d["frames"], d["eps"]
    x, y, times = d["x"] * 1e6, d["y"] * 1e6, d["times"] * 1e15
    wall = float(d["wall"])
    peak = float(d["peak_mb"])
    # Symmetric robust limits from the mid-run field (skip the quiet start).
    v = np.percentile(np.abs(frames[len(frames) // 3:]), 99.8)

    fig, ax = plt.subplots(figsize=(7.6, 5.9), dpi=100)
    fig.patch.set_facecolor("white")
    ext = [x[0], x[-1], y[0], y[-1]]
    im = ax.imshow(frames[0].T, origin="lower", extent=ext, cmap="RdBu_r",
                   vmin=-v, vmax=v, aspect="equal", interpolation="bilinear")
    ax.contour(x, y, eps.T, levels=[(1.444**2 + 2.0**2) / 2, (2.0**2 + 2.85**2) / 2],
               colors=MUTED, linewidths=0.5)
    ax.set_xlabel("x (µm)", color=INK, fontsize=10)
    ax.set_ylabel("y (µm)", color=INK, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ax.spines.values():
        s.set_color(MUTED)
        s.set_linewidth(0.6)
    ax.set_title(f"Elliptic grating coupler — {backend.upper()} backend\n"
                 f"{wall:.1f} s wall · {peak/1024:.2f} GB peak RSS · "
                 f"{int(d['n_steps'])} steps on {int(d['n_cells'])/1e6:.2f}M cells",
                 color=INK, fontsize=11, loc="left", pad=10)
    ax.title.set_linespacing(1.6)
    tcounter = ax.text(1.0, 1.02, "", transform=ax.transAxes, ha="right",
                       va="bottom", color=MUTED, fontsize=10)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Ez (arb.)", color=INK, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_edgecolor(MUTED)
    fig.tight_layout()

    def update(i):
        im.set_data(frames[i].T)
        tcounter.set_text(f"t = {times[i]:5.0f} fs")
        return im, tcounter

    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    out = f"{datadir}/photonfdtd_gc_{backend}.gif"
    anim.save(out, writer=animation.PillowWriter(fps=14))
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    datadir = sys.argv[1]
    for backend in sys.argv[2:]:
        make_gif(datadir, backend)
