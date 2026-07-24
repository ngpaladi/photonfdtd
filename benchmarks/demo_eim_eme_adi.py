"""Demo: the three layout-scale acceleration methods on one SOI platform.

  (1) 2.5-D EIM   - a 220 nm SOI slab mode -> effective indices for a 2-D run
  (2) EME         - a waveguide-width taper's transmission, S-matrix only
  (3) ADI-FDTD    - a resonator stepped at 4x the explicit CFL limit

    python benchmarks/demo_eim_eme_adi.py [outdir]
"""
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import photonfdtd as pf

INK, MUTED = "#2f2f2f", "#8a8a8a"
ACCENT, ACCENT2 = "#2f6fb0", "#b0472f"


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    lam = 1.55e-6
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    fig.patch.set_facecolor("white")

    # ---- (1) EIM: SOI slab mode + effective indices ---- #
    sm = pf.slab_modes([pf.Layer(0.22e-6, 3.48)], lam, polarization="TE",
                       num_modes=1, substrate_index=1.44, cladding_index=1.0)
    eim = pf.EffectiveIndex2D.from_stacks(
        [pf.Layer(0.22e-6, 3.48)], [pf.Layer(0.22e-6, 1.44)],
        lam, polarization="TE", substrate_index=1.44, cladding_index=1.0)
    z = sm.z * 1e9
    ax[0].plot(sm.profiles[0], z, color=ACCENT, lw=2, label="TE0 |Ez| profile")
    ax[0].axhspan(0, 220, color=MUTED, alpha=0.15, label="220 nm Si core")
    ax[0].set_xlabel("mode field (norm.)", color=INK)
    ax[0].set_ylabel("z (nm)", color=INK)
    ax[0].set_title(f"(1) 2.5-D EIM\nn_core={eim.n_core:.3f}  "
                    f"n_bg={eim.n_background:.3f}", color=INK, loc="left")
    ax[0].legend(fontsize=8, frameon=False)

    # ---- (2) EME: linear taper 0.5 -> 0.2 um, transmission vs #sections ---- #
    dy, ny = 15e-9, 260
    y = (np.arange(ny) - ny / 2) * dy
    def wg(width):
        e = np.full(ny, 1.44 ** 2)
        e[np.abs(y) <= width / 2] = 2.83 ** 2
        return e
    Ltaper = 6e-6
    counts = [2, 4, 8, 16, 32, 64]
    Ts = []
    for nsec in counts:
        widths = np.linspace(0.5e-6, 0.2e-6, nsec)
        secs = [pf.Section(wg(w), Ltaper / nsec) for w in widths]
        r = pf.eme_2d(secs, dy, lam, num_modes=6)
        Ts.append(abs(r.transmission(0, 0)) ** 2)
    ax[1].plot(counts, Ts, "o-", color=ACCENT, lw=2, ms=6)
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("taper sections (EME staircase)", color=INK)
    ax[1].set_ylabel("|T00|^2  (fundamental)", color=INK)
    ax[1].set_title("(2) EME taper transmission\n"
                    "adiabatic limit, S-matrix only", color=INK, loc="left")

    # ---- (3) ADI: ring-ish dielectric resonator at 4x CFL ---- #
    nx, nyg, dx = 120, 120, 40e-9
    eps = np.ones((nx, nyg))
    yy, xx = np.mgrid[0:nx, 0:nyg]
    cx, cy = nx / 2, nyg / 2
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    eps[(rr > 18) & (rr < 30)] = 2.6 ** 2            # dielectric ring
    def gauss(f0, fwhm):
        sig = fwhm / (2 * math.sqrt(2 * math.log(2))); t0 = 4 * sig
        return lambda t: np.exp(-((t - t0) / sig) ** 2) * np.sin(2 * math.pi * f0 * (t - t0))
    src = pf.ADISource(nx // 2, nyg // 2 + 24, gauss(pf.C_0 / 1e-6, 8e-15))
    sim = pf.ADISimulation2D(eps, dx, courant_factor=4.0, boundary="absorber",
                             sources=[src], pml_cells=16,
                             backend="rust" if _rust() else "numpy")
    res = sim.run(500, snapshot_interval=499)
    snap = res.snapshots["Ez"][-1]
    v = np.percentile(np.abs(snap), 99.5)
    ax[2].imshow(snap.T, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v,
                 aspect="equal")
    ax[2].contour(eps.T, levels=[3.0], colors=MUTED, linewidths=0.5)
    ax[2].set_xticks([]); ax[2].set_yticks([])
    ax[2].set_title(f"(3) ADI-FDTD @ {sim.courant_factor:.0f}x CFL\n"
                    f"unconditionally stable (dt={sim.dt*1e18:.0f} as)",
                    color=INK, loc="left")

    for a in ax[:2]:
        a.tick_params(colors=MUTED)
        for s in a.spines.values():
            s.set_color(MUTED)
    fig.tight_layout()
    out = f"{outdir}/demo_eim_eme_adi.png"
    fig.savefig(out, dpi=110)
    print("wrote", out)


def _rust():
    from photonfdtd.rustbackend import rust_available
    return rust_available()


if __name__ == "__main__":
    main()
