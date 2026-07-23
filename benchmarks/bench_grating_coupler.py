"""Backend benchmark: 2D elliptic (focusing) grating coupler, plan view.

A focusing grating coupler on an SOI-like slab, treated by the effective-
index method in the chip plane (Ez polarization): a single-mode access
waveguide opens into a slab wedge, and the grating lines are the standard
confocal-ellipse family

    q * lam0 = n_slab * r - F * x ,      F = n_c * sin(theta)

(ellipses with one focus at the waveguide aperture; Mekis-style layout), so
every line launches the cylindrical wave from the focus with the phase match
of a fiber tilted theta off vertical. The 2D plan-view sim can't diffract out
of plane, but it exercises exactly the update stencil, CPML, and a curved
multi-structure geometry - a realistic benchmark workload with a photogenic
field.

Run one backend per process (clean wall-clock + peak-RSS accounting):

    python benchmarks/bench_grating_coupler.py numpy|jax|rust|rust-cuda \
        [outdir] [float64|float32]

Writes <outdir>/gc_<tag>.npz with the Ez movie, timings, and peak RSS
(tag = backend, plus "_f32" for float32). For GPU JAX, put the nvidia pip
wheels' lib dirs on LD_LIBRARY_PATH.
"""
import json
import math
import resource
import sys
import time

import numpy as np

import photonfdtd as pf

# ----------------------------- geometry ---------------------------------- #
LAM0 = 1.55e-6
N_SLAB = 2.85          # 220 nm SOI slab, TE effective index
N_ETCH = 2.0           # 70 nm-etched grating region, TE effective index
N_BG = 1.444           # oxide background
THETA = math.radians(14.0)   # fiber tilt (in air)
F_PHASE = math.sin(THETA)    # n_c * sin(theta), n_c = 1 (air)

DX = 25e-9
SIZE = (22e-6, 16e-6)
PML = (12, 12, 0)
FOCUS = (-8e-6, 0.0)         # waveguide aperture = ellipse focus
WEDGE_HALF_ANGLE = math.radians(40.0)
GRATING_HALF_ANGLE = math.radians(35.0)
Q_FIRST, N_LINES = 8, 16     # first grating order index and number of lines
FILL = 0.5                   # etched fraction of the local period
RUN_TIME = 300e-15
N_FRAMES = 110               # movie frames to record
DOWNSAMPLE = 2


def _arc(r_of_phi, phi):
    """(x, y) polyline of a radial curve about FOCUS."""
    return np.column_stack([FOCUS[0] + r_of_phi * np.cos(phi),
                            FOCUS[1] + r_of_phi * np.sin(phi)])


def build_structures():
    slab = pf.Medium.from_index(N_SLAB)
    etch = pf.Medium.from_index(N_ETCH)

    structures = []
    # Access waveguide from the -x edge (through the PML) to the wedge apex.
    structures.append(pf.Box(center=((-11.5e-6 + FOCUS[0]) / 2, 0.0),
                             size=(FOCUS[0] + 11.5e-6, 0.5e-6), medium=slab))
    # Slab wedge (fan) with its apex at the focus.
    phi = np.linspace(-WEDGE_HALF_ANGLE, WEDGE_HALF_ANGLE, 81)
    fan = _arc(np.full_like(phi, 18e-6), phi)
    structures.append(pf.PolySlab(
        vertices=tuple(map(tuple, np.vstack([[FOCUS], fan]))),
        z_bounds=(-1.0, 1.0), medium=slab))
    # Confocal-ellipse grating trenches:  n_slab*r - F*(x-x_f) = q*lam0
    #   =>  r(phi) = q*lam0 / (n_slab - F*cos(phi)).
    phi = np.linspace(-GRATING_HALF_ANGLE, GRATING_HALF_ANGLE, 121)
    for q in range(Q_FIRST, Q_FIRST + N_LINES):
        r_in = q * LAM0 / (N_SLAB - F_PHASE * np.cos(phi))
        width = FILL * LAM0 / (N_SLAB - F_PHASE * np.cos(phi))
        outer = _arc(r_in + width, phi)
        inner = _arc(r_in, phi[::-1])
        structures.append(pf.PolySlab(
            vertices=tuple(map(tuple, np.vstack([outer, inner]))),
            z_bounds=(-1.0, 1.0), medium=etch))
    return structures


def build_sim(backend, precision="float64"):
    grid = pf.Grid(size=SIZE, cell_size=DX, pml_layers=PML)
    freq0 = pf.C_0 / LAM0
    yprof = np.linspace(-0.75e-6, 0.75e-6, 61)
    src = pf.ModeSource(
        center=(-9.5e-6, 0.0, 0.0), size=(0.0, 1.5e-6, 0.0), component="Ez",
        waveform=pf.GaussianPulse(freq0=freq0, fwhm=15e-15),
        profile=np.exp(-(yprof / 0.25e-6) ** 2), profile_coords=(yprof,))
    # interval chosen for ~N_FRAMES frames; exact value fixed by n_steps below.
    probe = pf.Simulation(grid, run_time=RUN_TIME, backend="numpy")
    interval = max(probe.n_steps // N_FRAMES, 1)
    mon = pf.FieldMonitor(name="movie", components=("Ez",),
                          interval=interval, downsample=DOWNSAMPLE)
    # Monitor storage stays float64 for the cross-backend parity probe even
    # when the compute precision is float32.
    prec = {"compute": precision, "monitors": "float64"}
    return pf.Simulation(grid, structures=build_structures(), sources=[src],
                         monitors=[mon], run_time=RUN_TIME, backend=backend,
                         precision=prec)


def main():
    backend = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else "."
    precision = sys.argv[3] if len(sys.argv) > 3 else "float64"
    tag = backend + ("_f32" if precision == "float32" else "")
    sim = build_sim(backend, precision)
    n_cells = int(np.prod(sim.grid.shape))

    t0 = time.perf_counter()
    result = sim.run()
    wall = time.perf_counter() - t0

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    frames = result.fields["movie"]["Ez"][..., 0]     # (n_frames, nx, ny)
    eps = np.asarray(sim.eps_r)[::DOWNSAMPLE, ::DOWNSAMPLE, 0]
    xs = sim.grid.coords[0][::DOWNSAMPLE]
    ys = sim.grid.coords[1][::DOWNSAMPLE]
    np.savez_compressed(
        f"{outdir}/gc_{tag}.npz",
        frames=frames.astype(np.float32), eps=eps,
        x=xs, y=ys, times=result.monitor_times["movie"],
        frames64_tail=frames[-1].astype(np.float64),   # full-precision parity probe
        wall=wall, peak_mb=peak_mb,
        n_steps=sim.n_steps, n_cells=n_cells)
    print(json.dumps(dict(
        backend=tag, wall_s=round(wall, 2), peak_rss_mb=round(peak_mb),
        n_steps=sim.n_steps, n_cells=n_cells,
        mcellsteps_per_s=round(n_cells * sim.n_steps / wall / 1e6),
    )))


if __name__ == "__main__":
    main()
