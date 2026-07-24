"""Peak-memory probe on a whole-chip-scale strip: 0.5 mm x 25 um at 25 nm
(20M cells, 2D). Few timesteps - this measures the resident stepping state,
not throughput.

    python benchmarks/bench_memory.py numpy|jax|rust
"""
import json
import resource
import sys
import time

import numpy as np

import photonfdtd as pf

backend = sys.argv[1]

grid = pf.Grid(size=(500e-6, 25e-6), cell_size=25e-9, pml_layers=(12, 12, 0))
wg = pf.Box(center=(0, 0), size=(2e-3, 0.5e-6), medium=pf.Medium.from_index(2.85))
src = pf.PointDipole(position=(-240e-6, 0.0), component="Ez",
                     waveform=pf.GaussianPulse(freq0=pf.C_0 / 1.55e-6, fwhm=15e-15))
mon = pf.FieldMonitor(name="s", components=("Ez",), times=[3.4e-15], downsample=8)
sim = pf.Simulation(grid, structures=[wg], sources=[src], monitors=[mon],
                    run_time=3.5e-15, backend=backend)

t0 = time.perf_counter()
sim.run()
wall = time.perf_counter() - t0
peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
print(json.dumps(dict(backend=backend, n_cells=int(np.prod(grid.shape)),
                      n_steps=sim.n_steps, wall_s=round(wall, 2),
                      peak_rss_gb=round(peak_mb / 1024, 2))))
