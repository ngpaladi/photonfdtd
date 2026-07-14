"""Peak-memory estimation and mode planning for large simulations.

Running a simulation that does not fit in memory is a choice among execution
modes, each trading memory for compute. This module estimates the peak
resident bytes of each and recommends one that fits a budget, so a large PIC
run can be sized *before* it OOMs. The modes:

* ``forward``            - a plain forward run (no adjoint): ~one working set.
* ``adjoint_none``       - reverse-mode AD storing the whole trajectory:
                           O(n_steps) working sets.
* ``adjoint_nested``     - L-level gradient checkpointing:
                           O(levels * n_steps**(1/levels)) working sets.
* ``adjoint_reversible`` - the reversible adjoint (:mod:`photonfdtd.reversible`):
                           O(1) working sets (a few), independent of n_steps.
* ``out_of_core``        - disk-tiled forward stepping
                           (:mod:`photonfdtd.outofcore`): the working set is
                           bounded by the tile, not the grid.

The estimate is the dominant field/coefficient/psi/monitor array memory; it is
approximate (it omits framework overhead and transient temporaries) but the
*ratios* between modes are exact by construction and are what matter for
choosing one. The GPU/host/disk hierarchy (``Simulation(use_gpu=True).run(out_of_core=True)``,
see :mod:`photonfdtd.outofcore`) processes each disk-backed tile on the device,
so the resident-on-GPU figure is the ``out_of_core`` tile working set while the
full arrays live on disk; peak device memory scales with ``tile_cells``, not the
grid (validated on an RTX 4080).
"""
from __future__ import annotations
from typing import Dict, Optional
import math
import numpy as np


def _working_set_bytes(sim) -> int:
    """Bytes of one resident 'working set': the six fields, the E/H update
    coefficient(s), the CPML psi state, evaluated at the sim's dtype."""
    shape = sim.grid.shape
    n_cell = int(np.prod(shape))
    itemsize = np.dtype(sim.dtypes["Ex"]).itemsize
    n_fields = 6
    n_coeff = 3 if getattr(sim, "subpixel", False) else 1
    # CPML psi: up to 12 running convolution terms (2 per E/H component), each of
    # order the domain size in the dense JAX layout. Zero when there is no PML.
    has_pml = any(int(p) > 0 for p in sim.grid.pml_layers)
    n_psi = 12 if has_pml else 0
    return (n_fields + n_coeff + n_psi) * n_cell * itemsize


def estimate_memory(sim, checkpoint_levels: int = 2,
                    tile_cells: Optional[int] = None) -> Dict[str, int]:
    """Estimated peak bytes per execution mode for ``sim`` (see module docstring)."""
    ws = _working_set_bytes(sim)
    n = max(int(sim.n_steps), 1)
    levels = max(1, int(checkpoint_levels))
    nested_states = levels * int(math.ceil(n ** (1.0 / levels)))
    if tile_cells is None:
        tile_cells = max(sim.grid.shape[0] // 8, 1)
    tile_frac = min(1.0, (tile_cells + 1) / max(sim.grid.shape[0], 1))
    return {
        "working_set": ws,
        "forward": ws,
        "adjoint_none": ws * n,
        "adjoint_nested": ws * nested_states,
        "adjoint_reversible": ws * 3,
        "out_of_core": int(ws * tile_frac),
    }


def available_host_ram():
    """Available host RAM in bytes (Linux ``MemAvailable``), or None if it can't
    be read on this platform."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    try:
        import os
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except Exception:
        return None


def _nvidia_total_bytes():
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3)
        return int(out.stdout.strip().splitlines()[0]) * 1024 * 1024
    except Exception:
        return None


def gpu_memory_budget():
    """Bytes a JAX program can allocate on the default GPU (the XLA allocator
    limit minus what's already in use), or None if there's no GPU / it can't be
    determined. Note: this initializes JAX's GPU backend if it hasn't been."""
    try:
        import jax
        for d in jax.devices():
            if d.platform in ("gpu", "cuda", "rocm"):
                try:
                    st = d.memory_stats() or {}
                    lim = st.get("bytes_limit")
                    if lim:
                        return max(0, int(lim) - int(st.get("bytes_in_use", 0)))
                except Exception:
                    pass
                tot = _nvidia_total_bytes()          # fall back to 70% of VRAM
                return int(tot * 0.7) if tot else None
    except Exception:
        pass
    return None


def available_memory():
    """``(bytes, "gpu"|"cpu")`` a run could use on the device JAX would target:
    GPU VRAM if a GPU is present, else host RAM. ``bytes`` is None if unknown."""
    gpu = gpu_memory_budget()
    if gpu is not None:
        return gpu, "gpu"
    return available_host_ram(), "cpu"


def recommend_mode(sim, budget_bytes=None, differentiable: bool = True,
                   checkpoint_levels: int = 2) -> str:
    """Cheapest-compute mode whose estimated peak fits ``budget_bytes``.

    ``budget_bytes`` defaults to the memory actually available on this machine
    (GPU VRAM if present, else host RAM). For a differentiable (adjoint) run the
    preference order is none -> nested -> reversible (increasing compute,
    decreasing memory); for a forward-only run, forward -> out_of_core. Returns
    the mode name, or ``'out_of_core'`` / ``'infeasible'`` as the last resort.
    """
    if budget_bytes is None:
        budget_bytes = available_memory()[0]
        if budget_bytes is None:
            budget_bytes = float("inf")             # unknown -> don't constrain
    est = estimate_memory(sim, checkpoint_levels)
    if differentiable:
        order = ["adjoint_none", "adjoint_nested", "adjoint_reversible"]
        for mode in order:
            if est[mode] <= budget_bytes:
                return mode
        # reversible is the least-memory adjoint; if even it doesn't fit, the
        # grid itself is too big for a single device -> needs tiling/offload.
        return "out_of_core" if est["out_of_core"] <= budget_bytes else "infeasible"
    if est["forward"] <= budget_bytes:
        return "forward"
    return "out_of_core" if est["out_of_core"] <= budget_bytes else "infeasible"


def format_report(sim, checkpoint_levels: int = 2) -> str:
    """A human-readable peak-memory table for ``sim``."""
    est = estimate_memory(sim, checkpoint_levels)
    lines = [f"Peak-memory estimate  (grid {tuple(sim.grid.shape)}, "
             f"{sim.n_steps} steps, {np.dtype(sim.dtypes['Ex']).name}):"]
    for k in ("forward", "adjoint_none", "adjoint_nested",
              "adjoint_reversible", "out_of_core"):
        lines.append(f"  {k:20s} {est[k] / 1e9:8.3f} GB")
    return "\n".join(lines)
