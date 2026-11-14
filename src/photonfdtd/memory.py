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
* ``disk_tiled``         - a forward run with the fields on disk, streamed
                           through the device in x-slabs
                           (:mod:`photonfdtd.tiled`): one slab's working set
                           on the device, the grid's on disk.
* ``adjoint_disk_tiled`` - the tiled adjoint
                           (:func:`photonfdtd.tiled.value_and_grad_eps_tiled`):
                           one slab and its VJP on the device, about
                           ``2 * sqrt(n_steps / steps_per_tile)`` checkpointed
                           states on disk.

The estimate is the dominant field/coefficient/psi/monitor array memory; it is
approximate (it omits framework overhead and transient temporaries) but the
*ratios* between modes are exact by construction and are what matter for
choosing one.
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
    n_coeff = 3 if (getattr(sim, "subpixel", False)
                    or getattr(sim, "_has_anisotropy", False)) else 1
    if getattr(sim, "_has_conductivity", False):
        n_coeff += 3                          # lossy decay factors ca
    if getattr(sim, "_has_magnetic", False):
        n_coeff += 3                          # per-cell H coefficients
        if sim.magnetic_conductivity_map() is not None:
            n_coeff += 3                      # magnetic decay factors da
    # CPML psi: up to 12 running convolution terms (2 per E/H component), each of
    # order the domain size in the dense JAX layout. Zero when there is no PML.
    has_pml = any(int(p) > 0 for p in sim.grid.pml_layers)
    n_psi = 12 if has_pml else 0
    if getattr(sim.grid, "is_complex", False):
        # Bloch runs step complex fields and psi (the coefficients stay real).
        n_fields, n_psi = 2 * n_fields, 2 * n_psi
    return (n_fields + n_coeff + n_psi) * n_cell * itemsize


def estimate_memory(sim, checkpoint_levels: int = 2,
                    tile_cells: Optional[int] = None,
                    steps_per_tile: int = 8) -> Dict[str, int]:
    """Estimated peak bytes per execution mode for ``sim`` (see module docstring).

    With ``tile_cells`` the result also has ``disk_tiled`` (device bytes of one
    slab) and ``disk_tiled_disk`` (scratch bytes on disk), and the same for
    the tiled adjoint as ``adjoint_disk_tiled`` and ``adjoint_disk_tiled_disk``.
    """
    ws = _working_set_bytes(sim)
    n = max(int(sim.n_steps), 1)
    levels = max(1, int(checkpoint_levels))
    nested_states = levels * int(math.ceil(n ** (1.0 / levels)))
    est = {
        "working_set": ws,
        "forward": ws,
        "adjoint_none": ws * n,
        "adjoint_nested": ws * nested_states,
        "adjoint_reversible": ws * 3,
    }
    if tile_cells is not None:
        from .tiled import tile_bytes, disk_bytes, adjoint_disk_bytes
        T = int(tile_cells)
        est["disk_tiled"] = tile_bytes(sim, T, steps_per_tile)
        est["disk_tiled_disk"] = disk_bytes(sim, T, steps_per_tile)
        est["adjoint_disk_tiled"] = tile_bytes(sim, T, steps_per_tile, adjoint=True)
        est["adjoint_disk_tiled_disk"] = adjoint_disk_bytes(sim, T, steps_per_tile)
    return est


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
    decreasing memory), then ``adjoint_disk_tiled``; for a forward-only run,
    forward, then ``disk_tiled``. A tiled mode is chosen if a one-plane,
    one-step slab fits half the budget, the rule the tiled runs apply.
    Returns the mode name, or ``'infeasible'`` if nothing fits the device.
    """
    if budget_bytes is None:
        budget_bytes = available_memory()[0]
        if budget_bytes is None:
            budget_bytes = float("inf")             # unknown -> don't constrain
    est = estimate_memory(sim, checkpoint_levels)
    from .tiled import choose_tile_cells
    if differentiable:
        order = ["adjoint_none", "adjoint_nested", "adjoint_reversible"]
        for mode in order:
            if est[mode] <= budget_bytes:
                return mode
        # Even the least-memory in-memory adjoint does not fit: tile it.
        return ("adjoint_disk_tiled"
                if choose_tile_cells(sim, 0.5 * budget_bytes, 1, adjoint=True) > 0
                else "infeasible")
    if est["forward"] <= budget_bytes:
        return "forward"
    return ("disk_tiled" if choose_tile_cells(sim, 0.5 * budget_bytes) > 0
            else "infeasible")


def format_report(sim, checkpoint_levels: int = 2, tile_cells: int = 16,
                  steps_per_tile: int = 8) -> str:
    """A human-readable peak-memory table for ``sim``, including a disk-tiled
    run with the given slab (device bytes, and scratch bytes on disk)."""
    est = estimate_memory(sim, checkpoint_levels, tile_cells=tile_cells,
                          steps_per_tile=steps_per_tile)
    lines = [f"Peak-memory estimate  (grid {tuple(sim.grid.shape)}, "
             f"{sim.n_steps} steps, {np.dtype(sim.dtypes['Ex']).name}):"]
    for k in ("forward", "adjoint_none", "adjoint_nested",
              "adjoint_reversible"):
        lines.append(f"  {k:20s} {est[k] / 1e9:8.3f} GB")
    for k in ("disk_tiled", "adjoint_disk_tiled"):
        lines.append(f"  {k:20s} {est[k] / 1e9:8.3f} GB  "
                     f"(slab {tile_cells} + halo, {steps_per_tile} steps/load; "
                     f"{est[k + '_disk'] / 1e9:.3f} GB on disk)")
    return "\n".join(lines)
