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
choosing one. A GPU/host/disk hierarchy (process each disk-backed tile on the
device) reduces the resident-on-device figure to the ``out_of_core`` tile
working set while the full arrays live on disk; that execution layer is
designed (see ``outofcore`` scope notes) and pairs with these estimates.
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


def recommend_mode(sim, budget_bytes: int, differentiable: bool = True,
                   checkpoint_levels: int = 2) -> str:
    """Cheapest-compute mode whose estimated peak fits ``budget_bytes``.

    For a differentiable (adjoint) run the preference order is
    none -> nested -> reversible (increasing compute, decreasing memory); for a
    forward-only run, forward -> out_of_core. Returns the mode name, or
    ``'out_of_core'`` / ``'infeasible'`` as the last resort.
    """
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
