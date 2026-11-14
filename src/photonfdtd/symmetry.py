"""Mirror symmetry planes: unfolding the stepped half back to the full domain.

A :class:`~photonfdtd.grid.Grid` with ``symmetry=(sx, sy, sz)`` steps only the
upper half of each symmetric axis (see the ``Grid`` docstring); monitor arrays
come back on that half. :func:`unfold` rebuilds the whole domain from them with
each component's mirror parity, and :func:`full_coords` gives the matching
coordinates. Flux monitors and :func:`photonfdtd.photon_spectrum` already
report whole-domain values.

Parity under the mirror ``u -> -u`` on a plane of type ``s`` (``-1`` PEC,
``+1`` PMC): tangential E and normal H take the factor ``s``, normal E and
tangential H take ``-s`` (H is a pseudovector). Where each sample lands follows
the Yee staggering. A component on the nodes (tangential E, normal H) at
coordinate ``x`` mirrors to ``-x``; one offset half a cell up along the axis
(normal E, tangential H) mirrors to the node coordinate ``-x - dx``. On a PEC
plane the node at 0 is shared; on a PMC plane (nodes at ``+-dx/2``) the offset
component at ``-dx/2`` lies on the plane itself, where its odd parity makes it
zero.
"""
from __future__ import annotations

import numpy as np

__all__ = ["unfold", "full_coords", "component_parity"]


def component_parity(component: str, axis: int, sym: int) -> int:
    """Mirror parity (+1 even, -1 odd) of ``component`` across a plane of type
    ``sym`` (-1 PEC, +1 PMC) normal to ``axis``."""
    tangential = "xyz".index(component[1]) != axis
    same = tangential if component[0] == "E" else not tangential
    return int(sym) if same else -int(sym)


def _staggered(component: str, axis: int) -> bool:
    """True if the component sits half a cell off the integer nodes along
    ``axis`` (normal E, tangential H)."""
    along = "xyz".index(component[1]) == axis
    return along if component[0] == "E" else not along


def full_coords(grid, axis: int) -> np.ndarray:
    """Node coordinates of the whole domain along ``axis`` (the stepped
    half's ``grid.coords[axis]`` when the axis is not symmetric)."""
    c = np.asarray(grid.coords[axis])
    s = grid.symmetry[axis]
    if not s:
        return c
    img = -c[::-1] if s == 1 else -c[:0:-1]
    return np.concatenate([img, c])


def unfold(grid, component: str, data) -> np.ndarray:
    """Whole-domain array from a monitor array of the stepped half.

    ``data`` has the grid's three spatial axes last (a ``FieldMonitor`` or
    ``DFTMonitor`` array; leading time/frequency axes are kept). Every
    symmetric axis on which ``data`` spans the stepped half is unfolded with
    the component's parity; a plane monitor normal to a symmetric axis (size
    1 there) is left as is on that axis. The result aligns with
    :func:`full_coords`. Monitors with ``downsample > 1`` are not supported.
    """
    out = np.asarray(data)
    nd = out.ndim
    for a, s in enumerate(grid.symmetry):
        if not s:
            continue
        ax = nd - 3 + a
        n = out.shape[ax]
        if n == 1:
            continue
        if n != grid.shape[a]:
            raise ValueError(
                f"axis {a} has {n} samples, the stepped half {grid.shape[a]}: "
                "unfold needs full-resolution (downsample=1) monitor data")
        p = component_parity(component, a, s)
        stag = _staggered(component, a)
        if s == 1 and not stag:
            img = np.flip(out, axis=ax)
        elif s == 1:
            # The sample at -dx/2 sits on the plane, where it is odd: zero.
            img = np.concatenate([
                np.flip(np.take(out, range(n - 1), axis=ax), axis=ax),
                np.zeros_like(np.take(out, [0], axis=ax))], axis=ax)
        elif stag:
            img = np.flip(np.take(out, range(n - 1), axis=ax), axis=ax)
        else:
            img = np.flip(np.take(out, range(1, n), axis=ax), axis=ax)
        out = np.concatenate([p * img, out], axis=ax)
    return out
