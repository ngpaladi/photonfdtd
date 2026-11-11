"""Run a :class:`~photonfdtd.circuit.Chip` in sax.

:func:`to_sax` gives the chip's optical netlist and models in sax's format,
with every instance's electrical state (electrode voltages, heater powers)
baked in from a DC solve, so ``sax.circuit`` reproduces
:meth:`Chip.s_matrix`. The models are plain functions of ``wl`` (um)
returning numpy arrays; sax evaluates them without ``jit``.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import numpy as np

__all__ = ["to_sax"]


def to_sax(chip, drives: Optional[Mapping] = None) -> Tuple[dict, Dict[str, callable]]:
    """``(netlist, models)`` for ``sax.circuit(netlist=..., models=...)``.

    Each instance becomes its own sax component (named after the instance)
    whose model has the instance's DC state at ``drives`` baked in.
    Photodetectors become absorbing terminations; electrical-only instances
    are left out.
    """
    fixed, _ = chip._fixed(drives)
    V, _ = chip._electrical(fixed, {})
    settings = chip._targets(V)
    instances, models = {}, {}
    for inst, m in chip.models.items():
        if not m.ports:
            continue
        s = settings.get(inst, {})

        def fn(wl=1.55, _m=m, _s=s):
            wl = np.atleast_1d(np.asarray(wl, float))
            sd = _m.sdict(wl, **_s)
            out = {k: v for k, v in sd.items()}
            for p in _m.ports:              # sax wants every port in the dict
                out.setdefault((p, p), np.zeros(len(wl), complex))
            return out
        instances[inst] = {"component": inst}
        models[inst] = fn
    net = chip.network
    connections = {}
    for i in range(len(net.ports)):
        for j in range(i + 1, len(net.ports)):
            if net.C[i, j]:
                connections["{},{}".format(*net.ports[i])] = "{},{}".format(*net.ports[j])
    ports = {k: "{},{}".format(*v) for k, v in net.external.items()}
    return {"instances": instances, "connections": connections, "ports": ports}, models
