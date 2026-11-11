"""Scattering-matrix circuit solver.

Every instance is a block of the global scattering matrix; nets join pairs of
instance ports. With ``a`` the waves entering the instance ports and ``b``
those leaving, ``b = S a``, and a net between ports ``i`` and ``j`` feeds one
into the other: ``a_i = b_j``, ``a_j = b_i``. Light injected at the circuit's
external ports adds ``a_ext``, so

    (I - C S) a = a_ext,      b = S a,

with ``C`` the permutation of the nets. One dense solve per wavelength gives
every wave in the circuit, including the light reaching a photodetector on an
internal port; the circuit's own S-matrix follows by injecting unit vectors.

Instance S-matrices use the sax convention: a dict ``{(p_in, p_out): S}``
with ``S`` the amplitude from ``p_in`` to ``p_out``, arrays over wavelength.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

SDict = Dict[Tuple[str, str], np.ndarray]


@dataclass
class Waves:
    """Every wave of a solved circuit. ``a[w, k]`` enters and ``b[w, k]``
    leaves global port ``k`` (``ports[k]`` is ``(instance, port)``) at
    wavelength ``w``."""
    ports: List[Tuple[str, str]]
    a: np.ndarray
    b: np.ndarray

    def index(self, inst: str, port: str) -> int:
        return self.ports.index((inst, port))

    def incident(self, inst: str, port: str) -> np.ndarray:
        """Amplitude arriving at ``inst``'s ``port`` (the wave entering it)."""
        return self.a[:, self.index(inst, port)]

    def outgoing(self, inst: str, port: str) -> np.ndarray:
        return self.b[:, self.index(inst, port)]


class Network:
    """The optical connectivity of a circuit.

    ``instance_ports`` maps instance name to its optical port names, ``nets``
    lists ``((inst, port), (inst, port))`` pairs, ``external`` maps external
    port name to ``(inst, port)``. Ports in neither a net nor ``external``
    are open: light leaving them is lost and none enters.
    """

    def __init__(self, instance_ports: Dict[str, Sequence[str]],
                 nets: Sequence[Tuple[Tuple[str, str], Tuple[str, str]]],
                 external: Dict[str, Tuple[str, str]]):
        self.ports = [(i, p) for i, ps in instance_ports.items() for p in ps]
        self._idx = {k: n for n, k in enumerate(self.ports)}
        self.instance_ports = {i: list(ps) for i, ps in instance_ports.items()}
        n = len(self.ports)
        C = np.zeros((n, n))
        used = set()
        for p, q in nets:
            for x in (p, q):
                if x not in self._idx:
                    raise KeyError(f"net endpoint {x} is not an optical port of the circuit")
                if x in used:
                    raise ValueError(f"port {x} is in more than one net")
                used.add(x)
            i, j = self._idx[p], self._idx[q]
            C[i, j] = C[j, i] = 1.0
        self.C = C
        self.external = dict(external)
        for name, x in self.external.items():
            if x not in self._idx:
                raise KeyError(f"external port {name!r} -> {x} is not an optical port")
            if x in used:
                raise ValueError(f"external port {name!r} is also in a net")

    def assemble(self, sdicts: Dict[str, SDict], nw: int) -> np.ndarray:
        """The block-diagonal global S, shape ``(nw, n, n)``, ``S[w, out, in]``."""
        n = len(self.ports)
        S = np.zeros((nw, n, n), complex)
        for inst, ports in self.instance_ports.items():
            sd = sdicts[inst]
            for (pi, po), v in sd.items():
                if pi not in ports or po not in ports:
                    raise KeyError(f"{inst}: model port pair {(pi, po)} not among {ports}")
                S[:, self._idx[(inst, po)], self._idx[(inst, pi)]] = np.broadcast_to(
                    np.asarray(v, complex), (nw,))
        return S

    def solve(self, S: np.ndarray, inputs: Dict[str, np.ndarray]) -> Waves:
        """All waves for the injected external amplitudes ``inputs``
        (``{external port: amplitude}``, scalars or arrays over wavelength)."""
        nw, n, _ = S.shape
        a_ext = np.zeros((nw, n), complex)
        for name, amp in inputs.items():
            if name not in self.external:
                raise KeyError(f"no external port {name!r}; have {sorted(self.external)}")
            a_ext[:, self._idx[self.external[name]]] = np.broadcast_to(np.asarray(amp, complex), (nw,))
        A = np.eye(n)[None] - self.C[None] @ S
        a = np.linalg.solve(A, a_ext[..., None])[..., 0]
        return Waves(self.ports, a, np.einsum("wij,wj->wi", S, a))

    def s_matrix(self, S: np.ndarray) -> SDict:
        """The circuit's own S-matrix between its external ports."""
        names = list(self.external)
        nw, n, _ = S.shape
        A = np.eye(n)[None] - self.C[None] @ S
        ext = [self._idx[self.external[k]] for k in names]
        rhs = np.zeros((nw, n, len(ext)), complex)
        for col, k in enumerate(ext):
            rhs[:, k, col] = 1.0
        a = np.linalg.solve(A, rhs)
        b = S @ a
        return {(pi, po): b[:, ext[o], i] for i, pi in enumerate(names)
                for o, po in enumerate(names)}
