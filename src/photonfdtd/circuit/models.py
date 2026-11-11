"""Component models for the circuit solver.

A model is a :class:`Model`: a function ``fn(wl, **settings) -> SDict`` in
the sax convention (``wl`` in micrometres, ``SDict`` a dict
``{(p_in, p_out): amplitude}``), the optical ports it has, and the
conductors that carry its electrical ports. Subclasses add the electrical
behaviour:

- :class:`Modulator`: its electrodes are conductors and the model takes their
  voltages, ``fn(wl, V={electrode: volts})``. Optional first-order bandwidth.
- :class:`Heater`: a resistor between two conductors; the model takes the
  dissipated power, ``fn(wl, P=watts)``. Optional thermal time constant.
- :class:`Photodetector`: absorbs the light at its optical port and drives a
  photocurrent between its anode and cathode.
- :class:`Resistor`: an electrical-only element.

Conductors are named groups of electrical ports that are one piece of metal
(an electrode's two ends, a pad's four sides).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .solver import SDict

__all__ = ["Model", "Modulator", "Heater", "Photodetector", "Resistor", "conductor",
           "waveguide", "directional_coupler", "mmi1x2", "mmi2x2", "terminator",
           "heater", "photodetector", "resistor", "from_lnoi", "reciprocal"]


@dataclass
class Model:
    """A passive component (optionally with metal: ``conductors``)."""
    fn: Callable[..., SDict]
    ports: Tuple[str, ...] = ()
    conductors: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    def sdict(self, wl, **settings) -> SDict:
        return self.fn(np.atleast_1d(np.asarray(wl, float)), **settings)


@dataclass
class Modulator(Model):
    """Electro-optic: ``fn(wl, V={electrode: volts})``; each conductor is an
    electrode. ``bandwidth`` (Hz) is the 3 dB frequency of a first-order
    low-pass on the electrode voltages, standing for the electrode's RF
    response in time-domain runs (``None``: instantaneous)."""
    bandwidth: Optional[float] = None


@dataclass
class Heater(Model):
    """Thermo-optic: ``fn(wl, P=watts)``, with the heater a ``resistance``
    (ohm) between conductors ``terminals``. ``tau`` (s) is the thermal time
    constant of a first-order response in time-domain runs."""
    terminals: Tuple[str, str] = ("a", "b")
    resistance: float = 1000.0
    tau: Optional[float] = None


@dataclass
class Photodetector(Model):
    """Absorbs the light arriving at ``port``; the photocurrent
    ``responsivity * P + dark_current`` (A) leaves the device at the anode
    and enters at the cathode. ``bandwidth`` (Hz) low-passes it in
    time-domain runs."""
    port: str = "o1"
    terminals: Tuple[str, str] = ("anode", "cathode")
    responsivity: float = 1.0
    dark_current: float = 0.0
    bandwidth: Optional[float] = None


@dataclass
class Resistor(Model):
    """A ``resistance`` (ohm) between conductors ``terminals``; no optics."""
    terminals: Tuple[str, str] = ("a", "b")
    resistance: float = 50.0


def reciprocal(pairs: Dict[Tuple[str, str], np.ndarray]) -> SDict:
    """Fill in the reverse direction of each entry (``S_ji = S_ij``)."""
    out = dict(pairs)
    for (p, q), v in pairs.items():
        out.setdefault((q, p), v)
    return out


def _no_optics(wl, **_):
    return {}


def conductor(ports: Sequence[str], name: str = "metal") -> Model:
    """A piece of metal (pad, wire, via) joining ``ports``; no optics."""
    return Model(_no_optics, (), {name: tuple(ports)})


# --------------------------------------------------------------------------- #
# Passive optics.
# --------------------------------------------------------------------------- #
def _propagation(wl, length, neff, ng, wl0, loss_db_cm):
    """Amplitude transmission of ``length`` um of waveguide: phase from
    n_eff with first-order dispersion set by n_g, loss in dB/cm."""
    neff_wl = neff + (neff - ng) * (wl - wl0) / wl0
    phase = 2 * np.pi * neff_wl * length / wl
    amp = 10 ** (-loss_db_cm * length * 1e-4 / 20)
    return amp * np.exp(-1j * phase)


def waveguide(length: float, neff: float = 1.9, ng: float = 2.3, wl0: float = 1.55,
              loss_db_cm: float = 0.0, ports: Tuple[str, str] = ("o1", "o2")) -> Model:
    """A straight, bend or taper of optical path ``length`` (um)."""
    def fn(wl, **_):
        return reciprocal({(ports[0], ports[1]): _propagation(wl, length, neff, ng, wl0, loss_db_cm)})
    return Model(fn, tuple(ports))


def directional_coupler(coupling: float = 0.5, loss_db: float = 0.0,
                        ports=("o1", "o2", "o3", "o4")) -> Model:
    """A lossless (up to ``loss_db``) 2 x 2 coupler moving ``coupling`` of the
    power across. Ports as gdsfactory's ``coupler``: o1 lower and o2 upper on
    the left, o3 upper and o4 lower on the right."""
    o1, o2, o3, o4 = ports
    t = np.sqrt(1 - coupling) * 10 ** (-loss_db / 20)
    k = -1j * np.sqrt(coupling) * 10 ** (-loss_db / 20)

    def fn(wl, **_):
        one = np.ones_like(wl)
        return reciprocal({(o1, o4): t * one, (o2, o3): t * one,
                           (o1, o3): k * one, (o2, o4): k * one})
    return Model(fn, tuple(ports))


def mmi1x2(loss_db: float = 0.0, ports=("o1", "o2", "o3")) -> Model:
    """An ideal 1 x 2 splitter (gdsfactory ports: o1 in, o2/o3 out)."""
    o1, o2, o3 = ports
    a = 10 ** (-loss_db / 20) / np.sqrt(2)

    def fn(wl, **_):
        one = np.ones_like(wl)
        return reciprocal({(o1, o2): a * one, (o1, o3): a * one})
    return Model(fn, tuple(ports))


def mmi2x2(loss_db: float = 0.0, ports=("o1", "o2", "o3", "o4")) -> Model:
    """An ideal 2 x 2 MMI coupler (50:50, 90 degrees between outputs)."""
    return directional_coupler(0.5, loss_db, ports)


def terminator(port: str = "o1") -> Model:
    """Absorbs everything at ``port``."""
    return Model(_no_optics, (port,))


# --------------------------------------------------------------------------- #
# Active elements.
# --------------------------------------------------------------------------- #
def heater(P_pi: float = 0.02, resistance: float = 1000.0, length: float = 0.0,
           neff: float = 1.9, ng: float = 2.3, wl0: float = 1.55, loss_db_cm: float = 0.0,
           tau: Optional[float] = 10e-6, ports=("o1", "o2"),
           terminals: Dict[str, Sequence[str]] = None) -> Heater:
    """A thermo-optic phase shifter: phase ``pi * P / P_pi`` on top of
    ``length`` um of waveguide. ``terminals`` maps the two terminal names to
    their electrical ports (default ``{"a": ("a",), "b": ("b",)}``)."""
    terminals = terminals or {"a": ("a",), "b": ("b",)}
    (ta, pa), (tb, pb) = terminals.items()

    def fn(wl, P=0.0, **_):
        t = _propagation(wl, length, neff, ng, wl0, loss_db_cm) * \
            np.exp(-1j * np.pi * float(P) / P_pi * wl0 / wl)
        return reciprocal({(ports[0], ports[1]): t})
    return Heater(fn, tuple(ports), {ta: tuple(pa), tb: tuple(pb)}, terminals=(ta, tb),
                  resistance=resistance, tau=tau)


def photodetector(responsivity: float = 1.0, dark_current: float = 1e-9,
                  bandwidth: Optional[float] = None, port: str = "o1",
                  anode: Sequence[str] = ("anode",),
                  cathode: Sequence[str] = ("cathode",)) -> Photodetector:
    """A photodetector absorbing at ``port``; ``anode`` / ``cathode`` are its
    electrical ports."""
    return Photodetector(_no_optics, (port,), {"anode": tuple(anode), "cathode": tuple(cathode)},
                         port=port, terminals=("anode", "cathode"),
                         responsivity=responsivity, dark_current=dark_current,
                         bandwidth=bandwidth)


def resistor(resistance: float = 50.0, a: Sequence[str] = ("e1",),
             b: Sequence[str] = ("e2",)) -> Resistor:
    return Resistor(_no_optics, (), {"a": tuple(a), "b": tuple(b)},
                    terminals=("a", "b"), resistance=resistance)


def from_lnoi(m, length: float = 0.0, bandwidth: Optional[float] = None,
              loss_db_cm: float = 0.0) -> Modulator:
    """A :class:`Modulator` from a :func:`photonfdtd.lnoi.model` result.

    ``length`` (um) is the optical path between the cell's input and output
    ports, for the common phase (``n_eff * length``) that the LN models
    leave out; it matters when light from one cell interferes with light
    that took another path. Away from the design wavelength the phases scale
    as 1/λ with the coupling and ``d n_eff / dV`` held (and no group-index
    dispersion).
    """
    from .. import lnoi
    if isinstance(m, lnoi.PhaseShifterModel):
        n0 = m.n_eff or 0.0

        def fn(wl, V=None, **_):
            lam = wl * 1e-6
            t = np.exp(-1j * (2 * np.pi / lam * n0 * length * 1e-6) - 1j * m.phase(V or {}, lam))
            t = t * 10 ** (-loss_db_cm * length * 1e-4 / 20)
            return reciprocal({("o1", "o2"): t})
        return Modulator(fn, ("o1", "o2"), {n: (f"{n}_w", f"{n}_e") for n in ("sig", "gnd")},
                         bandwidth=bandwidth)
    n0 = 0.5 * (m.info["n_even"] + m.info["n_odd"])
    names = list(m.electrode_names)

    def fn(wl, V=None, **_):
        V = {k: float(v) for k, v in (V or {}).items() if k in names}
        out = {}
        T = np.stack([m.transfer(V, wavelength=w * 1e-6) for w in wl])          # (nw, 2, 2)
        common = np.exp(-1j * 2 * np.pi / (wl * 1e-6) * n0 * length * 1e-6) * \
            10 ** (-loss_db_cm * length * 1e-4 / 20)
        T = T * common[:, None, None]
        # rows: outputs (o4 lower, o3 upper); columns: inputs (o1 lower, o2 upper)
        for i, pin in enumerate(("o1", "o2")):
            for o, pout in enumerate(("o4", "o3")):
                out[(pin, pout)] = T[:, o, i]
        return reciprocal(out)
    return Modulator(fn, ("o1", "o2", "o3", "o4"),
                     {n: (f"{n}_w", f"{n}_e") for n in names}, bandwidth=bandwidth)
