"""A whole chip: optics, electrodes, heaters and detectors in one circuit."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from . import electrical as el
from .models import (Heater, Model, Modulator, Photodetector, Resistor, conductor,
                     directional_coupler, from_lnoi, heater, mmi1x2, mmi2x2,
                     photodetector, waveguide)
from .solver import Network, SDict, Waves

Port = Tuple[str, str]


@dataclass
class ChipState:
    """A steady-state solution, arrays over ``wavelength`` (um).

    ``power`` / ``amplitude``: light leaving each external optical port (W,
    sqrt(W)). ``detector_power`` / ``detector_current``: per photodetector
    instance. ``voltage``: per net. ``source_current``: current each drive
    delivers (A). ``heater_power``: per heater instance (W). ``waves``: every
    wave in the circuit (one :class:`~photonfdtd.circuit.solver.Waves` per
    wavelength).
    """
    wavelength: np.ndarray
    power: Dict[str, np.ndarray]
    amplitude: Dict[str, np.ndarray]
    detector_power: Dict[str, np.ndarray]
    detector_current: Dict[str, np.ndarray]
    voltage: Dict[str, np.ndarray]
    source_current: Dict[str, np.ndarray]
    heater_power: Dict[str, np.ndarray]
    waves: list = field(default_factory=list, repr=False)


@dataclass
class Transient:
    """A time-domain run at one wavelength: the same quantities as
    :class:`ChipState`, as arrays over ``t`` (s). ``drive`` holds each
    drive's waveform."""
    t: np.ndarray
    wavelength: float
    power: Dict[str, np.ndarray]
    detector_power: Dict[str, np.ndarray]
    detector_current: Dict[str, np.ndarray]
    voltage: Dict[str, np.ndarray]
    source_current: Dict[str, np.ndarray]
    heater_power: Dict[str, np.ndarray]
    drive: Dict[str, np.ndarray]


class Chip:
    """A chip as a circuit of component models.

    Parameters
    ----------
    models
        The model of each instance, by instance name.
    optical_nets, electrical_nets
        Connections, each a pair of ``(inst, port)``.
    external
        The chip's optical ports, ``{name: (inst, port)}``.
    terminals
        Named electrical points (the chip's electrical ports), usable as
        drive keys, ``{name: (inst, port)}``.

    Notes
    -----
    Drives (``drives={key: value}``) hold a net at a voltage: a number, or a
    waveform of time (:mod:`photonfdtd.circuit.electrical`) for
    :meth:`transient`. A key is a terminal name, an instance whose model is a
    single conductor (a pad), ``"inst,port"``, or ``(inst, conductor)``.
    Optical ``inputs={external port: watts}`` inject light (or
    ``amplitudes=`` for coherent inputs with phases).
    """

    def __init__(self, models: Dict[str, Model], optical_nets: Sequence[Tuple[Port, Port]] = (),
                 electrical_nets: Sequence[Tuple[Port, Port]] = (),
                 external: Optional[Dict[str, Port]] = None,
                 terminals: Optional[Dict[str, Port]] = None):
        self.models = dict(models)
        self.instances: Dict[str, Any] = {}
        self.network = Network({k: m.ports for k, m in self.models.items()},
                               list(optical_nets), external or {})
        self.terminals = dict(terminals or {})
        # Electrical nodes: one per (instance, conductor), merged along nets.
        nodes, port_node = [], {}
        for inst, m in self.models.items():
            for cname, ports in m.conductors.items():
                nodes.append((inst, cname))
                for p in ports:
                    port_node[(inst, p)] = len(nodes) - 1
        parent = list(range(len(nodes)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for p, q in electrical_nets:
            if p not in port_node or q not in port_node:
                missing = [x for x in (p, q) if x not in port_node]
                warnings.warn(f"electrical net {p} - {q}: {missing} belong to no conductor "
                              "of their model; ignored", stacklevel=2)
                continue
            parent[find(port_node[p])] = find(port_node[q])
        roots = sorted({find(i) for i in range(len(nodes))})
        rid = {r: k for k, r in enumerate(roots)}
        self._node_net = [rid[find(i)] for i in range(len(nodes))]
        self._conductor_net = {nodes[i]: self._node_net[i] for i in range(len(nodes))}
        self._port_net = {p: self._node_net[i] for p, i in port_node.items()}
        self.n_nets = len(roots)
        # Net names: a terminal, else a pad or other lone piece of metal.
        names = [None] * self.n_nets
        for t, p in self.terminals.items():
            k = self._port_net.get(p)
            if k is not None and names[k] is None:
                names[k] = t
        # Pads first, then any other lone piece of metal (a route segment).
        lone = [(inst, k) for (inst, cname), k in self._conductor_net.items()
                if len(self.models[inst].conductors) == 1 and not self.models[inst].ports]
        for inst, k in sorted(lone, key=lambda x: "pad" not in x[0]):
            if names[k] is None:
                names[k] = inst
        # Otherwise after a conductor on it ("instance.conductor").
        for (inst, cname), k in self._conductor_net.items():
            if names[k] is None and self.models[inst].ports:
                names[k] = f"{inst}.{cname}"
        self.net_names = [n or f"net{k}" for k, n in enumerate(names)]
        self._check_floating()

    # ------------------------------------------------------------------ #
    @classmethod
    def from_gdsfactory(cls, component, models: Optional[Mapping[str, Any]] = None,
                        waveguide: Optional[Mapping[str, float]] = None,
                        open_ports: bool = True, **lnoi_kwargs) -> "Chip":
        """Build the circuit of a gdsfactory component from its netlist.

        Every instance gets a model: from ``models`` (keyed by instance name
        or by cell function name; a :class:`Model` or a callable
        ``f(cell, settings, info) -> Model``), else a built-in one:

        - :mod:`photonfdtd.lnoi` cells: their voltage models (solved once and
          cached);
        - :func:`photonfdtd.circuit.cells.photodetector`: a photodetector;
        - ``straight_heater_metal*``: a heater (``P_pi`` 20 mW and
          ``resistance`` from the cell's info, else 1 kohm; override through
          ``models``);
        - cells with only electrical ports (pads, metal routes, vias): one
          conductor;
        - two-port optical cells (straights, bends, tapers): a waveguide of
          their length, with ``waveguide`` parameters (``neff``, ``ng``,
          ``wl0``, ``loss_db_cm``);
        - ``mmi1x2`` / ``mmi2x2``: ideal splitters.

        Anything else raises, naming the cell, so pass a model for it (for
        example ``models={"coupler": directional_coupler(0.5)}``).

        The component's optical ports are the chip's external ports; with
        ``open_ports`` (the default) every optical port left unconnected in
        the layout is one too, named ``"instance,port"``, so a layout with no
        top-level ports can still take light in and give it out.
        ``chip.instances`` maps each netlist name to its gdsfactory instance.
        ``lnoi_kwargs`` (``stack``, ``cell_size``, ``cache``,
        ``bandwidth``) go to the LN models.
        """
        from gdsfactory.get_netlist import SmartNamer, function_namer

        class _Recorder:
            def __init__(self):
                self.inner = SmartNamer(function_namer)
                self.cells = {}
                self.instances = {}

            def __call__(self, inst):
                name = self.inner(inst)
                self.cells[name] = inst.cell
                self.instances[name] = inst
                return name

        rec = _Recorder()
        net = component.get_netlist(instance_namer=rec, on_dangling_port="ignore")
        models = dict(models or {})
        built, types = {}, {}
        for name, inst in net["instances"].items():
            cell = rec.cells[name]
            types[name] = {p.name: p.port_type for p in cell.ports}
            built[name] = _model_for(name, inst, cell, models, dict(waveguide or {}), lnoi_kwargs)

        def split(ref):
            i, p = ref.split(",")
            return i, p

        optical_nets, electrical_nets = [], []
        for n in net["nets"]:
            p, q = split(n["p1"]), split(n["p2"])
            if types[p[0]].get(p[1]) == "optical":
                optical_nets.append((p, q))
            else:
                electrical_nets.append((p, q))
        external, terminals = {}, {}
        for name, ref in net["ports"].items():
            i, p = split(ref)
            (external if types[i].get(p) == "optical" else terminals)[name] = (i, p)
        if open_ports:
            used = {x for pair in optical_nets for x in pair} | set(external.values())
            for i, m in built.items():
                for p in m.ports:
                    if (i, p) not in used:
                        external[f"{i},{p}"] = (i, p)
        chip = cls(built, optical_nets, electrical_nets, external, terminals)
        chip.instances = dict(rec.instances)
        return chip

    # ------------------------------------------------------------------ #
    def net(self, key) -> int:
        """The net a drive key refers to."""
        if isinstance(key, tuple):
            if key in self._conductor_net:
                return self._conductor_net[key]
            if key in self._port_net:
                return self._port_net[key]
            raise KeyError(f"{key} is neither (instance, conductor) nor (instance, port)")
        if key in self.terminals:
            return self._port_net[self.terminals[key]]
        if key in self.net_names:
            return self.net_names.index(key)
        if key in self.models:
            cs = self.models[key].conductors
            if len(cs) == 1:
                return self._conductor_net[(key, next(iter(cs)))]
            raise KeyError(f"instance {key!r} has several conductors {list(cs)}; "
                           "use (instance, conductor)")
        if isinstance(key, str) and "," in key:
            return self.net(tuple(key.split(",", 1)))
        raise KeyError(f"unknown drive {key!r}")

    def _check_floating(self):
        """Warn later, per solve, only about nets that matter; record which
        nets carry electrodes or heaters."""
        self._active_nets = set()
        for inst, m in self.models.items():
            if isinstance(m, (Modulator, Heater)):
                for c in m.conductors:
                    self._active_nets.add(self._conductor_net[(inst, c)])

    # ------------------------------------------------------------------ #
    def _electrical(self, fixed: Dict[int, float], det_I: Dict[str, float]):
        res, cur = [], {}
        for inst, m in self.models.items():
            if isinstance(m, (Heater, Resistor)):
                a, b = (self._conductor_net[(inst, t)] for t in m.terminals)
                res.append((a, b, m.resistance))
            elif isinstance(m, Photodetector):
                a, c = (self._conductor_net[(inst, t)] for t in m.terminals)
                i = det_I.get(inst, 0.0)
                cur[a] = cur.get(a, 0.0) + i
                cur[c] = cur.get(c, 0.0) - i
        return el.nodal(self.n_nets, fixed, res, cur)

    def _targets(self, V: np.ndarray):
        """Per-instance drive settings at net voltages ``V``."""
        out = {}
        for inst, m in self.models.items():
            if isinstance(m, Modulator):
                out[inst] = {"V": {c: float(V[self._conductor_net[(inst, c)]]) for c in m.conductors}}
            elif isinstance(m, Heater):
                a, b = (V[self._conductor_net[(inst, t)]] for t in m.terminals)
                out[inst] = {"P": float((a - b) ** 2 / m.resistance)}
        return out

    def _sdicts(self, wl, settings):
        return {inst: m.sdict(wl, **settings.get(inst, {})) for inst, m in self.models.items()}

    def _amplitudes(self, inputs, amplitudes):
        amp = {k: np.sqrt(np.asarray(v, float)) for k, v in (inputs or {}).items()}
        amp.update({k: np.asarray(v, complex) for k, v in (amplitudes or {}).items()})
        return amp

    def _detect(self, waves: Waves, w: int = 0):
        P, I = {}, {}
        for inst, m in self.models.items():
            if isinstance(m, Photodetector):
                p = float(abs(waves.a[w, waves.index(inst, m.port)]) ** 2)
                P[inst] = p
                I[inst] = m.responsivity * p + m.dark_current
        return P, I

    def _fixed(self, drives, t=None):
        fixed = {}
        for key, val in (drives or {}).items():
            k = self.net(key)
            v = el.value_at(val, 0.0 if t is None else t) if callable(val) else float(val)
            if k in fixed and fixed[k] != v:
                raise ValueError(f"drives {key!r} and another key set net "
                                 f"{self.net_names[k]} to different voltages")
            fixed[k] = v
        floating = self._active_nets - set(fixed)
        return fixed, floating

    def _warn_floating(self, floating, fixed, res_nets):
        # A net is held only through resistors to a driven net; otherwise it
        # floats (and sits at 0 V).
        loose = [self.net_names[k] for k in floating if k not in res_nets]
        if loose:
            warnings.warn(f"nets {loose} carry electrodes or heaters but no drive; "
                          "they float and are taken as 0 V", stacklevel=3)

    def _resistive_nets(self, fixed):
        """Nets connected to a driven net through resistors."""
        adj = {}
        for inst, m in self.models.items():
            if isinstance(m, (Heater, Resistor)):
                a, b = (self._conductor_net[(inst, t)] for t in m.terminals)
                adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
        seen, stack = set(fixed), list(fixed)
        while stack:
            for y in adj.get(stack.pop(), ()):
                if y not in seen:
                    seen.add(y); stack.append(y)
        return seen

    # ------------------------------------------------------------------ #
    def solve(self, wavelength=1.55, drives: Optional[Mapping] = None,
              inputs: Optional[Mapping[str, float]] = None,
              amplitudes: Optional[Mapping[str, complex]] = None,
              max_iter: int = 100, tol: float = 1e-9) -> ChipState:
        """Steady state at each ``wavelength`` (um) with DC ``drives`` and
        optical ``inputs`` (W). Photocurrents feed back into the electrical
        network (through load resistors, say); the two are iterated to
        self-consistency."""
        wl = np.atleast_1d(np.asarray(wavelength, float))
        fixed, floating = self._fixed(drives)
        self._warn_floating(floating, fixed, self._resistive_nets(fixed))
        amp = self._amplitudes(inputs, amplitudes)
        ext = list(self.network.external)
        res = dict(power={k: [] for k in ext}, amplitude={k: [] for k in ext},
                   detector_power={}, detector_current={}, voltage={k: [] for k in self.net_names},
                   source_current={k: [] for k in (drives or {})}, heater_power={})
        all_waves = []
        for w in wl:
            amp_w = {k: (v if np.ndim(v) == 0 else v[list(wl).index(w)]) for k, v in amp.items()}
            det_I = {}
            for it in range(max_iter):
                V, I_src = self._electrical(fixed, det_I)
                settings = self._targets(V)
                S = self.network.assemble(self._sdicts(np.array([w]), settings), 1)
                waves = self.network.solve(S, amp_w)
                P_det, new_I = self._detect(waves)
                if all(abs(new_I[k] - det_I.get(k, 0.0)) <= tol * max(1e-12, abs(new_I[k]))
                       for k in new_I):
                    det_I = new_I
                    break
                det_I = new_I
            else:
                warnings.warn("optoelectronic feedback did not converge", stacklevel=2)
            V, I_src = self._electrical(fixed, det_I)
            all_waves.append(waves)
            for k, (i, p) in self.network.external.items():
                b = waves.outgoing(i, p)[0]
                res["amplitude"][k].append(b)
                res["power"][k].append(abs(b) ** 2)
            for inst in P_det:
                res["detector_power"].setdefault(inst, []).append(P_det[inst])
                res["detector_current"].setdefault(inst, []).append(det_I[inst])
            for k, name in enumerate(self.net_names):
                res["voltage"][name].append(V[k])
            for key in res["source_current"]:
                res["source_current"][key].append(I_src[self.net(key)])
            for inst, s in settings.items():
                if "P" in s:
                    res["heater_power"].setdefault(inst, []).append(s["P"])
        arr = {k: {n: np.asarray(v) for n, v in d.items()} for k, d in res.items()}
        return ChipState(wavelength=wl, waves=all_waves, **arr)

    def s_matrix(self, wavelength=1.55, drives: Optional[Mapping] = None) -> SDict:
        """The chip's optical S-matrix between its external ports at DC
        ``drives`` (no photocurrent feedback)."""
        wl = np.atleast_1d(np.asarray(wavelength, float))
        fixed, floating = self._fixed(drives)
        self._warn_floating(floating, fixed, self._resistive_nets(fixed))
        V, _ = self._electrical(fixed, {})
        S = self.network.assemble(self._sdicts(wl, self._targets(V)), len(wl))
        return self.network.s_matrix(S)

    def transient(self, t, drives: Mapping, inputs: Optional[Mapping[str, float]] = None,
                  amplitudes: Optional[Mapping[str, complex]] = None,
                  wavelength: float = 1.55) -> Transient:
        """Step the chip through times ``t`` (s) with time-varying ``drives``.

        Quasi-static: at every time the optics are solved in steady state for
        the drives at that time, after first-order low-passes for each
        modulator's ``bandwidth``, each heater's ``tau`` and each detector's
        ``bandwidth``. Valid while the drives change slowly next to the time
        light takes to cross the chip (about 10 ps per mm of LN waveguide), so
        a GHz drive on a centimetre circuit is at the limit. Starts from the
        steady state at ``t[0]``.
        """
        t = np.asarray(t, float)
        amp = self._amplitudes(inputs, amplitudes)
        fixed0, floating = self._fixed(drives, t[0])
        self._warn_floating(floating, fixed0, self._resistive_nets(fixed0))
        st = self.solve(wavelength, {k: el.value_at(v, t[0]) for k, v in drives.items()},
                        amplitudes={k: v for k, v in amp.items()})
        det_I = {k: float(v[0]) for k, v in st.detector_current.items()}
        V0, _ = self._electrical(fixed0, det_I)
        filt = self._targets(V0)
        ext = list(self.network.external)
        out = dict(power={k: [] for k in ext}, detector_power={}, detector_current={},
                   voltage={k: [] for k in self.net_names},
                   source_current={k: [] for k in drives}, heater_power={})
        wl = np.array([wavelength])
        for n, tk in enumerate(t):
            dt = t[n] - t[n - 1] if n else 0.0
            fixed, _ = self._fixed(drives, tk)
            V, I_src = self._electrical(fixed, det_I)
            target = self._targets(V)
            for inst, m in self.models.items():
                if inst not in target:
                    continue
                tau = el.tau_of(m.bandwidth) if isinstance(m, Modulator) else m.tau
                a = el.lowpass_factor(dt, tau) if n else 1.0
                if "V" in target[inst]:
                    filt[inst]["V"] = {c: filt[inst]["V"][c] + a * (v - filt[inst]["V"][c])
                                       for c, v in target[inst]["V"].items()}
                else:
                    filt[inst]["P"] += a * (target[inst]["P"] - filt[inst]["P"])
            S = self.network.assemble(self._sdicts(wl, filt), 1)
            waves = self.network.solve(S, amp)
            P_det, raw_I = self._detect(waves)
            for inst, i in raw_I.items():
                m = self.models[inst]
                a = el.lowpass_factor(dt, el.tau_of(m.bandwidth)) if n else 1.0
                det_I[inst] = det_I.get(inst, i) + a * (i - det_I.get(inst, i))
            for k, (i, p) in self.network.external.items():
                out["power"][k].append(abs(waves.outgoing(i, p)[0]) ** 2)
            for inst in P_det:
                out["detector_power"].setdefault(inst, []).append(P_det[inst])
                out["detector_current"].setdefault(inst, []).append(det_I[inst])
            for k, name in enumerate(self.net_names):
                out["voltage"][name].append(V[k])
            for key in drives:
                out["source_current"][key].append(I_src[self.net(key)])
            for inst, s in filt.items():
                if "P" in s:
                    out["heater_power"].setdefault(inst, []).append(s["P"])
        arr = {k: {n: np.asarray(v) for n, v in d.items()} for k, d in out.items()}
        drive = {k: np.array([el.value_at(v, x) for x in t]) for k, v in drives.items()}
        return Transient(t=t, wavelength=wavelength, drive=drive, **arr)


# --------------------------------------------------------------------------- #
# Built-in models for gdsfactory cells.
# --------------------------------------------------------------------------- #
_WAVEGUIDES = ("straight", "bend_euler", "bend_circular", "bend_s", "taper",
               "bend_euler_s", "straight_all_angle", "bend_euler_all_angle")


def _port_distance(cell, a, b):
    pa, pb = cell.ports[a].center, cell.ports[b].center
    return float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))


def _model_for(name, inst, cell, models, wg, lnoi_kwargs) -> Model:
    fname = inst["component"]
    settings, info = inst.get("settings", {}), inst.get("info", {})
    user = models.get(name, models.get(fname))
    if user is not None:
        return user if isinstance(user, Model) else user(cell, settings, info)
    optical = [p.name for p in cell.ports if p.port_type == "optical"]
    elec = [p.name for p in cell.ports if p.port_type != "optical"]
    if "lnoi" in info:
        from .. import lnoi
        kw = dict(lnoi_kwargs)
        bw = kw.pop("bandwidth", None)
        m = lnoi.model(cell, **kw)
        o_in, o_out = ("o1", "o2") if len(optical) == 2 else ("o1", "o4")
        return from_lnoi(m, length=abs(cell.ports[o_out].center[0] - cell.ports[o_in].center[0]),
                         bandwidth=bw, loss_db_cm=wg.get("loss_db_cm", 0.0))
    if "photodetector" in info:
        d = dict(info["photodetector"])
        return photodetector(responsivity=d.get("responsivity", 1.0),
                             dark_current=d.get("dark_current", 0.0),
                             bandwidth=d.get("bandwidth"), port=optical[0],
                             anode=tuple(d.get("anode", ("anode",))),
                             cathode=tuple(d.get("cathode", ("cathode",))))
    if fname.startswith("straight_heater_metal"):
        R = float(info.get("resistance") or 0) or 1000.0
        left = tuple(p for p in elec if p.startswith("l_"))
        right = tuple(p for p in elec if p.startswith("r_"))
        return heater(P_pi=0.02, resistance=R, length=float(info.get("length", 0.0)),
                      terminals={"a": left, "b": right}, ports=tuple(optical[:2]),
                      **{k: v for k, v in wg.items() if k in ("neff", "ng", "wl0", "loss_db_cm")})
    if not optical:
        return conductor(elec)
    if fname == "mmi1x2" and len(optical) == 3:
        return mmi1x2()
    if fname == "mmi2x2" and len(optical) == 4:
        return mmi2x2()
    if len(optical) == 2 and not elec and (fname in _WAVEGUIDES or "length" in info):
        L = info.get("length", settings.get("length"))
        L = float(L) if L is not None else _port_distance(cell, *optical)
        return waveguide(L, ports=tuple(optical), **wg)
    raise ValueError(
        f"no model for instance {name!r} (cell {fname!r}, optical ports {optical}, "
        f"electrical {elec}); pass one in models={{{fname!r}: ...}}")
