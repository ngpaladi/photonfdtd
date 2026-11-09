"""Component files: a photonic part in one plain-text JSON file, like a KiCad
library part (footprint, symbol and simulation model together).

A component file (``*.pcomp.json``) holds:

``layout``
    The fixed geometry: polygons per GDS layer, in micrometres.
``ports``
    Every port: name, optical or electrical, position, width, orientation,
    layer.
``conductors``
    The electrical pins: which electrical ports are one piece of metal (an
    electrode's two ends, a pad's sides), by name.
``model``
    How the part behaves in a circuit, in up to three forms:

    - ``physics``: solved parameters for a known kind (an LN switch's
      ``d n_eff / dV`` and coupling, a detector's responsivity, a heater's
      ``P_pi``, a waveguide's indices, or a whole sub-circuit of such parts);
    - ``sparams``: S-matrices sampled over wavelength and drive voltages
      (from FDTD, measurement, or sampling the physics), interpolated;
    - ``recipe``: what the physics was solved from (stack, cross-section,
      mesh), so it can be re-solved at another wavelength or mesh.
``generator``
    Optionally, how to rebuild the part with other parameters: a function in
    an installed module (photonfdtd, gdsfactory), or Python source carried in
    the file, which only runs when the loader is told to trust it.

:func:`export_component` writes one from a gdsfactory component (a single
cell, or a sub-layout of several, which becomes a ``circuit`` model);
:func:`load_component` reads one back: ``.cell()`` is a placeable gdsfactory
component and ``.model()`` its circuit model. Cells built from a component
file carry ``info["pcomp"]``, so :meth:`Chip.from_gdsfactory` finds their
models by itself.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from . import models as cm

__all__ = ["FORMAT", "VERSION", "SUFFIX", "export_component", "load_component",
           "ComponentFile", "sample_sparams", "TableModel", "TRUSTED_MODULES"]

FORMAT = "photonfdtd-component"
VERSION = 1
SUFFIX = ".pcomp.json"
#: Modules whose functions a generator may name without ``trust=True``.
TRUSTED_MODULES = ("photonfdtd.", "gdsfactory.")

_REGISTRY: Dict[str, "ComponentFile"] = {}


# --------------------------------------------------------------------------- #
# Writing.
# --------------------------------------------------------------------------- #
def _layout(component) -> dict:
    dbu = component.kcl.dbu
    out = {}
    for layer, polys in component.get_polygons(by="tuple").items():
        key = f"{layer[0]}/{layer[1]}"
        for p in polys:
            simple = p.to_simple_polygon()   # holes joined by cut lines
            pts = [[round(q.x * dbu, 4), round(q.y * dbu, 4)] for q in simple.each_point()]
            out.setdefault(key, []).append(pts)
    return out


def _ports(component) -> list:
    out = []
    for p in component.ports:
        li = p.layer_info
        out.append(dict(name=p.name, type=p.port_type or "optical",
                        center=[round(float(p.center[0]), 4), round(float(p.center[1]), 4)],
                        width=round(float(p.width), 4), orientation=float(p.orientation),
                        layer=[int(li.layer), int(li.datatype)]))
    return out


def _scalar_settings(cell) -> dict:
    out = {}
    for k, v in dict(cell.settings).items():
        if isinstance(v, (bool, int, float, str)) and not (isinstance(v, str) and k == "layers"):
            out[k] = v
    return out


def _generator(cell, info) -> Optional[dict]:
    fname = getattr(cell, "function_name", None)
    if not fname:
        return None
    if "lnoi" in info:
        module = "photonfdtd.lnoi"
    elif "photodetector" in info:
        module = "photonfdtd.circuit.cells"
    else:
        import gdsfactory.components as gc
        if not hasattr(gc, fname):
            return None
        module = getattr(getattr(gc, fname), "__module__", "gdsfactory.components")
    return dict(module=module, function=fname, kwargs=_scalar_settings(cell))


def _physics(cell, info, stack, cell_size, waveguide) -> tuple:
    """``(physics, recipe)`` for a single cell of a kind photonfdtd models."""
    fname = getattr(cell, "function_name", "") or ""
    optical = [p.name for p in cell.ports if p.port_type == "optical"]
    elec = [p.name for p in cell.ports if p.port_type != "optical"]
    if "lnoi" in info:
        from .. import lnoi
        rec = dict(info["lnoi"])
        st = stack or (lnoi.Z_CUT if rec["kind"] == "dbeta_coupler_switch" else lnoi.X_CUT)
        m = lnoi.model(cell, stack=st, cell_size=cell_size)
        o_in, o_out = ("o1", "o2") if len(optical) == 2 else ("o1", "o4")
        length = abs(cell.ports[o_out].center[0] - cell.ports[o_in].center[0])
        physics = dict(kind="lnoi", data=lnoi._to_json(m), length=length,
                       loss_db_cm=(waveguide or {}).get("loss_db_cm", 0.0))
        recipe = dict(kind="lnoi", rec=rec, stack=dict(st.__dict__), cell_size=cell_size)
        return physics, recipe
    if "photodetector" in info:
        d = dict(info["photodetector"])
        return dict(kind="photodetector", data=dict(
            responsivity=d.get("responsivity", 1.0), dark_current=d.get("dark_current", 0.0),
            bandwidth=d.get("bandwidth"), port=optical[0],
            anode=list(d.get("anode", ["anode"])), cathode=list(d.get("cathode", ["cathode"])))), None
    wg = dict(neff=1.9, ng=2.3, wl0=1.55, loss_db_cm=0.0, **(waveguide or {}))
    if fname.startswith("straight_heater_metal"):
        R = float(info.get("resistance") or 0) or 1000.0
        return dict(kind="heater", data=dict(
            P_pi=0.02, resistance=R, length=float(info.get("length", 0.0)), **wg,
            ports=optical[:2], terminals={"a": [p for p in elec if p.startswith("l_")],
                                          "b": [p for p in elec if p.startswith("r_")]})), None
    if fname in ("mmi1x2", "mmi2x2"):
        return dict(kind=fname, data={}), None
    if len(optical) == 2 and not elec:
        L = info.get("length")
        if L is None:
            a, b = (cell.ports[n].center for n in optical)
            L = float(np.hypot(a[0] - b[0], a[1] - b[1]))
        return dict(kind="waveguide", data=dict(length=float(L), ports=optical, **wg)), None
    if not optical:
        return dict(kind="conductor", data=dict(ports=elec)), None
    return None, None


def _conductors_of(model: cm.Model) -> dict:
    return {k: list(v) for k, v in model.conductors.items()}


def sample_sparams(model: cm.Model, wavelengths, drives: Optional[Mapping[str, Sequence[float]]] = None,
                   settings: Optional[Mapping[str, Any]] = None) -> dict:
    """Tabulate ``model``'s S-matrix over ``wavelengths`` (um) and a grid of
    conductor voltages ``drives`` (``{conductor: values}``), for the
    ``sparams`` section of a component file."""
    wl = np.asarray(wavelengths, float)
    drives = {k: [float(x) for x in v] for k, v in (drives or {}).items()}
    names = list(drives)
    shape = tuple(len(drives[n]) for n in names)
    table: Dict[str, np.ndarray] = {}
    for idx in np.ndindex(*shape) if shape else [()]:
        V = {n: drives[n][i] for n, i in zip(names, idx)}
        extra = dict(settings or {})
        if names:
            extra["V"] = V
        sd = model.sdict(wl, **extra)
        for (a, b), v in sd.items():
            arr = table.setdefault(f"{a},{b}", np.zeros(shape + (len(wl),), complex))
            arr[idx] = np.broadcast_to(np.asarray(v, complex), (len(wl),))
    return dict(wavelength=wl.tolist(), drives=drives, ports=list(model.ports),
                S={k: {"re": v.real.tolist(), "im": v.imag.tolist()} for k, v in table.items()})


def export_component(component, path=None, *, name: Optional[str] = None,
                     description: str = "", author: str = "", library: str = "",
                     model: Any = "auto", sparams: Optional[dict] = None,
                     generator: Any = "auto", generator_source: Optional[str] = None,
                     stack=None, cell_size: float = 0.05,
                     waveguide: Optional[Mapping[str, float]] = None) -> dict:
    """Write ``component`` as a component file (to ``path``, if given) and
    return its dict.

    ``model="auto"`` derives the physics: a single photonfdtd/gdsfactory cell
    of a known kind gets its own (an LN cell is solved, or read from the
    cache); a component holding several instances becomes a ``circuit`` of
    its parts' physics, with the external ports and pads it exposes. Pass
    ``model=None`` to leave it out. ``sparams`` is a table from
    :func:`sample_sparams` (or measured data in the same form).
    ``generator="auto"`` records the cell function when it lives in an
    installed module; ``generator_source`` instead carries Python source
    defining ``generator["function"]`` (run only with ``trust=True`` on
    load). ``stack``, ``cell_size`` and ``waveguide`` go to the physics.
    """
    from .. import __version__
    info = dict(component.info) if component.info else {}
    name = name or getattr(component, "function_name", None) or component.name
    data: Dict[str, Any] = dict(
        format=FORMAT, version=VERSION, name=name, description=description, author=author,
        library=library, created=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        photonfdtd_version=__version__, layout=dict(unit="um", polygons=_layout(component)),
        ports=_ports(component))
    physics = recipe = None
    conductors: dict = {}
    if model == "auto":
        if len(list(component.insts)) and not info.get("lnoi") and not info.get("photodetector") \
                and not getattr(component, "function_name", None):
            physics, conductors = _circuit_physics(component, stack, cell_size, waveguide)
        else:
            physics, recipe = _physics(component, info, stack, cell_size, waveguide)
            if physics is not None:
                conductors = _conductors_of(_model_from_physics(physics))
    elif isinstance(model, dict):
        physics = model
        conductors = _conductors_of(_model_from_physics(physics))
    data["conductors"] = conductors
    data["model"] = {k: v for k, v in dict(physics=physics, sparams=sparams, recipe=recipe).items()
                     if v is not None}
    if generator == "auto":
        generator = None if generator_source else _generator(component, info)
    if generator_source:
        generator = dict(source=generator_source, function=(generator or {}).get("function", name),
                         kwargs=(generator or {}).get("kwargs", {}))
    if generator:
        data["generator"] = generator
    data["info"] = json.loads(json.dumps({k: v for k, v in info.items() if k != "pcomp"},
                                         default=str))
    if path is not None:
        p = Path(path)
        if not p.name.endswith(SUFFIX):
            p = p.with_name(p.name + SUFFIX)
        p.write_text(json.dumps(data, indent=1))
    return data


def _circuit_physics(component, stack, cell_size, waveguide):
    """A sub-layout as a ``circuit``: each instance's physics, the nets, and
    the ports the sub-layout exposes (its own ports, else its open optical
    ports and the open electrical ports of each net)."""
    from .chip import Chip
    chip = Chip.from_gdsfactory(component, waveguide=waveguide, stack=stack, cell_size=cell_size)
    instances = {}
    for inst, cell in ((n, i.cell) for n, i in chip.instances.items()):
        info = dict(cell.info) if cell.info else {}
        if "pcomp" in info:
            ph = _registry_get(info["pcomp"]).data["model"].get("physics")
        else:
            ph, _ = _physics(cell, info, stack, cell_size, waveguide)
        if ph is None:
            raise ValueError(f"instance {inst!r} has no physics to export; "
                             "export with model=None and an sparams table instead")
        instances[inst] = ph
    net = chip.network
    optical_nets = [[list(net.ports[i]), list(net.ports[j])]
                    for i in range(len(net.ports)) for j in range(i + 1, len(net.ports))
                    if net.C[i, j]]
    # Electrical connectivity as conductor membership per net.
    nets = {}
    for (inst, cond), k in chip._conductor_net.items():
        nets.setdefault(chip.net_names[k], []).append([inst, cond])
    # Optical ports: the component's own ports where it has them (matched by
    # position), else the open ports, named "instance,port".
    top = {p.name: p for p in component.ports if p.port_type == "optical"}
    external = {}
    for name, (i, pn) in net.external.items():
        ip = chip.instances[i].ports[pn]
        match = [tn for tn, tp in top.items()
                 if abs(tp.center[0] - ip.center[0]) < 1e-3 and abs(tp.center[1] - ip.center[1]) < 1e-3]
        if match:
            external[match[0]] = [i, pn]
        elif not top:
            external[name] = [i, pn]
    # Electrical pins: each net that reaches one of the component's own
    # electrical ports, with those ports, driven through one of its
    # conductors.
    terminals, conductors = {}, {}
    for tname, (i, p) in chip.terminals.items():
        k = chip._port_net.get((i, p))
        if k is None:
            continue
        net_name = chip.net_names[k]
        member = next([a, b] for (a, b), kk in chip._conductor_net.items() if kk == k)
        terminals.setdefault(net_name, dict(member=member, ports=[]))["ports"].append(tname)
        conductors.setdefault(net_name, []).append(tname)
    physics = dict(kind="circuit", data=dict(instances=instances, optical_nets=optical_nets,
                                             nets=nets, external=external,
                                             terminals=terminals))
    return physics, conductors


# --------------------------------------------------------------------------- #
# Reading.
# --------------------------------------------------------------------------- #
def _model_from_physics(ph: dict) -> cm.Model:
    kind, d = ph["kind"], ph.get("data", {})
    if kind == "lnoi":
        from .. import lnoi
        return cm.from_lnoi(lnoi._from_json(d), length=ph.get("length", 0.0),
                            bandwidth=ph.get("bandwidth"), loss_db_cm=ph.get("loss_db_cm", 0.0))
    if kind == "photodetector":
        return cm.photodetector(d.get("responsivity", 1.0), d.get("dark_current", 0.0),
                                d.get("bandwidth"), port=d.get("port", "o1"),
                                anode=tuple(d.get("anode", ["anode"])),
                                cathode=tuple(d.get("cathode", ["cathode"])))
    if kind == "heater":
        kw = {k: d[k] for k in ("P_pi", "resistance", "length", "neff", "ng", "wl0", "loss_db_cm")
              if k in d}
        return cm.heater(**kw, ports=tuple(d.get("ports", ("o1", "o2"))),
                         terminals={k: tuple(v) for k, v in d["terminals"].items()})
    if kind == "waveguide":
        return cm.waveguide(d["length"], d.get("neff", 1.9), d.get("ng", 2.3), d.get("wl0", 1.55),
                            d.get("loss_db_cm", 0.0), ports=tuple(d.get("ports", ("o1", "o2"))))
    if kind == "mmi1x2":
        return cm.mmi1x2()
    if kind == "mmi2x2":
        return cm.mmi2x2()
    if kind == "coupler":
        return cm.directional_coupler(d.get("coupling", 0.5), d.get("loss_db", 0.0))
    if kind == "conductor":
        return cm.conductor(d.get("ports", []))
    if kind == "circuit":
        return _circuit_model(d)
    raise ValueError(f"unknown physics kind {kind!r}")


def _circuit_model(d: dict) -> cm.Model:
    from .chip import Chip
    models = {k: _model_from_physics(v) for k, v in d["instances"].items()}
    enets = []
    for members in d["nets"].values():
        # Join the first port of each conductor on the net.
        ports = [(i, models[i].conductors[c][0]) for i, c in members if models[i].conductors.get(c)]
        enets += [(ports[0], q) for q in ports[1:]]
    ext = {k: tuple(v) for k, v in d["external"].items()}
    chip = Chip(models, [tuple(map(tuple, n)) for n in d["optical_nets"]], enets, ext)
    names = list(ext)
    # Each terminal net is a conductor of the composite, driven through one
    # of the conductors on it.
    terms = d.get("terminals", {})

    def fn(wl, V=None, **_):
        drives = {tuple(t["member"]): float((V or {}).get(n, 0.0)) for n, t in terms.items()}
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return chip.s_matrix(wl, drives)
    conductors = {n: tuple(t["ports"]) for n, t in terms.items()}
    nets = list(terms)
    cls = cm.Modulator if nets else cm.Model
    m = cls(fn, tuple(names), conductors)
    m.chip = chip
    return m


class TableModel:
    """Interpolates an ``sparams`` table: linear in wavelength and multilinear
    in the drive voltages (clamped at the table's edges). Axes with a single
    sample are held constant."""

    def __init__(self, table: dict, conductors: Mapping[str, Sequence[str]]):
        from scipy.interpolate import RegularGridInterpolator
        self.wl = np.asarray(table["wavelength"], float)
        self.drives = {k: np.asarray(v, float) for k, v in table.get("drives", {}).items()}
        self.ports = tuple(table["ports"])
        names = list(self.drives)
        self._live = [n for n in names if len(self.drives[n]) > 1]
        self._wl_live = len(self.wl) > 1
        axes = [self.drives[n] for n in self._live] + ([self.wl] if self._wl_live else [])
        take = tuple(slice(None) if n in self._live else 0 for n in names) + \
            ((slice(None),) if self._wl_live else (0,))
        self._interp = {}
        for key, v in table["S"].items():
            vals = (np.asarray(v["re"], float) + 1j * np.asarray(v["im"], float))[take]
            a, b = key.split(",", 1)
            self._interp[(a, b)] = RegularGridInterpolator(
                axes, vals, bounds_error=False, fill_value=None) if axes else complex(vals)
        cond = {k: tuple(v) for k, v in conductors.items() if k in self.drives}
        cls = cm.Modulator if self.drives else cm.Model
        self.model = cls(self._fn, self.ports, cond)

    def _fn(self, wl, V=None, **_):
        wl = np.atleast_1d(np.asarray(wl, float))
        point = [float(np.clip((V or {}).get(n, 0.0), self.drives[n][0], self.drives[n][-1]))
                 for n in self._live]
        if self._wl_live:
            pts = np.array([point + [w] for w in np.clip(wl, self.wl[0], self.wl[-1])])
        else:
            pts = np.array([point] * len(wl))
        out = {}
        for key, f in self._interp.items():
            out[key] = np.full(len(wl), f, complex) if not callable(f) else f(pts)
        return out


@dataclass
class ComponentFile:
    """A loaded component file. ``data`` is the parsed JSON."""
    data: dict
    path: Optional[Path] = None
    trust: bool = False
    _cell: Any = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return self.data["name"]

    @property
    def key(self) -> str:
        blob = json.dumps({k: v for k, v in self.data.items() if k != "created"}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def cell(self):
        """The fixed geometry as a gdsfactory component, with its ports and
        ``info["pcomp"]`` pointing back here."""
        if self._cell is not None:
            return self._cell
        import gdsfactory as gf
        import re
        safe = re.sub(r"[^0-9A-Za-z_]+", "_", self.name).strip("_") or "part"
        c = gf.Component(f"pcomp_{safe}_{self.key[:8]}")
        for key, polys in self.data["layout"]["polygons"].items():
            layer = tuple(int(x) for x in key.split("/"))
            for pts in polys:
                c.add_polygon([tuple(p) for p in pts], layer=layer)
        for p in self.data["ports"]:
            c.add_port(p["name"], center=tuple(p["center"]), width=p["width"],
                       orientation=p["orientation"], layer=tuple(p["layer"]),
                       port_type=p["type"])
        c.info["pcomp"] = dict(name=self.name, key=self.key,
                               path=str(self.path) if self.path else None)
        c.lock()
        self._cell = c
        return c

    def model(self, prefer: str = "physics") -> cm.Model:
        """The circuit model: from the ``physics`` section, else the
        ``sparams`` table (``prefer="sparams"`` reverses that)."""
        m = self.data.get("model", {})
        order = ("physics", "sparams") if prefer == "physics" else ("sparams", "physics")
        for kind in order:
            if kind == "physics" and m.get("physics"):
                return _model_from_physics(m["physics"])
            if kind == "sparams" and m.get("sparams"):
                return TableModel(m["sparams"], self.data.get("conductors", {})).model
        raise ValueError(f"component {self.name!r} has no model")

    def resolve(self, wavelength: Optional[float] = None, cell_size: Optional[float] = None) -> dict:
        """Re-solve the physics from the recipe (at another ``wavelength``
        in um or mesh) and return the new physics section."""
        r = self.data.get("model", {}).get("recipe")
        if not r:
            raise ValueError(f"component {self.name!r} carries no recipe")
        if r["kind"] != "lnoi":
            raise ValueError(f"no solver for recipe kind {r['kind']!r}")
        from .. import lnoi
        st = dict(r["stack"])
        if wavelength is not None:
            st["wavelength"] = float(wavelength)
        m = lnoi._solve_model(r["rec"], lnoi.Stack(**st), cell_size or r["cell_size"])
        old = self.data["model"].get("physics", {})
        return dict(kind="lnoi", data=lnoi._to_json(m), length=old.get("length", 0.0),
                    loss_db_cm=old.get("loss_db_cm", 0.0))

    def regenerate(self, **kwargs):
        """Rebuild the part from its generator with ``kwargs`` changed. A
        generator in an installed photonfdtd/gdsfactory module runs as is;
        source carried in the file runs only when loaded with
        ``trust=True``."""
        g = self.data.get("generator")
        if not g:
            raise ValueError(f"component {self.name!r} has no generator")
        kw = {**g.get("kwargs", {}), **kwargs}
        if "source" in g:
            if not self.trust:
                raise PermissionError(
                    f"component {self.name!r} carries Python source; load it with trust=True "
                    "to run it")
            ns: Dict[str, Any] = {}
            exec(compile(g["source"], f"<{self.name} generator>", "exec"), ns)
            return ns[g["function"]](**kw)
        mod = g["module"]
        if not (mod + ".").startswith(TRUSTED_MODULES) and not self.trust:
            raise PermissionError(f"generator module {mod!r} is not trusted; load with trust=True")
        return getattr(importlib.import_module(mod), g["function"])(**kw)


def load_component(source, trust: bool = False) -> ComponentFile:
    """Read a component file (a path, a JSON string or a dict) and register
    it, so circuits built from its cell find its model."""
    path = None
    if isinstance(source, dict):
        data = source
    elif isinstance(source, (str, Path)) and not str(source).lstrip().startswith("{"):
        path = Path(source)
        data = json.loads(path.read_text())
    else:
        data = json.loads(source)
    if data.get("format") != FORMAT:
        raise ValueError("not a photonfdtd component file")
    if int(data.get("version", 0)) > VERSION:
        raise ValueError(f"component file version {data['version']} is newer than this "
                         f"photonfdtd reads ({VERSION})")
    cf = ComponentFile(data, path, trust)
    known = _REGISTRY.get(cf.key)
    if known is not None:
        # The same part again: keep its built cell (cell names are unique in
        # a layout), and widen trust if asked.
        known.trust = known.trust or trust
        known.path = known.path or path
        return known
    _REGISTRY[cf.key] = cf
    return cf


def _registry_get(rec: Mapping) -> ComponentFile:
    cf = _REGISTRY.get(rec.get("key"))
    if cf is None and rec.get("path"):
        cf = load_component(rec["path"])
    if cf is None:
        raise KeyError(f"component file {rec.get('name')!r} is not loaded")
    return cf


def model_for_cell(info: Mapping) -> cm.Model:
    """The model of a cell built by :meth:`ComponentFile.cell`."""
    return _registry_get(info["pcomp"]).model()
