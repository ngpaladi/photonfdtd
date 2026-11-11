"""Thin-film lithium niobate (LNOI) switch cells for gdsfactory, with models.

The cells are ordinary gdsfactory components (``gf.cell``), so they place,
route and export like any other. Each one also records its electroded
cross-section in ``component.info["lnoi"]``. :func:`model` reads that record
and gives the component's voltage response, computed by photonfdtd:

- the electrostatic field of the electrodes on the waveguide cross-section and
  the Pockels index change it causes (:mod:`photonfdtd.active`);
- the mode indices with and without the drive (the mode solver), and from them
  the coupling between waveguides and each waveguide's ``d n_eff / dV``;
- coupled-mode theory along the device, giving the 2 x 2 transfer matrix at
  any drive voltage.

That runs in seconds for millimetre-long devices, which full FDTD can't reach.
:func:`to_simulation` builds the full 3-D FDTD model of a (short) component
instead.

The cells:

- :func:`phase_shifter`: an x-cut rib between a signal and a ground
  electrode.
- :func:`mzi_switch`: a 2 x 2 Mach-Zehnder switch, two directional couplers
  and push-pull ground-signal-ground electrodes on the arms (x-cut).
- :func:`dbeta_coupler_switch`: a directional coupler with electrodes on top
  of the two waveguides (z-cut, TM). The vertical fields under the electrodes
  point in opposite directions in the two waveguides, so the drive detunes
  them; ``sections=2`` or more reverses the detuning from section to section
  (the reversed delta-beta coupler), which reaches both the bar and the cross
  state electrically.

Lengths in the cells are in micrometres (gdsfactory units); the models work in
SI units. Requires gdsfactory (``pip install photonfdtd[gdsfactory]``).

Not modelled: optical absorption by the electrodes (keep the buffer under
top electrodes thick enough), coupling in the fan-in and fan-out bends, and
the electrodes' RF behaviour (velocity matching, microwave loss).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import gdsfactory as gf
except ImportError as exc:  # pragma: no cover
    raise ImportError("photonfdtd.lnoi needs gdsfactory: pip install gdsfactory") from exc

from . import active
from .geometry import Box
from .mode import ModeSolver

UM = 1e-6

__all__ = ["Layers", "LAYERS", "Stack", "rib", "phase_shifter", "mzi_switch",
           "dbeta_coupler_switch", "model", "SwitchModel", "PhaseShifterModel",
           "coupler_length", "layer_map", "to_simulation"]


# --------------------------------------------------------------------------- #
# Layers and the vertical stack.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Layers:
    """GDS layers the cells draw on: the unetched LN ridge and the electrode
    metal. The defaults are gdsfactory's generic ``WG`` and ``MTOP`` layers,
    so the generic pads and metal routing connect to the electrodes."""
    rib: Tuple[int, int] = (1, 0)
    metal: Tuple[int, int] = (49, 0)


LAYERS = Layers()


@dataclass(frozen=True)
class Stack:
    """The vertical stack, in micrometres.

    ``film`` of LN on ``box`` of buried oxide, with ``etch`` removed outside
    the ridges (the slab is ``film - etch`` thick). ``cladding`` of oxide
    covers the film (0: air). Lateral electrodes stand on the slab; top
    electrodes sit ``buffer`` above the ridge, which needs
    ``cladding >= buffer``. ``cut`` is the crystal cut: ``"x"`` puts the optic
    axis in the plane, across the waveguide, and uses the TE mode;
    ``"z"`` puts it vertical and uses the TM mode.
    """
    film: float = 0.6
    etch: float = 0.3
    box: float = 2.0
    cladding: float = 0.0
    buffer: float = 0.5
    metal: float = 0.5
    cut: str = "x"
    wavelength: float = 1.55

    def __post_init__(self):
        if not 0 < self.etch <= self.film:
            raise ValueError("need 0 < etch <= film")
        if self.cut not in ("x", "z"):
            raise ValueError("cut must be 'x' or 'z'")

    @property
    def slab(self) -> float:
        return self.film - self.etch

    @property
    def polarization(self) -> str:
        return "te" if self.cut == "x" else "tm"

    def media(self):
        lam = self.wavelength * UM
        return dict(ln=active.lithium_niobate_eo(lam, cut=self.cut),
                    sio2=active.silica_active(lam), metal=active.metal_active("Au"))


#: Stacks the cells are drawn for by default.
X_CUT = Stack(cut="x")
# z-cut uses the TM mode, which leaks into the slab's TE modes in a shallow
# etch, so its ridge is etched deeper.
Z_CUT = Stack(cut="z", etch=0.45, cladding=1.5, buffer=0.5)


def rib(width: float = 1.2, radius: float = 80.0, layers: Layers = LAYERS):
    """Cross-section of the LN ridge waveguide (``width`` and bend ``radius``
    in um)."""
    return gf.cross_section.cross_section(width=width, layer=layers.rib,
                                          radius=radius, radius_min=radius / 2)


# --------------------------------------------------------------------------- #
# Drawing helpers.
# --------------------------------------------------------------------------- #
def _strip(c, x0, x1, y, width, layer):
    c.add_polygon([(x0, y - width / 2), (x1, y - width / 2),
                   (x1, y + width / 2), (x0, y + width / 2)], layer=layer)


def _electrical_ports(c, name, x0, x1, y, width, layer):
    """Ports at both ends of an electrode strip (``<name>_w`` and
    ``<name>_e``)."""
    c.add_port(f"{name}_w", center=(x0, y), width=width, orientation=180,
               layer=layer, port_type="electrical")
    c.add_port(f"{name}_e", center=(x1, y), width=width, orientation=0,
               layer=layer, port_type="electrical")


def _pair(c, x0, length, gap, xs, width, dy, bend_length):
    """Two parallel waveguides from ``x0`` over ``length``, ``gap`` apart
    (edge to edge), with S-bends opening the ends to ``dy`` further out on
    each side. Returns the outer ports ``(lower_w, upper_w, upper_e,
    lower_e)``."""
    yc = (width + gap) / 2
    out = {}
    for side, sgn in (("lower", -1), ("upper", +1)):
        s = c << gf.components.straight(length=length, cross_section=xs)
        s.move((x0, sgn * yc))
        if dy > 0:
            bend = gf.components.bend_s(size=(bend_length, dy), cross_section=xs,
                                        allow_min_radius_violation=True)
            east = c << bend
            east.connect("o1", s.ports["o2"], mirror=(sgn < 0))
            west = c << bend
            west.connect("o1", s.ports["o1"], mirror=(sgn > 0))
            out[side] = (west.ports["o2"], east.ports["o2"])
        else:
            out[side] = (s.ports["o1"], s.ports["o2"])
    return out["lower"][0], out["upper"][0], out["upper"][1], out["lower"][1]


def _record(c, kind, **data):
    c.info["lnoi"] = dict(kind=kind, **data)


# --------------------------------------------------------------------------- #
# Cells.
# --------------------------------------------------------------------------- #
@gf.cell
def phase_shifter(length: float = 5000.0, width: float = 1.2, electrode_gap: float = 5.0,
                  signal_width: float = 20.0, ground_width: float = 60.0,
                  radius: float = 80.0, layers: Layers = LAYERS) -> gf.Component:
    """An x-cut phase shifter: a ridge of ``width`` centred in the
    ``electrode_gap`` between a signal electrode (above, +y) and a ground
    electrode (below), both ``length`` long. Optical ports ``o1``/``o2``,
    electrical ``sig_w``/``sig_e``/``gnd_w``/``gnd_e``."""
    c = gf.Component()
    xs = rib(width, radius, layers)
    s = c << gf.components.straight(length=length, cross_section=xs)
    c.add_ports(s.ports)
    y_sig = electrode_gap / 2 + signal_width / 2
    y_gnd = -(electrode_gap / 2 + ground_width / 2)
    for name, y, w in (("sig", y_sig, signal_width), ("gnd", y_gnd, ground_width)):
        _strip(c, 0, length, y, w, layers.metal)
        _electrical_ports(c, name, 0, length, y, w, layers.metal)
    _record(c, "phase_shifter", width=width, ribs=[0.0],
            strips=[dict(y=y_sig, width=signal_width, style="lateral"),
                    dict(y=y_gnd, width=ground_width, style="lateral")],
            segments=[dict(x0=0.0, x1=length, electrodes=dict(sig=0, gnd=1))])
    return c


@gf.cell
def mzi_switch(arm_length: float = 5000.0, width: float = 1.2,
               coupler_gap: float = 1.0, coupler_length: float = 276.0,
               electrode_gap: float = 5.0, signal_width: float = 20.0,
               ground_width: float = 60.0, bend_length: float = 150.0,
               radius: float = 80.0, layers: Layers = LAYERS) -> gf.Component:
    """A 2 x 2 Mach-Zehnder switch on x-cut LN.

    Two directional couplers (``coupler_gap`` edge to edge, ``coupler_length``
    long; the default is a 50:50 split for the default stack, see
    :func:`coupler_length`) and two arms of ``arm_length``, each centred in
    one gap of a ground-signal-ground electrode, so a voltage on the signal
    shifts the arms in opposite directions (push-pull). The arms are
    ``signal_width + electrode_gap`` apart, and so are the optical ports
    ``o1`` (lower left), ``o2`` (upper left), ``o3`` (upper right) and
    ``o4`` (lower right). Electrical ports: ``sig_*``, ``gnd_top_*``,
    ``gnd_bot_*``.
    """
    c = gf.Component()
    xs = rib(width, radius, layers)
    sep = signal_width + electrode_gap
    dy = sep / 2 - (width + coupler_gap) / 2
    if dy <= 0:
        raise ValueError("the arms must be further apart than the coupler waveguides")
    a1, a2, a3, a4 = _pair(c, 0.0, coupler_length, coupler_gap, xs, width, dy, bend_length)
    x_arm0 = a3.center[0]
    for p in (a3, a4):
        arm = c << gf.components.straight(length=arm_length, cross_section=xs)
        arm.connect("o1", p)
    x_arm1 = x_arm0 + arm_length
    x_dc2 = x_arm1 + bend_length
    b1, b2, b3, b4 = _pair(c, x_dc2, coupler_length, coupler_gap, xs, width, dy, bend_length)
    for name, p in (("o1", a1), ("o2", a2), ("o3", b3), ("o4", b4)):
        c.add_port(name, port=p)
    y_g = sep / 2 + electrode_gap / 2 + ground_width / 2
    strips = [dict(name="sig", y=0.0, width=signal_width),
              dict(name="gnd_top", y=y_g, width=ground_width),
              dict(name="gnd_bot", y=-y_g, width=ground_width)]
    for s in strips:
        _strip(c, x_arm0, x_arm1, s["y"], s["width"], layers.metal)
        _electrical_ports(c, s["name"], x_arm0, x_arm1, s["y"], s["width"], layers.metal)
    _record(c, "mzi_switch", width=width, ribs=[-sep / 2, sep / 2],
            coupler=dict(gap=coupler_gap, length=coupler_length),
            strips=[dict(y=s["y"], width=s["width"], style="lateral") for s in strips],
            segments=[dict(x0=x_arm0, x1=x_arm1,
                           electrodes=dict(sig=0, gnd_top=1, gnd_bot=2))])
    return c


@gf.cell
def dbeta_coupler_switch(length: float = 3987.0, width: float = 1.2, gap: float = 1.0,
                         sections: int = 1, section_gap: float = 10.0,
                         electrode_width: float = 1.2, port_spacing: float = 20.0,
                         bend_length: float = 150.0, radius: float = 80.0,
                         layers: Layers = LAYERS) -> gf.Component:
    """A delta-beta directional-coupler switch on z-cut LN (TM).

    Two waveguides ``gap`` apart (edge to edge) couple over ``length``; the
    default is three coupling lengths for the default z-cut stack, so the
    light crosses over at 0 V (see :func:`coupler_length`), and the bar state
    needs less detuning than over one coupling length (about 38 V). Electrodes of
    ``electrode_width`` run on top of each waveguide, split into ``sections``
    equal sections ``section_gap`` apart. In section ``k`` the signal
    electrode ``sig{k}`` is over the lower waveguide for even ``k`` and over
    the upper one for odd ``k``, with ground ``gnd{k}`` over the other: one
    voltage on all signals then detunes the sections alternately (reversed
    delta-beta). Driving the sections separately reaches the cross state
    electrically even when the coupling length comes out different from the
    design, which a single section can't. Optical ports ``o1``..``o4`` as in gdsfactory's coupler,
    ``port_spacing`` apart; electrical ``sig{k}_w/_e`` and ``gnd{k}_w/_e``.
    """
    if sections < 1:
        raise ValueError("sections must be >= 1")
    c = gf.Component()
    xs = rib(width, radius, layers)
    dy = port_spacing / 2 - (width + gap) / 2
    ports = _pair(c, 0.0, length, gap, xs, width, max(dy, 0.0), bend_length)
    for name, p in zip(("o1", "o2", "o3", "o4"), ports):
        c.add_port(name, port=p)
    yc = (width + gap) / 2
    seg = (length - (sections - 1) * section_gap) / sections
    if seg <= 0:
        raise ValueError("sections and section_gap leave no electrode length")
    segments = []
    for k in range(sections):
        x0 = k * (seg + section_gap)
        sig_lower = (k % 2 == 0)
        y_sig, y_gnd = (-yc, yc) if sig_lower else (yc, -yc)
        for name, y in ((f"sig{k}", y_sig), (f"gnd{k}", y_gnd)):
            _strip(c, x0, x0 + seg, y, electrode_width, layers.metal)
            _electrical_ports(c, name, x0, x0 + seg, y, electrode_width, layers.metal)
        segments.append(dict(x0=x0, x1=x0 + seg, electrodes={
            f"sig{k}": 0 if sig_lower else 1, f"gnd{k}": 1 if sig_lower else 0}))
    _record(c, "dbeta_coupler_switch", width=width, ribs=[-yc, yc],
            coupler=dict(gap=gap, length=length),
            strips=[dict(y=-yc, width=electrode_width, style="top"),
                    dict(y=yc, width=electrode_width, style="top")],
            segments=segments)
    return c


# --------------------------------------------------------------------------- #
# The cross-section and its solves.
# --------------------------------------------------------------------------- #
def _cross_section(rec, stack: Stack, ribs: Sequence[float], cell_size: float,
                   margin: float = 8.0, with_strips: bool = True):
    """Optical layout, static layout and ModeSolver geometry (SI) of the
    record's cross-section with the ridges at ``ribs`` (um)."""
    m = stack.media()
    w = rec["width"]
    lo, hi = min(ribs) - margin, max(ribs) + margin
    span = hi - lo
    z_top = stack.film + max(stack.cladding, 0.0)
    strips = rec["strips"] if with_strips else []
    if any(s["style"] == "top" for s in strips):
        if stack.cladding < stack.buffer:
            raise ValueError("top electrodes need stack.cladding >= stack.buffer")
        z_top = max(z_top, stack.film + stack.buffer + stack.metal)
    else:
        z_top = max(z_top, stack.slab + (stack.metal if strips else 0.0))
    z_bot = -min(stack.box, 1.5)
    z_hi = z_top + 1.0
    # The ModeSolver window is centred on the origin: shift the layout to it.
    ym, zm = 0.5 * (lo + hi), 0.5 * (z_hi + z_bot)
    B = lambda y0, z0, wy, wz, med: Box(((y0 - ym) * UM, (z0 - zm) * UM),
                                        (wy * UM, wz * UM), med)
    wide = 4 * span
    optics = [B(0, (z_bot - 1) / 2, wide, -z_bot + 1, m["sio2"])]
    if stack.cladding > 0:
        optics.append(B(0, stack.film + stack.cladding / 2 - stack.etch / 2,
                        wide, stack.cladding + stack.etch, m["sio2"]))
    optics.append(B(0, stack.slab / 2, wide, stack.slab, m["ln"]))
    for y in ribs:
        optics.append(B(y, stack.slab + stack.etch / 2, w, stack.etch, m["ln"]))
    metal = []
    for i, s in enumerate(strips):
        # Clip wide electrodes to the window; their far edges hardly matter.
        y0, y1 = max(s["y"] - s["width"] / 2, lo), min(s["y"] + s["width"] / 2, hi)
        if y1 <= y0:
            continue
        z0 = stack.film + stack.buffer if s["style"] == "top" else stack.slab
        metal.append((i, B((y0 + y1) / 2, z0 + stack.metal / 2, y1 - y0, stack.metal,
                           m["metal"])))
    ms_kw = dict(size=(span * UM, (z_hi - z_bot) * UM), cell_size=cell_size * UM, wavelength=stack.wavelength * UM,
                 polarization=stack.polarization)
    return optics, metal, ms_kw


def _modesolver(optics, ms_kw, num_modes=1):
    return ModeSolver(structures=optics, num_modes=num_modes, **ms_kw)


def _slopes(rec, stack, cell_size):
    """``S[r][s]``: d n_eff of ridge ``r`` alone per volt on strip ``s``
    (others grounded), from the electrostatic solve and the mode solver."""
    S = np.zeros((len(rec["ribs"]), len(rec["strips"])))
    for r, y in enumerate(rec["ribs"]):
        optics, metal, kw = _cross_section(rec, stack, [y], cell_size)
        ms = _modesolver(optics, kw)
        electrodes = [active.Electrode(b, name=f"s{i}") for i, b in metal]
        region = active.ActiveRegion(optics + [b for _, b in metal], electrodes=electrodes,
                                     wavelength=stack.wavelength * UM)
        prep = region.prepare(ms)
        for i, _ in metal:
            S[r, i] = prep.dneff(f"s{i}")
    return S


def _supermodes(gap, width, stack, cell_size):
    rec = dict(width=width, ribs=[-(width + gap) / 2, (width + gap) / 2], strips=[])
    optics, _, kw = _cross_section(rec, stack, rec["ribs"], cell_size, with_strips=False)
    n = np.real(_modesolver(optics, kw, num_modes=4).solve().n_eff)[:2]
    if len(n) < 2:
        raise RuntimeError("the coupled waveguides have no second supermode")
    return float(n[0]), float(n[1])


def coupler_length(gap: float = 1.0, width: float = 1.2, stack: Stack = X_CUT,
                   split: float = 1.0, cell_size: float = 0.05) -> float:
    """Length (um) of a directional coupler of two ridges ``gap`` apart that
    moves ``split`` of the power across (1: the full coupling length, 0.5: a
    50:50 coupler), from the supermode splitting. The fan-in and fan-out
    bends add a little coupling that this leaves out."""
    n_even, n_odd = _supermodes(gap, width, stack, cell_size)
    kappa = np.pi * (n_even - n_odd) / (stack.wavelength * UM)
    return float(np.arcsin(np.sqrt(split)) / kappa / UM)


# --------------------------------------------------------------------------- #
# Models.
# --------------------------------------------------------------------------- #
def _coupled(kappa, delta, L):
    """Coupled-mode transfer over ``L`` for amplitudes (lower, upper) with
    coupling ``kappa`` and half-detuning ``delta = (beta_lower - beta_upper)/2``
    (common phase dropped)."""
    s = np.sqrt(kappa ** 2 + delta ** 2 + 0j)
    c, sn = np.cos(s * L), np.sinc(s * L / np.pi) * L   # sin(sL)/s, safe at s=0
    return np.array([[c - 1j * delta * sn, -1j * kappa * sn],
                     [-1j * kappa * sn, c + 1j * delta * sn]])


def _drives(V, names, signals):
    if isinstance(V, dict):
        unknown = set(V) - set(names)
        if unknown:
            raise KeyError(f"unknown electrodes {sorted(unknown)}; have {names}")
        return {n: float(V.get(n, 0.0)) for n in names}
    return {n: (float(V) if n in signals else 0.0) for n in names}


@dataclass
class PhaseShifterModel:
    """Voltage response of a :func:`phase_shifter`. ``dneff_dV`` is per volt
    on the signal electrode; ``vpi`` is for the drawn length."""
    length: float
    wavelength: float
    dneff_dV: float

    @property
    def vpi_l(self) -> float:
        """V_pi * L in V m."""
        return self.wavelength / (2 * abs(self.dneff_dV))

    @property
    def vpi(self) -> float:
        return self.vpi_l / self.length

    def phase(self, V) -> np.ndarray:
        """Optical phase shift (rad) at signal voltage ``V``."""
        return 2 * np.pi / self.wavelength * self.dneff_dV * np.asarray(V, float) * self.length

    def transfer(self, V) -> complex:
        return np.exp(-1j * self.phase(V))


@dataclass
class SwitchModel:
    """Voltage response of a 2 x 2 switch cell.

    ``transfer(V)`` is the 2 x 2 amplitude matrix from the inputs
    (``o1`` lower, ``o2`` upper) to the outputs (``o4`` lower, ``o3``
    upper), common phase dropped. ``V`` is a dict of electrode voltages, or a
    number applied to every signal electrode (grounds at 0).
    """
    kind: str
    wavelength: float
    kappa: float
    slopes: np.ndarray
    rec: dict
    electrode_names: list
    signals: list
    info: dict = field(default_factory=dict)

    def _delta_n(self, seg, V):
        """d n_eff of each ridge in segment ``seg`` at voltages ``V``."""
        d = np.zeros(self.slopes.shape[0])
        for name, strip in seg["electrodes"].items():
            d += self.slopes[:, strip] * V[name]
        return d

    def transfer(self, V=0.0) -> np.ndarray:
        V = _drives(V, self.electrode_names, self.signals)
        k0 = 2 * np.pi / self.wavelength
        rec = self.rec
        if self.kind == "mzi_switch":
            dc = _coupled(self.kappa, 0.0, rec["coupler"]["length"] * UM)
            seg = rec["segments"][0]
            dn = self._delta_n(seg, V)
            L = (seg["x1"] - seg["x0"]) * UM
            # ribs are ordered (lower, upper)
            arms = np.diag(np.exp(-1j * k0 * dn * L))
            return dc @ arms @ dc
        M = np.eye(2, dtype=complex)
        x = 0.0
        for seg in rec["segments"]:
            if seg["x0"] > x:      # an unelectroded gap: coupling only
                M = _coupled(self.kappa, 0.0, (seg["x0"] - x) * UM) @ M
            dn = self._delta_n(seg, V)
            delta = 0.5 * k0 * (dn[0] - dn[1])
            M = _coupled(self.kappa, delta, (seg["x1"] - seg["x0"]) * UM) @ M
            x = seg["x1"]
        return M

    def cross(self, V=0.0, port: str = "o1") -> float:
        """Power fraction from input ``port`` (o1 or o2) to the other
        waveguide's output."""
        i = {"o1": 0, "o2": 1}[port]
        return float(abs(self.transfer(V)[1 - i, i]) ** 2)

    def bar(self, V=0.0, port: str = "o1") -> float:
        i = {"o1": 0, "o2": 1}[port]
        return float(abs(self.transfer(V)[i, i]) ** 2)

    def sweep(self, voltages) -> Tuple[np.ndarray, np.ndarray]:
        """Bar and cross fractions from ``o1`` over ``voltages`` (on every
        signal electrode)."""
        v = np.asarray(voltages, float)
        return (np.array([self.bar(x) for x in v]), np.array([self.cross(x) for x in v]))

    def switching_voltage(self, state: str = "bar", v_max: float = 100.0,
                          n: int = 2001) -> float:
        """Smallest ``abs(V) <= v_max`` (both polarities scanned) that best reaches
        ``state`` ("bar" or "cross") from o1."""
        v = np.linspace(-v_max, v_max, n)
        bar, cross = self.sweep(v)
        f = bar if state == "bar" else cross
        best = f.max()
        ok = np.flatnonzero(f >= best - 1e-3)
        return float(v[ok[np.argmin(np.abs(v[ok]))]])


def model(component, stack: Optional[Stack] = None, cell_size: float = 0.05,
          wavelength: Optional[float] = None):
    """The voltage response of an LNOI cell (from its ``info["lnoi"]``).

    ``stack`` defaults to :data:`X_CUT` for the phase shifter and the MZI
    switch and :data:`Z_CUT` for the delta-beta coupler. ``wavelength`` (um)
    overrides the stack's. ``cell_size`` (um) is the cross-section mesh;
    0.05 gives a few per cent on ``d n_eff / dV``. The coupling between
    waveguides depends exponentially on the gap, so keep the gap and the ridge
    edges on whole cells: a staircased gap at 0.08 um moves the coupling
    length by tens of per cent.
    """
    rec = dict(component.info["lnoi"])
    kind = rec["kind"]
    if stack is None:
        stack = Z_CUT if kind == "dbeta_coupler_switch" else X_CUT
    if wavelength is not None:
        stack = Stack(**{**stack.__dict__, "wavelength": wavelength})
    lam = stack.wavelength * UM
    if kind == "dbeta_coupler_switch" and stack.cut != "z":
        raise ValueError("electrodes on top of the waveguides detune them only on "
                         "z-cut LN (an x-cut film sees the same lateral field in both)")
    S = _slopes(rec, stack, cell_size)
    names = [n for seg in rec["segments"] for n in seg["electrodes"]]
    signals = [n for n in names if n.startswith("sig")]
    if kind == "phase_shifter":
        seg = rec["segments"][0]
        return PhaseShifterModel(length=(seg["x1"] - seg["x0"]) * UM, wavelength=lam,
                                 dneff_dV=float(S[0, seg["electrodes"]["sig"]]))
    cp = rec["coupler"]
    n_even, n_odd = _supermodes(cp["gap"], rec["width"], stack, cell_size)
    kappa = np.pi * (n_even - n_odd) / lam
    info = dict(n_even=n_even, n_odd=n_odd, coupling_length=np.pi / (2 * kappa))
    if kind == "mzi_switch":
        dn = S[:, 0]   # per volt on the signal: (lower, upper)
        seg = rec["segments"][0]
        info["vpi"] = lam / (2 * abs(dn[0] - dn[1]) * (seg["x1"] - seg["x0"]) * UM)
    return SwitchModel(kind=kind, wavelength=lam, kappa=kappa, slopes=S, rec=rec,
                       electrode_names=names, signals=signals, info=info)


# --------------------------------------------------------------------------- #
# Full FDTD.
# --------------------------------------------------------------------------- #
def layer_map(stack: Stack = X_CUT, layers: Layers = LAYERS):
    """``layers`` and ``background_slabs`` for
    :func:`photonfdtd.from_gdsfactory` (SI units). The metal is left out of
    the optics; it enters the static solve as electrodes."""
    m = stack.media()
    lm = {layers.rib: (m["ln"], (stack.slab * UM, stack.film * UM))}
    bg = [(m["sio2"], (-stack.box * UM, 0.0))]
    if stack.cladding > 0:
        bg.append((m["sio2"], (stack.slab * UM, (stack.film + stack.cladding) * UM)))
    bg.append((m["ln"], (0.0, stack.slab * UM)))
    return lm, bg


def to_simulation(component, stack: Optional[Stack] = None, cell_size: float = 0.05,
                  layers: Layers = LAYERS, **kwargs):
    """The 3-D FDTD model of an LNOI cell: ``(Simulation, ActiveRegion)`` from
    :func:`photonfdtd.from_gdsfactory`, with every electrode of the cell
    named as in its ports (``sig``, ``gnd0``, ...). ``cell_size`` in um; the
    other keywords go to ``from_gdsfactory``. Only short components fit in
    FDTD; use :func:`model` for millimetre devices."""
    from .adapters import from_gdsfactory
    rec = component.info["lnoi"]
    if stack is None:
        stack = Z_CUT if rec["kind"] == "dbeta_coupler_switch" else X_CUT
    lm, bg = layer_map(stack, layers)
    electrodes = {}
    for seg in rec["segments"]:
        xm = 0.5 * (seg["x0"] + seg["x1"])
        for name, i in seg["electrodes"].items():
            s = rec["strips"][i]
            z0 = stack.film + stack.buffer if s["style"] == "top" else stack.slab
            electrodes[name] = (layers.metal, (z0 * UM, (z0 + stack.metal) * UM),
                                (xm, s["y"]))
    # Bring the optical ``layers``/``background_slabs`` defaults, overridable.
    kwargs.setdefault("background_slabs", bg)
    return from_gdsfactory(component, layers=lm, cell_size=cell_size * UM,
                           electrodes=electrodes, wavelength=stack.wavelength * UM,
                           **kwargs)
