"""Component files: export, load, models, generators."""
import json

import numpy as np
import pytest

from photonfdtd import circuit as cc
from photonfdtd.circuit import library as lib
from photonfdtd.circuit import models as cm


def _phase_modulator(dn_dV=1e-4, length=100.0):
    def fn(wl, V=None, **_):
        v = (V or {}).get("sig", 0.0) - (V or {}).get("gnd", 0.0)
        t = np.exp(-1j * 2 * np.pi / (wl * 1e-6) * dn_dV * v * length * 1e-6)
        return cm.reciprocal({("o1", "o2"): 0.9 * t})
    return cm.Modulator(fn, ("o1", "o2"), {"sig": ("s",), "gnd": ("g",)})


def test_sparams_table_reproduces_and_interpolates():
    m = _phase_modulator()
    wl = np.linspace(1.5, 1.6, 11)
    table = lib.sample_sparams(m, wl, drives={"sig": np.linspace(0, 10, 21), "gnd": [0.0]})
    table = json.loads(json.dumps(table))          # survives JSON
    tm = lib.TableModel(table, {"sig": ["s"], "gnd": ["g"]}).model
    assert isinstance(tm, cm.Modulator) and set(tm.conductors) == {"sig", "gnd"}
    for v in (0.0, 5.0, 10.0):
        ref = m.sdict(wl, V={"sig": v})[("o1", "o2")]
        assert np.allclose(tm.sdict(wl, V={"sig": v})[("o1", "o2")], ref, atol=1e-12)
    # Between samples: close to the truth (fine grid), and |S| stays bounded.
    got = tm.sdict(np.array([1.555]), V={"sig": 5.25})[("o1", "o2")]
    ref = m.sdict(np.array([1.555]), V={"sig": 5.25})[("o1", "o2")]
    assert abs(got - ref) < 0.02 and abs(got) <= 0.9 + 1e-9


gf = pytest.importorskip("gdsfactory")
gf.gpdk.PDK.activate()
from photonfdtd import lnoi  # noqa: E402
from photonfdtd.circuit.cells import photodetector  # noqa: E402

V_PI = 5.0


@pytest.fixture
def fast_lnoi(monkeypatch):
    def fake_model(cell, **_):
        rec = dict(cell.info["lnoi"])
        seg = rec["segments"][0]
        L = (seg["x1"] - seg["x0"]) * 1e-6
        d = 1.55e-6 / (2 * V_PI * L) / 2
        kappa = np.pi / (4 * rec["coupler"]["length"] * 1e-6)
        return lnoi.SwitchModel(kind=rec["kind"], wavelength=1.55e-6, kappa=kappa,
                                slopes=np.array([[-d, 0, 0], [d, 0, 0]]), rec=rec,
                                electrode_names=["sig", "gnd_top", "gnd_bot"], signals=["sig"],
                                info=dict(n_even=1.9, n_odd=1.9, coupling_length=0.0, vpi=V_PI))
    monkeypatch.setattr(lnoi, "model", fake_model)


def test_lnoi_cell_round_trip(tmp_path, fast_lnoi):
    c = lnoi.mzi_switch(arm_length=500)
    data = lib.export_component(c, tmp_path / "sw", description="test switch")
    path = tmp_path / ("sw" + lib.SUFFIX)
    assert path.exists() and json.loads(path.read_text())["name"] == "mzi_switch"
    assert set(data["model"]) == {"physics", "recipe"}
    assert data["generator"]["module"] == "photonfdtd.lnoi"
    assert data["conductors"]["sig"] == ["sig_w", "sig_e"]
    cf = lib.load_component(path)
    cell = cf.cell()
    assert {p.name for p in cell.ports} == {p.name for p in c.ports}
    a, b = c.get_polygons(by="tuple"), cell.get_polygons(by="tuple")
    assert {k: len(v) for k, v in a.items()} == {k: len(v) for k, v in b.items()}
    assert cell.dbbox().width() == pytest.approx(c.dbbox().width(), abs=1e-3)
    # Same model as the cell's own.
    ref = cm.from_lnoi(lnoi.model(c), length=data["model"]["physics"]["length"])
    wl = np.array([1.55])
    for V in (0.0, V_PI):
        s1, s2 = ref.sdict(wl, V={"sig": V}), cf.model().sdict(wl, V={"sig": V})
        for k in s1:
            assert np.allclose(s1[k], s2[k])
    # The generator rebuilds it with other parameters.
    c2 = cf.regenerate(arm_length=800)
    assert c2.settings.arm_length == 800
    # A circuit built from the loaded cell finds its model by itself.
    top = gf.Component("uses_file")
    top << cell
    chip = cc.Chip.from_gdsfactory(top)
    st0 = chip.solve(1.55, drives={(next(iter(chip.models)), "sig"): 0.0},
                     inputs={next(n for n in chip.network.external if n.endswith(",o1")): 1.0})
    assert max(st0.power.values())[0] > 0.99


def test_photodetector_round_trip(tmp_path):
    data = lib.export_component(photodetector(responsivity=0.6), tmp_path / "pd")
    cf = lib.load_component(tmp_path / ("pd" + lib.SUFFIX))
    m = cf.model()
    assert isinstance(m, cm.Photodetector) and m.responsivity == 0.6
    assert data["generator"]["module"] == "photonfdtd.circuit.cells"


def test_sub_layout_exports_as_a_circuit(tmp_path, fast_lnoi):
    c = gf.Component("sub")
    sw = c << lnoi.mzi_switch(arm_length=500)
    pad = c << gf.components.pad()
    pad.move((200, 300))
    gf.routing.route_single(c, pad.ports["e4"], sw.ports["sig_w"],
                            cross_section="metal_routing", allow_width_mismatch=True)
    for n, p in (("in_lo", "o1"), ("in_up", "o2"), ("out_up", "o3"), ("out_lo", "o4")):
        c.add_port(n, port=sw.ports[p])
    c.add_port("drive", port=pad.ports["e2"])
    c.add_port("gnd_a", port=sw.ports["gnd_top_w"])
    c.add_port("gnd_b", port=sw.ports["gnd_bot_w"])
    data = lib.export_component(c, tmp_path / "sub", name="switch_on_pad")
    ph = data["model"]["physics"]
    assert ph["kind"] == "circuit" and set(ph["data"]["external"]) == {"in_lo", "in_up", "out_up", "out_lo"}
    cf = lib.load_component(tmp_path / ("sub" + lib.SUFFIX))
    m = cf.model()
    assert isinstance(m, cm.Modulator)
    pad_pin = next(n for n, ports in data["conductors"].items() if "drive" in ports)
    wl = np.array([1.55])
    off = m.sdict(wl, V={pad_pin: 0.0})
    on = m.sdict(wl, V={pad_pin: V_PI})
    assert abs(off[("in_lo", "out_up")][0]) ** 2 > 0.99     # cross at 0 V
    assert abs(on[("in_lo", "out_lo")][0]) ** 2 > 0.99      # bar at V_pi
    # Placed in another layout, it is driven through its pad port.
    top = gf.Component("top_with_sub")
    ref = top << cf.cell()
    other_pad = top << gf.components.pad()
    other_pad.move((-400, 600))
    gf.routing.route_single(top, other_pad.ports["e4"], ref.ports["drive"],
                            cross_section="metal_routing", allow_width_mismatch=True)
    chip = cc.Chip.from_gdsfactory(top)
    inst = next(n for n, mm in chip.models.items() if mm.ports)
    for v, port in ((0.0, "out_up"), (V_PI, "out_lo")):
        st = chip.solve(1.55, drives={"pad": v}, inputs={f"{inst},in_lo": 1.0})
        assert st.power[f"{inst},{port}"][0] > 0.99


def test_generator_source_needs_trust(tmp_path):
    src = ("import gdsfactory as gf\n"
           "def block(size=2.0):\n"
           "    c = gf.Component()\n"
           "    c.add_polygon([(0, 0), (size, 0), (size, size), (0, size)], layer=(1, 0))\n"
           "    return c\n")
    c = gf.Component("block_src")
    c.add_polygon([(0, 0), (2, 0), (2, 2), (0, 2)], layer=(1, 0))
    lib.export_component(c, tmp_path / "blk", name="block", model=None,
                         generator={"function": "block", "kwargs": {"size": 2.0}},
                         generator_source=src)
    with pytest.raises(PermissionError):
        lib.load_component(tmp_path / ("blk" + lib.SUFFIX)).regenerate(size=3.0)
    big = lib.load_component(tmp_path / ("blk" + lib.SUFFIX), trust=True).regenerate(size=3.0)
    assert big.dbbox().width() == pytest.approx(3.0)


def test_recipe_resolves_at_another_wavelength(tmp_path, fast_lnoi, monkeypatch):
    c = lnoi.phase_shifter(length=1000)
    monkeypatch.setattr(lnoi, "model", lambda cell, **kw: lnoi.PhaseShifterModel(
        length=1e-3, wavelength=1.55e-6, dneff_dV=1e-5, n_eff=1.9))
    data = lib.export_component(c, tmp_path / "ps")
    seen = {}

    def fake_solve(rec, stack, cell_size):
        seen.update(wavelength=stack.wavelength, cell=cell_size)
        return lnoi.PhaseShifterModel(length=1e-3, wavelength=stack.wavelength * 1e-6,
                                      dneff_dV=2e-5, n_eff=1.8)
    monkeypatch.setattr(lnoi, "_solve_model", fake_solve)
    new = lib.load_component(data).resolve(wavelength=1.31, cell_size=0.03)
    assert seen == {"wavelength": 1.31, "cell": 0.03}
    assert new["data"]["dneff_dV"] == 2e-5


def test_rejects_other_files():
    with pytest.raises(ValueError):
        lib.load_component({"format": "something-else"})
