"""Chip-scale circuit simulation: solver, electrical network, time domain."""
import numpy as np
import pytest

from photonfdtd import circuit as cc
from photonfdtd.circuit import models as cm


def _mzi(phase_upper_model, coupling=0.5):
    """Two couplers joined by two arms: lower a plain waveguide, upper the
    given model."""
    models = dict(dc1=cm.directional_coupler(coupling), dc2=cm.directional_coupler(coupling),
                  lo=cm.waveguide(100.0), up=phase_upper_model)
    nets = [(("dc1", "o4"), ("lo", "o1")), (("lo", "o2"), ("dc2", "o1")),
            (("dc1", "o3"), ("up", "o1")), (("up", "o2"), ("dc2", "o2"))]
    ext = {"in1": ("dc1", "o1"), "in2": ("dc1", "o2"),
           "out_up": ("dc2", "o3"), "out_lo": ("dc2", "o4")}
    return models, nets, ext


def test_mzi_matches_analytic():
    dphi = np.linspace(0, 2 * np.pi, 9)
    for d in dphi:
        up = cm.Model(lambda wl, d=d, **_: cm.reciprocal(
            {("o1", "o2"): cm.waveguide(100.0).sdict(wl)[("o1", "o2")] * np.exp(-1j * d)}),
            ("o1", "o2"))
        chip = cc.Chip(*_mzi(up)[:2], external=_mzi(up)[2])
        st = chip.solve(1.55, inputs={"in1": 1.0})
        # lower input, two 50:50 couplers: bar output (lower) gets sin^2(d/2)
        assert st.power["out_lo"][0] == pytest.approx(np.sin(d / 2) ** 2, abs=1e-12)
        assert st.power["out_up"][0] == pytest.approx(np.cos(d / 2) ** 2, abs=1e-12)


def test_lossless_chain_is_unitary_and_single_instance_reproduces_model():
    models, nets, ext = _mzi(cm.waveguide(37.0))
    chip = cc.Chip(models, nets, external=ext)
    wl = np.linspace(1.5, 1.6, 7)
    S = chip.s_matrix(wl)
    names = list(ext)
    M = np.array([[S[(i, o)] for i in names] for o in names]).transpose(2, 0, 1)
    for m in M:
        assert np.allclose(m.conj().T @ m, np.eye(4), atol=1e-12)
    one = cc.Chip({"dc": cm.directional_coupler(0.3)},
                  external={p: ("dc", p) for p in ("o1", "o2", "o3", "o4")})
    S1 = one.s_matrix(wl)
    ref = cm.directional_coupler(0.3).sdict(wl)
    for k, v in ref.items():
        assert np.allclose(S1[k], v)


def test_internal_waves_reach_detector():
    models = dict(wg=cm.waveguide(10.0, loss_db_cm=1000.0), pd=cm.photodetector(0.5, 0.0))
    chip = cc.Chip(models, [(("wg", "o2"), ("pd", "o1"))], external={"in": ("wg", "o1")})
    st = chip.solve(1.55, inputs={"in": 2e-3}, drives={("pd", "cathode"): 0.0,
                                                       ("pd", "anode"): 0.0})
    loss = 10 ** (-1000 * 10e-4 / 10)
    assert st.detector_power["pd"][0] == pytest.approx(2e-3 * loss, rel=1e-12)
    assert st.detector_current["pd"][0] == pytest.approx(0.5 * 2e-3 * loss, rel=1e-12)
    # The current leaves at the anode: its source sinks it.
    assert st.source_current[("pd", "anode")][0] == pytest.approx(
        -st.detector_current["pd"][0], rel=1e-9)


def test_detector_into_load_resistor():
    # Photocurrent into a 10 kohm load from anode to ground: V = I R.
    models = dict(pd=cm.photodetector(1.0, 0.0), load=cm.resistor(1e4, a=("a",), b=("b",)),
                  gnd=cm.conductor(["g"]))
    enets = [(("pd", "anode"), ("load", "a")), (("load", "b"), ("gnd", "g")),
             (("pd", "cathode"), ("gnd", "g"))]
    chip = cc.Chip(models, (), enets, external={"in": ("pd", "o1")})
    st = chip.solve(1.55, inputs={"in": 1e-4}, drives={"gnd": 0.0})
    anode = chip.net_names[chip.net(("pd", "anode"))]
    assert st.voltage[anode][0] == pytest.approx(1e-4 * 1e4, rel=1e-6)


def test_unconnected_detector_is_an_open_circuit():
    # No drive and no load between the contacts: no current, no runaway voltage.
    chip = cc.Chip(dict(pd=cm.photodetector(1.0, 0.0)), (), (), external={"in": ("pd", "o1")})
    with pytest.warns(UserWarning, match="no circuit between"):
        st = chip.solve(1.55, inputs={"in": 1e-3})
    assert st.detector_current["pd"][0] == 0.0
    assert max(abs(v[0]) for v in st.voltage.values()) < 1e-6


def test_heater_switches_mzi_through_its_resistance():
    h = cm.heater(P_pi=0.02, resistance=500.0, length=100.0)
    models, nets, ext = _mzi(h)
    models["pad"] = cm.conductor(["e"])
    enets = [(("pad", "e"), ("up", "a"))]
    chip = cc.Chip(models, nets, enets, ext)
    V_pi = np.sqrt(0.02 * 500.0)            # P = V^2 / R
    # With the far terminal floating no current flows: no power.
    st = chip.solve(1.55, drives={"pad": 5.0}, inputs={"in1": 1.0})
    assert st.heater_power["up"][0] < 1e-12
    st0 = chip.solve(1.55, drives={"pad": 0.0, ("up", "b"): 0.0}, inputs={"in1": 1.0})
    st1 = chip.solve(1.55, drives={"pad": V_pi, ("up", "b"): 0.0}, inputs={"in1": 1.0})
    assert st1.heater_power["up"][0] == pytest.approx(0.02)
    assert st0.power["out_up"][0] == pytest.approx(1.0, abs=1e-9)
    assert st1.power["out_lo"][0] == pytest.approx(1.0, abs=1e-9)
    assert st1.source_current["pad"][0] == pytest.approx(V_pi / 500.0, rel=1e-6)


def _phase_modulator(dn_dV=1e-4, length=100.0, bandwidth=None):
    def fn(wl, V=None, **_):
        v = (V or {}).get("sig", 0.0) - (V or {}).get("gnd", 0.0)
        t = np.exp(-1j * 2 * np.pi / (wl * 1e-6) * dn_dV * v * length * 1e-6) * \
            cm.waveguide(length).sdict(wl)[("o1", "o2")]
        return cm.reciprocal({("o1", "o2"): t})
    return cm.Modulator(fn, ("o1", "o2"), {"sig": ("s",), "gnd": ("g",)}, bandwidth=bandwidth)


def test_transient_square_wave_and_bandwidth():
    dn, L = 1e-4, 100.0
    v_pi = 1.55e-6 / (2 * dn * L * 1e-6)
    f3 = 1e9
    models, nets, ext = _mzi(_phase_modulator(dn, L, bandwidth=f3))
    chip = cc.Chip(models, nets, (), ext)
    t = np.linspace(0, 4e-9, 4001)
    drive = cc.Square(0.0, v_pi, freq=0.25e9, duty=0.5)   # high for the first 2 ns
    tr = chip.transient(t, {("up", "sig"): drive, ("up", "gnd"): 0.0}, inputs={"in1": 1.0})
    tau = 1 / (2 * np.pi * f3)
    # Starts in the high state (switched to bar); after the falling edge at
    # 2 ns the effective voltage decays as exp(-t/tau).
    k = np.searchsorted(t, 2e-9 + tau)
    v_eff = v_pi * np.exp(-(t[k] - 2e-9) / tau)
    expected_lo = np.sin(np.pi * v_eff / v_pi / 2) ** 2
    assert tr.power["out_lo"][0] == pytest.approx(1.0, abs=1e-9)
    assert tr.power["out_lo"][k] == pytest.approx(expected_lo, abs=0.01)
    assert tr.power["out_up"][-1] > 0.999
    assert np.allclose(tr.power["out_up"] + tr.power["out_lo"], 1.0)


def test_waveforms():
    t = np.array([0.0, 0.2e-9, 0.6e-9, 1.2e-9])
    assert np.allclose(cc.Square(0, 1, 1e9)(t), [1, 1, 0, 1])
    assert np.allclose(cc.PWL((0, 1), (0, 2))(np.array([0.5])), [1.0])
    bits = cc.PRBS(order=7).bits(127)
    assert 60 <= bits.sum() <= 67                     # balanced maximal sequence
    assert np.allclose(cc.Sine(1, 2, 1e9)(np.array([0.25e-9])), [3.0])


def test_from_gdsfactory_chip(monkeypatch, tmp_path):
    gf = pytest.importorskip("gdsfactory")
    gf.gpdk.PDK.activate()
    from photonfdtd import lnoi
    from photonfdtd.circuit.cells import photodetector

    # A stand-in for the minute-long cross-section solve: an ideal MZI
    # switch with V_pi = 5 V.
    def fake_model(cell, **_):
        rec = dict(cell.info["lnoi"])
        lam = 1.55e-6
        L = (rec["segments"][0]["x1"] - rec["segments"][0]["x0"]) * 1e-6
        d = lam / (2 * 5.0 * L) / 2
        kappa = np.pi / (4 * rec["coupler"]["length"] * 1e-6)   # 50:50
        return lnoi.SwitchModel(kind=rec["kind"], wavelength=lam, kappa=kappa,
                                slopes=np.array([[-d, 0, 0], [d, 0, 0]]), rec=rec,
                                electrode_names=["sig", "gnd_top", "gnd_bot"], signals=["sig"],
                                info=dict(n_even=1.9, n_odd=1.9, coupling_length=0.0, vpi=5.0))
    monkeypatch.setattr(lnoi, "model", fake_model)

    c = gf.Component("chip_test")
    sw = c << lnoi.mzi_switch(arm_length=500)
    pad = c << gf.components.pad()
    pad.move((200, 300))
    gf.routing.route_single(c, pad.ports["e4"], sw.ports["sig_w"], cross_section="metal_routing",
                            allow_width_mismatch=True)
    pds = {}
    for name, port in (("pd_up", "o3"), ("pd_lo", "o4")):
        wg = c << gf.components.straight(length=30, cross_section=lnoi.rib())
        wg.connect("o1", sw.ports[port])
        pd = c << photodetector()
        pd.connect("o1", wg.ports["o2"])
        pds[name] = pd
    c.add_port("in", port=sw.ports["o1"])
    for name, pd in pds.items():
        c.add_port(f"{name}_a", port=pd.ports["anode_w"])
        c.add_port(f"{name}_k", port=pd.ports["cathode_w"])
    c.add_port("gnd_top", port=sw.ports["gnd_top_w"])
    c.add_port("gnd_bot", port=sw.ports["gnd_bot_w"])

    chip = cc.Chip.from_gdsfactory(c)
    assert "in" in chip.network.external
    bias = {k: 0.0 for k in ("gnd_top", "gnd_bot", "pd_up_a", "pd_lo_a")}
    bias.update(pd_up_k=2.0, pd_lo_k=2.0)
    pd_names = {n for n, m in chip.models.items() if isinstance(m, cc.Photodetector)}
    assert len(pd_names) == 2

    def detector_currents(v):
        st = chip.solve(1.55, drives={**bias, "pad": v}, inputs={"in": 1e-3})
        return {n: st.detector_current[n][0] for n in pd_names}

    off, on = detector_currents(0.0), detector_currents(5.0)
    # The light moves from one detector to the other at V_pi.
    hi_off = max(off, key=off.get)
    assert off[hi_off] > 0.7e-3 and on[hi_off] < 0.02e-3


def test_sax_matches_builtin_solver():
    sax = pytest.importorskip("sax")
    h = cm.heater(P_pi=0.02, resistance=500.0, length=100.0)
    models, nets, ext = _mzi(h, coupling=0.4)
    models["pad"] = cm.conductor(["e"])
    chip = cc.Chip(models, nets, [(("pad", "e"), ("up", "a"))], ext)
    drives = {"pad": 2.0, ("up", "b"): 0.0}
    wl = np.linspace(1.5, 1.6, 5)
    S = chip.s_matrix(wl, drives)
    netlist, sax_models = cc.to_sax(chip, drives)
    circuit, _ = sax.circuit(netlist=netlist, models=sax_models)
    Ssax = circuit(wl=wl)
    for k in (("in1", "out_up"), ("in1", "out_lo"), ("in2", "out_up")):
        assert np.allclose(np.asarray(Ssax[k]), S[k], atol=1e-6)
