"""LNOI switch cells for gdsfactory and their voltage models."""
import numpy as np
import pytest

gf = pytest.importorskip("gdsfactory")
gf.gpdk.PDK.activate()

from photonfdtd import lnoi  # noqa: E402


def test_coupled_mode_limits():
    kappa = np.pi / (2 * 500e-6)            # L_c = 500 um
    M = lnoi._coupled(kappa, 0.0, 500e-6)
    assert abs(M[1, 0]) ** 2 == pytest.approx(1.0, abs=1e-12)
    # Kogelnik-Schmidt: delta = sqrt(3) kappa over L_c returns all light to bar.
    M = lnoi._coupled(kappa, np.sqrt(3) * kappa, 500e-6)
    assert abs(M[0, 0]) ** 2 == pytest.approx(1.0, abs=1e-12)
    # Unitary for any detuning.
    M = lnoi._coupled(kappa, 0.7 * kappa, 321e-6)
    assert np.allclose(M.conj().T @ M, np.eye(2))


@pytest.mark.parametrize("cell", [lnoi.phase_shifter, lnoi.mzi_switch,
                                  lnoi.mzi_switch_1x2, lnoi.dbeta_coupler_switch])
def test_cells_draw(cell):
    c = cell()
    names = {p.name for p in c.ports}
    rec = c.info["lnoi"]
    assert rec["kind"] == cell.__name__
    for seg in rec["segments"]:
        for e in seg["electrodes"]:
            assert {f"{e}_w", f"{e}_e"} <= names
    optical = {p.name for p in c.ports if p.port_type == "optical"}
    expected = {lnoi.phase_shifter: {"o1", "o2"}, lnoi.mzi_switch_1x2: {"o1", "o3", "o4"}}
    assert optical == expected.get(cell, {"o1", "o2", "o3", "o4"})
    assert c.get_polygons()             # draws something


def test_dbeta_sections_alternate():
    rec = lnoi.dbeta_coupler_switch(sections=2).info["lnoi"]
    s0, s1 = rec["segments"]
    assert s0["electrodes"]["sig0"] != s1["electrodes"]["sig1"]
    assert s0["x1"] < s1["x0"]


def test_phase_shifter_model():
    m = lnoi.model(lnoi.phase_shifter(length=10000), cell_size=0.05)
    # x-cut TFLN, 5 um gap: about 4 V cm per arm (example 8).
    assert 3.0 < m.vpi_l / 1e-2 < 6.0
    assert m.vpi == pytest.approx(m.vpi_l / 1e-2, rel=1e-9)
    assert abs(m.phase(m.vpi)) == pytest.approx(np.pi, rel=1e-9)


def test_switch_models():
    mz = lnoi.model(lnoi.mzi_switch(), cell_size=0.05)
    # Push-pull: the arms move in opposite directions by the same amount.
    a, b = mz.slopes[:, 0]
    assert a * b < 0 and abs(a + b) < 0.05 * abs(a - b)
    # Near-50:50 couplers with equal arms: cross at 0 V, bar at V_pi.
    assert mz.cross(0) > 0.95
    assert mz.bar(mz.info["vpi"]) > 0.95
    db = lnoi.model(lnoi.dbeta_coupler_switch(), cell_size=0.05)
    assert db.cross(0) > 0.9
    v = db.switching_voltage("bar")
    assert db.bar(v) > 0.95


def test_1x2_switch_geometry():
    """The Y-junction opens from the single input, centred between the arms,
    to the arm separation; the outputs are where the 2 x 2's are."""
    c = lnoi.mzi_switch_1x2(arm_length=1000)
    two = lnoi.mzi_switch(arm_length=1000)
    p = {q.name: q for q in c.ports}
    assert p["o1"].center[1] == pytest.approx(0.0)
    assert p["o1"].orientation == 180
    for name in ("o3", "o4"):
        assert p[name].center[1] == pytest.approx(two.ports[name].center[1])
    # The first coupler and its bends (150 + 276 + 150 um) give way to the
    # 50 um input and the 150 um Y.
    span_two = two.ports["o3"].center[0] - two.ports["o1"].center[0]
    assert p["o3"].center[0] - p["o1"].center[0] == pytest.approx(span_two - 576.0 + 200.0)
    (seg,), (seg2,) = c.info["lnoi"]["segments"], two.info["lnoi"]["segments"]
    assert seg["x1"] - seg["x0"] == pytest.approx(seg2["x1"] - seg2["x0"])
    assert seg["electrodes"] == seg2["electrodes"]
    assert c.info["lnoi"]["ribs"] == two.info["lnoi"]["ribs"]


def _ideal_1x2():
    """A 1 x 2 model with exact 50:50 output coupler and V_pi = 4 V."""
    rec = lnoi.mzi_switch_1x2(arm_length=1000).info["lnoi"]
    L = rec["coupler"]["length"] * 1e-6
    kappa = np.pi / (4 * L)
    seg = rec["segments"][0]
    d = 1.55e-6 / (2 * 4.0 * (seg["x1"] - seg["x0"]) * 1e-6) / 2
    return lnoi.SwitchModel(kind="mzi_switch_1x2", wavelength=1.55e-6, kappa=kappa,
                            slopes=np.array([[-d, 0, 0], [d, 0, 0]]), rec=rec,
                            electrode_names=["sig", "gnd_top", "gnd_bot"], signals=["sig"],
                            info=dict(n_even=2.0, n_odd=2.0, vpi=4.0))


def test_1x2_transfer_splits_at_0v_and_switches_at_half_vpi():
    m = _ideal_1x2()
    assert m.transfer(0).shape == (2, 1)
    assert m.bar(0) == pytest.approx(0.5, abs=1e-9)
    assert m.cross(0) == pytest.approx(0.5, abs=1e-9)
    lo, hi = sorted([m.bar(2.0), m.bar(-2.0)])
    assert hi == pytest.approx(1.0, abs=1e-9) and lo == pytest.approx(0.0, abs=1e-9)
    for V in (-3.0, 0.7, 2.0):
        assert m.bar(V) + m.cross(V) == pytest.approx(1.0, abs=1e-12)   # lossless
    assert abs(m.switching_voltage("bar")) == pytest.approx(2.0, abs=0.01)
    with pytest.raises(ValueError):
        m.bar(0, port="o2")


def test_1x2_circuit_model_is_a_three_port():
    from photonfdtd.circuit.models import from_lnoi

    m = _ideal_1x2()
    mod = from_lnoi(m, length=100.0)
    assert set(mod.ports) == {"o1", "o3", "o4"}
    s = mod.fn(np.array([1.55]), V={"sig": 2.0})
    p4, p3 = abs(s[("o1", "o4")][0]) ** 2, abs(s[("o1", "o3")][0]) ** 2
    assert p4 + p3 == pytest.approx(1.0, abs=1e-9)
    assert max(p4, p3) == pytest.approx(1.0, abs=1e-9)
    assert s[("o4", "o1")][0] == s[("o1", "o4")][0]            # reciprocal


def test_1x2_json_round_trip():
    m = _ideal_1x2()
    back = lnoi._from_json(lnoi._to_json(m))
    assert back.kind == "mzi_switch_1x2"
    assert np.allclose(back.transfer(1.3), m.transfer(1.3))


def test_unknown_kind_has_no_transfer():
    m = _ideal_1x2()
    m.kind = "something_else"
    with pytest.raises(ValueError):
        m.transfer(0)


def test_1x2_switch_model_solved():
    """The real cross-section solve: the default 1 x 2 is 50:50 at 0 V (within
    the default coupler's split) and switches at +-V_pi/2."""
    m = lnoi.model(lnoi.mzi_switch_1x2(), cell_size=0.05)
    assert m.bar(0) == pytest.approx(0.5, abs=0.03)
    v = m.switching_voltage("bar")
    assert abs(v) == pytest.approx(m.info["vpi"] / 2, rel=0.05)
    assert m.bar(v) > 0.95 and m.cross(-v) > 0.95
