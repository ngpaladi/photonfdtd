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
                                  lnoi.dbeta_coupler_switch])
def test_cells_draw(cell):
    c = cell()
    names = {p.name for p in c.ports}
    rec = c.info["lnoi"]
    assert rec["kind"] == cell.__name__
    for seg in rec["segments"]:
        for e in seg["electrodes"]:
            assert {f"{e}_w", f"{e}_e"} <= names
    optical = {p.name for p in c.ports if p.port_type == "optical"}
    assert optical == ({"o1", "o2"} if cell is lnoi.phase_shifter else {"o1", "o2", "o3", "o4"})
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
