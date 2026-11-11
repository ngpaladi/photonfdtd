"""Near-to-far-field projection (photonfdtd.farfield)."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import hankel1

import photonfdtd as pf
from photonfdtd.farfield import _component_coords

LAM = 1e-6
F0 = pf.C_0 / LAM
K = 2 * np.pi / LAM
WAVE = pf.GaussianPulse(freq0=F0, fwhm=6e-15)
FACES_2D = {"xm": "-x", "xp": "+x", "ym": "-y", "yp": "+y"}


def _contour_monitors(h, comps=("Ez", "Hx", "Hy"), axes="xy"):
    mons = []
    for a in axes:
        for s, tag in ((-1, "m"), (1, "p")):
            mons.append(pf.DFTMonitor(name=a + tag, components=comps, freqs=[F0],
                                      plane_axis=a, plane_position=s * h))
    return mons


def _run_2d(ys, ppw=30, h=1.5e-6, pol="TM"):
    g = pf.Grid(size=(5e-6, 5e-6), cell_size=LAM / ppw, pml_layers=(15, 15, 0))
    comp, comps = ("Ez", ("Ez", "Hx", "Hy")) if pol == "TM" else ("Hz", ("Hz", "Ex", "Ey"))
    srcs = [pf.PointDipole(position=(0, y), component=comp, waveform=WAVE) for y in ys]
    sim = pf.Simulation(g, sources=srcs, monitors=_contour_monitors(h, comps),
                        run_time=80e-15)
    return sim, sim.run()


def test_projection_of_exact_line_source_fields():
    """Fed the exact Hankel-function fields of a 2-D line source, the
    projection reproduces its isotropic far field H0 ~ sqrt(2/(pi k rho))
    e^{i(k rho - pi/4)} in magnitude and phase - this checks the propagator,
    the currents, the Yee sample positions and the box closure on their own."""
    w = 2 * np.pi * F0
    g = pf.Grid(size=(5e-6, 5e-6), cell_size=LAM / 30, pml_layers=(15, 15, 0))
    h = 1.5e-6                                 # not on the grid: faces must snap

    def fields(x, y):
        r = np.hypot(x, y)
        dEdr = -K * hankel1(1, K * r)
        return {"Ez": hankel1(0, K * r),
                "Hx": dEdr * y / r / (1j * w * pf.MU_0),
                "Hy": -dEdr * x / r / (1j * w * pf.MU_0)}

    mons = _contour_monitors(h)
    dft = {}
    for m in mons:
        ax = "xyz".index(m.plane_axis)
        xp = g.coords[ax][int(np.argmin(abs(g.coords[ax] - m.plane_position)))]
        t = 1 - ax
        out = {}
        for c in ("Ez", "Hx", "Hy"):
            ct = _component_coords(g, c, t)
            X, Y = (np.full_like(ct, xp), ct) if ax == 0 else (ct, np.full_like(ct, xp))
            shp = [1, 1, 1, 1]; shp[1 + t] = -1
            out[c] = fields(X, Y)[c].reshape(shp)
        dft[m.name] = out
    sim = SimpleNamespace(grid=g, monitors=mons)
    res = SimpleNamespace(dft=dft, dft_freqs={n: np.array([F0]) for n in dft})
    phi = np.linspace(0, 2 * np.pi, 73)
    ff = pf.project_far_field(sim, res, FACES_2D, pf.polar_directions(phi))
    exact = np.sqrt(2 / (np.pi * K)) * np.exp(-0.25j * np.pi)
    assert np.abs(ff.E[0, :, 2] / exact - 1).max() < 2e-3
    assert np.abs(ff.E[0, :, :2]).max() < 1e-6 * np.abs(ff.E).max()


@pytest.mark.parametrize("pol", ["TM", "TE"])
def test_2d_dipole_isotropic_and_power_balanced(pol):
    """Both 2-D polarizations: TM (Ez, Hx, Hy) exercises the eta N_theta + L_phi
    branch, TE (Hz, Ex, Ey) the eta N_phi - L_theta one."""
    sim, res = _run_2d([0.0], pol=pol)
    phi = np.linspace(0, 2 * np.pi, 361)[:-1]
    ff = pf.project_far_field(sim, res, FACES_2D, pf.polar_directions(phi))
    U = ff.intensity[0]
    assert (U.max() - U.min()) / U.mean() < 0.02
    # Far-field power equals the near-field Poynting flux out of the contour.
    P_ff = U.sum() * (phi[1] - phi[0])
    g, h, P = sim.grid, 1.5e-6, 0.0
    for name, ax, s in (("xm", 0, -1), ("xp", 0, 1), ("ym", 1, -1), ("yp", 1, 1)):
        d = {c: np.squeeze(v[0]) for c, v in res.dft[name].items()}
        c = g.coords[1 - ax]
        inside = np.abs(c) <= h
        if pol == "TM":      # S = E x H*: Sx = -Ez Hy*, Sy = Ez Hx*
            Sn = (-0.5 * np.real(d["Ez"] * np.conj(d["Hy"])) if ax == 0
                  else 0.5 * np.real(d["Ez"] * np.conj(d["Hx"])))
        else:                # Sx = Ey Hz*, Sy = -Ex Hz*
            Sn = (0.5 * np.real(d["Ey"] * np.conj(d["Hz"])) if ax == 0
                  else -0.5 * np.real(d["Ex"] * np.conj(d["Hz"])))
        P += s * Sn[inside].sum() * g.cell_size[0]
    assert abs(P_ff / P - 1) < 0.01


def test_2d_two_dipole_interference():
    """Two in-phase line sources a distance d apart along y radiate
    4 U0 cos^2(k d sin(phi) / 2)."""
    phi = np.linspace(0, 2 * np.pi, 181)[:-1]
    sim1, res1 = _run_2d([0.0])
    U0 = pf.project_far_field(sim1, res1, FACES_2D,
                              pf.polar_directions(phi)).intensity[0].mean()
    sim, res = _run_2d([-0.4e-6, 0.4e-6])
    ys = [sim.grid.coords[1][sim.grid.index_at(s.position)[1]] for s in sim.sources]
    d = ys[1] - ys[0]
    U = pf.project_far_field(sim, res, FACES_2D, pf.polar_directions(phi)).intensity[0]
    model = 4 * U0 * np.cos(K * d * np.sin(phi) / 2) ** 2
    assert np.abs(U - model).max() / model.max() < 0.02


def test_3d_dipole_pattern():
    """A z-dipole radiates U ~ sin^2(theta), polarized along theta."""
    g = pf.Grid(size=(3e-6,) * 3, cell_size=LAM / 20, pml_layers=(10, 10, 10))
    mons = _contour_monitors(0.9e-6, comps=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
                             axes="xyz")
    sim = pf.Simulation(g, sources=[pf.PointDipole(position=(0, 0, 0),
                                                   component="Ez", waveform=WAVE)],
                        monitors=mons, run_time=50e-15)
    res = sim.run()
    faces = {m.name: ("-" if m.name[1] == "m" else "+") + m.name[0] for m in mons}
    th = np.linspace(0.01, np.pi - 0.01, 40)
    ph = np.linspace(0, 2 * np.pi, 25)[:-1]
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    ff = pf.project_far_field(sim, res, faces, pf.spherical_directions(TH, PH))
    U = ff.intensity[0].reshape(TH.shape)
    model = np.sin(TH) ** 2
    c = (U * model).sum() / (model ** 2).sum()
    assert np.abs(U - c * model).max() / c < 0.02
    Eth, Eph = ff.components()
    assert np.abs(Eph).max() < 0.01 * np.abs(Eth).max()


def test_far_field_distance_scaling():
    sim, res = _run_2d([0.0])
    ff = pf.project_far_field(sim, res, FACES_2D, pf.polar_directions([0.3]))
    E1, E2 = ff.field_at(1.0), ff.field_at(4.0)
    assert np.isclose(np.abs(E1).max() / np.abs(E2).max(), 2.0)


def test_projection_validation():
    sim, res = _run_2d([0.0])
    with pytest.raises(ValueError, match="normal"):
        pf.project_far_field(sim, res, {"xm": "-y"}, pf.polar_directions([0.0]))
    with pytest.raises(ValueError, match="xy plane"):
        pf.project_far_field(sim, res, FACES_2D, [[0, 0, 1.0]])
    with pytest.raises(ValueError, match="below"):
        pf.project_far_field(sim, res, {"xm": "+x", "xp": "-x"},
                             pf.polar_directions([0.0]))


@pytest.mark.parametrize("index", [1.0, 1.5])
def test_far_field_power_equals_near_field_flux(index):
    """Absolute normalisation: the far-field intensity integrated over all
    directions is the Poynting flux through the monitor box."""
    lam = 1e-6
    f0 = pf.C_0 / lam
    h = 1.5e-6
    grid = pf.Grid(size=(5e-6, 5e-6), cell_size=lam / 30, pml_layers=(15, 15, 0))
    mons = [pf.DFTMonitor(name=a + t, components=("Ez", "Hx", "Hy"), freqs=[f0],
                          plane_axis=a, plane_position=s * h)
            for a in "xy" for s, t in ((-1, "m"), (1, "p"))]
    structures = [pf.Box(center=(0, 0, 0), size=(1, 1, 1), medium=pf.Medium.from_index(index))] \
        if index != 1.0 else []
    sim = pf.Simulation(grid, structures=structures, monitors=mons, run_time=120e-15,
                        sources=[pf.PointDipole(position=(0, 0), component="Ez",
                                                waveform=pf.GaussianPulse(freq0=f0, fwhm=6e-15))])
    res = sim.run()

    flux = 0.0
    for ax, a in enumerate("xy"):
        t = np.asarray(grid.coords[1 - ax])
        inside = np.abs(t) <= h + 1e-12
        w = inside * grid.cell_size[1 - ax]
        w = np.where(np.isclose(np.abs(t), np.abs(t)[inside].max()), 0.5 * w, w)   # box corners
        for s, tag in ((-1, "m"), (1, "p")):
            d = {c: np.asarray(v)[0].ravel() for c, v in res.dft[a + tag].items()}
            # S_x = -Ez Hy*, S_y = Ez Hx*
            sn = 0.5 * np.real(-d["Ez"] * np.conj(d["Hy"]) if a == "x" else d["Ez"] * np.conj(d["Hx"]))
            flux += s * np.sum(sn * w)

    phi = np.linspace(0, 2 * np.pi, 361)[:-1]
    ff = pf.project_far_field(sim, res, {"xm": "-x", "xp": "+x", "ym": "-y", "yp": "+y"},
                              pf.polar_directions(phi), index=index)
    power = ff.intensity[0].sum() * (phi[1] - phi[0])
    assert power == pytest.approx(flux, rel=0.03)
