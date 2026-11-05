"""JAX engine: regression against golden values from the retired NumPy core,
plus differentiability.
"""
import numpy as np
import pytest

import photonfdtd as pf
from _golden import assert_golden

import jax
import jax.numpy as jnp

# Golden fingerprints (see _golden.py), keyed case -> output -> fingerprint.
GOLDEN = {
    "core_3D": {
        "Ex": [2.3790322493203195, 0.08574770175790544],
        "Ey": [2.7990120635328175, -2.9760315377197313],
        "Ez": [2.8097465163561797, -0.6609005570929383],
        "Hx": [0.0036432965025395307, 0.0015943895971755136],
        "Hy": [0.002541258826352217, -0.00016123611648570063],
        "Hz": [0.002542547120480419, 0.0002472295528734639],
        "dft_Ez": [3.978601132396091e-15, -1.5232747870322774e-15, 4.2665161149418196e-15, -2.680172624226342e-16],
        "dft_Hx": [5.511213677274633e-18, -2.6471382936250085e-18, 5.2383244299485196e-18, -5.524142285467087e-18],
    },
    "core_2D-xy": {
        "Ex": [2.499036241888193, 0.1828705326497246],
        "Ey": [3.4771790423306004, 3.114806499739099],
        "Ez": [2.5705918321080183, -2.4231585346993105],
        "Hx": [0.005449959472299247, 0.004643215947461147],
        "Hy": [0.005785729760597819, -0.0006828183646737353],
        "Hz": [0.005796003276885297, -0.00019750029104669762],
        "dft_Ez": [3.762825805314455e-15, 3.436236301368369e-15, 3.791307102062782e-15, -4.035876054442532e-15],
        "dft_Hx": [8.339126715726948e-18, 1.1522300957951674e-17, 7.839980649123509e-18, -2.951028659085948e-17],
    },
    "core_2D-xz": {
        "Ex": [2.462710717607479, 3.1364804927386727],
        "Ey": [2.3547178021988855, -0.9185826140636251],
        "Ez": [3.442969895047212, -0.32856565660013703],
        "Hx": [0.0049580840035622015, 0.0026737562285275686],
        "Hy": [0.005578854672333303, 0.007044554814282675],
        "Hz": [0.005552695224396404, 0.004012909768851156],
        "dft_Ez": [4.9311367319035905e-15, 2.2405386892156744e-16, 5.191513219078655e-15, 6.536371513630602e-16],
        "dft_Hx": [7.601152097919333e-18, -8.887882108072324e-19, 7.103877972096134e-18, 2.0448707074680765e-18],
    },
    "core_1D": {
        "Ex": [0.0, 0.0],
        "Ey": [5.224693503510797, -0.034827528995645396],
        "Ez": [5.2246932960209005, -0.009357794125038787],
        "Hx": [0.0, 0.0],
        "Hy": [0.013869288787439355, -4.1858019502282796e-05],
        "Hz": [0.013869289444399063, 4.359951346705291e-05],
        "dft_Ez": [8.059484715031908e-15, 2.0181223427766505e-15, 7.713555045357042e-15, 5.542975049325677e-15],
        "dft_Hx": [0.0, 0.0, 0.0, 0.0],
    },
    # The flux entries are re-recorded from the JAX engine after FluxMonitor
    # began averaging tangential H onto the E plane in space and time (the
    # NumPy core paired E^n with H^{n+1/2} at the next cell, which biased
    # one-sided flux by O(dx)). The Ez entries are unchanged.
    "flux_float64": {
        "flux": [3.9053139687874964e-26, 3.9053139687874964e-26],
        "Ez": [2.36222298190397, 0.47592662919627327],
    },
    "flux_float32": {
        "flux": [3.905313535120555e-26, 3.905313535120555e-26],
        "Ez": [2.362223083707841, 0.4759267158782559],
    },
    "cherenkov_2D": {
        "Ex": [8577592751.526555, -2859183.3721070215],
        "Ey": [5688449131.542524, 7247399.802071292],
        "Hz": [49453573.248694524, -88749.73641461255],
    },
    "cherenkov_3D": {
        "Ex": [1.2937563520598902e+16, -193899503912603.22],
        "Ey": [6667149287177222.0, 79623868167619.5],
        "Hz": [54870180434555.85, 313707299902.9114],
    },
    "mode_source": {
        "Ez": [22.670078008666177, 0.015239062027684868],
        "Hy": [0.05491677226961353, -2.271646498229857e-05],
    },
}


def _build(size, cell, pml, precision="float64", run_time=30e-15):
    freq0 = pf.C_0 / 1.0e-6
    wf = pf.GaussianPulse(freq0=freq0, fwhm=6e-15)
    pos1 = tuple(0.1e-6 if s > 0 else 0.0 for s in size)
    pos2 = tuple(-0.15e-6 if s > 0 else 0.0 for s in size)
    srcs = [pf.PointDipole(position=pos1, component="Ez", waveform=wf),
            pf.PointDipole(position=pos2, component="Ey", waveform=wf)]
    mons = [pf.FieldMonitor(name="fm",
                            components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
                            interval=5),
            pf.DFTMonitor(name="dm", components=("Ez", "Hx"),
                          freqs=[0.9 * freq0, freq0], interval=1)]
    return pf.Simulation(pf.Grid(size=size, cell_size=cell, pml_layers=pml),
                         sources=srcs, monitors=mons, run_time=run_time,
                         precision=precision)


CORE_CASES = {
    "3D": ((1.6e-6, 1.3e-6, 1.1e-6), 80e-9, (4, 4, 4)),
    "2D-xy": ((1.8e-6, 1.4e-6), 80e-9, (5, 5, 0)),
    "2D-xz": ((1.8e-6, 0, 1.2e-6), 80e-9, (5, 0, 5)),
    "1D": ((1.8e-6, 0, 0), 80e-9, (6, 0, 0)),
}


def case_core(label):
    r = _build(*CORE_CASES[label]).run()
    out = {c: r.fields["fm"][c] for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
    out.update({"dft_" + c: r.dft["dm"][c] for c in ("Ez", "Hx")})
    return out


@pytest.mark.parametrize("label", list(CORE_CASES))
def test_jax_matches_golden(label):
    for k, v in case_core(label).items():
        assert_golden(v, GOLDEN["core_" + label][k], 1e-8, f"{label} {k}")


def case_flux(prec):
    freq0 = pf.C_0 / 1.0e-6
    wf = pf.GaussianPulse(freq0=freq0, fwhm=6e-15)
    grid = pf.Grid(size=(1.8e-6, 1.4e-6), cell_size=80e-9, pml_layers=(5, 5, 0))
    src = pf.PointDipole(position=(0, 0), component="Ez", waveform=wf)
    mons = [pf.FluxMonitor(name="flux", plane_axis="x", plane_position=0.4e-6),
            pf.FieldMonitor(name="fm", components=("Ez",), interval=6)]
    r = pf.Simulation(grid, sources=[src], monitors=mons, run_time=40e-15,
                      precision=prec).run()
    return {"flux": np.array([r.flux["flux"]]), "Ez": r.fields["fm"]["Ez"]}


@pytest.mark.parametrize("prec,tol", [("float64", 1e-8), ("float32", 3e-4)])
def test_jax_flux_and_precision(prec, tol):
    for k, v in case_flux(prec).items():
        assert_golden(v, GOLDEN["flux_" + prec][k], tol, f"{prec} {k}")


def test_jax_grad_matches_finite_difference():
    """jax.grad of a monitor-based loss w.r.t. eps_r is a correct adjoint."""
    freq0 = pf.C_0 / 1.0e-6

    def make():
        grid = pf.Grid(size=(1.2e-6, 1.0e-6), cell_size=100e-9, pml_layers=(3, 3, 0))
        wf = pf.GaussianPulse(freq0=freq0, fwhm=6e-15)
        src = pf.PointDipole(position=(-0.2e-6, 0), component="Ez", waveform=wf)
        mon = pf.FieldMonitor(name="fm", components=("Ez",), interval=4)
        return pf.Simulation(grid, sources=[src], monitors=[mon],
                             run_time=24e-15)

    def loss(out):
        return jnp.sum(out["fields"]["fm"]["Ez"] ** 2)

    sim = make()
    val, grad = pf.jax_value_and_grad_eps(sim, loss)
    assert np.isfinite(val) and np.isfinite(grad).all()

    base = sim.eps_r.copy()

    def loss_only(eps):
        s = make(); s.eps_r = eps
        r = s.run()
        return float(np.sum(r.fields["fm"]["Ez"].astype(np.float64) ** 2))

    h = 1e-3
    for (ci, cj) in [(6, 5), (4, 7), (8, 3)]:
        ep = base.copy(); ep[ci, cj, 0] += h
        em = base.copy(); em[ci, cj, 0] -= h
        fd = (loss_only(ep) - loss_only(em)) / (2 * h)
        an = float(grad[ci, cj, 0])
        assert abs(an - fd) / (abs(fd) + 1e-30) < 1e-5, (ci, cj, an, fd)


def case_cherenkov(ndim):
    n = 2.0
    if ndim == 2:
        Lx, Ly, dx = 6e-6, 4.5e-6, 60e-9
        grid = pf.Grid(size=(Lx, Ly), cell_size=dx, pml_layers=(10, 10, 0))
        med = pf.Box(center=(0, 0), size=(3 * Lx, 3 * Ly),
                     medium=pf.Medium.from_index(n))
        p = pf.ChargedParticle(charge=2e-9, velocity=(0.9 * pf.C_0, 0.0),
                               start=(-2e-6, 0.0))
        rt = 24e-15
    else:
        L, dx = 2.2e-6, 70e-9
        grid = pf.Grid(size=(L, L, L), cell_size=dx, pml_layers=(6, 6, 6))
        med = pf.Box(center=(0, 0, 0), size=(3 * L, 3 * L, 3 * L),
                     medium=pf.Medium.from_index(n))
        p = pf.ChargedParticle(charge=2e-9, velocity=(0.9 * pf.C_0, 0.0, 0.0),
                               start=(-0.8e-6, 0.0, 0.0))
        rt = 12e-15
    assert n * 0.9 > 1.0                        # above Cherenkov threshold
    mon = pf.FieldMonitor(name="s", components=("Ex", "Ey", "Hz"), interval=6)
    r = pf.Simulation(grid, structures=[med], sources=[p], monitors=[mon],
                      run_time=rt).run()
    return {c: r.fields["s"][c] for c in ("Ex", "Ey", "Hz")}


@pytest.mark.parametrize("ndim", [2, 3])
def test_jax_cherenkov_matches_golden(ndim):
    """Moving-charge (Cherenkov) current injection."""
    for k, v in case_cherenkov(ndim).items():
        assert_golden(v, GOLDEN[f"cherenkov_{ndim}D"][k], 1e-8, f"{ndim}D {k}")


def case_mode_source():
    freq0 = pf.C_0 / 1.0e-6
    grid = pf.Grid(size=(2.4e-6, 2.0e-6), cell_size=60e-9, pml_layers=(8, 8, 0))
    yc = np.linspace(-0.6e-6, 0.6e-6, 13)
    prof = np.exp(-(yc / 0.4e-6) ** 2)
    src = pf.ModeSource(center=(-0.6e-6, 0.0), size=(0.0, 1.2e-6),
                        component="Ez", waveform=pf.GaussianPulse(freq0=freq0,
                        fwhm=6e-15), profile=prof, profile_coords=[yc])
    mon = pf.FieldMonitor(name="s", components=("Ez", "Hy"), interval=6)
    r = pf.Simulation(grid, sources=[src], monitors=[mon], run_time=30e-15).run()
    return {c: r.fields["s"][c] for c in ("Ez", "Hy")}


def test_jax_mode_source():
    """An expandable source (ModeSource -> PointDipoles)."""
    for k, v in case_mode_source().items():
        assert_golden(v, GOLDEN["mode_source"][k], 1e-8, k)
