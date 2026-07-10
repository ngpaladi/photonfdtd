"""JAX backend: parity with the in-core solver + differentiability.

Skipped entirely when jax is not installed.
"""
import numpy as np
import pytest

import photonfdtd as pf

jax = pytest.importorskip("jax")
import jax.numpy as jnp   # noqa: E402


def _build(size, cell, pml, use_jax, precision="float64", run_time=30e-15):
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
                         use_jax=use_jax, precision=precision)


@pytest.mark.parametrize("label,size,cell,pml", [
    ("3D", (1.6e-6, 1.3e-6, 1.1e-6), 80e-9, (4, 4, 4)),
    ("2D-xy", (1.8e-6, 1.4e-6), 80e-9, (5, 5, 0)),
    ("2D-xz", (1.8e-6, 0, 1.2e-6), 80e-9, (5, 0, 5)),
    ("1D", (1.8e-6, 0, 0), 80e-9, (6, 0, 0)),
])
def test_jax_matches_in_core(label, size, cell, pml):
    ref = _build(size, cell, pml, use_jax=False).run()
    jx = _build(size, cell, pml, use_jax=True).run()
    escale = max(np.abs(ref.fields["fm"][c]).max() for c in ("Ex", "Ey", "Ez"))
    hscale = max(np.abs(ref.fields["fm"][c]).max() for c in ("Hx", "Hy", "Hz"))
    for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        scale = hscale if c[0] == "H" else escale
        rel = np.abs(ref.fields["fm"][c] - jx.fields["fm"][c]).max() / scale
        assert rel < 1e-9, f"{label} {c}: field rel={rel:.2e}"
    for c in ("Ez", "Hx"):
        a = ref.dft["dm"][c]; b = jx.dft["dm"][c]
        rel = np.abs(a - b).max() / (np.abs(a).max() + 1e-300)
        assert rel < 1e-9, f"{label} DFT {c}: rel={rel:.2e}"


def test_jax_flux_and_float32():
    freq0 = pf.C_0 / 1.0e-6
    wf = pf.GaussianPulse(freq0=freq0, fwhm=6e-15)

    def build(prec):
        grid = pf.Grid(size=(1.8e-6, 1.4e-6), cell_size=80e-9, pml_layers=(5, 5, 0))
        src = pf.PointDipole(position=(0, 0), component="Ez", waveform=wf)
        mons = [pf.FluxMonitor(name="flux", plane_axis="x", plane_position=0.4e-6),
                pf.FieldMonitor(name="fm", components=("Ez",), interval=6)]
        return grid, src, mons

    for prec, tol in [("float64", 1e-6), ("float32", 3e-4)]:
        g, s, m = build(prec)
        ref = pf.Simulation(g, sources=[s], monitors=m, run_time=40e-15,
                            precision=prec).run()
        g, s, m = build(prec)
        jx = pf.Simulation(g, sources=[s], monitors=m, run_time=40e-15,
                           precision=prec, use_jax=True).run()
        frel = abs(ref.flux["flux"] - jx.flux["flux"]) / abs(ref.flux["flux"])
        mrel = (np.abs(ref.fields["fm"]["Ez"] - jx.fields["fm"]["Ez"]).max()
                / np.abs(ref.fields["fm"]["Ez"]).max())
        assert frel < tol and mrel < tol, (prec, frel, mrel)


def test_jax_grad_matches_finite_difference():
    """jax.grad of a monitor-based loss w.r.t. eps_r is a correct adjoint."""
    freq0 = pf.C_0 / 1.0e-6

    def make():
        grid = pf.Grid(size=(1.2e-6, 1.0e-6), cell_size=100e-9, pml_layers=(3, 3, 0))
        wf = pf.GaussianPulse(freq0=freq0, fwhm=6e-15)
        src = pf.PointDipole(position=(-0.2e-6, 0), component="Ez", waveform=wf)
        mon = pf.FieldMonitor(name="fm", components=("Ez",), interval=4)
        return pf.Simulation(grid, sources=[src], monitors=[mon],
                             run_time=24e-15, use_jax=True)

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


@pytest.mark.parametrize("ndim", [2, 3])
def test_jax_cherenkov_matches_in_core(ndim):
    """Moving-charge (Cherenkov) current injection matches the in-core solver."""
    n = 2.0

    def build(use_jax):
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
        mon = pf.FieldMonitor(name="s", components=("Ex", "Ey", "Hz"), interval=6)
        return pf.Simulation(grid, structures=[med], sources=[p], monitors=[mon],
                             run_time=rt, use_jax=use_jax)

    assert n * 0.9 > 1.0                        # above Cherenkov threshold
    ref = build(False).run()
    jx = build(True).run()
    escale = max(np.abs(ref.fields["s"][c]).max() for c in ("Ex", "Ey"))
    for c in ("Ex", "Ey", "Hz"):
        scale = escale if c[0] == "E" else np.abs(ref.fields["s"]["Hz"]).max()
        rel = np.abs(ref.fields["s"][c] - jx.fields["s"][c]).max() / (scale + 1e-300)
        assert rel < 1e-9, f"{ndim}D Cherenkov {c}: rel={rel:.2e}"


def test_jax_mode_source():
    """An expandable source (ModeSource -> PointDipoles) runs under JAX."""
    freq0 = pf.C_0 / 1.0e-6

    def build(use_jax):
        grid = pf.Grid(size=(2.4e-6, 2.0e-6), cell_size=60e-9, pml_layers=(8, 8, 0))
        yc = np.linspace(-0.6e-6, 0.6e-6, 13)
        prof = np.exp(-(yc / 0.4e-6) ** 2)
        src = pf.ModeSource(center=(-0.6e-6, 0.0), size=(0.0, 1.2e-6),
                            component="Ez", waveform=pf.GaussianPulse(freq0=freq0,
                            fwhm=6e-15), profile=prof, profile_coords=[yc])
        mon = pf.FieldMonitor(name="s", components=("Ez", "Hy"), interval=6)
        return pf.Simulation(grid, sources=[src], monitors=[mon],
                             run_time=30e-15, use_jax=use_jax)

    ref = build(False).run()
    jx = build(True).run()
    for c in ("Ez", "Hy"):
        a = ref.fields["s"][c]; b = jx.fields["s"][c]
        assert np.abs(a - b).max() / (np.abs(a).max() + 1e-300) < 1e-9


def test_jax_rejects_unsupported():
    grid = pf.Grid(size=(1.2e-6, 1.0e-6), cell_size=100e-9, pml_layers=(3, 3, 0))
    # compressed monitor is host-side, not part of the device path
    m = pf.FieldMonitor(name="c", components=("Ez",), compression=True)
    with pytest.raises(NotImplementedError):
        pf.Simulation(grid, monitors=[m], run_time=5e-15, use_jax=True).run()
