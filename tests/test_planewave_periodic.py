"""One-way plane-wave / Gaussian-beam sources and periodic boundaries."""
import numpy as np
import pytest

import photonfdtd as pf

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

LAM = 1e-6
F0 = pf.C_0 / LAM


def _wave(fwhm=8e-15):
    return pf.GaussianPulse(freq0=F0, fwhm=fwhm)


def _peak(w):
    t = np.linspace(0, 80e-15, 400001)
    return np.abs(w(t)).max()


def _run_1d(direction, n, ppw=60, periodic=False, run_time=None, amplitude=1.0):
    dx = LAM / ppw
    if periodic:
        g = pf.Grid(size=(396 * dx, 0, 0), cell_size=(dx, 0, 0),
                    periodic=(True, False, False))
    else:
        g = pf.Grid(size=(30e-6, 0, 0), cell_size=(dx, 0, 0), pml_layers=(20, 0, 0))
    st = ([pf.Box(center=(0, 0, 0), size=(1, 1, 1), medium=pf.Medium.from_index(n))]
          if n != 1 else [])
    src = pf.PlaneWave(center=(0, 0, 0), direction=direction, polarization="Ey",
                       waveform=_wave(), amplitude=amplitude, index=n)
    mon = pf.FieldMonitor(name="m", components=("Ey",), interval=1)
    sim = pf.Simulation(g, structures=st, sources=[src], monitors=[mon],
                        run_time=run_time or 100e-15 * n)
    E = np.asarray(sim.run().fields["m"]["Ey"]).reshape(sim.n_steps, -1)
    return sim, g.coords[0], E


@pytest.mark.parametrize("direction", ["+x", "-x"])
@pytest.mark.parametrize("n", [1.0, 1.5])
def test_plane_wave_one_way_and_calibrated(direction, n):
    """The Huygens sheet launches peak ``amplitude`` forward (to O((k dx)^2))
    and essentially nothing backward."""
    _, x, E = _run_1d(direction, n, amplitude=2.0)
    fwd = x > 8e-6 if direction == "+x" else x < -8e-6
    bwd = x < -8e-6 if direction == "+x" else x > 8e-6
    pf_ = np.abs(E[:, fwd]).max()
    assert abs(pf_ / (2.0 * _peak(_wave())) - 1) < 0.015
    assert np.abs(E[:, bwd]).max() / pf_ < 1e-4


def test_plane_wave_amplitude_converges_second_order():
    errs = []
    for ppw in (30, 60):
        _, x, E = _run_1d("+x", 1.5, ppw=ppw)
        errs.append(abs(np.abs(E[:, x > 8e-6]).max() / _peak(_wave()) - 1))
    assert 3.0 < errs[0] / errs[1] < 5.0


def test_plane_wave_validation():
    g = pf.Grid(size=(4e-6, 4e-6), cell_size=LAM / 20, pml_layers=(10, 0, 0))
    with pytest.raises(ValueError, match="transverse"):
        pf.Simulation(g, sources=[pf.PlaneWave(center=(0, 0), direction="+x",
                                               polarization="Ex", waveform=_wave())])
    with pytest.raises(ValueError, match="direction"):
        pf.Simulation(g, sources=[pf.PlaneWave(center=(0, 0), direction="x",
                                               polarization="Ez", waveform=_wave())])
    with pytest.raises(ValueError, match="dt"):
        pf.PlaneWave(center=(0, 0), direction="+x", polarization="Ez",
                     waveform=_wave()).expand(g)


def test_periodic_grid_validation():
    with pytest.raises(ValueError, match="periodic"):
        pf.Grid(size=(4e-6, 4e-6), cell_size=LAM / 20, pml_layers=(10, 10, 0),
                periodic=(False, True, False))
    g = pf.Grid(size=(4e-6, 4e-6, 0), cell_size=LAM / 20, periodic=True)
    assert g.periodic == (True, True, False)     # collapsed axis is never periodic


def test_periodic_ring_returns_pulse():
    """On a periodic 1-D ring a pulse comes back after one lap (exactly 400
    steps at S = 0.99 on 396 cells) and its energy is conserved."""
    sim, x, E = _run_1d("+x", 1.0, ppw=30, periodic=True,
                        run_time=1000 * (LAM / 30) / pf.C_0)
    assert np.isclose(396 * (LAM / 30) / (pf.C_0 * sim.dt), 400)
    n1 = 200
    assert np.abs(E[n1 + 400] - E[n1]).max() / np.abs(E[n1]).max() < 0.01
    assert np.isclose((E[n1 + 800] ** 2).sum(), (E[n1] ** 2).sum(), rtol=1e-4)


@pytest.mark.parametrize("dims", [2, 3])
def test_plane_wave_uniform_on_periodic_cell(dims):
    """A plane wave in a cell periodic on its transverse axes stays exactly
    uniform across them (no edge diffraction)."""
    dx = LAM / 20
    size = (6e-6, 1e-6, 0) if dims == 2 else (5e-6, 0.8e-6, 0.8e-6)
    cs = (dx, dx, 0) if dims == 2 else (dx, dx, dx)
    g = pf.Grid(size=size, cell_size=cs, pml_layers=(12, 0, 0),
                periodic=(False, True, dims == 3))
    pol = "Ez" if dims == 2 else "Ey"
    src = pf.PlaneWave(center=(-1.5e-6, 0, 0), direction="+x", polarization=pol,
                       waveform=_wave(6e-15))
    mon = pf.FieldMonitor(name="m", components=(pol,), interval=4)
    sim = pf.Simulation(g, sources=[src], monitors=[mon], run_time=30e-15)
    F = np.asarray(sim.run().fields["m"][pol]).reshape((-1,) + tuple(g.shape))
    spread = F.max(axis=(2, 3)) - F.min(axis=(2, 3))
    assert np.abs(F).max() > 0.5
    assert np.abs(spread).max() < 1e-12 * np.abs(F).max()


def test_periodic_slab_transmission_matches_fresnel():
    """Normal-incidence transmission through a half-space on a periodic cell:
    |t|^2 n2 / n1 = 4 n1 n2 / (n1 + n2)^2, from the flux of a one-way wave."""
    dx = LAM / 40
    n2 = 2.0
    g = pf.Grid(size=(12e-6, 0.5e-6, 0), cell_size=(dx, dx, 0),
                pml_layers=(30, 0, 0), periodic=(False, True, False))
    half = pf.Box(center=(4e-6, 0, 0), size=(8e-6, 1, 1),
                  medium=pf.Medium.from_index(n2))
    src = pf.PlaneWave(center=(-3e-6, 0, 0), direction="+x", polarization="Ez",
                       waveform=_wave(6e-15))
    fin = pf.FluxMonitor(name="in", plane_axis="x", plane_position=-2e-6)
    fout = pf.FluxMonitor(name="out", plane_axis="x", plane_position=2e-6)
    kw = dict(sources=[src], monitors=[fin, fout], run_time=90e-15)
    ref = pf.Simulation(g, **kw).run()
    res = pf.Simulation(g, structures=[half], **kw).run()
    T = float(res.flux["out"]) / float(ref.flux["in"])
    assert abs(T / (4 * n2 / (1 + n2) ** 2) - 1) < 0.01


def test_gaussian_beam_profile():
    """The Gaussian-beam sheet carries exp(-r^2/w0^2) across the plane."""
    dx = LAM / 20
    g = pf.Grid(size=(4e-6, 6e-6), cell_size=dx, pml_layers=(10, 10, 0))
    w0 = 1.2e-6
    src = pf.GaussianBeam(center=(-1e-6, 0.5e-6), direction="+x",
                          polarization="Ez", waveform=_wave(), waist_radius=w0)
    sim = pf.Simulation(g, sources=[src], run_time=1e-15)
    ez = [d for d in sim.sources if d.component == "Ez"]
    y = np.array([d.position[1] for d in ez])
    a = np.array([d.amplitude for d in ez])
    assert np.allclose(a / a.max(), np.exp(-(y - 0.5e-6) ** 2 / w0 ** 2), atol=0.02)


def test_reversible_gradient_on_periodic_cell():
    """The O(1)-memory reversible adjoint runs on a periodic (PML-free on the
    periodic axes, closed elsewhere) domain and matches the stored-trajectory
    gradient."""
    dx = LAM / 14
    g = pf.Grid(size=(2e-6, 1e-6, 0), cell_size=(dx, dx, 0),
                periodic=(False, True, False))
    box = pf.Box(center=(0, 0, 0), size=(0.6e-6, 0.4e-6, 1),
                 medium=pf.Medium.from_index(2.0))
    src = pf.PlaneWave(center=(-0.6e-6, 0, 0), direction="+x", polarization="Ez",
                       waveform=_wave(6e-15))
    mon = pf.DFTMonitor(name="d", components=("Ez",), freqs=[F0])
    sim = pf.Simulation(g, structures=[box], sources=[src], monitors=[mon],
                        run_time=40e-15)
    loss = lambda out: jnp.sum(jnp.abs(out["dft"]["d"]["Ez"]) ** 2)
    v_ref, g_ref = pf.jax_value_and_grad_eps(sim, loss, remat="none")
    v_rev, g_rev = pf.jax_value_and_grad_eps_reversible(sim, loss)
    assert np.isclose(v_ref, v_rev)
    assert np.abs(g_ref - g_rev).max() / np.abs(g_ref).max() < 1e-10


def test_reversible_pml_gradient_on_periodic_cell():
    """The grating/metasurface adjoint: periodic transversely, PML along the
    propagation axis, through the PML-shell reversible adjoint."""
    dx = LAM / 14
    g = pf.Grid(size=(3e-6, 1e-6, 0), cell_size=(dx, dx, 0),
                pml_layers=(8, 0, 0), periodic=(False, True, False))
    box = pf.Box(center=(0.2e-6, 0, 0), size=(0.6e-6, 0.4e-6, 1),
                 medium=pf.Medium.from_index(2.0))
    src = pf.PlaneWave(center=(-0.6e-6, 0, 0), direction="+x", polarization="Ez",
                       waveform=_wave(6e-15))
    mon = pf.DFTMonitor(name="d", components=("Ez",), freqs=[F0],
                        plane_axis="x", plane_position=0.8e-6)
    sim = pf.Simulation(g, structures=[box], sources=[src], monitors=[mon],
                        run_time=40e-15)
    loss = lambda out: jnp.sum(jnp.abs(out["dft"]["d"]["Ez"]) ** 2)
    v_ref, g_ref = pf.jax_value_and_grad_eps(sim, loss, remat="none")
    v_rev, g_rev = pf.jax_value_and_grad_eps_reversible_pml(sim, loss)
    assert np.isclose(v_ref, v_rev)
    assert np.abs(g_ref - g_rev).max() / np.abs(g_ref).max() < 1e-9


def test_tiled_refuses_periodic():
    g = pf.Grid(size=(2e-6, 1e-6), cell_size=LAM / 14, periodic=(False, True, False))
    sim = pf.Simulation(g, run_time=5e-15)
    with pytest.raises(NotImplementedError, match="periodic"):
        sim.run(disk_tiled=True)


def test_source_table_is_separable():
    """A plane-wave sheet stores one waveform per sheet, not one per cell."""
    from photonfdtd.jaxbackend import _build_static
    g = pf.Grid(size=(3e-6, 2e-6, 2e-6), cell_size=LAM / 20, pml_layers=(8, 0, 0),
                periodic=(False, True, True))
    src = pf.PlaneWave(center=(0, 0, 0), direction="+x", polarization="Ey",
                       waveform=_wave())
    sim = pf.Simulation(g, sources=[src], run_time=20e-15)
    st = _build_static(sim)
    for comp, I, J, K, tab in st["src_E"] + st["src_H"]:
        assert tab.W.shape == (sim.n_steps, 1)
        assert tab.n_src == g.shape[1] * g.shape[2]
        assert np.allclose(tab.dense()[7], tab[7])
