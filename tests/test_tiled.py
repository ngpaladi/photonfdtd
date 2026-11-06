"""Disk-tiled forward stepping (photonfdtd.tiled) against the in-memory engine."""
import os
import warnings

import numpy as np
import pytest

import photonfdtd as pf
from photonfdtd.tiled import run_tiled, value_and_grad_eps_tiled

F0 = pf.C_0 / 1e-6
COMPS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


LORENTZ = pf.DispersiveMedium.lorentz(2.0, [(1.5, 1.4 * F0, 0.1 * F0),
                                           (0.4, 0.7 * F0, 0.0)])
DRUDE = pf.DispersiveMedium.drude(1.5, 1.2 * F0, 0.05 * F0)


def _build(ndim=3, particle=False, subpixel=False, precision="float64",
           dispersive=False, run_time=25e-15):
    wf = pf.GaussianPulse(freq0=F0, fwhm=6e-15)
    if ndim == 3:
        grid = pf.Grid(size=(1.84e-6, 1.2e-6, 0.96e-6), cell_size=80e-9,
                       pml_layers=(4, 4, 4))
        srcs = [pf.PointDipole(position=(0.1e-6, 0.05e-6, 0), component="Ez", waveform=wf),
                pf.PointDipole(position=(-0.3e-6, 0, 0.1e-6), component="Hy", waveform=wf)]
        box = pf.Box(center=(0.2e-6, 0, 0), size=(0.5e-6, 0.37e-6, 0.3e-6),
                     medium=pf.Medium.from_index(2.0))
        v, start = (0.8 * pf.C_0, 0.1 * pf.C_0, 0), (-0.7e-6, 0, 0)
    else:
        grid = pf.Grid(size=(2.4e-6, 1.6e-6), cell_size=80e-9, pml_layers=(5, 5, 0))
        srcs = [pf.PointDipole(position=(0.1e-6, 0.05e-6), component="Ez", waveform=wf),
                pf.PointDipole(position=(-0.3e-6, 0.1e-6), component="Ey", waveform=wf)]
        box = pf.Box(center=(0.2e-6, 0), size=(0.5e-6, 0.37e-6),
                     medium=pf.Medium.from_index(2.0))
        v, start = (0.8 * pf.C_0, 0.1 * pf.C_0), (-0.9e-6, 0)
    structures = [box]
    if dispersive:
        # Two dispersive media (three poles) and a plain box stamped over part
        # of one, so the dispersive cells are not one block.
        c, sz = box.center, box.size
        structures = [pf.Box(center=c, size=sz, medium=LORENTZ),
                      pf.Box(center=(-c[0],) + tuple(c[1:]),
                             size=tuple(0.6 * a for a in sz), medium=DRUDE),
                      pf.Box(center=c, size=tuple(0.4 * a for a in sz),
                             medium=pf.Medium.from_index(1.7))]
    if particle:
        srcs.append(pf.ChargedParticle(charge=1e-12, velocity=v, start=start))
    mons = [pf.FieldMonitor(name="fm", components=COMPS, interval=3),
            pf.DFTMonitor(name="dv", components=("Ez", "Hx"), freqs=[0.9 * F0, F0],
                          downsample=2),
            pf.DFTMonitor(name="dx", components=COMPS, freqs=[F0], downsample=2,
                          plane_axis="x", plane_position=0.3e-6),
            pf.DFTMonitor(name="dy", components=COMPS, freqs=[F0], downsample=2,
                          plane_axis="y", plane_position=-0.2e-6),
            pf.FieldMonitor(name="ft", components=("Ez", "Hy"),
                            times=[3e-15, 11e-15, 11.05e-15, 20e-15]),
            pf.FluxMonitor(name="flx", plane_axis="x", plane_position=0.4e-6),
            pf.FluxMonitor(name="fly", plane_axis="y", plane_position=0.3e-6)]
    if ndim == 3:
        mons += [pf.FieldMonitor(name="fmz", components=("Ez",), interval=2,
                                 downsample=2, plane_z=0.1e-6),
                 pf.FluxMonitor(name="flz", plane_axis="z", plane_position=-0.2e-6)]
    return pf.Simulation(grid, structures=structures, sources=srcs, monitors=mons,
                         run_time=run_time, subpixel=subpixel, precision=precision)


def _tol(sim):
    """Tiled and in-memory runs differ only by XLA's reassociation of the same
    arithmetic: measured up to ~90 eps (CPU) and ~10 eps (GPU)."""
    return 128 * np.finfo(sim.dtypes["Ex"]).eps


def _worst(a, b):
    worst = 0.0
    for store in ("fields", "dft"):
        for n, d in getattr(a, store).items():
            for c, x in d.items():
                y = getattr(b, store)[n][c]
                assert x.shape == y.shape and x.dtype == y.dtype, (n, c)
                assert np.abs(x).max() > 0, (n, c)
                worst = max(worst, np.abs(x - y).max() / np.abs(x).max())
    for n, x in a.flux.items():
        assert x != 0.0
        worst = max(worst, abs(x - b.flux[n]) / abs(x))
    return worst


@pytest.mark.parametrize("tile,steps", [(5, 3), (1, 1), (2, 4), (40, 2)])
def test_tiled_matches_in_memory(tile, steps):
    """Slabs narrower than the halo, wider than the grid, and in between."""
    sim = _build()
    ref = sim.run(disk_tiled=False)
    out = _build().run(disk_tiled=True, tile_cells=tile, steps_per_tile=steps)
    assert _worst(ref, out) < _tol(sim)


@pytest.mark.parametrize("ndim,subpixel", [(3, True), (2, False)])
def test_tiled_particle_and_subpixel(ndim, subpixel):
    sim = _build(ndim, particle=True, subpixel=subpixel)
    ref = sim.run()
    out = _build(ndim, particle=True, subpixel=subpixel).run(
        disk_tiled=True, tile_cells=4, steps_per_tile=5)
    assert _worst(ref, out) < _tol(sim)


def test_tiled_1d():
    wf = pf.GaussianPulse(freq0=F0, fwhm=6e-15)
    grid = pf.Grid(size=(3e-6, 0, 0), cell_size=50e-9, pml_layers=(8, 0, 0))

    def build():
        mons = [pf.FieldMonitor(name="fm", components=("Ey", "Hz"), interval=4),
                pf.DFTMonitor(name="d", components=("Ey", "Hz"),
                              freqs=[F0], plane_axis="x", plane_position=0.6e-6),
                pf.FluxMonitor(name="f", plane_axis="x", plane_position=0.6e-6)]
        return pf.Simulation(grid, sources=[pf.PointDipole(
            position=(-0.2e-6, 0, 0), component="Ey", waveform=wf)],
            monitors=mons, run_time=30e-15)
    sim = build()
    ref = sim.run()
    out = build().run(disk_tiled=True, tile_cells=7, steps_per_tile=6)
    assert _worst(ref, out) < _tol(sim)


def test_tiled_float32():
    sim = _build(precision="float32")
    ref = sim.run()
    out = _build(precision="float32").run(disk_tiled=True, tile_cells=6,
                                          steps_per_tile=4)
    assert out.fields["fm"]["Ez"].dtype == np.float32
    assert _worst(ref, out) < _tol(sim)


def test_tiled_cleans_workdir(tmp_path):
    sim = _build(2)
    sim.run(disk_tiled=True, tile_cells=8, steps_per_tile=4, workdir=str(tmp_path))
    assert os.listdir(tmp_path) == []


@pytest.mark.parametrize("ndim,tile,steps", [(3, 5, 3), (3, 1, 2), (2, 3, 6)])
def test_tiled_dispersion(ndim, tile, steps):
    """Lossy and lossless poles in two media, with a moving charge, across
    slabs narrower and wider than the halo."""
    sim = _build(ndim, particle=True, dispersive=True)
    assert sim._has_dispersion
    ref = sim.run()
    out = _build(ndim, particle=True, dispersive=True).run(
        disk_tiled=True, tile_cells=tile, steps_per_tile=steps)
    assert _worst(ref, out) < _tol(sim)


def test_auto_fallback_warns(monkeypatch):
    """A forward run that does not fit the device falls back to tiling."""
    import photonfdtd.memory as mem
    monkeypatch.setattr(mem, "available_memory",
                        lambda: (pf.estimate_memory(_build(2))["forward"] - 1, "cpu"))
    sim = _build(2)
    ref = sim.run(disk_tiled=False)
    with pytest.warns(RuntimeWarning, match="disk-tiled"):
        out = _build(2).run()
    assert _worst(ref, out) < _tol(sim)


# --------------------------------------------------------------------------- #
# Adjoint.
# --------------------------------------------------------------------------- #
import jax.numpy as jnp  # noqa: E402

# One loss per kind of monitor output, so each output's cotangent path is
# checked on its own (a sum would let the largest term hide the others).
LOSSES = {
    "dft_plane_x": lambda o: jnp.sum(jnp.abs(o["dft"]["dx"]["Ey"]) ** 2),
    "dft_plane_y": lambda o: jnp.sum(o["dft"]["dy"]["Hz"].imag),
    "dft_volume": lambda o: jnp.sum(jnp.abs(o["dft"]["dv"]["Ez"]) ** 2),
    "flux": lambda o: o["flux"]["flx"] - 2.0 * o["flux"]["fly"],
    "fields": lambda o: jnp.sum(o["fields"]["fm"]["Hx"] ** 2)
    + jnp.sum(o["fields"]["ft"]["Ez"]),
}


def _rel(a, b):
    return np.abs(a - b).max() / np.abs(a).max()


@pytest.mark.parametrize("term", sorted(LOSSES))
def test_tiled_gradient_matches_in_memory(term):
    """The tiled adjoint equals the in-memory reverse pass (2-D, dispersive
    media and a moving charge), for every kind of monitor output."""
    kw = dict(ndim=2, particle=True, dispersive=True, run_time=15e-15)
    loss = LOSSES[term]
    v0, g0 = pf.jax_value_and_grad_eps(_build(**kw), loss, remat="none")
    v1, g1 = value_and_grad_eps_tiled(_build(**kw), loss, tile_cells=5,
                                      steps_per_tile=3)
    assert g1.shape == g0.shape and np.abs(g0).max() > 0
    assert abs(v1 - v0) <= 1e-13 * abs(v0)
    assert _rel(g0, g1) < 1e-12


@pytest.mark.parametrize("tile,steps,every", [(2, 4, 1), (40, 2, 3), (3, 3, None)])
def test_tiled_gradient_3d(tile, steps, every):
    """3-D, every monitor output at once, with checkpointing at every chunk,
    every third chunk and the default sqrt schedule; slabs narrower than the
    halo and wider than the grid."""
    kw = dict(ndim=3, particle=True, dispersive=True, run_time=12e-15)

    def loss(o):
        return sum(f(o) / abs(float(f(ref))) for f in LOSSES.values())
    ref = _out(_build(**kw).run())
    v0, g0 = pf.jax_value_and_grad_eps(_build(**kw), loss, remat="none")
    v1, g1 = value_and_grad_eps_tiled(_build(**kw), loss, tile_cells=tile,
                                      steps_per_tile=steps, checkpoint_every=every)
    assert abs(v1 - v0) <= 1e-13 * abs(v0)
    assert _rel(g0, g1) < 1e-12


def _out(res):
    from photonfdtd.tiled import _out_jax
    return _out_jax(res)


def test_tiled_gradient_schedules_agree_exactly():
    """Recomputing chunks from checkpoints is exact: the default schedule and
    keeping every chunk's state give the same gradient bit for bit."""
    kw = dict(ndim=2, dispersive=True, run_time=10e-15)
    loss = LOSSES["dft_volume"]
    _, ga = value_and_grad_eps_tiled(_build(**kw), loss, tile_cells=6,
                                     steps_per_tile=3)
    _, gb = value_and_grad_eps_tiled(_build(**kw), loss, tile_cells=6,
                                     steps_per_tile=3, checkpoint_every=1)
    assert np.array_equal(ga, gb)


def test_tiled_gradient_rejects_subpixel():
    with pytest.raises(NotImplementedError, match="subpixel"):
        value_and_grad_eps_tiled(_build(2, subpixel=True), LOSSES["flux"],
                                 tile_cells=4)


def test_tiled_gradient_cleans_workdir(tmp_path):
    value_and_grad_eps_tiled(_build(2, run_time=6e-15), LOSSES["flux"],
                             tile_cells=8, steps_per_tile=2, workdir=str(tmp_path))
    assert os.listdir(tmp_path) == []
