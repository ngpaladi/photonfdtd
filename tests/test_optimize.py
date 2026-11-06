"""Geometry optimization and chip training: design regions, parametric shapes,
the compile-once gradient, the optimizer loop, robustness, GDS export, and
training a chip on a dataset through its transfer matrix."""
import os

import numpy as np
import pytest

import photonfdtd as pf

import jax
import jax.numpy as jnp   # noqa: E402
jax.config.update("jax_enable_x64", True)

LAM = 1.55e-6
F0 = pf.C_0 / LAM
DX = 80e-9
CORE = pf.Medium.from_index(3.0)
PORTS_Y = (-0.4e-6, 0.4e-6)


def _sim(amps=(1.0, 0.0), freqs=(F0,), run_time=40e-15):
    """2-D chip: two input stubs on the left, two output stubs on the right, a
    design box in between; dipoles in the input stubs scaled by ``amps``."""
    grid = pf.Grid(size=(3.2e-6, 2.4e-6), cell_size=DX, pml_layers=(6, 6, 0))
    wgs = [pf.Box(center=(sx, y), size=(1.12e-6, 0.32e-6), medium=CORE)
           for sx in (-1.2e-6, 1.2e-6) for y in PORTS_Y]
    wf = pf.GaussianPulse(freq0=F0, fwhm=6e-15)
    srcs = [pf.PointDipole(position=(-1.3e-6, y), component="Ez", waveform=wf,
                           amplitude=a) for y, a in zip(PORTS_Y, amps) if a != 0]
    mon = pf.DFTMonitor(name="out", components=("Ez",), freqs=list(freqs),
                        plane_axis="x", plane_position=1.2e-6)
    return pf.Simulation(grid, structures=wgs, sources=srcs, monitors=[mon],
                         run_time=run_time)


def _region(**kw):
    kw.setdefault("filter_radius", 160e-9)
    kw.setdefault("beta", 4.0)
    return pf.DesignRegion(center=(0, 0), size=(1.2e-6, 1.6e-6), core_z=(-1, 1),
                           eps_solid=CORE.eps_r, eps_void=1.0, **kw)


def _readouts(sim):
    y = np.asarray(sim.grid.coords[1])
    return [pf.FieldProjection("out", "Ez",
                               np.exp(-((y - yc) / 0.25e-6) ** 2)[None, :, None])
            for yc in PORTS_Y]


def _rho(reg, sim, seed=0):
    return jnp.asarray(np.random.default_rng(seed).uniform(
        0.3, 0.7, reg.shape(sim.grid)))


# --------------------------------------------------------------------------- #
# Design region
# --------------------------------------------------------------------------- #

def test_region_overlay_keeps_layout_outside():
    sim = _sim()
    reg = _region()
    i0, i1, j0, j1, _ = reg.cells(sim.grid)
    ce = reg.ce_builder(sim)(_rho(reg, sim))
    base = sim.dt / (np.asarray(sim.eps_r) * pf.EPS_0)
    outside = np.ones(sim.grid.shape, bool)
    outside[i0:i1, j0:j1, :] = False
    for c in ("Ex", "Ey", "Ez"):
        assert np.array_equal(np.asarray(ce[c])[outside], base[outside])


def test_region_solid_matches_stamped_box():
    """rho = 1 with no filter/projection reproduces a stamped solid block."""
    sim = _sim()
    reg = _region(filter_radius=0.0, beta=0.0)
    ce = reg.ce_builder(sim)(jnp.ones(reg.shape(sim.grid)))
    block = pf.Box(center=(0, 0), size=(1.2e-6, 1.6e-6), medium=CORE)
    ref = pf.Simulation(sim.grid, structures=list(sim.structures) + [block],
                        run_time=1e-15)
    want = sim.dt / (np.asarray(ref.eps_r) * pf.EPS_0)
    for c in ("Ex", "Ey", "Ez"):
        np.testing.assert_allclose(np.asarray(ce[c]), want, rtol=1e-14)


def test_region_padding_joins_waveguide_stubs():
    """The filter sees the fixed stubs around the box: with rho = 0 the
    physical density next to a stub is pulled up, elsewhere it stays ~0."""
    sim = _sim()
    reg = _region(beta=0.0)
    rp = np.asarray(reg.rho_phys(sim, jnp.zeros(reg.shape(sim.grid))))
    _, ys = reg.coords(sim.grid)
    j_stub = int(np.argmin(np.abs(ys - PORTS_Y[1])))
    j_mid = int(np.argmin(np.abs(ys)))
    assert rp[0, j_stub] > 0.2          # stub enters from the left edge
    assert rp[0, j_mid] < 1e-12         # no stub there
    assert rp[rp.shape[0] // 2, j_stub] < 1e-12


def test_traced_projection_matches_python_beta():
    from photonfdtd.design import tanh_projection
    r = jnp.linspace(0, 1, 7)
    for b in (0.0, 3.0):
        np.testing.assert_allclose(np.asarray(tanh_projection(r, jnp.asarray(b))),
                                   np.asarray(tanh_projection(r, b)), atol=1e-12)


# --------------------------------------------------------------------------- #
# Compile-once gradient
# --------------------------------------------------------------------------- #

def _loss(sim):
    ro = _readouts(sim)
    return lambda out: -ro[1](out)[0] / 1e-30


def test_region_gradient_matches_fd_and_compiles_once():
    sim = _sim()
    reg = _region()
    ce_of = reg.ce_builder(sim)
    traces = []

    def counted(rho, **aux):
        traces.append(1)
        return ce_of(rho, **aux)

    vg = pf.make_value_and_grad_params(sim, counted, _loss(sim))
    rho = _rho(reg, sim)
    v, g = vg(rho, beta=4.0)
    vg(rho, beta=6.0)                      # new beta: no retrace
    assert len(traces) == 1

    ref_v, ref_g = pf.value_and_grad_params(
        sim, lambda p: ce_of(p, beta=4.0), rho, _loss(sim))
    assert abs(float(v) - ref_v) <= 1e-12 * abs(ref_v)
    assert float(jnp.max(jnp.abs(g - ref_g))) <= 1e-12 * float(jnp.max(jnp.abs(g)))

    h = 1e-4
    nxd, nyd = rho.shape
    for (i, j) in [(nxd // 2, nyd // 2 + 3), (2, nyd // 3)]:
        fd = (float(vg(rho.at[i, j].add(h), beta=4.0)[0])
              - float(vg(rho.at[i, j].add(-h), beta=4.0)[0])) / (2 * h)
        assert float(g[i, j]) == pytest.approx(fd, rel=1e-6,
                                               abs=1e-9 * float(jnp.abs(g).max()))


# --------------------------------------------------------------------------- #
# Parametric shapes
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("theta", [0.0, 0.3, 1.1])
def test_polygon_density_area(theta):
    xs = np.arange(-1, 1, 0.01) + 0.005
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    v = np.asarray(pf.design.rect(0.1, 0.0, 0.8, 0.5)) @ R.T
    assert float(pf.polygon_density(v, xs, xs).sum()) * 1e-4 == pytest.approx(
        0.4, rel=1e-3)
    tri = np.array([[-0.5, -0.4], [0.6, -0.3], [0.0, 0.7]])
    area = 0.5 * abs((tri[1, 0] - tri[0, 0]) * (tri[2, 1] - tri[0, 1])
                     - (tri[1, 1] - tri[0, 1]) * (tri[2, 0] - tri[0, 0]))
    assert float(pf.polygon_density(tri @ R.T, xs, xs).sum()) * 1e-4 == \
        pytest.approx(area, rel=1e-3)


def _shape_design():
    from photonfdtd.design import rect
    reg = _region(filter_radius=0.0, beta=0.0)
    # Two parallel bars bridging the stubs: params = (width_lower, width_upper).
    return pf.ShapeDesign(reg, lambda p: [rect(0.0, PORTS_Y[0], 1.4e-6, p[0]),
                                          rect(0.0, PORTS_Y[1], 1.4e-6, p[1])])


def test_shape_gradient_matches_fd():
    sim = _sim()
    sd = _shape_design()
    vg = pf.make_value_and_grad_params(sim, sd.ce_builder(sim), _loss(sim))
    p = jnp.asarray([0.30e-6, 0.37e-6])
    v, g = vg(p)
    # The sidewall normals make the response curved on the nm scale, so
    # Richardson-extrapolate the central difference (O(h^2) -> O(h^4)).
    h = 2e-9

    def cd(k, step):
        e = jnp.zeros(2).at[k].set(step)
        return (float(vg(p + e)[0]) - float(vg(p - e)[0])) / (2 * step)
    for k in range(2):
        fd = (4 * cd(k, h / 2) - cd(k, h)) / 3
        assert float(g[k]) == pytest.approx(fd, rel=1e-5)


# --------------------------------------------------------------------------- #
# Optimizer and loop
# --------------------------------------------------------------------------- #

def _quadratic():
    target = jnp.asarray([0.2, 0.9, -0.5, 1.7])
    f = lambda x: jnp.sum((x - target) ** 2 * jnp.arange(1, 5))
    return target, jax.jit(jax.value_and_grad(f))


def test_adam_matches_reference_steps():
    """Built-in Adam reproduces the textbook update (and optax, if present)."""
    _, vg = _quadratic()
    x = jnp.zeros(4)
    opt = pf.Adam(0.1)
    st = opt.init(x)
    m = v = np.zeros(4)
    xr = np.zeros(4)
    for t in range(1, 6):
        _, g = vg(x)
        u, st = opt.update(g, st, x)
        x = x + u
        gr = np.asarray(vg(jnp.asarray(xr))[1])
        m = 0.9 * m + 0.1 * gr
        v = 0.999 * v + 0.001 * gr ** 2
        xr = xr - 0.1 * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    np.testing.assert_allclose(np.asarray(x), xr, rtol=1e-12, atol=1e-14)
    optax = pytest.importorskip("optax")
    xo = jnp.zeros(4)
    so = optax.adam(0.1).init(xo)
    for _ in range(5):
        u, so = optax.adam(0.1).update(vg(xo)[1], so, xo)
        xo = xo + u
    np.testing.assert_allclose(np.asarray(xo), xr, rtol=1e-6)


def test_minimize_bounds_and_resume(tmp_path):
    target, vg = _quadratic()
    wrap = lambda x, **aux: vg(x)
    full = pf.minimize(wrap, jnp.full(4, 0.5), pf.Adam(0.05), n_iter=40)
    x = np.asarray(full.x)
    assert x.min() >= 0.0 and x.max() <= 1.0
    np.testing.assert_allclose(x, np.clip(np.asarray(target), 0, 1), atol=0.05)
    assert full.values()[-1] < full.values()[0]

    ck = str(tmp_path / "ck.npz")
    pf.minimize(wrap, jnp.full(4, 0.5), pf.Adam(0.05), n_iter=40, checkpoint=ck,
                callback=lambda it, x, rec: it == 14)       # "interrupted"
    resumed = pf.minimize(wrap, jnp.full(4, 0.5), pf.Adam(0.05), n_iter=40,
                          checkpoint=ck)
    assert np.array_equal(np.asarray(resumed.x), x)
    assert len(resumed.history) == 40
    assert [h["value"] for h in resumed.history] == [h["value"] for h in full.history]


def test_beta_continuation_and_binarization():
    s = pf.beta_continuation(1.0, 2.0, every=10, beta_max=8.0, eta=0.5)
    assert [s(i)["beta"] for i in (0, 9, 10, 25, 1000)] == [1, 1, 2, 4, 8]
    assert s(0)["eta"] == 0.5
    assert pf.binarization(np.array([0.0, 1.0])) == 1.0
    assert pf.binarization(np.array([0.5, 0.5])) == 0.0


def test_robust_variants_order_and_worst_case():
    sim = _sim()
    reg = _region(beta=8.0)
    rho = _rho(reg, sim)
    fill = [float(jnp.mean(reg.rho_phys(sim, rho, eta=e)))
            for e in (0.55, 0.5, 0.45)]
    assert fill[0] < fill[1] < fill[2]      # eroded < nominal < dilated

    vg = pf.make_value_and_grad_params(sim, reg.ce_builder(sim), _loss(sim))
    vals = [float(vg(rho, eta=e)[0]) for e in (0.45, 0.5, 0.55)]
    v, g = pf.robust(vg, (0.45, 0.5, 0.55), "max")(rho)
    assert float(v) == max(vals)
    vm, _ = pf.robust(vg, (0.45, 0.5, 0.55), "mean")(rho)
    assert float(vm) == pytest.approx(np.mean(vals), rel=1e-12)


def test_mode_transmission_wraps_overlap():
    core = pf.Box(center=(0, 0), size=(0.45e-6, 0.22e-6),
                  medium=pf.Medium.from_index(3.48))
    ms = pf.ModeSolver(size=(2.0e-6, 1.4e-6), cell_size=(50e-9, 50e-9),
                       structures=[core], wavelength=LAM,
                       background_eps=1.44 ** 2, num_modes=1)
    mr = ms.solve()
    from photonfdtd.smatrix import mode_port_fields
    m0 = mode_port_fields(mr, 0)
    out = {"dft": {"p": {k: jnp.asarray(0.6 * m0[k][None]) for k in m0}}}
    obj = pf.ModeTransmission("p", mr, dA=ms.dy * ms.dz)
    assert float(obj(out)[0]) == pytest.approx(0.36, rel=1e-6)
    back = pf.ModeTransmission("p", mr, dA=ms.dy * ms.dz, direction="-")
    assert float(back(out)[0]) < 1e-10
    f = pf.combine([obj, back], weights=[1.0, 1.0])
    assert float(f(out)) == pytest.approx(0.18, rel=1e-6)


# --------------------------------------------------------------------------- #
# GDS export
# --------------------------------------------------------------------------- #

def test_gds_export_density_and_polygons(tmp_path):
    gdstk = pytest.importorskip("gdstk")
    sim = _sim()
    reg = _region()
    xs, ys = reg.coords(sim.grid)
    # A binary disc + a hole: contoured area matches the pixel area.
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    r = np.hypot(X, Y)
    rho = ((r < 0.5e-6) & (r > 0.2e-6)).astype(float)
    path = str(tmp_path / "d.gds")
    pf.export_gds(path, region=reg, rho=rho, grid=sim.grid)
    lib = gdstk.read_gds(path)
    area = sum(p.area() for p in lib.cells[0].polygons) * 1e-12   # um^2 -> m^2
    assert area == pytest.approx(rho.sum() * DX * DX, rel=0.06)

    sd = _shape_design()
    polys = sd.polygons(jnp.asarray([0.3e-6, 0.4e-6]))
    pf.export_gds(path, polygons=polys)
    lib = gdstk.read_gds(path)
    area = sum(p.area() for p in lib.cells[0].polygons) * 1e-12
    assert area == pytest.approx(1.4e-6 * 0.7e-6, rel=1e-6)


# --------------------------------------------------------------------------- #
# Training a chip like a neural network
# --------------------------------------------------------------------------- #

def _model(freqs=(F0,), detection="power"):
    sims = [_sim((1.0, 0.0), freqs), _sim((0.0, 1.0), freqs)]
    reg = _region()
    return pf.PhotonicModel(sims, reg, _readouts(sims[0]), detection=detection), reg


def test_transfer_matrix_is_exact_superposition():
    """T @ x equals a direct run with every input driven at once."""
    model, reg = _model(freqs=(0.95 * F0, F0))
    rho = _rho(reg, model.sims[0])
    T = model.transfer(rho, beta=4.0)
    assert T.shape == (2, 2, 2)
    x = np.array([0.7, -0.4])
    sx = _sim(tuple(x), (0.95 * F0, F0))
    from photonfdtd.jaxbackend import _prepare, _simulate_out
    out = _simulate_out(sx, _prepare(sx), reg.ce_builder(sx)(rho, beta=4.0))
    direct = jnp.stack([r.amplitude(out) for r in _readouts(sx)], -1)
    lin = jnp.einsum("fmn,n->fm", T, x)
    assert float(jnp.max(jnp.abs(direct - lin))) <= 1e-10 * float(jnp.max(jnp.abs(direct)))


def test_batch_gradient_matches_per_sample_simulation():
    """The transfer-matrix route gives the same training gradient as
    differentiating one direct simulation per sample."""
    model, reg = _model()
    sim0 = model.sims[0]
    rho = _rho(reg, sim0)
    X = np.array([[1.0, 0.5], [-0.3, 1.0], [0.8, -0.8]])
    Y = np.array([0, 1, 1])
    from photonfdtd.train import _cross_entropy
    from photonfdtd.jaxbackend import _prepare, _simulate_out

    def via_T(r):
        return _cross_entropy(model.apply(model._transfer(r, {}), X), jnp.asarray(Y))

    sims = [_sim(tuple(x)) for x in X]
    preps = [_prepare(s) for s in sims]
    ce_of = reg.ce_builder(sim0)
    ro = _readouts(sim0)

    def direct(r):
        ce = ce_of(r)
        P = jnp.stack([jnp.stack([jnp.sum(jnp.abs(q.amplitude(_simulate_out(s, p, ce))) ** 2)
                                  for q in ro]) for s, p in zip(sims, preps)])
        return _cross_entropy(P, jnp.asarray(Y))

    v1, g1 = jax.value_and_grad(via_T)(rho)
    v2, g2 = jax.value_and_grad(direct)(rho)
    assert float(v1) == pytest.approx(float(v2), rel=1e-10)
    assert float(jnp.max(jnp.abs(g1 - g2))) <= 1e-10 * float(jnp.max(jnp.abs(g2)))


def _phase_dataset(n=48, seed=1):
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, 2, n)
    phi = np.where(lab == 0, 0.0, np.pi) + rng.normal(0, 0.4, n)
    X = np.stack([np.ones(n), np.exp(1j * phi)], 1) * rng.uniform(0.5, 1.5, (n, 1))
    return X, lab


def test_train_learns_phase_classifier_and_resumes(tmp_path):
    """Two inputs whose relative phase is the class: the trained etch routes
    in-phase light to port 0 and anti-phase light to port 1."""
    model, reg = _model()
    X, Y = _phase_dataset()
    rho0 = jnp.full(reg.shape(model.sims[0].grid), 0.5)
    kw = dict(optimizer=pf.Adam(0.05), epochs=6, batch_size=12,
              schedule=pf.beta_continuation(2.0, 2.0, 8, 8.0))
    res = pf.train(model, rho0, X, Y, validation=(X[:12], Y[:12]), **kw)
    assert len(res.history) == 24
    assert res.history[3]["val_accuracy"] >= 0.0
    assert res.history[-1]["value"] < 0.5 * res.history[0]["value"]
    P = model.predict(res.x, X, beta=8.0)
    assert pf.accuracy(P, Y) >= 0.9

    ck = str(tmp_path / "t.npz")
    pf.train(model, rho0, X, Y, checkpoint=ck,
             callback=lambda it, x, rec: it == 9, **kw)
    again = pf.train(model, rho0, X, Y, checkpoint=ck, **kw)
    assert np.array_equal(np.asarray(again.x), np.asarray(
        pf.train(model, rho0, X, Y, **kw).x))


def test_fit_transfer_reduces_mismatch():
    model, reg = _model(detection="coherent")
    rho0 = jnp.full(reg.shape(model.sims[0].grid), 0.5)
    T0 = np.asarray(model.transfer(rho0))
    s = np.abs(T0).max()
    target = s * np.array([[0.0, 1.0], [1.0, 0.0]])      # a crossing
    res = pf.fit_transfer(model, rho0, target, optimizer=pf.Adam(0.05), n_iter=12)
    assert res.history[-1]["value"] < 0.7 * res.history[0]["value"]
