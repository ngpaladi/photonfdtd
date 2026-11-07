"""Active devices: anisotropic and conductive media in FDTD, electro-optic,
thermo-optic and free-carrier index changes, and their coupling to the mode
solver and the FDTD engine (photonfdtd.active)."""
import numpy as np
import pytest

import photonfdtd as pf
from photonfdtd import active as A
from photonfdtd import statics
from photonfdtd.jaxbackend import update_coefficients

C0 = pf.C_0
UM = 1e-6


# --------------------------------------------------------------------------- #
# FDTD engine: anisotropy, conductivity, coefficient overrides, modulation.
# --------------------------------------------------------------------------- #
def _line(medium, comp, freqs, L=30e-6, dx=40e-9, run_time=250e-15):
    grid = pf.Grid(size=(L,), cell_size=dx, pml_layers=(30,))
    box = pf.Box(center=(0.0,), size=(2 * L,), medium=medium)
    src = pf.PointDipole(position=(-10e-6,), component=comp,
                         waveform=pf.GaussianPulse(freq0=200e12, fwhm=8e-15))
    mon = pf.DFTMonitor(name="d", components=(comp,), freqs=freqs)
    return pf.Simulation(grid, structures=[box], sources=[src], monitors=[mon],
                         run_time=run_time, precision="float64")


def _phase_index(sim, res, comp, f, x1=-5e-6, x2=5e-6):
    D = np.asarray(res.dft["d"][comp])[0, :, 0, 0]
    x = sim.grid.coords[0]
    i1, i2 = (int(np.argmin(np.abs(x - p))) for p in (x1, x2))
    ph = np.unwrap(np.angle(D))
    return abs(ph[i2] - ph[i1]) / (x[i2] - x[i1]) * C0 / (2 * np.pi * f), D, x, i1, i2


def test_anisotropic_medium_each_component_sees_its_index():
    med = pf.AnisotropicMedium(permittivity=(4.0, 2.25, 6.25))
    f = 200e12
    for comp, n in (("Ey", 1.5), ("Ez", 2.5)):
        sim = _line(med, comp, np.array([f]))
        assert sim.per_component
        n_fdtd = _phase_index(sim, sim.run(), comp, f)[0]
        assert n_fdtd == pytest.approx(n, rel=0.01)


def test_anisotropic_guards():
    med = pf.AnisotropicMedium(permittivity=(4.0, 2.25, 6.25))
    grid = pf.Grid(size=(4e-6,), cell_size=50e-9)
    box = pf.Box(center=(0.0,), size=(1e-6,), medium=med)
    with pytest.raises(NotImplementedError):
        pf.Simulation(grid, structures=[box], subpixel=True)
    sim = pf.Simulation(grid, structures=[box])
    with pytest.raises(ValueError):
        sim.eps_r = np.ones(grid.shape)
    with pytest.raises(NotImplementedError):
        pf.jax_value_and_grad_eps(sim, lambda out: 0.0)
    assert sim.eps_r[40] == pytest.approx((4 + 2.25 + 6.25) / 3)


def test_coefficient_override_reproduces_the_plain_run():
    sim = _line(pf.Medium(permittivity=2.25), "Ez", np.array([200e12]), run_time=100e-15)
    a = np.asarray(sim.run().dft["d"]["Ez"])
    ce, ca = update_coefficients(sim)
    assert ca is None
    b = np.asarray(sim.run(ce=ce).dft["d"]["Ez"])
    assert np.array_equal(a, b)


def test_conductive_medium_attenuates_as_the_complex_wavenumber():
    eps, sigma, f = 2.25, 2.0e3, 200e12
    med = pf.Medium(permittivity=eps, conductivity=sigma)
    sim = _line(med, "Ez", np.array([f]))
    assert sim._has_conductivity
    res = sim.run()
    _, D, x, i1, i2 = _phase_index(sim, res, "Ez", f)
    w = 2 * np.pi * f
    k = w / C0 * np.sqrt(eps + 1j * sigma / (w * pf.EPS_0))
    decay = np.abs(D[i2]) / np.abs(D[i1])
    assert decay == pytest.approx(np.exp(-k.imag * (x[i2] - x[i1])), rel=0.02)
    with pytest.raises(NotImplementedError):
        sim.run(disk_tiled=True)


def test_constant_modulation_equals_the_static_run():
    """A modulation drive held at a constant voltage gives the same fields as
    the static coefficients at that voltage."""
    lam = 1.55e-6
    ln = A.lithium_niobate_eo(lam, cut="x")   # optic axis along grid y
    grid = pf.Grid(size=(6e-6, 2e-6), cell_size=50e-9, pml_layers=(12, 0, 0))
    slab = pf.Box((0.0, 0.0), (3e-6, 1.2e-6), ln)
    top = pf.Box((0.0, 0.8e-6), (3e-6, 0.2e-6), A.metal_active("Au"))
    bot = pf.Box((0.0, -0.8e-6), (3e-6, 0.2e-6), A.metal_active("Au"))
    src = pf.PointDipole(position=(-2e-6, 0.0), component="Ey",
                         waveform=pf.GaussianPulse(freq0=C0 / lam, fwhm=6e-15))
    mon = pf.DFTMonitor(name="d", components=("Ey",), freqs=np.array([C0 / lam]))
    # The electrodes' metal is left out of the optical layout here (the
    # modulated box must not contain conductive cells).
    sim = pf.Simulation(grid, structures=[slab], sources=[src], monitors=[mon],
                        run_time=60e-15, precision="float64")
    reg = A.ActiveRegion([slab, top, bot], electrodes=[A.Electrode(top, "s"),
                                                       A.Electrode(bot, "g")],
                         wavelength=lam)
    prep = reg.prepare(sim)
    V = 50.0
    static = prep.run(V={"s": V})
    mod, ce, ca = prep.modulation({"s": lambda t: np.full_like(t, V)})
    dyn = sim.run(modulation=mod, ce=ce, ca=ca)
    a = np.asarray(static.dft["d"]["Ey"]); b = np.asarray(dyn.dft["d"]["Ey"])
    assert np.allclose(a, b, rtol=1e-10, atol=1e-12 * np.abs(a).max())
    # A different voltage does change the field.
    other = np.asarray(prep.run(V={"s": -V}).dft["d"]["Ey"])
    assert np.abs(other - a).max() > 1e-6 * np.abs(a).max()


def test_sinusoidal_modulation_makes_sidebands():
    """Phase-modulating a CW wave at f_m puts power at f0 +/- f_m."""
    eps_b, deps = 4.0, 0.2
    grid = pf.Grid(size=(20e-6,), cell_size=40e-9, pml_layers=(30,))
    f0, fm = 200e12, 20e12
    src = pf.PointDipole(position=(-7e-6,), component="Ez",
                         waveform=pf.GaussianPulse(freq0=f0, fwhm=200e-15))
    freqs = np.array([f0 - fm, f0 - 0.5 * fm, f0 + fm])
    mon = pf.DFTMonitor(name="d", components=("Ez",), freqs=freqs)
    box = pf.Box((0.0,), (40e-6,), pf.Medium(permittivity=eps_b))
    sim = pf.Simulation(grid, structures=[box], sources=[src], monitors=[mon],
                        run_time=600e-15)
    i = np.flatnonzero(np.abs(grid.coords[0]) < 5e-6)
    sl = (slice(int(i[0]), int(i[-1]) + 1), slice(0, 1), slice(0, 1))
    shape = (i.size, 1, 1)
    t = (np.arange(sim.n_steps) + 0.5) * sim.dt
    mod = dict(box=sl,
               eps_b={c: np.full(shape, eps_b) for c in ("Ex", "Ey", "Ez")},
               deps={c: np.full((1,) + shape, deps) for c in ("Ex", "Ey", "Ez")},
               v=np.sin(2 * np.pi * fm * t)[:, None])
    D = np.abs(np.asarray(sim.run(modulation=mod).dft["d"]["Ez"]))[:, :, 0, 0]
    out = int(np.argmin(np.abs(grid.coords[0] - 8e-6)))
    side_lo, between, side_hi = D[:, out]
    assert side_lo > 20 * between and side_hi > 20 * between


# --------------------------------------------------------------------------- #
# Materials and the index response.
# --------------------------------------------------------------------------- #
def test_lithium_niobate_orientation():
    lam = 1.55e-6
    n_e = np.sqrt(pf.lithium_niobate("e").eps_model(C0 / lam).real)
    n_o = np.sqrt(pf.lithium_niobate("o").eps_model(C0 / lam).real)
    x = A.lithium_niobate_eo(lam, cut="x")              # optic axis along grid y
    assert x.permittivity == pytest.approx((n_o ** 2, n_e ** 2, n_o ** 2))
    assert x.crystal_axes == (2, 0, 1)
    z = A.lithium_niobate_eo(lam, cut="z")              # optic axis vertical
    assert z.permittivity == pytest.approx((n_o ** 2, n_o ** 2, n_e ** 2))
    xy = A.lithium_niobate_eo(lam, cut="x", propagation="y")
    assert xy.permittivity[0] == pytest.approx(n_e ** 2)
    with pytest.raises(ValueError):
        A.lithium_niobate_eo(lam, cut="q")


def _plate_region(medium, V=10.0, gap=2e-6):
    """A block of ``medium`` between two horizontal plates (uniform E_z)."""
    lam = 1.55e-6
    block = pf.Box((0.0, 0.0), (4e-6, gap), medium)
    top = pf.Box((0.0, gap / 2 + 0.05e-6), (4e-6, 0.1e-6), A.metal_active("Au"))
    bot = pf.Box((0.0, -gap / 2 - 0.05e-6), (4e-6, 0.1e-6), A.metal_active("Au"))
    ms = pf.ModeSolver(size=(4e-6, gap + 0.2e-6), cell_size=0.05e-6,
                       structures=[block, top, bot], wavelength=lam)
    reg = A.ActiveRegion([block, top, bot], electrodes=[A.Electrode(top, "t"),
                                                        A.Electrode(bot, "b")],
                         wavelength=lam)
    return reg.prepare(ms), ms, gap


def test_pockels_uniform_field_on_z_cut():
    """E along the optic axis: d eps_zz = -eps_zz^2 r33 E (so dn_e = -n_e^3 r33 E/2)."""
    ln = A.lithium_niobate_eo(1.55e-6, cut="z")
    prep, ms, gap = _plate_region(ln)
    V = 10.0
    d, _ = prep.delta_eps(V={"t": V, "b": 0.0})
    j = int(np.argmin(np.abs(ms.y)))
    k = int(np.argmin(np.abs(ms.z)))
    Ez = prep.electrostatics.field([V, 0.0])[2][0, j, k]
    assert Ez == pytest.approx(-V / gap, rel=0.01)       # between the plate faces
    r13, r33 = 8.6e-12, 30.8e-12
    assert d[2][0, j, k] == pytest.approx(-ln.permittivity[2] ** 2 * r33 * Ez, rel=1e-9)
    assert d[0][0, j, k] == pytest.approx(-ln.permittivity[0] ** 2 * r13 * Ez, rel=1e-9)


def test_thermo_optic_change_and_missing_conductivity():
    lam = 1.55e-6
    si = A.silicon_active(lam)
    block = pf.Box((0.0, 0.0), (4e-6, 2e-6), si)
    heater = pf.Box((0.0, 0.95e-6), (4e-6, 0.1e-6), A.metal_active("TiN"))
    ms = pf.ModeSolver(size=(4e-6, 2e-6), cell_size=0.05e-6,
                       structures=[block, heater], wavelength=lam)
    reg = A.ActiveRegion([block, heater], heaters=[A.Heater(heater, name="h")],
                         wavelength=lam, sinks=("zmin",))
    prep = reg.prepare(ms)
    d, _ = prep.delta_eps(P={"h": 2.0})
    dT = 2.0 * prep.thermal.basis_dT[0]
    n = np.sqrt(si.permittivity[1])
    m = statics.region_mask(block, prep.mesh) & ~statics.region_mask(heater, prep.mesh)
    assert np.allclose(d[1][m], 2 * n * 1.86e-4 * dT[m])
    # A plain Medium has no thermal conductivity: a heater solve says so.
    bare = pf.Box((0.0, 0.0), (4e-6, 2e-6), pf.Medium(permittivity=12.0, name="mystery"))
    reg2 = A.ActiveRegion([bare, heater], heaters=[A.Heater(heater)], wavelength=lam)
    with pytest.raises(ValueError, match="mystery"):
        reg2.prepare(ms)


def test_plasma_dispersion_soref_bennett():
    dn, alpha = A.plasma_dispersion(1e24, 1e24, 1.55e-6)       # 1e18 cm^-3 each
    assert dn == pytest.approx(-(8.8e-22 * 1e18 + 8.5e-18 * 1e18 ** 0.8))
    assert alpha == pytest.approx((8.5e-18 + 6.0e-18) * 1e18 * 100)
    with pytest.warns(RuntimeWarning):
        A.plasma_dispersion(1e24, 0.0, 2.0e-6)


def test_pn_depletion_width():
    region = pf.Box((0.0, 0.0), (2e-6, 0.2e-6), pf.Medium(12.0))
    pn = A.PNJunction(region, axis="y", position=0.0, N_a=1e24, N_d=1e24)
    xp, xn = pn.depletion(2.0)
    q, eps = pf.Q_E, 11.7 * pf.EPS_0
    w = np.sqrt(2 * eps * (pn.built_in() + 2.0) * 2e24 / (q * 1e48))
    assert xp + xn == pytest.approx(w)
    assert xp == pytest.approx(xn)
    assert 0.8 < pn.built_in() < 1.0
    y = (np.arange(80) - 39.5) * 0.025e-6
    mesh = statics.Mesh((np.array([0.0]), y, np.array([0.0])), (0.0, 0.025e-6, 0.0))
    Ne, Nh = pn.carriers(mesh, 2.0)
    assert Nh[0, 0, 0] == 1e24 and Ne[0, -1, 0] == 1e24
    assert Ne[0, 40, 0] == 0 and Nh[0, 39, 0] == 0
    with pytest.raises(ValueError):
        pn.depletion(-5.0)


# --------------------------------------------------------------------------- #
# Mode solver and the drives.
# --------------------------------------------------------------------------- #
def test_mode_solver_isotropic_tensor_is_unchanged():
    lam = 1.55e-6
    core = pf.Box((0.0, 0.0), (0.5e-6, 0.22e-6), pf.Medium(permittivity=12.1))
    ms = pf.ModeSolver(size=(2e-6, 1.5e-6), cell_size=0.04e-6, structures=[core],
                       wavelength=lam)
    n0 = ms.solve().n_eff[0]
    ms.eps_components = (ms.eps_r, ms.eps_r, ms.eps_r)
    assert ms.solve().n_eff[0] == pytest.approx(n0, rel=1e-12)
    # Raising only eps_yy raises a TE (Ey) mode's index.
    ms.eps_components = (ms.eps_r, ms.eps_r * 1.02, ms.eps_r)
    assert ms.solve().n_eff[0] > n0


def _tfln(cell=0.05e-6):
    lam = 1.55e-6
    ln = A.lithium_niobate_eo(lam, cut="x")
    ox = A.silica_active(lam)
    au = A.metal_active("Au")
    S = [pf.Box((0, -1.0 * UM), (20 * UM, 2 * UM), ox),
         pf.Box((0, 0.15 * UM), (20 * UM, 0.3 * UM), ln),
         pf.Box((0, 0.45 * UM), (1.2 * UM, 0.3 * UM), ln)]
    g = 5 * UM
    eL = pf.Box((-(g / 2 + 2 * UM), 0.55 * UM), (4 * UM, 0.5 * UM), au)
    eR = pf.Box(((g / 2 + 2 * UM), 0.55 * UM), (4 * UM, 0.5 * UM), au)
    S += [eL, eR]
    ms = pf.ModeSolver(size=(12 * UM, 3.6 * UM), cell_size=cell, structures=S,
                       wavelength=lam, polarization="te")
    reg = A.ActiveRegion(S, electrodes=[A.Electrode(eR, "signal"),
                                        A.Electrode(eL, "ground")], wavelength=lam)
    return reg.prepare(ms)


def test_tfln_half_wave_voltage_length():
    """x-cut thin-film LN, 5 um gap: about 4-6 V cm per arm (2-3 V cm push-pull
    in the literature)."""
    prep = _tfln(0.06e-6)
    vpl = prep.vpi_l("signal") / 1e-2
    assert 3.5 < vpl < 7.0
    assert prep.offdiagonal_ratio is not None and prep.offdiagonal_ratio < 0.05
    # Linear in V: twice the voltage, twice the shift.
    n0 = prep.n_eff()
    d1 = prep.n_eff(V={"signal": 5.0}) - n0
    d2 = prep.n_eff(V={"signal": 10.0}) - n0
    assert d2 == pytest.approx(2 * d1, rel=1e-3)


def test_eim_transfer_applies_the_index_shift():
    lam = 1.55e-6
    prep = _tfln(0.08e-6)
    slope = prep.dneff("signal")
    core = pf.Box((0.0, 0.0), (4e-6, 1.2e-6), pf.Medium(permittivity=1.9 ** 2))
    eim = prep.eim([core], drives=["signal"])
    grid = pf.Grid(size=(6e-6, 4e-6), cell_size=0.05e-6, pml_layers=(10, 10, 0))
    sim = pf.Simulation(grid, structures=[core])
    e = eim.eps_components(sim, V={"signal": 20.0})
    j = grid.index_at((0.0, 0.0))
    assert np.sqrt(e[1][j]) - 1.9 == pytest.approx(20.0 * slope, rel=1e-9)
    assert e[1][grid.index_at((0.0, 1.5e-6))] == pytest.approx(1.0)


def test_extruded_3d_preparation_matches_the_cross_section():
    lam = 1.55e-6
    ln = A.lithium_niobate_eo(lam, cut="z")
    grid = pf.Grid(size=(2e-6, 2.4e-6, 2.4e-6), cell_size=0.1e-6)
    slab = pf.Box((0.0, 0.0, 0.0), (1e-6, 2e-6, 2e-6), ln)
    top = pf.Box((0.0, 0.0, 1.05e-6), (1e-6, 2e-6, 0.1e-6), A.metal_active("Au"))
    bot = pf.Box((0.0, 0.0, -1.05e-6), (1e-6, 2e-6, 0.1e-6), A.metal_active("Au"))
    sim = pf.Simulation(grid, structures=[slab])
    reg = A.ActiveRegion([slab, top, bot], electrodes=[A.Electrode(top, "t"),
                                                       A.Electrode(bot, "b")],
                         wavelength=lam)
    prep = reg.prepare(sim, extrude="x")
    assert prep.mesh.shape == (1,) + grid.shape[1:]
    e = prep.eps_components(V={"t": 10.0})
    base = sim.eps_components()
    inside = grid.index_at((0.0, 0.0, 0.0))
    outside = grid.index_at((0.9e-6, 0.0, 0.0))
    d, _ = prep.delta_eps(V={"t": 10.0})
    assert d[2][0, inside[1], inside[2]] > 0          # field points down: Ez < 0
    assert e[2][inside] == pytest.approx(base[2][inside] + d[2][0, inside[1], inside[2]])
    assert e[2][outside] == base[2][outside]


def test_voltage_gradient_through_the_coefficient_builder():
    """d(transmitted intensity)/dV through ce_builder matches a central
    difference of full runs."""
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    lam = 1.55e-6
    ln = A.lithium_niobate_eo(lam, cut="x")   # optic axis along grid y
    grid = pf.Grid(size=(4e-6, 1.6e-6), cell_size=0.05e-6, pml_layers=(10, 0, 0))
    slab = pf.Box((0.0, 0.0), (2e-6, 1.2e-6), ln)
    top = pf.Box((0.0, 0.7e-6), (2e-6, 0.1e-6), A.metal_active("Au"))
    bot = pf.Box((0.0, -0.7e-6), (2e-6, 0.1e-6), A.metal_active("Au"))
    f = C0 / lam
    src = pf.PointDipole(position=(-1.5e-6, 0.0), component="Ey",
                         waveform=pf.GaussianPulse(freq0=f, fwhm=5e-15))
    mon = pf.DFTMonitor(name="d", components=("Ey",), freqs=np.array([f]))
    sim = pf.Simulation(grid, structures=[slab], sources=[src], monitors=[mon],
                        run_time=40e-15, precision="float64")
    reg = A.ActiveRegion([slab, top, bot], electrodes=[A.Electrode(top, "t"),
                                                       A.Electrode(bot, "b")],
                         wavelength=lam)
    prep = reg.prepare(sim)
    j = grid.index_at((1.5e-6, 0.0))

    def loss(out):
        return jnp.abs(out["dft"]["d"]["Ey"][0, j[0], j[1], 0]) ** 2

    vg = pf.make_value_and_grad_params(sim, prep.ce_builder(), loss, jit=False)
    V0 = 30.0
    val, g = vg({"V": jnp.array([V0, 0.0])})
    h = 5.0
    fp, _ = vg({"V": jnp.array([V0 + h, 0.0])})
    fm, _ = vg({"V": jnp.array([V0 - h, 0.0])})
    assert float(g["V"][0]) == pytest.approx(float((fp - fm) / (2 * h)), rel=1e-3)
    # The builder at V reproduces prep.run(V).
    r = prep.run(V={"t": V0})
    assert float(val) == pytest.approx(
        abs(np.asarray(r.dft["d"]["Ey"])[0, j[0], j[1], 0]) ** 2, rel=1e-9)
