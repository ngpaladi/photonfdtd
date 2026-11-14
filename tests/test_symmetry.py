"""Mirror-symmetry planes: a run on the symmetric half reproduces the
full-domain run, field for field, once unfolded."""
import numpy as np
import pytest

import photonfdtd as pf
from photonfdtd.symmetry import component_parity, _staggered

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

LAM = 1e-6
F0 = pf.C_0 / LAM
COMPS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _wave():
    return pf.GaussianPulse(freq0=F0, fwhm=3e-15)


def _images(dipoles, grid, cell):
    """The full-domain source set of mirror-symmetric ``dipoles`` (snapped to
    the nodes of the symmetric ``grid``): each one's images across every
    symmetric axis, at the image's Yee position and with the component's
    parity. On-plane node sources are their own image."""
    grid_sym = grid.symmetry
    snapped = []
    for d in dipoles:
        idx = grid.index_at(d.position)
        snapped.append(pf.PointDipole(
            position=tuple(float(grid.coords[a][idx[a]]) for a in range(3)),
            component=d.component, waveform=d.waveform, amplitude=d.amplitude))
    out = snapped
    for a, s in enumerate(grid_sym):
        if not s:
            continue
        new = []
        for d in out:
            pos = list(d.position) + [0.0] * (3 - len(d.position))
            x = pos[a]
            xi = -x - cell[a] if _staggered(d.component, a) else -x
            if abs(xi - x) < 1e-3 * cell[a]:
                continue                       # on the plane: its own image
            pos[a] = xi
            new.append(pf.PointDipole(position=tuple(pos), component=d.component,
                                      waveform=d.waveform,
                                      amplitude=d.amplitude
                                      * component_parity(d.component, a, s)))
        out += new
    return out


def _compare(size, cell, pml, sym, dipoles, structures=(), particles=(),
             run_time=14e-15, extra_monitors=(), tol=1e-10):
    full_g = pf.Grid(size=size, cell_size=cell, pml_layers=pml,
                     symmetry=(0, 0, 0))
    half_g = pf.Grid(size=size, cell_size=cell, pml_layers=pml, symmetry=sym)
    # The full grid must have the parity the symmetric one chose.
    full_g = pf.Grid(size=tuple(n * c if n > 1 else 0 for n, c in
                                zip(half_g.full_shape, half_g.cell_size)),
                     cell_size=cell, pml_layers=pml)
    assert full_g.shape == half_g.full_shape
    mon = pf.FieldMonitor(name="f", components=COMPS, times=[run_time * 0.98])
    full = pf.Simulation(full_g, structures=structures,
                         sources=_images(dipoles, half_g, half_g.cell_size)
                         + list(particles),
                         monitors=[mon] + list(extra_monitors), run_time=run_time)
    half = pf.Simulation(half_g, structures=structures,
                         sources=list(dipoles) + list(particles),
                         monitors=[mon] + list(extra_monitors), run_time=run_time)
    assert full.n_steps == half.n_steps
    # Per-node permittivity makes the full-domain Yee problem itself slightly
    # asymmetric for the half-cell-offset components (normal E sees eps_m on
    # one side and eps_{m+1} on the mirror side). The half run is exactly the
    # mirror-symmetric discretization, so compare with the full run given the
    # mirrored coefficients; test_structures_agree_with_plain_full_run checks
    # the plain full run agrees to discretization accuracy.
    from photonfdtd.jaxbackend import update_coefficients
    ce_half, _ = update_coefficients(half)
    ce_full = {c: np.abs(pf.unfold(half_g, c, v)) for c, v in ce_half.items()}
    rf, rh = full.run(ce=ce_full if structures else None), half.run()
    full._rf_plain = full.run() if structures and extra_monitors else None
    for c in COMPS:
        F = np.asarray(rf.fields["f"][c])
        U = pf.unfold(half_g, c, rh.fields["f"][c])
        assert U.shape == F.shape, c
        scale = max(np.abs(F).max(), 1e-300)
        err = np.abs(U - F).max() / scale
        assert err < tol, (c, err)
    for a in range(3):
        assert np.allclose(pf.full_coords(half_g, a), full_g.coords[a])
    return full, half, rf, rh


def test_grid_symmetry_parity_and_validation():
    g = pf.Grid(size=(4e-6, 4e-6), cell_size=0.1e-6, symmetry=(-1, 1, 0),
                pml_layers=(5, 5, 0))
    assert g.full_shape[0] % 2 == 1 and g.full_shape[1] % 2 == 0
    assert np.isclose(g.coords[0][0], 0.0)              # PEC: node on the plane
    assert np.isclose(g.coords[1][0], 0.05e-6)          # PMC: nodes at +-d/2
    assert g.pml_low == (0, 0, 0)
    with pytest.raises(ValueError, match="periodic"):
        pf.Grid(size=(4e-6, 4e-6), cell_size=0.1e-6, symmetry=(1, 0, 0),
                periodic=(True, False, False))
    with pytest.raises(ValueError, match="collapsed"):
        pf.Grid(size=(4e-6, 4e-6), cell_size=0.1e-6, symmetry=(0, 0, 1))


@pytest.mark.parametrize("sym", [(-1, 1, 0), (1, -1, 0), (1, 1, 0), (-1, -1, 0)])
def test_2d_half_domain_matches_full(sym):
    """Ez/Hz dipoles off the planes plus a dielectric block straddling both
    planes: the quarter-domain run, unfolded, is the full run."""
    dx = LAM / 20
    w = _wave()
    dip = [pf.PointDipole(position=(0.6e-6, 0.45e-6), component="Ez", waveform=w),
           pf.PointDipole(position=(0.3e-6, 0.8e-6), component="Hz", waveform=w,
                          amplitude=1 / pf.ETA_0)]
    block = pf.Box(center=(0, 0, 0), size=(1.0e-6, 0.6e-6, 1),
                   medium=pf.Medium.from_index(2.0))
    _compare((10e-6, 10e-6), (dx, dx, 0), (10, 10, 0), sym, dip, [block])


def test_sources_on_the_plane():
    """A source on a PEC plane: normal E (even) is stepped, tangential E (odd,
    zero there) is dropped. On a PMC plane a source at the plane coordinate
    lands on the first node pair, +-dx/2."""
    dx = LAM / 20
    w = _wave()
    g = pf.Grid(size=(6e-6, 6e-6), cell_size=dx, pml_layers=(8, 8, 0),
                symmetry=(-1, 0, 0))
    sim = pf.Simulation(g, sources=[
        pf.PointDipole(position=(0, 0.3e-6), component="Ey", waveform=w),
        pf.PointDipole(position=(0, 0.3e-6), component="Ex", waveform=w),
        pf.PointDipole(position=(-1e-6, 0.3e-6), component="Ex", waveform=w)],
        run_time=2e-15)
    assert [d.component for d in sim.sources] == ["Ex"]
    # Normal Ex on the PEC plane (even) against its full-domain image pair.
    _compare((10e-6, 10e-6), (dx, dx, 0), (10, 10, 0), (-1, 0, 0),
             [pf.PointDipole(position=(0, 0.3e-6), component="Ex", waveform=w)])
    # Tangential Ez on a PMC plane (even): the snapped pair at +-dx/2.
    _compare((10e-6, 10e-6), (dx, dx, 0), (10, 10, 0), (1, 0, 0),
             [pf.PointDipole(position=(dx / 2, 0.3e-6), component="Ez",
                             waveform=w)])


def test_3d_charge_on_the_axis_of_a_tube():
    """The Cherenkov-tube geometry: a charge moving along z on the axis of a
    dielectric tube, PMC planes in x and y (Jz is even across both). The
    quarter-domain run reproduces the full one. Not to round-off: on the
    full grid the charge sits midway between nodes and its index-window
    stencil reaches one node further on one side (a ~1e-3 tail), while the
    half run deposits the exactly symmetric cloud."""
    dx = LAM / 10
    tube = [pf.Box(center=(0, 0, 0), size=(1.6e-6, 1.6e-6, 10e-6),
                   medium=pf.Medium.from_index(1.5)),
            pf.Box(center=(0, 0, 0), size=(0.8e-6, 0.8e-6, 10e-6),
                   medium=pf.Medium.from_index(1.0))]
    q = pf.ChargedParticle(charge=1e-12, velocity=(0, 0, 0.9 * pf.C_0),
                           start=(0, 0, -1.5e-6), cloud_cells=1.0)
    _compare((7e-6, 7e-6, 6e-6), (dx, dx, dx), (6, 6, 6), (1, 1, 0), [],
             tube, particles=[q], run_time=8e-15, tol=5e-3)


def test_flux_reports_the_whole_domain():
    """FluxMonitor on the half domain reports the full-domain flux (here
    with no structure, so the two discretizations are the same)."""
    dx = LAM / 20
    w = _wave()
    dip = [pf.PointDipole(position=(0.6e-6, 0.45e-6), component="Ez", waveform=w)]
    mons = [pf.FluxMonitor(name="py", plane_axis="y", plane_position=2e-6),
            pf.FluxMonitor(name="px", plane_axis="x", plane_position=2e-6)]
    # Run until the pulse has crossed the flux planes and reached the PML; the
    # full grid's low-side PML is not an exact mirror of its high side (the
    # Yee staggering), so the fields there agree to ~1e-4, not round-off.
    full, half, rf, rh = _compare((10e-6, 10e-6), (dx, dx, 0), (10, 10, 0),
                                  (-1, 1, 0), dip, run_time=20e-15,
                                  extra_monitors=mons, tol=1e-3)
    for k in ("py", "px"):
        assert abs(rh.flux[k] / rf.flux[k] - 1) < 1e-4, k


def test_structures_agree_with_plain_full_run():
    """With a dielectric block straddling both planes the half run and the
    plain full run differ only by the per-node rasterization of the block
    (a discretization error), so their fluxes agree closely."""
    dx = LAM / 20
    w = _wave()
    dip = [pf.PointDipole(position=(0.6e-6, 0.45e-6), component="Hz", waveform=w)]
    block = pf.Box(center=(0, 0, 0), size=(1.0e-6, 0.6e-6, 1),
                   medium=pf.Medium.from_index(2.0))
    mons = [pf.FluxMonitor(name="py", plane_axis="y", plane_position=2e-6),
            pf.FluxMonitor(name="px", plane_axis="x", plane_position=2e-6)]
    full, half, rf, rh = _compare((10e-6, 10e-6), (dx, dx, 0), (10, 10, 0),
                                  (-1, 1, 0), dip, [block], run_time=20e-15,
                                  extra_monitors=mons, tol=1e-3)
    for k in ("py", "px"):
        assert abs(rh.flux[k] / full._rf_plain.flux[k] - 1) < 0.03, k


def test_gradients_on_a_symmetric_grid():
    """The checkpointed and PML-reversible adjoints agree on the half grid."""
    dx = LAM / 14
    g = pf.Grid(size=(4e-6, 4e-6), cell_size=dx, pml_layers=(8, 8, 0),
                symmetry=(1, 0, 0))
    box = pf.Box(center=(0.3e-6, 0.4e-6, 0), size=(0.6e-6, 0.4e-6, 1),
                 medium=pf.Medium.from_index(2.0))
    src = pf.PointDipole(position=(0.2e-6, -0.6e-6), component="Ez",
                         waveform=pf.GaussianPulse(freq0=F0, fwhm=6e-15))
    mon = pf.DFTMonitor(name="d", components=("Ez",), freqs=[F0],
                        plane_axis="y", plane_position=1e-6)
    sim = pf.Simulation(g, structures=[box], sources=[src], monitors=[mon],
                        run_time=30e-15)
    loss = lambda out: jnp.sum(jnp.abs(out["dft"]["d"]["Ez"]) ** 2)
    v_ref, g_ref = pf.jax_value_and_grad_eps(sim, loss, remat="none")
    v_rev, g_rev = pf.jax_value_and_grad_eps_reversible_pml(sim, loss)
    assert np.isclose(v_ref, v_rev)
    assert np.abs(g_ref - g_rev).max() / np.abs(g_ref).max() < 1e-9


def test_symmetry_refusals():
    g = pf.Grid(size=(4e-6, 4e-6), cell_size=LAM / 14, pml_layers=(6, 6, 0),
                symmetry=(1, 0, 0))
    sim = pf.Simulation(g, run_time=5e-15)
    with pytest.raises(NotImplementedError, match="symmetry"):
        sim.run(disk_tiled=True)
    with pytest.raises(NotImplementedError, match="symmetry"):
        pf.project_far_field(sim, None, {"a": "+x"}, [[1, 0, 0]])


def test_symmetric_closed_surface_and_energy_match_full_domain():
    """A closed box and an energy box centred on the planes report the
    whole-domain values on the quarter grid."""
    dx = LAM / 20
    w = _wave()
    dip = [pf.PointDipole(position=(0.6e-6, 0.45e-6), component="Ez", waveform=w)]
    half_g = pf.Grid(size=(10e-6, 10e-6), cell_size=(dx, dx, 0),
                     pml_layers=(10, 10, 0), symmetry=(-1, 1, 0))
    full_g = pf.Grid(size=tuple(n * dx for n in half_g.full_shape[:2]),
                     cell_size=(dx, dx, 0), pml_layers=(10, 10, 0))
    box = dict(center=(0, 0), size=(3e-6, 3e-6))
    # Bounds off the midpoint between nodes, so the full grid snaps them
    # symmetrically (a tie would round both ends the same way).
    U = pf.EnergyMonitor(name="u", interval=10,
                         bounds=[(-1.51e-6, 1.51e-6), (-1.51e-6, 1.51e-6), None])
    full = pf.Simulation(full_g, sources=_images(dip, half_g, half_g.cell_size),
                         monitors=pf.closed_surface("b", grid=full_g, **box) + [U],
                         run_time=25e-15)
    half = pf.Simulation(half_g, sources=dip,
                         monitors=pf.closed_surface("b", grid=half_g, **box) + [U],
                         run_time=25e-15)
    assert len(half.monitors) == 3
    rf, rh = full.run(), half.run()
    assert abs(pf.closed_surface_flux(rh, "b") / pf.closed_surface_flux(rf, "b") - 1) < 1e-6
    # The bounded energy agrees to the box's edge resolution: on the full grid
    # the half-weighted edge cells of the half-cell-offset components are not
    # mirror images of each other (an O(dx) edge effect of the full-domain
    # measure); the unbounded energy agrees to round-off
    # (test_energy_on_symmetric_grid_is_whole_domain).
    uf, uh = rf.energy["u"], rh.energy["u"]
    assert np.abs(uh - uf).max() / uf.max() < 2e-2


def test_monitors_in_the_mirrored_half_are_refused():
    g = pf.Grid(size=(4e-6, 4e-6), cell_size=LAM / 14, pml_layers=(6, 6, 0),
                symmetry=(1, 0, 0))
    bad = [pf.FluxMonitor(name="f", plane_axis="x", plane_position=-1e-6),
           pf.DFTMonitor(name="d", components=("Ez",), freqs=[F0],
                         plane_axis="x", plane_position=-1e-6),
           pf.EnergyMonitor(name="u", bounds=[(-1e-6, 0.5e-6), None, None])]
    for m in bad:
        with pytest.raises(ValueError, match="symmetric|mirrored"):
            pf.Simulation(g, monitors=[m], run_time=1e-15).run()
    with pytest.raises(ValueError, match="centred"):
        pf.closed_surface("b", center=(0.5e-6, 0), size=(1e-6, 1e-6), grid=g)
