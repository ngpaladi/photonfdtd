"""Tests for anisotropic subpixel (sub-cell) smoothing.

Covers three things:

1. Tensor correctness on a flat interface: the field component normal to the
   interface must see the *harmonic* mean of permittivity and the tangential
   components the *arithmetic* mean (Farjadpour et al., Opt. Lett. 31, 2972).
2. Grid-aligned geometry is a no-op: when every interface lands on a cell
   boundary there are no partially filled cells, so smoothing reproduces plain
   staircasing exactly.
3. The accuracy payoff: as a dielectric interface is swept across one cell in
   sub-cell steps, a transmitted-phase observable staircases (jumps all at once
   at the cell crossing) without smoothing but varies smoothly *with* it,
   while both accumulate the same net physical change.
"""
import numpy as np
import pytest

import photonfdtd as pf
from photonfdtd.grid import Grid
from photonfdtd.geometry import Box
from photonfdtd.materials import Medium
from photonfdtd.smoothing import smooth_permittivity


def test_flat_interface_tensor():
    """A flat interface normal to x: eps_xx harmonic, eps_yy = eps_zz arithmetic."""
    dx = 1e-6
    n = 20
    g = Grid(size=(n * dx, n * dx), cell_size=dx, pml_layers=(0, 0, 0))
    eps_core = 12.0
    xc = g.coords[0]
    face = xc[10] + 0.25 * dx                     # partial fill in column i=10
    left = xc[0] - 5 * dx
    box = Box(center=((left + face) / 2, 0.0),
              size=(face - left, 100 * dx), medium=Medium(eps_core))
    exx, eyy, ezz = smooth_permittivity(g, [box], background_eps=1.0, factor=40)

    i = 10
    # Recover the (discretised) fill fraction from the arithmetic mean itself,
    # then predict the harmonic mean from the SAME fraction -> exact regardless
    # of the finite supersampling factor.
    f = (eyy[i, 0, 0] - 1.0) / (eps_core - 1.0)
    assert 0.0 < f < 1.0
    harm = 1.0 / (f / eps_core + (1 - f) / 1.0)
    assert exx[i, 0, 0] == pytest.approx(harm, rel=1e-6)          # normal -> harmonic
    assert ezz[i, 0, 0] == pytest.approx(eyy[i, 0, 0], rel=1e-12)  # out-of-plane = tangential
    assert exx[i, 0, 0] < eyy[i, 0, 0]                            # harmonic < arithmetic

    # Cells fully inside / outside are untouched.
    assert exx[3, 0, 0] == pytest.approx(eps_core)
    assert eyy[3, 0, 0] == pytest.approx(eps_core)
    assert exx[17, 0, 0] == pytest.approx(1.0)
    assert ezz[17, 0, 0] == pytest.approx(1.0)


def test_grid_aligned_is_noop():
    """A box whose faces land on cell boundaries has no partial cells, so the
    smoothed tensor is isotropic and equal to plain stamping everywhere."""
    dx = 1e-6
    n = 24
    g = Grid(size=(n * dx, n * dx), cell_size=dx, pml_layers=(0, 0, 0))
    eps_core = 6.0
    # coords are cell centres at (i-(n-1)/2)*dx; a face at a half-integer
    # multiple of dx sits exactly on a cell boundary.
    box = Box(center=(0.0, 0.0), size=(6 * dx, 8 * dx), medium=Medium(eps_core))
    exx, eyy, ezz = smooth_permittivity(g, [box], background_eps=1.0, factor=5)

    plain = np.ones(g.shape)
    box.stamp(g, plain)
    for comp in (exx, eyy, ezz):
        assert np.allclose(comp, plain)


def _transmitted_phase(face_x, subpixel, factor=6):
    lam0 = 1.0e-6
    freq0 = pf.C_0 / lam0
    dx = lam0 / 30
    L = 24e-6
    grid = pf.Grid(size=(L,), cell_size=dx, pml_layers=(20,))
    slab = pf.Box(center=((-4e-6 + face_x) / 2,),
                  size=(face_x - (-4e-6),), medium=pf.Medium.from_index(2.0))
    src = pf.PointDipole(position=(-7e-6,), component="Ez",
                         waveform=pf.GaussianPulse(freq0=freq0, fwhm=12e-15))
    mon = pf.DFTMonitor(name="d", components=("Ez",), freqs=[freq0])
    sim = pf.Simulation(grid, structures=[slab], sources=[src], monitors=[mon],
                        run_time=200e-15, subpixel=subpixel, subpixel_factor=factor)
    r = sim.run()
    v = np.asarray(r.dft["d"]["Ez"]).reshape(-1)
    idx = int(np.argmin(np.abs(grid.coords[0] - 6e-6)))
    return np.angle(v[idx])


def test_subcell_interface_smoothness():
    """Sweep a slab face across one cell: the transmitted phase staircases
    without smoothing (one big jump at the cell crossing) but ramps smoothly
    with it, and both span the same net physical change.

    This is the Farjadpour et al. (2006) demonstration that subpixel smoothing
    restores sensitivity to sub-cell geometry.
    """
    dx = (1.0e-6) / 30
    shifts = np.linspace(0.0, dx, 6)

    ph_stair = np.unwrap([_transmitted_phase(1e-6 + d, False) for d in shifts])
    ph_smooth = np.unwrap([_transmitted_phase(1e-6 + d, True) for d in shifts])

    jump_stair = np.abs(np.diff(ph_stair))
    jump_smooth = np.abs(np.diff(ph_smooth))
    net_stair = abs(ph_stair[-1] - ph_stair[0])
    net_smooth = abs(ph_smooth[-1] - ph_smooth[0])

    # Same net physics captured either way (moving the face a full cell).
    assert net_smooth == pytest.approx(net_stair, rel=0.25)
    assert net_stair > 0.05                       # the sweep actually does something

    # Staircase concentrates the whole change into one step; smoothing spreads
    # it out -> a markedly smaller worst-case sub-cell jump.
    assert jump_smooth.max() < 0.4 * jump_stair.max()
    # Smoothing tracks the interface monotonically (total variation ~ net move).
    assert jump_smooth.sum() == pytest.approx(net_smooth, rel=0.2)


def test_backend_guards():
    """subpixel is rejected on the fused kernels that assume a scalar coeff."""
    grid = pf.Grid(size=(3e-6, 3e-6), cell_size=1e-7, pml_layers=(4, 4, 0))
    with pytest.raises(ValueError):
        pf.Simulation(grid, subpixel=True, use_jax=True)
