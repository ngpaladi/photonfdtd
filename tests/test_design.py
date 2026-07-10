"""Differentiable etched-core geometry: pipeline + gradient-to-density."""
import numpy as np
import pytest

import photonfdtd as pf

jax = pytest.importorskip("jax")
import jax.numpy as jnp   # noqa: E402
jax.config.update("jax_enable_x64", True)


def test_projection_binarizes():
    from photonfdtd.design import tanh_projection
    rho = jnp.linspace(0, 1, 11)
    p = np.asarray(tanh_projection(rho, beta=20.0, eta=0.5))
    assert p[0] == pytest.approx(0.0, abs=1e-6)
    assert p[-1] == pytest.approx(1.0, abs=1e-6)
    assert p[5] == pytest.approx(0.5, abs=1e-6)          # midpoint fixed at eta
    # sharper than identity: values away from eta pushed toward 0/1
    assert p[3] < 0.3 and p[7] > 0.7


def test_layer_stack_and_extremes():
    grid = pf.Grid(size=(2e-6, 2e-6, 1.2e-6), cell_size=100e-9, pml_layers=(0, 0, 0))
    nx, ny, nz = grid.shape
    ec = pf.EtchedCore(eps_solid=12.0, eps_void=2.0, eps_sub=2.25, eps_clad=1.0,
                       core_z=(-0.1e-6, 0.1e-6))
    # rho == 1 everywhere -> core is solid; substrate/cladding as specified.
    exx, eyy, ezz = ec.eps_components(grid, jnp.ones((nx, ny)))
    exx = np.asarray(exx)
    zc = np.asarray(grid.coords[2])
    kcore = int(np.argmin(np.abs(zc)))
    ksub = 0
    ktop = nz - 1
    assert np.allclose(exx[:, :, kcore], 12.0)
    assert np.allclose(exx[:, :, ksub], 2.25)
    assert np.allclose(exx[:, :, ktop], 1.0)
    # rho == 0 -> core is void.
    exx0, _, _ = ec.eps_components(grid, jnp.zeros((nx, ny)))
    assert np.allclose(np.asarray(exx0)[:, :, kcore], 2.0)


def _grad_setup():
    lam = 1.55e-6
    f0 = pf.C_0 / lam
    dx = 70e-9
    grid = pf.Grid(size=(3e-6, 3e-6, 1.2e-6), cell_size=dx, pml_layers=(8, 8, 8))
    src = pf.PointDipole(position=(-0.8e-6, 0, 0), component="Ez",
                         waveform=pf.GaussianPulse(freq0=f0, fwhm=8e-15))
    mon = pf.DFTMonitor(name="d", components=("Ez",), freqs=[f0],
                        plane_axis="x", plane_position=0.8e-6)
    sim = pf.Simulation(grid, sources=[src], monitors=[mon],
                        run_time=90e-15, use_jax=True)
    ec = pf.EtchedCore(eps_solid=3.48 ** 2, eps_void=1.44 ** 2,
                       eps_sub=1.44 ** 2, eps_clad=1.0,
                       core_z=(-0.11e-6, 0.11e-6), filter_radius=140e-9, beta=6.0)
    return sim, ec, grid


@pytest.mark.slow
def test_density_gradient_matches_finite_difference():
    """d(loss)/d(rho) through filter+projection+subpixel+FDTD matches finite
    differences - the end-to-end topology-optimization gradient."""
    sim, ec, grid = _grad_setup()
    nx, ny, _ = grid.shape
    rng = np.random.default_rng(0)
    rho = jnp.asarray(rng.uniform(0.2, 0.8, size=(nx, ny)))
    loss = lambda out: jnp.sum(jnp.abs(out["dft"]["d"]["Ez"]) ** 2)

    val, grad = pf.value_and_grad_density(sim, ec, rho, loss)
    assert np.isfinite(val)
    grad = np.asarray(grad)

    def L(r):
        v, _ = pf.value_and_grad_density(sim, ec, r, loss)
        return v

    # Central finite differences through the (nonlinear) filter+projection have
    # their own O(h^2) truncation error, so a few-per-mille agreement already
    # confirms the adjoint gradient (a wrong gradient would be off by O(1)).
    h = 1e-3
    for (i, j) in [(nx // 2, ny // 2), (nx // 3, ny // 2 + 4)]:
        fd = (L(rho.at[i, j].add(h)) - L(rho.at[i, j].add(-h))) / (2 * h)
        assert grad[i, j] == pytest.approx(fd, rel=3e-3, abs=1e-6 * abs(grad).max())
