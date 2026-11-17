"""Fabrication constraints: symmetry, length-scale penalties, brush checks."""
import numpy as np
import pytest

import photonfdtd as pf

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


def test_symmetrize_is_invariant_and_differentiable():
    rng = np.random.default_rng(0)
    r = rng.random((12, 12))
    s = pf.symmetrize(r, mirror_x=True, mirror_y=True, diagonal=True)
    assert np.allclose(s, s[::-1]) and np.allclose(s, s[:, ::-1]) and np.allclose(s, s.T)
    p = pf.symmetrize(r, rotate180=True)
    assert np.allclose(p, p[::-1, ::-1])
    g = jax.grad(lambda x: jnp.sum(pf.symmetrize(x, mirror_x=True) * jnp.arange(12.0)[:, None]))(
        jnp.asarray(r))
    assert np.allclose(np.asarray(g), np.asarray(g)[::-1])
    with pytest.raises(ValueError):
        pf.symmetrize(np.ones((3, 4)), diagonal=True)


def _line(width, n=60):
    b = np.zeros((n, n))
    c = n // 2
    b[:, c - width // 2: c - width // 2 + width] = 1.0
    return b


def test_brush_and_minimum_lengthscale():
    thin = _line(3)
    wide = _line(9)
    s, v = pf.brush_violations(thin, 5)
    assert s.any() and not v.any()
    s, v = pf.brush_violations(wide, 5)
    assert not s.any() and not v.any()
    solid, void = pf.minimum_lengthscale(thin, max_cells=20)
    assert solid == 3.0 and void >= 20
    # A narrow gap is a void violation.
    gap = 1.0 - _line(2)
    assert pf.brush_violations(gap, 4)[1].any()


def test_lengthscale_penalty_flags_thin_lines_only():
    R = 4.0
    pen = []
    for w in (3, 14):
        rho = jnp.asarray(_line(w))
        rf = pf.conic_filter(rho, R)
        rp = pf.tanh_projection(rf, beta=64.0)
        gs, gv = pf.lengthscale_constraints(rf, rp, radius_cells=R)
        pen.append(float(gs))
    assert pen[0] > 1e3 * max(pen[1], 1e-30)
    gs_grad = jax.grad(lambda x: pf.lengthscale_constraints(
        pf.conic_filter(x, R), pf.tanh_projection(pf.conic_filter(x, R), 64.0),
        radius_cells=R)[0])(jnp.asarray(_line(3)))
    assert np.isfinite(np.asarray(gs_grad)).all() and np.abs(gs_grad).max() > 0


def test_remove_floating_and_fill_voids():
    b = np.zeros((20, 20), bool)
    b[:, 0:3] = True                 # attached to the edge
    b[8:11, 8:11] = True             # floating island
    out = pf.remove_floating(b)
    assert out[:, 0:3].all() and not out[8:11, 8:11].any()
    anchored = pf.remove_floating(b, anchors=np.pad(np.ones((1, 1), bool), ((9, 10), (9, 10))))
    assert anchored[8:11, 8:11].all()
    ring = np.ones((20, 20), bool)
    ring[5:8, 5:8] = False           # enclosed hole
    ring[:, 15:] = False             # open void at the edge
    f = pf.fill_enclosed_voids(ring)
    assert f[5:8, 5:8].all() and not f[:, 15:].any()
