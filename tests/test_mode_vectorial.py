"""Acceptance test for the full-vectorial FDFD mode solver.

The full-vectorial solver is validated against the EXACT analytic dispersion of
a symmetric dielectric slab waveguide.  For a slab of core index ``n1`` and
thickness ``d`` embedded in cladding ``n2`` (interfaces perpendicular to z,
invariant along y, propagation along x), the fundamental even guided modes
satisfy the standard transcendental equations

    TE:  kappa * tan(kappa*d/2) = gamma
    TM:  kappa * tan(kappa*d/2) = (n1**2/n2**2) * gamma

with ``kappa = k0*sqrt(n1**2 - neff**2)`` and ``gamma = k0*sqrt(neff**2 - n2**2)``.

A *scalar* Helmholtz solver cannot reproduce the TM result (it has no
polarization dependence, so it predicts TE == TM).  The vectorial solver must
reproduce BOTH the TE0 and the TM0 effective index, which for a high-index-
contrast slab differ substantially.  A low-contrast case where TE ~= TM ~=
scalar is included as a sanity check.

The waveguide is invariant along y, so it is modelled in a wide-y box with PEC
(Dirichlet) walls; the walls impose a small transverse-y momentum ~(pi/Ly)**2.
With Ly large this shift is well under the 0.5% tolerance, and we compare the
numerical TE0/TM0 directly to the 1D analytic values.
"""
import math
import numpy as np
import pytest
from scipy.optimize import brentq

import photonfdtd as pf

LAM0 = 1.55e-6
K0 = 2 * math.pi / LAM0


def analytic_slab_neff(n1, n2, d, kind):
    """Fundamental even TE0/TM0 effective index of a symmetric slab."""
    fac = 1.0 if kind == "te" else (n1 ** 2 / n2 ** 2)
    # The fundamental even mode has kappa*d/2 < pi/2, i.e. neff above the value
    # where kappa*d/2 hits pi/2; restrict the bracket so tan() has one branch.
    lim = (math.pi / (K0 * d)) ** 2
    neff_min = math.sqrt(n1 ** 2 - lim) if lim < n1 ** 2 - n2 ** 2 else n2
    lo = max(n2, neff_min) + 1e-6
    hi = n1 - 1e-9

    def g(neff):
        kap = K0 * math.sqrt(n1 ** 2 - neff ** 2)
        gam = K0 * math.sqrt(neff ** 2 - n2 ** 2)
        return kap * math.tan(kap * d / 2.0) - fac * gam

    return brentq(g, lo, hi)


def solve_slab(n1, n2, d, Ly, dy, Lz, ncell_slab, num_modes):
    """Run the vectorial solver on a y-invariant slab; return (TE0, TM0) neff.

    Modes are classified by their dominant transverse E-component: Ey-dominant
    is TE (E parallel to interfaces and to the invariant axis), Ez-dominant is
    TM.  The highest-neff mode of each polarization is the fundamental.
    """
    dz = d / ncell_slab
    core = pf.Box(center=(0.0, 0.0), size=(3 * Ly, d),
                  medium=pf.Medium.from_index(n1))
    ms = pf.ModeSolver(
        size=(Ly, Lz), cell_size=(dy, dz), structures=[core],
        wavelength=LAM0, background_eps=n2 ** 2, num_modes=num_modes,
    )
    res = ms.solve()
    # Fields must be present and finite.
    assert res.Ey is not None and res.Ez is not None
    assert np.isfinite(res.n_eff).all()
    te_num = tm_num = None
    for i in range(len(res.n_eff)):
        py = float(np.sum(np.abs(res.Ey[i]) ** 2))
        pz = float(np.sum(np.abs(res.Ez[i]) ** 2))
        if py >= pz:
            if te_num is None:
                te_num = float(res.n_eff[i])
        else:
            if tm_num is None:
                tm_num = float(res.n_eff[i])
    return te_num, tm_num, res


def test_high_contrast_slab_te_and_tm():
    """High-index-contrast slab: TE0 and TM0 differ a lot; the vectorial solver
    must reproduce BOTH analytic values to within 0.5%."""
    n1, n2, d = 3.0, 1.0, 0.30e-6
    te0 = analytic_slab_neff(n1, n2, d, "te")
    tm0 = analytic_slab_neff(n1, n2, d, "tm")
    # sanity: the polarizations really are far apart for this contrast
    assert te0 - tm0 > 0.4

    te_num, tm_num, res = solve_slab(
        n1, n2, d, Ly=8e-6, dy=0.1e-6, Lz=2.0e-6, ncell_slab=30, num_modes=26,
    )
    assert te_num is not None, "no TE mode found"
    assert tm_num is not None, "no TM mode found"

    te_err = abs(te_num - te0) / te0
    tm_err = abs(tm_num - tm0) / tm0
    assert te_err < 5e-3, f"TE0 num={te_num:.6f} analytic={te0:.6f} err={te_err:.3%}"
    assert tm_err < 5e-3, f"TM0 num={tm_num:.6f} analytic={tm0:.6f} err={tm_err:.3%}"

    # The vectorial solver must distinguish TE from TM (the whole point): the
    # numerical split must match the analytic split, unlike a scalar solver.
    assert (te_num - tm_num) > 0.4
    assert abs((te_num - tm_num) - (te0 - tm0)) < 0.02

    # back-compat: psi present, real, unit-max normalised, right shape.
    assert res.psi.shape[1:] == (res.y.size, res.z.size)
    assert np.isclose(res.psi[0].max(), 1.0)


def test_low_contrast_slab_te_equals_tm():
    """Low-contrast slab: TE0 ~= TM0 ~= scalar. Confirms the vectorial solver
    collapses to the weakly-guiding limit and both match analytic."""
    n1, n2, d = 1.50, 1.45, 0.70e-6
    te0 = analytic_slab_neff(n1, n2, d, "te")
    tm0 = analytic_slab_neff(n1, n2, d, "tm")
    assert abs(te0 - tm0) < 5e-3  # nearly degenerate at low contrast

    te_num, tm_num, _ = solve_slab(
        n1, n2, d, Ly=8e-6, dy=0.1e-6, Lz=4.0e-6, ncell_slab=30, num_modes=6,
    )
    assert te_num is not None and tm_num is not None
    assert abs(te_num - te0) / te0 < 5e-3
    assert abs(tm_num - tm0) / tm0 < 5e-3
    # both close to each other, as expected in the scalar limit
    assert abs(te_num - tm_num) < 5e-3


def test_scalar_solver_would_fail_tm():
    """Document the motivation: the analytic TE and TM effective indices of the
    high-contrast slab are genuinely different, so any solver that returns a
    single polarization-independent n_eff (a scalar solver) is necessarily
    wrong for at least one of them by far more than the 0.5% tolerance."""
    n1, n2, d = 3.0, 1.0, 0.30e-6
    te0 = analytic_slab_neff(n1, n2, d, "te")
    tm0 = analytic_slab_neff(n1, n2, d, "tm")
    # A scalar solver has no n1^2/n2^2 factor -> it returns the TE value for
    # both. Its TM error would be:
    scalar_tm_err = abs(te0 - tm0) / tm0
    assert scalar_tm_err > 0.30  # >30%: hopelessly wrong, unlike the vector solver
