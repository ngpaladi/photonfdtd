"""Mode-decomposition S-parameters from port-plane DFT fields.

A waveguide port is a plane normal to the propagation axis (x, matching the
mode solver's convention). Given the frequency-domain tangential fields on that
plane - accumulated cheaply by a plane-restricted :class:`DFTMonitor` recording
``('Ey','Ez','Hy','Hz')`` - the complex amplitude of a guided mode is its
overlap with the simulated field via the unconjugated/conjugated cross-product
inner product (Meep's convention):

    <psi_a, psi_b> = integral_S [ E_a* x H_b + E_b x H_a* ] . n_hat  dA

For n_hat = x_hat this is ``E_y H_z - E_z H_y``. The forward amplitude of mode m
is the normalization-agnostic projection

    alpha_m^+ = <psi_m^+, psi_sim> / <psi_m^+, psi_m^+>,

and the backward amplitude uses the reversed mode ``psi_m^-`` (transverse E kept,
transverse H sign-flipped), which is orthogonal to ``psi_m^+`` under this inner
product - that orthogonality is what separates the forward-transmitted wave from
any counter-propagating reflection on the same plane. With a unit-power
unidirectional source ``S_ij = alpha_i^{out}``.

The functions are written with plain array operations (``*``, ``.conj()``,
``.sum()``), so they run unchanged on NumPy, CuPy, and JAX arrays. Passing the
simulated fields as JAX arrays (e.g. ``out['dft'][name][comp]`` inside a loss)
makes ``|S_ij|**2`` differentiable through the time-domain adjoint - the modes
are treated as constants, as in every adjoint FDTD.

References
----------
* Meep, "Mode Decomposition," https://meep.readthedocs.io/en/latest/Mode_Decomposition/
* A. Snyder & J. Love, *Optical Waveguide Theory* (orthogonality of guided modes).
"""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

_TANGENTIAL_X = ("Ey", "Ez", "Hy", "Hz")


def _overlap(a, b, dA):
    """<psi_a, psi_b> = integral (E_a* x H_b + E_b x H_a*).x_hat dA.

    Each of ``a``/``b`` is a 4-tuple of tangential (Ey, Ez, Hy, Hz), with the
    transverse plane flattened into a single trailing axis (and an optional
    leading frequency axis). The integral is a sum over that trailing axis, so
    it works whether or not a frequency axis is present.
    """
    Eay, Eaz, Hay, Haz = a
    Eby, Ebz, Hby, Hbz = b
    integrand = ((Eay.conj() * Hbz - Eaz.conj() * Hby)
                 + (Eby * Haz.conj() - Ebz * Hay.conj()))
    return integrand.sum(axis=-1) * dA


def mode_amplitudes(sim_fields: Dict, mode_fields: Dict, dA: float):
    """Forward/backward modal amplitudes ``(alpha_plus, alpha_minus)``.

    Parameters
    ----------
    sim_fields : dict
        ``'Ey','Ez','Hy','Hz'`` on the port plane, each shaped ``(n_freq, ...)``
        (any spatial layout; a size-1 plane axis is fine). NumPy/CuPy/JAX.
    mode_fields : dict
        The solved mode's ``'Ey','Ez','Hy','Hz'`` on the same transverse grid,
        broadcastable against one frequency slice.
    dA : float
        Transverse cell area (e.g. ``dy*dz``); cancels in the ratio but keeps
        the intermediate integrals physical.
    """
    # Flatten the transverse plane to one trailing axis; keep the leading
    # frequency axis on the simulated fields. This makes sim (n_freq, N) and
    # mode (N,) integrate consistently and tolerates a size-1 plane axis.
    sim = tuple(sim_fields[k].reshape(sim_fields[k].shape[0], -1)
                for k in _TANGENTIAL_X)
    Emy, Emz, Hmy, Hmz = (mode_fields[k].reshape(-1) for k in _TANGENTIAL_X)
    fwd = (Emy, Emz, Hmy, Hmz)               # psi_m^+
    bwd = (Emy, Emz, -Hmy, -Hmz)             # psi_m^- : transverse H flipped

    # alpha = <psi_mode, psi_sim> / <psi_mode, psi_mode>. The self-overlap in the
    # denominator (NOT the sim) is what normalizes; <psi^+, psi^-> = 0 makes the
    # two directions separable on a plane carrying both.
    alpha_plus = _overlap(fwd, sim, dA) / _overlap(fwd, fwd, dA)
    alpha_minus = _overlap(bwd, sim, dA) / _overlap(bwd, bwd, dA)
    return alpha_plus, alpha_minus


def port_fields(result, monitor_name: str) -> Dict:
    """Pull the four tangential components of a port DFTMonitor from a Result.

    Returns a dict of ``'Ey','Ez','Hy','Hz'`` arrays shaped ``(n_freq, ...)``.
    The monitor must have recorded those components on its port plane.
    """
    dft = result.dft[monitor_name]
    missing = [c for c in _TANGENTIAL_X if c not in dft]
    if missing:
        raise ValueError(
            f"port monitor {monitor_name!r} is missing components {missing}; "
            f"record components={_TANGENTIAL_X} on the port plane")
    return {c: dft[c] for c in _TANGENTIAL_X}


def mode_port_fields(mode_result, mode_index: int = 0) -> Dict:
    """The four tangential components of a solved mode, ready for overlap."""
    return {c: np.asarray(getattr(mode_result, c)[mode_index])
            for c in _TANGENTIAL_X}


def s_parameters(result, out_monitor: str, mode_result, mode_index: int = 0,
                 dA: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience: forward/backward amplitudes of ``mode_index`` at a port.

    With a unit-power unidirectional input at another port, ``alpha_plus`` is the
    S-parameter into this port. Returns ``(alpha_plus, alpha_minus)`` over the
    monitor's frequencies.
    """
    return mode_amplitudes(port_fields(result, out_monitor),
                           mode_port_fields(mode_result, mode_index), dA)
