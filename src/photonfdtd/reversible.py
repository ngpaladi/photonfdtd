"""Reversible-in-time adjoint for the JAX FDTD (O(1)-in-steps memory).

The lossless, non-dispersive Yee leapfrog is *exactly time-reversible*: the H
and E passes only add curl/source increments, so given the end-of-step state
``(E^{n+1}, H^{n+1/2})`` the previous state is recovered by subtracting the same
increments in reverse order - no division, no lost information (contrast the
dissipative CPML psi recursion, which is why this path requires *no absorbing
boundaries*). A running DFT/flux monitor accumulates by ``+=`` and so is
reversible too.

That reversibility lets the reverse-mode adjoint reconstruct the entire forward
trajectory on the fly instead of storing it: the backward pass keeps only the
final state, one working state, and one cotangent - O(1) memory in the number
of timesteps N, versus O(N) for a plain reverse pass or O(N^{1/L}) for
checkpointing. This is the memory-optimal adjoint for closed / periodic
differentiable runs (resonators, photonic crystals, periodic metasurfaces).

Scope / gating: no PML (``pml_layers`` all zero), no dispersion, no moving
charges, and DFT/Flux monitors only (a FieldMonitor's ``.set`` is not a running
accumulation). Absorbing-boundary (PML) support needs the interior reconstructed
while the thin PML shell's state is stored per step - designed, not yet built.
Implemented as a ``jax.custom_vjp`` whose backward pass steps the physics
backward; validated by ``forward-then-reverse`` returning to machine precision
and by matching the plain (checkpointed) gradient.

References
----------
* Y. Tang et al., "Time-reversal differentiation of FDTD for photonic inverse
  design," *ACS Photonics* (2023) - reversible-FDTD adjoint.
* Reversible-network / RevNet adjoints; Griewank & Walther, Revolve (2000).
"""
from __future__ import annotations
import numpy as np

from .constants import EPS_0
from .monitors import DFTMonitor, FluxMonitor, FieldMonitor
from .jaxbackend import (_build_static, _monitor_plan, _validate,
                         _enable_x64_if_needed, _snap, _FLUX_AXIS)


def reversible_available(sim):
    """Whether the reversible adjoint applies to ``sim`` (else use checkpointing)."""
    if any(int(p) > 0 for p in sim.grid.pml_layers):
        return False, "reversible adjoint requires no PML (pml_layers all 0)"
    if getattr(sim, "_has_dispersion", False):
        return False, "reversible adjoint does not support dispersive media"
    if getattr(sim, "particle_sources", None):
        return False, "reversible adjoint does not support moving-charge sources"
    for m in sim.monitors:
        if isinstance(m, FieldMonitor):
            return False, "reversible adjoint supports DFT/Flux monitors only"
    return True, ""


def _pass_increment(fields, is_E_pass, ce_e, static, neg_ch):
    """Per-component field increment ``K * curl`` for one Yee pass (no psi).

    Returns ``{Fkey: (region, increment)}``. Identical arithmetic to the forward
    stepper's ``apply_pass`` with the (all-zero, no-PML) psi terms dropped.
    """
    out = {}
    for comp in static["comps"]:
        if comp["is_E"] != is_E_pass:
            continue
        curl = None
        for t in comp["terms"]:
            ax = t["ax"]
            src = fields[t["src"]]
            hi = [slice(None)] * 3
            lo = [slice(None)] * 3
            hi[ax] = slice(1, None)
            lo[ax] = slice(0, -1)
            d = (src[tuple(hi)] - src[tuple(lo)]) / t["dl"]
            d = d[t["osl"]]
            if curl is None:
                curl = d if t["sign"] > 0 else -d
            else:
                curl = curl + d if t["sign"] > 0 else curl - d
        if curl is None:
            continue
        reg = comp["region"]
        K = ce_e[comp["Fkey"]][reg] if is_E_pass else neg_ch
        out[comp["Fkey"]] = (reg, K * curl)
    return out


def _make_reversible(sim, static, plans, ce_e):
    """Build ``forward_step`` / ``reverse_step`` / init / output-extractor for a
    lossless no-PML run. Monitors (DFT, flux) live in the carry and accumulate
    reversibly. ``ce_e`` is a dict of per-component coefficients (JAX arrays)."""
    import jax.numpy as jnp
    dt = static["dt"]
    shape = static["shape"]
    jdt = ce_e["Ex"].dtype
    cdt = jnp.complex128 if jdt == jnp.float64 else jnp.complex64
    neg_ch = jnp.asarray(-static["ch_field"], dtype=jdt)

    src_H = [(c, i, j, k, jnp.asarray(v)) for c, i, j, k, v in static["src_H"]]
    src_E = [(c, i, j, k, jnp.asarray(v)) for c, i, j, k, v in static["src_E"]]

    dft_plans = [p for p in plans if p["kind"] == "dft"]
    flux_plans = [p for p in plans if p["kind"] == "flux"]

    def init_carry():
        fields = {c: jnp.zeros(shape, dtype=jdt)
                  for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
        dft = {p["m"].name: {c: jnp.zeros(
            (len(p["m"].freqs),) + _snap(fields[c], p["m"], p["plane"]).shape,
            dtype=cdt) for c in p["m"].components} for p in dft_plans}
        flux = {p["m"].name: jnp.asarray(0.0, dtype=jdt) for p in flux_plans}
        return (fields, dft, flux)

    def _accumulate(fields, dft, flux, n, sgn):
        dft = {k: dict(v) for k, v in dft.items()}
        flux = dict(flux)
        for p in dft_plans:
            m = p["m"]
            for c in m.components:
                t = (n + (1.0 if c[0] == "E" else 0.5)) * dt
                ph = jnp.exp(1j * jnp.asarray(p["omega"]) * t).astype(cdt)
                s = _snap(fields[c], m, p["plane"]).astype(cdt)
                ph = ph.reshape((-1,) + (1,) * s.ndim)
                dft[m.name][c] = dft[m.name][c] + sgn * ph * s[None] * dt
        for p in flux_plans:
            flux[p["m"].name] = flux[p["m"].name] + sgn * _flux(
                fields, p["m"], sim.grid, jnp) * dt
        return dft, flux

    def forward_step(carry, ce_e, n):
        fields, dft, flux = dict(carry[0]), carry[1], carry[2]
        for F, (reg, v) in _pass_increment(fields, False, ce_e, static, neg_ch).items():
            fields[F] = fields[F].at[reg].add(v)
        for c, i, j, k, vals in src_H:
            fields[c] = fields[c].at[i, j, k].add(vals[n])
        for F, (reg, v) in _pass_increment(fields, True, ce_e, static, neg_ch).items():
            fields[F] = fields[F].at[reg].add(v)
        for c, i, j, k, vals in src_E:
            fields[c] = fields[c].at[i, j, k].add(vals[n])
        dft, flux = _accumulate(fields, dft, flux, n, +1.0)
        return (fields, dft, flux)

    def reverse_step(carry, ce_e, n):
        fields, dft, flux = dict(carry[0]), carry[1], carry[2]
        dft, flux = _accumulate(fields, dft, flux, n, -1.0)
        for c, i, j, k, vals in src_E:
            fields[c] = fields[c].at[i, j, k].add(-vals[n])
        for F, (reg, v) in _pass_increment(fields, True, ce_e, static, neg_ch).items():
            fields[F] = fields[F].at[reg].add(-v)
        for c, i, j, k, vals in src_H:
            fields[c] = fields[c].at[i, j, k].add(-vals[n])
        for F, (reg, v) in _pass_increment(fields, False, ce_e, static, neg_ch).items():
            fields[F] = fields[F].at[reg].add(-v)
        return (fields, dft, flux)

    def extract_outputs(carry):
        _, dft, flux = carry
        return {"dft": dft, "flux": flux}

    return init_carry, forward_step, reverse_step, extract_outputs


def value_and_grad_eps_reversible(sim, loss):
    """``(value, d loss / d eps_r)`` via the reversible adjoint - O(1) memory in
    the number of timesteps.

    Same contract as :func:`photonfdtd.jax_value_and_grad_eps` but the backward
    pass reconstructs the forward field history by stepping the physics backward
    (see module docstring) instead of storing or recomputing it. Requires
    :func:`reversible_available`; raises otherwise.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax

    ok, why = reversible_available(sim)
    if not ok:
        raise NotImplementedError(why + "; use jax_value_and_grad_eps instead.")
    _validate(sim)
    _enable_x64_if_needed(sim)
    npdt = np.dtype(sim.dtypes["Ex"])
    static = _build_static(sim)
    plans = _monitor_plan(sim)
    n_steps = static["n_steps"]

    @jax.custom_vjp
    def rev_run(ce_e):
        init, fwd, _, extract = _make_reversible(sim, static, plans, ce_e)
        c, _ = lax.scan(lambda cc, n: (fwd(cc, ce_e, n), None),
                        init(), jnp.arange(n_steps))
        return extract(c)

    def rev_fwd(ce_e):
        init, fwd, _, extract = _make_reversible(sim, static, plans, ce_e)
        c, _ = lax.scan(lambda cc, n: (fwd(cc, ce_e, n), None),
                        init(), jnp.arange(n_steps))
        return extract(c), (c, ce_e)

    def rev_bwd(res, g):
        final_carry, ce_e = res
        _, fwd, rev_step, _ = _make_reversible(sim, static, plans, ce_e)
        # Seed the adjoint: the outputs are exactly the carry's monitor
        # accumulators, so their cotangents map straight in; fields start at 0.
        bar_fields = {c: jnp.zeros_like(final_carry[0][c]) for c in final_carry[0]}
        bar = (bar_fields, g["dft"], g["flux"])
        bar_ce = {c: jnp.zeros_like(ce_e[c]) for c in ce_e}

        def body(carry, n):
            state, bar, bar_ce = carry
            state = rev_step(state, ce_e, n)          # reconstruct carry_n
            _, vjp = jax.vjp(lambda s, ce: fwd(s, ce, n), state, ce_e)
            bar_state, bar_ce_n = vjp(bar)
            bar_ce = {c: bar_ce[c] + bar_ce_n[c] for c in bar_ce}
            return (state, bar_state, bar_ce), None

        (state, bar, bar_ce), _ = lax.scan(
            body, (final_carry, bar, bar_ce), jnp.arange(n_steps - 1, -1, -1))
        return (bar_ce,)

    rev_run.defvjp(rev_fwd, rev_bwd)

    def loss_of_eps(eps_r):
        ce = sim.dt / (eps_r * EPS_0)
        return loss(rev_run({c: ce for c in ("Ex", "Ey", "Ez")}))

    eps0 = jnp.asarray(np.asarray(sim.eps_r, dtype=npdt))
    val, grad = jax.value_and_grad(loss_of_eps)(eps0)
    return float(val), np.asarray(grad)


def _flux(fields, m, grid, jnp):
    axis = _FLUX_AXIS[m.plane_axis]
    coord = grid.coords[axis]
    idx = 0 if coord.size == 1 else int(np.argmin(np.abs(coord - m.plane_position)))
    Ex, Ey, Ez = fields["Ex"], fields["Ey"], fields["Ez"]
    Hx, Hy, Hz = fields["Hx"], fields["Hy"], fields["Hz"]
    if axis == 0:
        S = Ey[idx] * Hz[idx] - Ez[idx] * Hy[idx]
        dA = (grid.cell_size[1] if grid.shape[1] > 1 else 1.0) * \
             (grid.cell_size[2] if grid.shape[2] > 1 else 1.0)
    elif axis == 1:
        S = Ez[:, idx] * Hx[:, idx] - Ex[:, idx] * Hz[:, idx]
        dA = (grid.cell_size[0] if grid.shape[0] > 1 else 1.0) * \
             (grid.cell_size[2] if grid.shape[2] > 1 else 1.0)
    else:
        S = Ex[:, :, idx] * Hy[:, :, idx] - Ey[:, :, idx] * Hx[:, :, idx]
        dA = (grid.cell_size[0] if grid.shape[0] > 1 else 1.0) * \
             (grid.cell_size[1] if grid.shape[1] > 1 else 1.0)
    return jnp.sum(S) * dA
