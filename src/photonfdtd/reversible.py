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

Scope / gating: no PML (``pml_layers`` all zero), no moving charges, and
DFT/Flux monitors only (a FieldMonitor's ``.set`` is not a running
accumulation). Dispersive media are supported when every pole is lossless
(``gamma == 0``, e.g. Sellmeier glasses such as :func:`~photonfdtd.silica`):
the pole recursion ``P^{n+1} = a P^n + b P^{n-1} + c E^n`` then has ``b = -1``
and is inverted exactly for ``P^{n-1}``. A lossy pole has ``|b| < 1``, so the
inverse amplifies roundoff by ``1/|b|`` per step and is refused. With PML, see
:mod:`photonfdtd.reversible_pml`.
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
                         _enable_x64_if_needed, _snap, _dft_snap, _FLUX_AXIS,
                         _flux, _build_ade_jax)


def reversible_available(sim):
    """Whether the reversible adjoint applies to ``sim`` (else use checkpointing)."""
    if any(int(p) > 0 for p in sim.grid.pml_layers):
        return False, "reversible adjoint requires no PML (pml_layers all 0)"
    if not lossless_poles(sim):
        return False, ("reversible adjoint needs lossless dispersion poles "
                       "(gamma == 0); a lossy pole cannot be stepped backward "
                       "stably")
    if getattr(sim, "particle_sources", None):
        return False, "reversible adjoint does not support moving-charge sources"
    for m in sim.monitors:
        if isinstance(m, FieldMonitor):
            return False, "reversible adjoint supports DFT/Flux monitors only"
    return True, ""


def lossless_poles(sim):
    """True if ``sim`` has no dispersive medium with a lossy (gamma != 0) pole."""
    for s in sim.structures:
        med = getattr(s, "medium", None)
        if getattr(med, "is_dispersive", False) and \
                any(p.gamma != 0.0 for p in med.poles):
            return False
    return True


class _ADE:
    """Lossless-pole polarization stepping, forward and exactly backward.

    Mirrors the ADE update in :func:`photonfdtd.jaxbackend._device_simulate`.
    The state is ``(Pcur, Pprev)``, each ``(npole, 3, n_disp)``, or ``()``
    without dispersion (every method is then a no-op).
    """

    def __init__(self, ade, ce_e, dt):
        import jax.numpy as jnp
        self.on = ade is not None
        if not self.on:
            return
        self.idx = tuple(jnp.asarray(i) for i in ade["idx"])
        self.A = jnp.asarray(ade["A"]).reshape(-1, 1, 1)
        self.B = jnp.asarray(ade["B"]).reshape(-1, 1, 1)
        self.C = jnp.asarray(ade["C"])[:, None, :]
        self.inv = [ce_e[c][self.idx] / dt for c in ("Ex", "Ey", "Ez")]
        self.shape = (ade["npole"], 3, ade["n_disp"])

    def init(self, jdt):
        import jax.numpy as jnp
        if not self.on:
            return ()
        return (jnp.zeros(self.shape, jdt), jnp.zeros(self.shape, jdt))

    def gather(self, fields):
        import jax.numpy as jnp
        if not self.on:
            return None
        return jnp.stack([fields[c][self.idx] for c in ("Ex", "Ey", "Ez")])

    def _correct(self, fields, delta, sgn):
        for a, c in enumerate(("Ex", "Ey", "Ez")):
            fields[c] = fields[c].at[self.idx].add(-sgn * self.inv[a] * delta[a])
        return fields

    def forward(self, fields, pol, En):
        """After the E curl pass: advance P and apply the correction to E."""
        import jax.numpy as jnp
        if not self.on:
            return fields, pol
        Pcur, Pprev = pol
        Pnew = self.A * Pcur + self.B * Pprev + self.C * En[None]
        fields = self._correct(fields, jnp.sum(Pnew - Pcur, axis=0), +1.0)
        return fields, (Pnew, Pcur)

    def undo_correction(self, fields, pol):
        """Remove the correction ``forward`` added to E (state unchanged)."""
        import jax.numpy as jnp
        if not self.on:
            return fields
        Pnew, Pcur = pol
        return self._correct(fields, jnp.sum(Pnew - Pcur, axis=0), -1.0)

    def backward(self, fields, pol):
        """Given E^n (restored) and ``(P^{n+1}, P^n)``, return ``(P^n, P^{n-1})``."""
        if not self.on:
            return pol
        Pnew, Pcur = pol
        Pprev = (Pnew - self.A * Pcur - self.C * self.gather(fields)[None]) / self.B
        return (Pcur, Pprev)


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


def _make_reversible(sim, static, plans, ce_e, ade=None):
    """Build ``forward_step`` / ``reverse_step`` / init / output-extractor for a
    lossless no-PML run. Monitors (DFT, flux) live in the carry and accumulate
    reversibly. ``ce_e`` is a dict of per-component coefficients (JAX arrays);
    ``ade`` the lossless-pole plan from :func:`_build_ade_jax` (or None). The
    carry is ``(fields, dft, flux, pol)``."""
    import jax.numpy as jnp
    dt = static["dt"]
    shape = static["shape"]
    jdt = ce_e["Ex"].dtype
    cdt = jnp.complex128 if jdt == jnp.float64 else jnp.complex64
    neg_ch = jnp.asarray(-static["ch_field"], dtype=jdt)
    pole = _ADE(ade, ce_e, dt)

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
        return (fields, dft, flux, pole.init(jdt))

    def _accumulate(fields, dft, flux, n, sgn):
        dft = {k: dict(v) for k, v in dft.items()}
        flux = dict(flux)
        for p in dft_plans:
            m = p["m"]
            for c in m.components:
                t = (n + (1.0 if c[0] == "E" else 0.5)) * dt
                ph = jnp.exp(1j * jnp.asarray(p["omega"]) * t).astype(cdt)
                s = _dft_snap(fields[c], c, m, p["plane"]).astype(cdt)
                ph = ph.reshape((-1,) + (1,) * s.ndim)
                dft[m.name][c] = dft[m.name][c] + sgn * ph * s[None] * dt
        flux = _flux_half(fields, flux, sgn)
        return dft, flux

    def _flux_half(fields, flux, sgn):
        # Half of the time-centred flux increment: called on (E^n, H^{n+1/2})
        # before the E update and on (E^{n+1}, H^{n+1/2}) after it, as in
        # jaxbackend.
        flux = dict(flux)
        for p in flux_plans:
            flux[p["m"].name] = flux[p["m"].name] + sgn * 0.5 * _flux(
                fields, p["m"], sim.grid, jnp) * dt
        return flux

    def forward_step(carry, ce_e, n):
        pole = _ADE(ade, ce_e, dt)        # its 1/eps factor depends on ce_e
        fields, dft, flux, pol = dict(carry[0]), carry[1], carry[2], carry[3]
        for F, (reg, v) in _pass_increment(fields, False, ce_e, static, neg_ch).items():
            fields[F] = fields[F].at[reg].add(v)
        for c, i, j, k, vals in src_H:
            fields[c] = fields[c].at[i, j, k].add(vals[n])
        flux = _flux_half(fields, flux, +1.0)
        En = pole.gather(fields)
        for F, (reg, v) in _pass_increment(fields, True, ce_e, static, neg_ch).items():
            fields[F] = fields[F].at[reg].add(v)
        fields, pol = pole.forward(fields, pol, En)
        for c, i, j, k, vals in src_E:
            fields[c] = fields[c].at[i, j, k].add(vals[n])
        dft, flux = _accumulate(fields, dft, flux, n, +1.0)
        return (fields, dft, flux, pol)

    def reverse_step(carry, ce_e, n):
        pole = _ADE(ade, ce_e, dt)
        fields, dft, flux, pol = dict(carry[0]), carry[1], carry[2], carry[3]
        dft, flux = _accumulate(fields, dft, flux, n, -1.0)
        for c, i, j, k, vals in src_E:
            fields[c] = fields[c].at[i, j, k].add(-vals[n])
        fields = pole.undo_correction(fields, pol)
        for F, (reg, v) in _pass_increment(fields, True, ce_e, static, neg_ch).items():
            fields[F] = fields[F].at[reg].add(-v)
        pol = pole.backward(fields, pol)
        flux = _flux_half(fields, flux, -1.0)
        for c, i, j, k, vals in src_H:
            fields[c] = fields[c].at[i, j, k].add(-vals[n])
        for F, (reg, v) in _pass_increment(fields, False, ce_e, static, neg_ch).items():
            fields[F] = fields[F].at[reg].add(-v)
        return (fields, dft, flux, pol)

    def extract_outputs(carry):
        _, dft, flux, _ = carry
        return {"dft": dft, "flux": flux}

    return init_carry, forward_step, reverse_step, extract_outputs


def reversible_run(sim):
    """``ce_e -> {'fields', 'dft', 'flux'}``, differentiable with the
    reversible adjoint (O(1) memory in the number of timesteps).

    ``ce_e`` is a dict ``{'Ex','Ey','Ez'}`` of per-component E-update
    coefficients, as for :func:`photonfdtd.jaxbackend.value_and_grad_params`;
    ``jax.grad`` through the returned function reconstructs the forward
    history by stepping the physics backward. Requires
    :func:`reversible_available`.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax

    ok, why = reversible_available(sim)
    if not ok:
        raise NotImplementedError(why + "; use jax_value_and_grad_eps instead.")
    _validate(sim)
    _enable_x64_if_needed(sim)
    static = _build_static(sim)
    plans = _monitor_plan(sim)
    ade = _build_ade_jax(sim, static)
    n_steps = static["n_steps"]

    @jax.custom_vjp
    def rev_run(ce_e):
        init, fwd, _, extract = _make_reversible(sim, static, plans, ce_e, ade)
        c, _ = lax.scan(lambda cc, n: (fwd(cc, ce_e, n), None),
                        init(), jnp.arange(n_steps))
        return extract(c)

    def rev_fwd(ce_e):
        init, fwd, _, extract = _make_reversible(sim, static, plans, ce_e, ade)
        c, _ = lax.scan(lambda cc, n: (fwd(cc, ce_e, n), None),
                        init(), jnp.arange(n_steps))
        return extract(c), (c, ce_e)

    def rev_bwd(res, g):
        final_carry, ce_e = res
        _, fwd, rev_step, _ = _make_reversible(sim, static, plans, ce_e, ade)
        # Seed the adjoint: the outputs are exactly the carry's monitor
        # accumulators, so their cotangents map straight in; fields start at 0.
        zeros = lambda t: jax.tree_util.tree_map(jnp.zeros_like, t)
        bar = (zeros(final_carry[0]), g["dft"], g["flux"], zeros(final_carry[3]))
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

    def run(ce_e):
        out = rev_run(dict(ce_e))
        return {"fields": {}, "dft": out["dft"], "flux": out["flux"]}
    return run


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
    run = reversible_run(sim)
    npdt = np.dtype(sim.dtypes["Ex"])

    def loss_of_eps(eps_r):
        ce = sim.dt / (eps_r * EPS_0)
        return loss(run({c: ce for c in ("Ex", "Ey", "Ez")}))

    eps0 = jnp.asarray(np.asarray(sim.eps_r, dtype=npdt))
    val, grad = jax.value_and_grad(loss_of_eps)(eps0)
    return float(val), np.asarray(grad)
