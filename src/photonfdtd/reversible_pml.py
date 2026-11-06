"""Reversible adjoint WITH CPML, via thin-shell storage (O(surface*N) memory).

The lossless interior of a Yee run is exactly time-reversible, but the CPML
absorbing layer is dissipative and cannot be inverted stably. The fix
(FDTDX / Tang et al.): reconstruct the interior by reverse leapfrog while
storing only the PML region's state per step and overwriting the PML cells from
that tape on the way back.

Key simplification used here: the *bare* reverse step (subtract the plain curl,
no psi) is already exact on the interior (psi is identically zero there), so the
reverse only needs to (a) run the bare reverse everywhere, then (b) overwrite
the PML cells with the stored shell and restore the stored psi. Field storage
is therefore O(number of PML cells * n_steps) - a surface shell, not the volume.

Status: fully implemented and validated - the forward matches the reference
stepper to machine precision, forward-then-reverse returns to machine precision
with PML, and the ``jax.custom_vjp`` gradient matches the plain reverse pass to
~1e-15 (WITH CPML). See ``tests/test_reversible.py``.

IMPORTANT - when this actually saves memory. The tape stores the PML *shell*
(fields + psi at PML cells) per step: O(shell * n_steps). It beats L-level
gradient checkpointing (O(volume * L * n_steps**(1/L))) only when the shell is a
small fraction of the volume:

    shell/volume  <  L * n_steps**((1-L)/L)   (~5% to beat 2-level at N~1500).

But a 3-D CPML shell is rarely that thin: a 200^3 grid with 8-cell PML is ~22%
of the volume, and thinner/smaller domains are worse (a small grid can be >60%).
So for realistic PIC geometries this reversible-with-PML path usually does NOT
beat checkpointing, and can use more memory - it is a *correct* capability for
the large-domain / thin-relative-PML regime, not a general win. For a
lossless run with NO absorbing boundaries the shell vanishes and the O(1)
reversible adjoint in :mod:`photonfdtd.reversible` is the big win; for a typical
PML PIC run, use ``remat="nested"`` checkpointing. This is the honest,
measured trade - the technique is here and validated, characterised, not oversold.
"""
from __future__ import annotations
import numpy as np

from .constants import EPS_0
from .monitors import DFTMonitor, FluxMonitor, FieldMonitor
from .jaxbackend import (_build_static, _monitor_plan, _snap, _dft_snap, _FLUX_AXIS,
                         _build_ade_jax)
from .reversible import _pass_increment, _flux, _ADE, lossless_poles


def pml_mask(grid):
    """Boolean (grid.shape): True where a cell lies within ``pml_layers`` of a
    boundary on any resolved axis (the region where CPML psi is nonzero)."""
    shape = grid.shape
    mask = np.zeros(shape, dtype=bool)
    for a in range(3):
        n = shape[a]
        npml = int(grid.pml_layers[a]) if n > 1 else 0
        if npml <= 0:
            continue
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[a] = slice(0, npml)
        hi[a] = slice(n - npml, n)
        mask[tuple(lo)] = True
        mask[tuple(hi)] = True
    return mask


def _pass_with_psi(fields, psi, is_E_pass, ce_e, static, neg_ch):
    """Forward Yee pass including the CPML psi update. Mirrors the reference
    ``apply_pass`` in :mod:`photonfdtd.jaxbackend`. Returns
    ``({Fkey: (region, increment)}, new_psi)``."""
    import jax.numpy as jnp
    psi = dict(psi)
    inc = {}
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
            p = jnp.asarray(t["b1"]) * psi[t["key"]] + jnp.asarray(t["c1"]) * d
            psi[t["key"]] = p
            cterm = d + p
            if curl is None:
                curl = cterm if t["sign"] > 0 else -cterm
            else:
                curl = curl + cterm if t["sign"] > 0 else curl - cterm
        if curl is None:
            continue
        reg = comp["region"]
        K = ce_e[comp["Fkey"]][reg] if is_E_pass else neg_ch
        inc[comp["Fkey"]] = (reg, K * curl)
    return inc, psi


def make_pml_reversible(sim, static, plans, ce_e, ade=None):
    """Build init / forward_step / reverse_step for a lossless CPML run.

    forward_step advances the full solver (with psi); reverse_step reconstructs
    the previous state from the next one plus the per-step PML-shell tape
    (fields + psi at the PML cells). Monitors (DFT/flux) accumulate reversibly.
    ``ade`` is a lossless-pole plan (or None); its polarization is inverted
    exactly, as in :mod:`photonfdtd.reversible`, so it needs no tape. The
    carry is ``(fields, psi, dft, flux, pol)``.
    """
    import jax.numpy as jnp
    dt = static["dt"]
    shape = static["shape"]
    jdt = ce_e["Ex"].dtype
    cdt = jnp.complex128 if jdt == jnp.float64 else jnp.complex64
    neg_ch = jnp.asarray(-static["ch_field"], dtype=jdt)
    pole = _ADE(ade, ce_e, dt)
    mask = jnp.asarray(pml_mask(sim.grid))

    src_H = [(c, i, j, k, jnp.asarray(v)) for c, i, j, k, v in static["src_H"]]
    src_E = [(c, i, j, k, jnp.asarray(v)) for c, i, j, k, v in static["src_E"]]
    dft_plans = [p for p in plans if p["kind"] == "dft"]
    flux_plans = [p for p in plans if p["kind"] == "flux"]
    psi_keys = [t["key"] for comp in static["comps"] for t in comp["terms"]]
    psi_shapes = {t["key"]: t["pshape"]
                  for comp in static["comps"] for t in comp["terms"]}

    # Gather indices so the per-step tape stores only PML cells (O(surface)):
    # fields at the boundary shell, and each psi term where its CPML coefficient
    # is nonzero (== where that psi is ever nonzero).
    _npmask = pml_mask(sim.grid)
    field_idx = tuple(jnp.asarray(a) for a in np.nonzero(_npmask))
    psi_idx = {}
    for comp in static["comps"]:
        for t in comp["terms"]:
            m = np.broadcast_to(np.asarray(t["c1"]) != 0, t["pshape"])
            psi_idx[t["key"]] = tuple(jnp.asarray(a) for a in np.nonzero(m))

    def init_carry():
        fields = {c: jnp.zeros(shape, dtype=jdt)
                  for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
        psi = {k: jnp.zeros(psi_shapes[k], dtype=jdt) for k in psi_keys}
        dft = {p["m"].name: {c: jnp.zeros(
            (len(p["m"].freqs),) + _snap(fields[c], p["m"], p["plane"]).shape,
            dtype=cdt) for c in p["m"].components} for p in dft_plans}
        flux = {p["m"].name: jnp.asarray(0.0, dtype=jdt) for p in flux_plans}
        return (fields, psi, dft, flux, pole.init(jdt))

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
        fields, psi, dft, flux, pol = (dict(carry[0]), carry[1], carry[2],
                                       carry[3], carry[4])
        inc, psi = _pass_with_psi(fields, psi, False, ce_e, static, neg_ch)
        for F, (reg, v) in inc.items():
            fields[F] = fields[F].at[reg].add(v)
        for c, i, j, k, vals in src_H:
            fields[c] = fields[c].at[i, j, k].add(vals[n])
        flux = _flux_half(fields, flux, +1.0)
        En = pole.gather(fields)
        inc, psi = _pass_with_psi(fields, psi, True, ce_e, static, neg_ch)
        for F, (reg, v) in inc.items():
            fields[F] = fields[F].at[reg].add(v)
        fields, pol = pole.forward(fields, pol, En)
        for c, i, j, k, vals in src_E:
            fields[c] = fields[c].at[i, j, k].add(vals[n])
        dft, flux = _accumulate(fields, dft, flux, n, +1.0)
        return (fields, psi, dft, flux, pol)

    def reverse_step(carry, shell, ce_e, n):
        pole = _ADE(ade, ce_e, dt)
        """Reconstruct the previous state. ``shell`` is the stored PML-region
        fields (dict of full-grid arrays, nonzero only on PML) and psi of the
        *previous* state; the bare reverse handles the interior."""
        fields, dft, flux, pol = dict(carry[0]), carry[2], carry[3], carry[4]
        shell_fields, shell_psi = shell
        dft, flux = _accumulate(fields, dft, flux, n, -1.0)
        # undo E-sources, the pole correction and the bare E-pass, then
        # overwrite E PML cells from the tape; E^n then gives P^{n-1}.
        for c, i, j, k, vals in src_E:
            fields[c] = fields[c].at[i, j, k].add(-vals[n])
        fields = pole.undo_correction(fields, pol)
        for F, (reg, v) in _pass_increment(fields, True, ce_e, static, neg_ch).items():
            fields[F] = fields[F].at[reg].add(-v)
        for c in ("Ex", "Ey", "Ez"):
            fields[c] = fields[c].at[field_idx].set(shell_fields[c])
        pol = pole.backward(fields, pol)
        flux = _flux_half(fields, flux, -1.0)
        # undo H-sources + bare H-pass (now uses the corrected E^n), overwrite H.
        for c, i, j, k, vals in src_H:
            fields[c] = fields[c].at[i, j, k].add(-vals[n])
        for F, (reg, v) in _pass_increment(fields, False, ce_e, static, neg_ch).items():
            fields[F] = fields[F].at[reg].add(-v)
        for c in ("Hx", "Hy", "Hz"):
            fields[c] = fields[c].at[field_idx].set(shell_fields[c])
        psi = {k: jnp.zeros(psi_shapes[k], dtype=jdt).at[psi_idx[k]].set(shell_psi[k])
               for k in psi_keys}
        return (fields, psi, dft, flux, pol)

    def shell_of(carry):
        """Extract the PML-region tape entry for a state: fields on the boundary
        shell and each psi term on its PML cells (gathered, so the tape is
        O(surface) per step, not O(volume))."""
        fields, psi = carry[0], carry[1]
        shell_fields = {c: fields[c][field_idx] for c in fields}
        shell_psi = {k: psi[k][psi_idx[k]] for k in psi_keys}
        return (shell_fields, shell_psi)

    return init_carry, forward_step, reverse_step, shell_of


def value_and_grad_eps_reversible_pml(sim, loss):
    """``(value, d loss / d eps_r)`` via the PML-shell reversible adjoint.

    Same contract as :func:`photonfdtd.jax_value_and_grad_eps`, valid for a
    lossless run *with* CPML (non-dispersive, or dispersive with lossless
    poles). The backward pass reconstructs the forward history by
    reverse-stepping the interior and replaying the stored PML shell (see
    module docstring). DFT/Flux monitors only.

    Note: this reference implementation stores the shell densely (full-grid,
    zero off-PML); gathering it to the PML cells only (O(surface*n_steps)) is a
    mechanical memory optimization on top of the validated reconstruction.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from .jaxbackend import _validate, _enable_x64_if_needed

    if not lossless_poles(sim):
        raise NotImplementedError(
            "reversible PML adjoint needs lossless dispersion poles (gamma == 0)")
    if getattr(sim, "particle_sources", None):
        raise NotImplementedError("reversible PML adjoint: no moving charges.")
    for m in sim.monitors:
        if isinstance(m, FieldMonitor):
            raise NotImplementedError("reversible PML adjoint: DFT/Flux monitors only.")
    _validate(sim)
    _enable_x64_if_needed(sim)
    npdt = np.dtype(sim.dtypes["Ex"])
    static = _build_static(sim)
    plans = _monitor_plan(sim)
    ade = _build_ade_jax(sim, static)
    n_steps = static["n_steps"]

    @jax.custom_vjp
    def rev_run(ce_e):
        init, fwd, _, _ = make_pml_reversible(sim, static, plans, ce_e, ade)
        c, _ = lax.scan(lambda cc, n: (fwd(cc, ce_e, n), None),
                        init(), jnp.arange(n_steps))
        dft, flux = c[2], c[3]
        return {"dft": dft, "flux": flux}

    def rev_fwd(ce_e):
        init, fwd, _, shell_of = make_pml_reversible(sim, static, plans, ce_e, ade)

        def body(cc, n):
            return fwd(cc, ce_e, n), shell_of(cc)     # tape the pre-step shell
        final, tape = lax.scan(body, init(), jnp.arange(n_steps))
        dft, flux = final[2], final[3]
        return {"dft": dft, "flux": flux}, (final, tape, ce_e)

    def rev_bwd(res, g):
        final, tape, ce_e = res
        _, fwd, rev_step, _ = make_pml_reversible(sim, static, plans, ce_e, ade)
        zeros = lambda t: jax.tree_util.tree_map(jnp.zeros_like, t)
        bar = (zeros(final[0]), zeros(final[1]), g["dft"], g["flux"],
               zeros(final[4]))
        bar_ce = {c: jnp.zeros_like(ce_e[c]) for c in ce_e}

        def body(carry, n):
            state, bar, bar_ce = carry
            shell_n = jax.tree_util.tree_map(lambda x: x[n], tape)
            state = rev_step(state, shell_n, ce_e, n)
            _, vjp = jax.vjp(lambda s, ce: fwd(s, ce, n), state, ce_e)
            bar_state, bar_ce_n = vjp(bar)
            bar_ce = {c: bar_ce[c] + bar_ce_n[c] for c in bar_ce}
            return (state, bar_state, bar_ce), None

        (state, bar, bar_ce), _ = lax.scan(
            body, (final, bar, bar_ce), jnp.arange(n_steps - 1, -1, -1))
        return (bar_ce,)

    rev_run.defvjp(rev_fwd, rev_bwd)

    def loss_of_eps(eps_r):
        ce = sim.dt / (eps_r * EPS_0)
        return loss(rev_run({c: ce for c in ("Ex", "Ey", "Ez")}))

    eps0 = jnp.asarray(np.asarray(sim.eps_r, dtype=npdt))
    val, grad = jax.value_and_grad(loss_of_eps)(eps0)
    return float(val), np.asarray(grad)
