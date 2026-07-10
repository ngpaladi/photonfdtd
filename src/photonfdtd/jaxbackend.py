"""JAX backend: a pure-functional FDTD time step under ``jax.lax.scan``.

This is a functional reimplementation of the Yee + CPML update the NumPy/Numba
backends run imperatively. JAX arrays are immutable, so every field/psi update
is expressed functionally (``x.at[region].add(...)``) and the whole time loop is
a ``lax.scan`` over a carry of (fields, psi, monitor accumulators). That makes
the solver:

* **JIT-compiled** through XLA (CPU / GPU / TPU from one code path), and
* **differentiable**: the stepper is a pure function of the per-component
  E-update coefficients (hence of ``eps_r`` / the subpixel-smoothed tensor) and
  of the dispersive-pole coefficients, so ``jax.grad`` of any scalar built from
  the monitor outputs flows back to the permittivity, the smoothed geometry, or
  the pole strengths - a time-domain adjoint, the engine of gradient-based
  inverse design. See :func:`value_and_grad_eps` (scalar-eps case) and
  ``tests/test_jax_accuracy.py`` (a pole-strength gradient).

Accuracy features: anisotropic subpixel smoothing (``subpixel=True``, via a
per-component coefficient) and dispersive media (Lorentz/Drude/Sellmeier poles
advanced by the ADE recursion inside the ``lax.scan`` carry) both run on this
path and match the NumPy reference to floating-point reordering.

The per-cell arithmetic mirrors the fused Numba kernel's ``cy - cz`` form (dense
CPML psi), so results track the in-core backends to floating reordering
(~1e-6 single / ~1e-11 double), like the Numba backend.

Sources: soft point/dipole sources (so expanded ModeSource / SinglePhoton
sources work too) and moving ``ChargedParticle`` charges (Cherenkov). The
particle trajectory is deterministic and field-independent, so its Gaussian
current cloud is precomputed per step on host and injected inside the scan via
``dynamic_slice`` / ``dynamic_update_slice``; only the ``ce_field`` factor stays
symbolic, so gradients still flow through it.

Monitors: ``FieldMonitor`` / ``DFTMonitor`` / ``FluxMonitor``. CPML, 1D/2D/3D,
float32/float64. The disk-side ``FieldMonitor(compression=)`` / ``out_of_core``
machinery is not part of the JAX device path and raises a clear error.
"""
from __future__ import annotations
import math
from typing import Dict, List
import numpy as np

from .constants import EPS_0, MU_0
from .monitors import FieldMonitor, FluxMonitor, DFTMonitor

# Component update table, identical in structure to Simulation.run's _table:
# (F, aligned_axis, (axA, srcA), (axB, srcB), is_E). curl = termA - termB.
_TABLE = [
    ("Hx", 0, (1, "Ez"), (2, "Ey"), False),
    ("Hy", 1, (2, "Ex"), (0, "Ez"), False),
    ("Hz", 2, (0, "Ey"), (1, "Ex"), False),
    ("Ex", 0, (1, "Hz"), (2, "Hy"), True),
    ("Ey", 1, (2, "Hx"), (0, "Hz"), True),
    ("Ez", 2, (0, "Hy"), (1, "Hx"), True),
]
_AXNAME = {0: "x", 1: "y", 2: "z"}
_FLUX_AXIS = {"x": 0, "y": 1, "z": 2}


def _crop(n, is_E):
    """Region slice on a derivative axis: [1:] for E, [:-1] for H, all if flat."""
    if n <= 1:
        return slice(None)
    return slice(1, None) if is_E else slice(0, -1)


def _region_shape(region, shape):
    out = []
    for i in range(3):
        s = region[i]
        if isinstance(s, slice) and s != slice(None):
            lo, hi, _ = s.indices(shape[i])
            out.append(hi - lo)
        else:
            out.append(shape[i])
    return tuple(out)


def _build_static(sim):
    """Fixed-for-the-run data: regions, per-term coeff slices, source tables."""
    grid = sim.grid
    shape = tuple(grid.shape)
    cell = tuple(d if d > 0 else 1.0 for d in grid.cell_size)
    dt = sim.dt
    n_steps = sim.n_steps
    npdt = np.dtype(sim.dtypes["Ex"])

    be = [np.asarray(sim._b_e[a]) for a in range(3)]
    ce = [np.asarray(sim._c_e[a]) for a in range(3)]
    bh = [np.asarray(sim._b_h[a]) for a in range(3)]
    ch = [np.asarray(sim._c_h[a]) for a in range(3)]

    comps = []
    for Fkey, aligned, (axA, srcA), (axB, srcB), is_E in _TABLE:
        region = [slice(None), slice(None), slice(None)]
        region[axA] = _crop(shape[axA], is_E)
        region[axB] = _crop(shape[axB], is_E)
        region = tuple(region)
        b_src = be if is_E else bh
        c_src = ce if is_E else ch
        rshape = _region_shape(region, shape)
        terms = []
        for (ax, src, other_ax, sign) in ((axA, srcA, axB, +1.0),
                                          (axB, srcB, axA, -1.0)):
            if shape[ax] <= 1:
                continue
            osl = [slice(None), slice(None), slice(None)]
            osl[other_ax] = _crop(shape[other_ax], is_E)
            ridx = region[ax]
            b1 = b_src[ax][ridx].astype(npdt).reshape(
                [-1 if i == ax else 1 for i in range(3)])
            c1 = c_src[ax][ridx].astype(npdt).reshape(
                [-1 if i == ax else 1 for i in range(3)])
            terms.append(dict(ax=ax, src=src, sign=sign, osl=tuple(osl),
                              b1=b1, c1=c1, key=f"{Fkey}_{_AXNAME[ax]}",
                              dl=cell[ax], pshape=rshape))
        comps.append(dict(Fkey=Fkey, region=region, is_E=is_E, terms=terms))

    src_H, src_E = [], []
    steps = np.arange(n_steps)
    for src in sim.sources:
        i, j, k = grid.index_at(src.position)
        t = (steps + (0.5 if src.component[0] == "H" else 1.0)) * dt
        vals = np.asarray(src.amplitude * src.waveform(t), dtype=npdt)
        entry = (src.component, int(i), int(j), int(k), vals)
        (src_H if src.component[0] == "H" else src_E).append(entry)

    return dict(shape=shape, cell=cell, dt=dt, n_steps=n_steps, npdt=npdt,
                comps=comps, ch_field=float(dt / MU_0), src_H=src_H, src_E=src_E)


def _particle_plan(sim, static):
    """Precompute each moving charge's per-step Gaussian-cloud deposition.

    The trajectory is deterministic and field-independent, so the stencil
    location and the normalised weights are fully known on host - only the
    ``ce_field`` factor stays symbolic (kept in the kernel via a dynamic slice),
    so gradients w.r.t. ``eps_r`` still flow through the deposit. Reproduces the
    in-core :meth:`Simulation._inject_particle_currents` exactly: each step's
    deposit is embedded in a fixed-size window ``R`` (so ``lax.dynamic_slice``
    can place it), zero-padded outside the actual [i0,i1) stencil, and zero on
    steps where the particle centre is outside the physical interior / time
    window.
    """
    grid = sim.grid
    shape = static["shape"]
    dt = static["dt"]
    n_steps = static["n_steps"]
    npdt = static["npdt"]
    if not sim.particle_sources:
        return []
    dV = 1.0
    for a in range(3):
        if shape[a] > 1:
            dV *= grid.cell_size[a]

    plans = []
    for p in sim.particle_sources:
        radius = max(1, int(math.ceil(3.0 * p.cloud_cells)))
        R = tuple(min(2 * radius + 1, shape[a]) if shape[a] > 1 else 1
                  for a in range(3))
        coef3 = np.zeros(3, dtype=npdt)
        for a in range(3):
            if p.velocity3[a] != 0.0 and shape[a] > 1:
                coef3[a] = -(p.charge * p.velocity3[a]) / dV
        starts = np.zeros((n_steps, 3), dtype=np.int32)
        wblocks = np.zeros((n_steps,) + R, dtype=npdt)

        for si in range(n_steps):
            t_now = (si + 1.0) * dt                       # E-time, as in-core
            if not (p.t_start <= t_now <= p.t_stop):
                continue
            pos = p.position_at(t_now)
            wvecs = [None, None, None]
            inside = True
            for a in range(3):
                n = shape[a]
                coord = np.asarray(grid.coords[a])
                if n == 1:
                    wvecs[a] = np.array([1.0]); starts[si, a] = 0
                    continue
                npml = grid.pml_layers[a]
                if pos[a] < coord[npml] or pos[a] > coord[n - 1 - npml]:
                    inside = False
                    break
                dl = grid.cell_size[a]
                sigma = max(p.cloud_cells, 1e-6) * dl
                ic = int(round((pos[a] - coord[0]) / dl))
                i0 = max(0, ic - radius)
                i1 = min(n, ic + radius + 1)
                w = np.exp(-0.5 * ((coord[i0:i1] - pos[a]) / sigma) ** 2)
                wsum = w.sum()
                if wsum == 0.0:
                    inside = False
                    break
                w = w / wsum
                start = int(np.clip(i0, 0, n - R[a]))     # window fits [i0,i1)
                vec = np.zeros(R[a])
                vec[i0 - start:i0 - start + (i1 - i0)] = w
                wvecs[a] = vec
                starts[si, a] = start
            if not inside:
                continue
            wblocks[si] = (wvecs[0].reshape(-1, 1, 1)
                           * wvecs[1].reshape(1, -1, 1)
                           * wvecs[2].reshape(1, 1, -1)).astype(npdt)

        plans.append(dict(R=R, coef3=coef3, starts=starts, wblocks=wblocks,
                          axes=[a for a in range(3) if coef3[a] != 0.0]))
    return plans


def _build_ade_jax(sim, static):
    """Dispersive-media (ADE) plan for the JAX device path, stored *masked*.

    Polarization lives only on the dispersive cells (their flat indices), so the
    scan carry is (npole, 3, n_disp) rather than (npole, 3, *grid) - for a metal
    nanostructure in a large domain that is orders of magnitude smaller. Each
    step gathers E at those cells, advances the poles, and scatters the
    correction back; gather/scatter are differentiable in JAX, so this keeps the
    inverse-design gradients while cutting memory to the material's footprint.

    Returns None if nothing dispersive occupies the grid. The recursion
    coefficients a, b are per-pole scalars; the driving coefficient c is masked
    per pole (zero on cells belonging to a different material).
    """
    if not getattr(sim, "_has_dispersion", False):
        return None
    grid = sim.grid
    shape = static["shape"]
    npdt = static["npdt"]
    dt = static["dt"]

    id_grid = np.full(shape, -1, dtype=np.int32)
    media: List = []
    med_id: Dict[int, int] = {}
    for s in sim.structures:
        med = getattr(s, "medium", None)
        mask = s.region_mask(grid)
        if getattr(med, "is_dispersive", False):
            key = id(med)
            if key not in med_id:
                med_id[key] = len(media)
                media.append(med)
            id_grid[mask] = med_id[key]
        else:
            id_grid[mask] = -1

    max_w0dt = max((m.max_pole_omega() for m in media), default=0.0) * dt
    if max_w0dt >= 2.0:
        raise ValueError(
            f"A dispersive medium has a pole with omega0*dt = {max_w0dt:.2f} "
            ">= 2, unstable for the explicit ADE update at this resolution. Use "
            "a finer grid, restrict the medium to in-band poles, or "
            "DispersiveMedium.at_wavelength(lambda) for a fixed-index Medium."
        )

    disp = id_grid >= 0
    idx = np.nonzero(disp)                          # (I, J, K), each (n_disp,)
    n_disp = idx[0].size
    if n_disp == 0:
        return None
    cell_mat = id_grid[disp]                        # (n_disp,) material id per cell

    a_list, b_list, c_rows = [], [], []
    inv_eps = np.zeros(n_disp, dtype=npdt)
    for mid, med in enumerate(media):
        inv_eps[cell_mat == mid] = 1.0 / (EPS_0 * med.eps_inf)
        for p in med.poles:
            denom = 1.0 + p.gamma * dt
            a_list.append((2.0 - p.omega0 ** 2 * dt ** 2) / denom)
            b_list.append((p.gamma * dt - 1.0) / denom)
            c = (EPS_0 * p.strength * dt ** 2) / denom
            c_rows.append(np.where(cell_mat == mid, c, 0.0).astype(npdt))
    return dict(
        idx=tuple(i.astype(np.int32) for i in idx),
        A=np.asarray(a_list, dtype=npdt),          # (npole,)
        B=np.asarray(b_list, dtype=npdt),          # (npole,)
        C=np.stack(c_rows, axis=0),                # (npole, n_disp)
        inv_eps=inv_eps,                           # (n_disp,)
        npole=len(a_list),
        n_disp=n_disp,
    )


def _monitor_plan(sim):
    grid = sim.grid
    dt = sim.dt
    n_steps = sim.n_steps
    plans = []
    for m in sim.monitors:
        if isinstance(m, FieldMonitor):
            if m.compression is not None:
                raise NotImplementedError(
                    "FieldMonitor(compression=) is a host-side store, not part "
                    "of the JAX device path; drop compression for use_jax.")
            rec = (sorted({int(round(t / dt)) for t in m.times
                           if 0 <= int(round(t / dt)) < n_steps})
                   if m.times is not None else list(range(0, n_steps, m.interval)))
            plane = None
            if m.plane_z is not None:
                zi = int(np.argmin(np.abs(np.asarray(grid.coords[2]) - m.plane_z)))
                plane = (2, zi)
            plans.append(dict(kind="field", m=m, rec=rec, plane=plane))
        elif isinstance(m, DFTMonitor):
            plane = None
            pl = m.plane()
            if pl is not None:
                ax, pos = pl
                ci = int(np.argmin(np.abs(np.asarray(grid.coords[ax]) - pos)))
                plane = (ax, ci)
            plans.append(dict(kind="dft", m=m, rec=list(range(0, n_steps, m.interval)),
                              plane=plane, omega=2 * np.pi * np.asarray(m.freqs, float)))
        elif isinstance(m, FluxMonitor):
            plans.append(dict(kind="flux", m=m))
    return plans


def _snap(field, m, plane):
    """Strided, optionally single-plane (any axis) view of a component."""
    ds = m.downsample
    sl = [slice(None, None, ds) if ds > 1 else slice(None) for _ in range(3)]
    if plane is not None:
        ax, ci = plane
        sl[ax] = slice(ci, ci + 1)
    return field[tuple(sl)]


def _device_simulate(sim, static, plans, ce_e, particle_plans=(),
                     ade=None, ce_particle=None, remat="none", checkpoint_levels=2):
    """Pure device-side time loop. Returns the raw monitor accumulators (field
    buffers, DFT accumulators, flux scalars) as JAX arrays.

    ``ce_e`` is a dict ``{'Ex','Ey','Ez'}`` of per-component E-update
    coefficients (equal arrays unless subpixel smoothing is on). ``ce_particle``
    is the scalar coefficient used for moving-charge deposition (defaults to the
    Ex coefficient). ``ade`` is the dispersive-media plan from
    :func:`_build_ade_jax` (or None). The result is differentiable in ``ce_e``,
    in ``ce_particle``, and in the ADE coefficient arrays (``ade['C']`` etc.),
    so gradients flow to permittivity, the smoothed tensor, and pole strengths.
    """
    import jax.numpy as jnp
    from jax import lax

    shape = static["shape"]
    dt = static["dt"]
    n_steps = static["n_steps"]
    jdt = ce_e["Ex"].dtype
    cdt = jnp.complex128 if jdt == jnp.float64 else jnp.complex64
    neg_ch = jnp.asarray(-static["ch_field"], dtype=jdt)
    if ce_particle is None:
        ce_particle = ce_e["Ex"]

    # Dispersive-media coefficients on device (masked: state lives only on the
    # dispersive cells, gathered/scattered by their flat indices each step).
    if ade is not None:
        ade_idx = tuple(jnp.asarray(i) for i in ade["idx"])     # (I,J,K)
        ade_A = jnp.asarray(ade["A"]).reshape(-1, 1, 1)         # (npole,1,1)
        ade_B = jnp.asarray(ade["B"]).reshape(-1, 1, 1)
        ade_C = jnp.asarray(ade["C"])[:, None, :]               # (npole,1,n_disp)
        ade_inv = jnp.asarray(ade["inv_eps"])                   # (n_disp,)
        npole = ade["npole"]
        n_disp = ade["n_disp"]

    fields = {c: jnp.zeros(shape, dtype=jdt)
              for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
    psis = {}
    for comp in static["comps"]:
        for t in comp["terms"]:
            psis[t["key"]] = jnp.zeros(t["pshape"], dtype=jdt)

    # Dispersive polarization state: (Pcur, Pprev), each (npole, 3, n_disp) -
    # only the dispersive cells; an empty tuple when there is no dispersion.
    if ade is not None:
        pol_shape = (npole, 3, n_disp)
        ade_state = (jnp.zeros(pol_shape, dtype=jdt),
                     jnp.zeros(pol_shape, dtype=jdt))
    else:
        ade_state = ()

    # Monitor accumulators + per-plan static schedules.
    state_field, field_slot = {}, {}
    state_dft, dft_mask = {}, {}
    state_flux = {}
    for p in plans:
        m = p["m"]
        if p["kind"] == "field":
            n_rec = len(p["rec"])
            slot = np.full(n_steps, n_rec, dtype=np.int32)
            for r, s in enumerate(p["rec"]):
                slot[s] = r
            field_slot[m.name] = jnp.asarray(slot)
            state_field[m.name] = {
                c: jnp.zeros((n_rec + 1,) + _snap(fields[c], m, p["plane"]).shape,
                             dtype=jdt) for c in m.components}
        elif p["kind"] == "dft":
            mask = np.zeros(n_steps, dtype=np.float64)
            for s in p["rec"]:
                mask[s] = 1.0
            dft_mask[m.name] = jnp.asarray(mask, dtype=jdt)
            state_dft[m.name] = {
                c: jnp.zeros((len(m.freqs),) + _snap(fields[c], m, p["plane"]).shape,
                             dtype=cdt) for c in m.components}
        elif p["kind"] == "flux":
            state_flux[m.name] = jnp.asarray(0.0, dtype=jdt)

    def apply_pass(fields, psis, is_E_pass):
        fields = dict(fields)
        psis = dict(psis)
        for comp in static["comps"]:
            if comp["is_E"] != is_E_pass:
                continue
            curl = None
            for t in comp["terms"]:
                ax = t["ax"]
                src = fields[t["src"]]
                hi = [slice(None)] * 3; hi[ax] = slice(1, None)
                lo = [slice(None)] * 3; lo[ax] = slice(0, -1)
                d = (src[tuple(hi)] - src[tuple(lo)]) / t["dl"]
                d = d[t["osl"]]
                p = jnp.asarray(t["b1"]) * psis[t["key"]] + jnp.asarray(t["c1"]) * d
                psis[t["key"]] = p
                cterm = d + p
                curl = (cterm if t["sign"] > 0 else -cterm) if curl is None \
                    else (curl + cterm if t["sign"] > 0 else curl - cterm)
            if curl is None:
                continue
            reg = comp["region"]
            k = ce_e[comp["Fkey"]][reg] if comp["is_E"] else neg_ch
            fields[comp["Fkey"]] = fields[comp["Fkey"]].at[reg].add(k * curl)
        return fields, psis

    # Source waveform tables on device so a traced step index can gather them.
    src_H = [(c, i, j, k, jnp.asarray(v)) for c, i, j, k, v in static["src_H"]]
    src_E = [(c, i, j, k, jnp.asarray(v)) for c, i, j, k, v in static["src_E"]]

    # Moving-charge deposition tables (per particle), on device.
    E_names = ("Ex", "Ey", "Ez")
    pplans = [dict(R=pp["R"], axes=pp["axes"],
                   coef3=[jnp.asarray(pp["coef3"][a]) for a in range(3)],
                   starts=jnp.asarray(pp["starts"]),
                   wblocks=jnp.asarray(pp["wblocks"]))
              for pp in particle_plans]

    def deposit_particles(fields, step_idx):
        for pp in pplans:
            R = pp["R"]
            st = pp["starts"][step_idx]                   # (3,) traced ints
            start = (st[0], st[1], st[2])
            wb = pp["wblocks"][step_idx]
            ce_blk = lax.dynamic_slice(ce_particle, start, R)
            for a in pp["axes"]:
                comp = E_names[a]
                add = pp["coef3"][a] * wb * ce_blk
                cur = lax.dynamic_slice(fields[comp], start, R)
                fields[comp] = lax.dynamic_update_slice(fields[comp], cur + add,
                                                        start)
        return fields

    def step(carry, step_idx):
        fields, psis, ade_state, sf, sd, sfx = carry
        fields, psis = apply_pass(fields, psis, is_E_pass=False)
        for comp, i, j, k, vals in src_H:
            fields[comp] = fields[comp].at[i, j, k].add(vals[step_idx])
        # Gather E^n at the dispersive cells before the curl E-update, so the
        # pole recursion uses the pre-update field (matches the in-core ADE).
        if ade is not None:
            En = jnp.stack([fields["Ex"][ade_idx], fields["Ey"][ade_idx],
                            fields["Ez"][ade_idx]], axis=0)   # (3, n_disp)
        fields, psis = apply_pass(fields, psis, is_E_pass=True)
        if ade is not None:
            # P^{n+1} = a P^n + b P^{n-1} + c E^n; scatter the increment into E.
            Pcur, Pprev = ade_state
            Pnew = ade_A * Pcur + ade_B * Pprev + ade_C * En[None]
            delta = jnp.sum(Pnew - Pcur, axis=0)           # (3, n_disp)
            fields["Ex"] = fields["Ex"].at[ade_idx].add(-ade_inv * delta[0])
            fields["Ey"] = fields["Ey"].at[ade_idx].add(-ade_inv * delta[1])
            fields["Ez"] = fields["Ez"].at[ade_idx].add(-ade_inv * delta[2])
            ade_state = (Pnew, Pcur)
        for comp, i, j, k, vals in src_E:
            fields[comp] = fields[comp].at[i, j, k].add(vals[step_idx])
        # Moving-charge currents, injected at the E-time as in the in-core loop.
        if pplans:
            fields = deposit_particles(fields, step_idx)

        for p in plans:
            m = p["m"]
            if p["kind"] == "field":
                slot = field_slot[m.name][step_idx]
                for c in m.components:
                    sf[m.name][c] = sf[m.name][c].at[slot].set(
                        _snap(fields[c], m, p["plane"]))
            elif p["kind"] == "dft":
                active = dft_mask[m.name][step_idx]
                for c in m.components:
                    tt = (step_idx + (1.0 if c[0] == "E" else 0.5)) * dt
                    ph = jnp.exp(1j * jnp.asarray(p["omega"]) * tt).astype(cdt)
                    s = _snap(fields[c], m, p["plane"]).astype(cdt)
                    ph = ph.reshape((-1,) + (1,) * s.ndim)
                    sd[m.name][c] = sd[m.name][c] + active * ph * s[None] * dt
            elif p["kind"] == "flux":
                sfx[m.name] = sfx[m.name] + _flux(fields, m, sim.grid, jnp) * dt
        return (fields, psis, ade_state, sf, sd, sfx), None

    init = (fields, psis, ade_state, state_field, state_dft, state_flux)
    final, _ = _run_time_loop(step, init, n_steps, remat, levels=checkpoint_levels)
    (fields, psis, ade_state, sf, sd, sfx) = final
    return sf, sd, sfx


def _run_time_loop(step, init, n_steps, remat, levels=2):
    """Drive the time loop, optionally with recursive gradient checkpointing.

    ``remat``:

    * ``"none"`` - a plain ``lax.scan``. Reverse-mode AD then stores the state
      trajectory for every step (O(n_steps) memory) - fine for forward runs and
      short gradients.
    * ``"step"`` - ``jax.checkpoint`` the single step (recomputes each step's
      interior in the backward pass; helps the constant factor only).
    * ``"nested"`` - an ``L``-level tree of nested scans, each level's segment
      wrapped in ``jax.checkpoint``. With ``levels`` equal factors of
      ``n_steps`` the backward pass holds only ~``levels * n_steps**(1/levels)``
      states and recomputes the rest, so adjoint memory drops from O(n_steps) to
      O(levels * n_steps**(1/levels)) - i.e. O(sqrt(n_steps)) at ``levels=2``,
      O(n_steps**(1/3)) at ``levels=3`` - for ~``levels``x the forward compute.
      This is the Griewank/Volin-style multi-level checkpointing that lets long
      differentiable runs (inverse design) fit in memory.
    """
    import jax
    from jax import lax
    import jax.numpy as jnp

    if remat != "nested":
        step_fn = jax.checkpoint(step) if remat == "step" else step
        return lax.scan(step_fn, init, jnp.arange(n_steps))

    levels = max(1, int(levels))
    base = max(1, int(math.ceil(n_steps ** (1.0 / levels))))
    factors = [base] * levels                         # product >= n_steps
    strides = [1] * levels
    acc = 1
    for i in range(levels - 1, -1, -1):
        strides[i] = acc
        acc *= factors[i]

    def guarded(carry, g):
        # Padded tail iterations (global index g >= n_steps) are a no-op; the
        # clamped index keeps the source/monitor table lookups in-bounds.
        def do(c):
            return step(c, jnp.minimum(g, n_steps - 1))[0]
        return lax.cond(g < n_steps, do, lambda c: c, carry), None

    def build(level):
        f, s = factors[level], strides[level]
        if level == levels - 1:                       # innermost: real steps
            def leaf(carry, base_idx):
                carry, _ = lax.scan(guarded, carry, base_idx + jnp.arange(f))
                return carry
            return leaf
        child = build(level + 1)

        def node(carry, base_idx):
            def body(c, i):
                return child(c, base_idx + i * s), None
            carry, _ = lax.scan(jax.checkpoint(body), carry, jnp.arange(f))
            return carry
        return node

    return build(0)(init, 0), None


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


def _enable_x64_if_needed(sim):
    import jax
    if np.dtype(sim.dtypes["Ex"]) == np.float64:
        jax.config.update("jax_enable_x64", True)


def _validate(sim):
    if sim.use_gpu or sim.use_numba:
        raise NotImplementedError("use_jax is exclusive of use_gpu / use_numba.")
    if len({sim.dtypes[k] for k in
            ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "eps_r")}) > 1:
        raise NotImplementedError(
            "the JAX backend requires one uniform precision for the fields.")


def run_jax(sim, jit: bool = True):
    """Run ``sim`` on the JAX backend and return a Result (see module docstring)."""
    import jax
    import jax.numpy as jnp
    from .simulation import Result

    _validate(sim)
    _enable_x64_if_needed(sim)
    npdt = np.dtype(sim.dtypes["Ex"])
    static = _build_static(sim)
    plans = _monitor_plan(sim)
    pplans = _particle_plan(sim, static)
    ade = _build_ade_jax(sim, static)
    shape = static["shape"]
    dt = static["dt"]
    n_steps = static["n_steps"]

    # Scalar coefficient from eps_r (== eps_inf in dispersive cells); used for
    # moving-charge deposition and as the E-update coefficient when smoothing is
    # off.
    ce_scalar = (dt / (np.asarray(sim.eps_r, dtype=npdt) * EPS_0)).astype(npdt)
    if sim.subpixel:
        exx, eyy, ezz = sim._smoothed_eps_components()
        ce_e_np = {c: (dt / (np.asarray(e, dtype=npdt) * EPS_0)).astype(npdt)
                   for c, e in zip(("Ex", "Ey", "Ez"), (exx, eyy, ezz))}
    else:
        ce_e_np = {c: ce_scalar for c in ("Ex", "Ey", "Ez")}

    def fn(ce_e, ce_particle):
        return _device_simulate(sim, static, plans, ce_e, pplans,
                                ade=ade, ce_particle=ce_particle)
    fn = jax.jit(fn) if jit else fn
    ce_e = {c: jnp.asarray(v) for c, v in ce_e_np.items()}
    sf, sd, sfx = fn(ce_e, jnp.asarray(ce_scalar))

    result = Result(times=np.arange(n_steps) * dt)
    for p in plans:
        m = p["m"]
        if p["kind"] == "field":
            n_rec = len(p["rec"])
            if n_rec > 0:
                result.fields[m.name] = {
                    c: np.asarray(sf[m.name][c])[:n_rec].astype(sim.dtypes["monitors"])
                    for c in m.components}
                result.monitor_times[m.name] = np.asarray(p["rec"], float) * dt
            else:
                result.fields[m.name] = {c: np.zeros((0,) + shape)
                                         for c in m.components}
                result.monitor_times[m.name] = np.array([])
        elif p["kind"] == "dft":
            result.dft[m.name] = {c: np.asarray(sd[m.name][c]) for c in m.components}
            result.dft_freqs[m.name] = np.asarray(m.freqs, float)
        elif p["kind"] == "flux":
            result.flux[m.name] = float(np.asarray(sfx[m.name]))
    return result


def value_and_grad_eps(sim, loss, remat="nested", checkpoint_levels=2):
    """Return ``(value, d value / d eps_r)`` for a scalar ``loss(out)``.

    ``out`` is a dict of JAX arrays mirroring the monitor outputs::

        {'fields': {name: {comp: (n_rec, *snap) real}},
         'dft':    {name: {comp: (n_freq, *snap) complex}},
         'flux':   {name: scalar}}

    ``loss`` maps that to a scalar. The returned gradient w.r.t. the per-cell
    permittivity ``eps_r`` propagates through ``ce_field = dt/(eps_r*EPS_0)`` and
    the entire time evolution - a time-domain adjoint, i.e. exactly the gradient
    an inverse-design / topology-optimization loop needs.

    ``remat`` controls adjoint memory (see :func:`_run_time_loop`). The default
    ``"nested"`` uses two-level gradient checkpointing so the backward pass
    stores only ~sqrt(n_steps) segment-boundary states instead of the full
    field trajectory, cutting peak memory from O(n_steps) to O(sqrt(n_steps))
    for ~2x compute (measured ~3-4x lower peak on a few-thousand-step run). Pass
    ``remat="none"`` for the plain reverse pass on short runs.
    """
    import jax
    import jax.numpy as jnp

    _validate(sim)
    if sim.subpixel or getattr(sim, "_has_dispersion", False):
        raise NotImplementedError(
            "value_and_grad_eps differentiates the scalar per-cell eps_r, which "
            "does not parameterize a subpixel-smoothed tensor or a dispersive "
            "pole model. Both forward-run and are differentiable under the JAX "
            "backend: build the coefficient / pole arrays symbolically and call "
            "jax.grad on _device_simulate directly (see tests/test_jax_accuracy "
            "for a worked pole-strength gradient)."
        )
    _enable_x64_if_needed(sim)
    npdt = np.dtype(sim.dtypes["Ex"])
    static = _build_static(sim)
    plans = _monitor_plan(sim)
    pplans = _particle_plan(sim, static)

    def loss_of_eps(eps_r):
        ce = sim.dt / (eps_r * EPS_0)
        ce_e = {c: ce for c in ("Ex", "Ey", "Ez")}
        sf, sd, sfx = _device_simulate(sim, static, plans, ce_e, pplans,
                                       ade=None, ce_particle=ce, remat=remat,
                                       checkpoint_levels=checkpoint_levels)
        n_rec = {p["m"].name: len(p["rec"]) for p in plans if p["kind"] == "field"}
        out = {
            "fields": {n: {c: sf[n][c][:n_rec[n]] for c in sf[n]} for n in sf},
            "dft": {n: {c: sd[n][c] for c in sd[n]} for n in sd},
            "flux": {n: sfx[n] for n in sfx},
        }
        return loss(out)

    eps0 = jnp.asarray(np.asarray(sim.eps_r, dtype=npdt))
    val, grad = jax.value_and_grad(loss_of_eps)(eps0)
    return float(val), np.asarray(grad)


def _out_dict(sf, sd, sfx, plans):
    """Assemble the loss-facing monitor-output dict (JAX arrays)."""
    n_rec = {p["m"].name: len(p["rec"]) for p in plans if p["kind"] == "field"}
    return {
        "fields": {n: {c: sf[n][c][:n_rec[n]] for c in sf[n]} for n in sf},
        "dft": {n: {c: sd[n][c] for c in sd[n]} for n in sd},
        "flux": {n: sfx[n] for n in sfx},
    }


def value_and_grad_params(sim, ce_of_params, params, loss,
                          remat="nested", checkpoint_levels=2):
    """Differentiate a monitor loss w.r.t. arbitrary design ``params``.

    ``ce_of_params(params)`` must return a dict ``{'Ex','Ey','Ez'}`` of
    per-component E-update coefficients (JAX arrays) built symbolically from
    ``params`` - e.g. a density field mapped through filtering, projection and
    subpixel smoothing to a permittivity tensor (see
    :mod:`photonfdtd.design`). ``jax.grad`` then flows the loss back through the
    whole time evolution to ``params`` - true topology / shape optimization,
    not just per-cell eps. Returns ``(value, grad_params)`` (``grad_params`` a
    JAX array matching ``params``).

    The subpixel per-component coefficient path is used, so a plain
    ``Simulation`` (``subpixel=False``) is fine; dispersion is not supported here
    (the pole state is not parameterized by ``params``).
    """
    import jax
    import jax.numpy as jnp

    _validate(sim)
    if getattr(sim, "_has_dispersion", False):
        raise NotImplementedError(
            "value_and_grad_params does not support dispersive media.")
    _enable_x64_if_needed(sim)
    static = _build_static(sim)
    plans = _monitor_plan(sim)
    pplans = _particle_plan(sim, static)

    def loss_of_params(params):
        ce_e = ce_of_params(params)
        sf, sd, sfx = _device_simulate(
            sim, static, plans, ce_e, pplans, ade=None,
            ce_particle=ce_e["Ex"], remat=remat,
            checkpoint_levels=checkpoint_levels)
        return loss(_out_dict(sf, sd, sfx, plans))

    val, grad = jax.value_and_grad(loss_of_params)(params)
    return float(val), grad
