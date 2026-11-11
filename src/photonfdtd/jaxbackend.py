"""JAX engine: a pure-functional FDTD time step under ``jax.lax.scan``.

This is the time-stepping core behind :meth:`Simulation.run`. JAX arrays are
immutable, so every field/psi update is expressed functionally (``x.at[region].add(...)``) and the whole time loop is
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
advanced by the ADE recursion inside the ``lax.scan`` carry).

The per-cell update uses the ``cy - cz`` curl form with dense CPML psi.

Sources: soft point/dipole sources (so expanded ModeSource / SinglePhoton
sources work too) and moving ``ChargedParticle`` charges (Cherenkov). The
particle trajectory is deterministic and field-independent, so its Gaussian
current cloud is precomputed per step on host and injected inside the scan via
``dynamic_slice`` / ``dynamic_update_slice``; only the ``ce_field`` factor stays
symbolic, so gradients still flow through it.

Monitors: ``FieldMonitor`` / ``DFTMonitor`` / ``FluxMonitor``. CPML, 1D/2D/3D,
float32/float64.
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

    # Sources grouped per component: (comp, I, J, K, V) with index arrays of
    # length n_src and V of shape (n_steps, n_src), so each step adds them all
    # with one scatter, ``F.at[I, J, K].add(V[n])``. One op per source instead
    # (a mode source expands to thousands of dipoles) makes XLA's compile time
    # explode: 7000 dipoles took ~40 s to compile on jax 0.6 and >10 min on 0.12.
    steps = np.arange(n_steps)
    groups: Dict[str, list] = {}
    for src in sim.sources:
        i, j, k = grid.index_at(src.position)
        t = (steps + (0.5 if src.component[0] == "H" else 1.0)) * dt
        vals = np.asarray(src.amplitude * src.waveform(t), dtype=npdt)
        groups.setdefault(src.component, []).append((i, j, k, vals))
    src_H, src_E = [], []
    for comp in sorted(groups):
        g = groups[comp]
        entry = (comp,
                 np.array([e[0] for e in g], dtype=np.int32),
                 np.array([e[1] for e in g], dtype=np.int32),
                 np.array([e[2] for e in g], dtype=np.int32),
                 np.stack([e[3] for e in g], axis=1).astype(npdt))
        (src_H if comp[0] == "H" else src_E).append(entry)

    return dict(shape=shape, cell=cell, dt=dt, n_steps=n_steps, npdt=npdt,
                comps=comps, ch_field=float(dt / MU_0), src_H=src_H, src_E=src_E)


def _particle_plan(sim, static):
    """Precompute each moving charge's per-step Gaussian-cloud deposition.

    The trajectory is deterministic and field-independent, so the stencil
    location and the normalised weights are fully known on host - only the
    ``ce_field`` factor stays symbolic (kept in the kernel via a dynamic slice),
    so gradients w.r.t. ``eps_r`` still flow through the deposit. A charge ``q``
    at velocity ``v`` is a current density ``J = q v``; it is smeared over a
    Gaussian cloud (``cloud_cells`` wide) to band-limit the radiated spectrum and
    deposited as ``dE = -(q v / dV) * weight * ce_field`` only while the
    particle centre is inside the physical (non-PML) interior. Each step's
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
            t_now = (si + 1.0) * dt                       # E-time
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
    per pole (zero on cells belonging to a different material). The cells are
    in C order, so sorted by x (the disk-tiled stepper relies on this). The
    ``1/(eps_inf*EPS_0)`` factor of the polarization correction is not stored:
    the stepper takes it from the E-update coefficient ``ce/dt`` at the cell
    (eps_r there is eps_inf), so gradients w.r.t. eps_r include it.
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
    for mid, med in enumerate(media):
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


def _dft_snap(field, comp, m, plane):
    """Sample ``comp`` for a DFT monitor, on the monitor plane itself.

    A plane monitor sits at ``coord[ci]`` along its normal axis. Components
    staggered half a cell along that axis (E along it, H across it) are
    averaged over their two half-offset neighbours ``ci-1`` and ``ci``, so
    every recorded component lives on the same plane and ``E x H`` products
    are second-order accurate. Without a plane restriction, or at the first
    plane, the raw samples are returned.
    """
    s = _snap(field, m, plane)
    if plane is None:
        return s
    ax, ci = plane
    a = "xyz".index(comp[1])
    if ci == 0 or (a == ax) != (comp[0] == "E"):
        return s
    return 0.5 * (s + _snap(field, m, (ax, ci - 1)))


def _device_simulate(sim, static, plans, ce_e, particle_plans=(),
                     ade=None, ce_particle=None, remat="none", checkpoint_levels=2,
                     ca_e=None, modulation=None):
    """Pure device-side time loop. Returns the raw monitor accumulators (field
    buffers, DFT accumulators, flux scalars) as JAX arrays.

    ``ce_e`` is a dict ``{'Ex','Ey','Ez'}`` of per-component E-update
    coefficients (equal arrays unless subpixel smoothing is on). ``ce_particle``
    is the scalar coefficient used for moving-charge deposition (defaults to the
    Ex coefficient). ``ade`` is the dispersive-media plan from
    :func:`_build_ade_jax` (or None). The result is differentiable in ``ce_e``,
    in ``ce_particle``, and in the ADE coefficient arrays (``ade['C']`` etc.),
    so gradients flow to permittivity, the smoothed tensor, and pole strengths.

    ``ca_e`` (optional, same layout as ``ce_e``) is the lossy-medium decay
    factor: the E update becomes ``E <- ca*E + ce*curl H``. ``None`` (no
    conductivity anywhere) keeps the plain ``E <- E + ce*curl H``.

    ``modulation`` (optional) makes the permittivity inside a box follow
    applied voltages in time: a dict with ``box`` (tuple of index slices),
    ``eps_b`` ({comp: base eps in the box}), ``deps`` ({comp: (n_drive, *box)}
    permittivity change per unit drive) and ``v`` ((n_steps, n_drive) drive
    values per step). Each step the coefficient in the box is rebuilt as
    ``dt / (EPS_0 * (eps_b + v[n] . deps))`` (an instantaneous-permittivity
    model, valid when the drive varies slowly next to the optical period).
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
        # 1/(eps_inf*EPS_0) per component, from the E-update coefficient.
        ade_inv = [ce_e[c][ade_idx] / dt for c in ("Ex", "Ey", "Ez")]
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

    if ca_e is not None:
        ca_e = {c: jnp.asarray(v, dtype=jdt) for c, v in ca_e.items()}
    if modulation is not None:
        mod_box = tuple(modulation["box"])
        mod_eps = {c: jnp.asarray(modulation["eps_b"][c], dtype=jdt) for c in ("Ex", "Ey", "Ez")}
        mod_deps = {c: jnp.asarray(modulation["deps"][c], dtype=jdt) for c in ("Ex", "Ey", "Ez")}
        mod_v = jnp.asarray(modulation["v"], dtype=jdt)

    def coefficients_at(step_idx):
        if modulation is None:
            return ce_e
        v = mod_v[step_idx]
        return {c: ce_e[c].at[mod_box].set(
                    dt / (EPS_0 * (mod_eps[c] + jnp.tensordot(v, mod_deps[c], 1))))
                for c in ("Ex", "Ey", "Ez")}

    def apply_pass(fields, psis, is_E_pass, ce_now=None):
        ce_now = ce_e if ce_now is None else ce_now
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
            key = comp["Fkey"]
            k = ce_now[key][reg] if comp["is_E"] else neg_ch
            if comp["is_E"] and ca_e is not None:
                F = fields[key]
                fields[key] = F.at[reg].set(ca_e[key][reg] * F[reg] + k * curl)
            else:
                fields[key] = fields[key].at[reg].add(k * curl)
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
        # pole recursion uses the pre-update field (P^{n+1} = aP^n + bP^{n-1} + cE^n).
        if ade is not None:
            En = jnp.stack([fields["Ex"][ade_idx], fields["Ey"][ade_idx],
                            fields["Ez"][ade_idx]], axis=0)   # (3, n_disp)
        # FluxMonitor is time-centred: half of E^n x H^{n+1/2} (here) plus half
        # of E^{n+1} x H^{n+1/2} (after the E update), which sums to the Yee
        # discrete energy flux E^n x (H^{n-1/2} + H^{n+1/2}) / 2.
        flux_pre = {p["m"].name: _flux(fields, p["m"], sim.grid, jnp)
                    for p in plans if p["kind"] == "flux"}
        fields, psis = apply_pass(fields, psis, is_E_pass=True,
                                  ce_now=coefficients_at(step_idx))
        if ade is not None:
            # P^{n+1} = a P^n + b P^{n-1} + c E^n; scatter the increment into E.
            Pcur, Pprev = ade_state
            Pnew = ade_A * Pcur + ade_B * Pprev + ade_C * En[None]
            delta = jnp.sum(Pnew - Pcur, axis=0)           # (3, n_disp)
            fields["Ex"] = fields["Ex"].at[ade_idx].add(-ade_inv[0] * delta[0])
            fields["Ey"] = fields["Ey"].at[ade_idx].add(-ade_inv[1] * delta[1])
            fields["Ez"] = fields["Ez"].at[ade_idx].add(-ade_inv[2] * delta[2])
            ade_state = (Pnew, Pcur)
        for comp, i, j, k, vals in src_E:
            fields[comp] = fields[comp].at[i, j, k].add(vals[step_idx])
        # Moving-charge currents, injected at the E-time.
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
                    s = _dft_snap(fields[c], c, m, p["plane"]).astype(cdt)
                    ph = ph.reshape((-1,) + (1,) * s.ndim)
                    sd[m.name][c] = sd[m.name][c] + active * ph * s[None] * dt
            elif p["kind"] == "flux":
                sfx[m.name] = sfx[m.name] + 0.5 * (
                    flux_pre[m.name] + _flux(fields, m, sim.grid, jnp)) * dt
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
    """Poynting flux through the plane of ``m`` at one instant.

    The plane sits on the tangential-E sub-grid (``coord[idx]`` along the
    normal), where the two tangential H components are half a cell off. Each
    H is averaged over the two half-offset neighbours ``idx-1`` and ``idx``, so
    every ``E x H`` product pairs samples at the same point, which is
    second-order accurate in the cell size. Along a collapsed normal axis there
    is no offset to correct.
    """
    axis = _FLUX_AXIS[m.plane_axis]
    coord = grid.coords[axis]
    idx = 0 if coord.size == 1 else int(np.argmin(np.abs(coord - m.plane_position)))
    lo = max(idx - 1, 0)

    def plane(F):
        return jnp.take(F, idx, axis=axis)

    def plane_avg(F):
        if coord.size == 1:
            return plane(F)
        return 0.5 * (jnp.take(F, lo, axis=axis) + jnp.take(F, idx, axis=axis))

    E1, E2, H1, H2 = {0: ("Ey", "Ez", "Hy", "Hz"),
                      1: ("Ez", "Ex", "Hz", "Hx"),
                      2: ("Ex", "Ey", "Hx", "Hy")}[axis]
    S = (plane(fields[E1]) * plane_avg(fields[H2])
         - plane(fields[E2]) * plane_avg(fields[H1]))
    dA = 1.0
    for a in range(3):
        if a != axis and grid.shape[a] > 1:
            dA *= grid.cell_size[a]
    return jnp.sum(S) * dA


def _enable_x64_if_needed(sim):
    import jax
    if np.dtype(sim.dtypes["Ex"]) == np.float64:
        jax.config.update("jax_enable_x64", True)


def _validate(sim):
    if len({sim.dtypes[k] for k in
            ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "eps_r")}) > 1:
        raise NotImplementedError(
            "the stepping arrays (fields, eps_r) must share one precision.")


def update_coefficients(sim, eps_components=None, sigma=None, dtype=None):
    """E-update coefficients ``(ce, ca)`` for ``sim``.

    ``ce`` is ``{'Ex','Ey','Ez'}`` of ``dt / (eps EPS_0)``, from
    ``eps_components`` (default: :meth:`Simulation.eps_components`). With a
    conductivity map ``sigma`` (default: :meth:`Simulation.conductivity_map`)
    the lossy-medium form is used, ``ce = (dt/eps EPS_0) / (1 + s)`` and
    ``ca = (1 - s) / (1 + s)`` with ``s = sigma dt / (2 eps EPS_0)``; ``ca`` is
    None when nothing is conductive. Arrays are NumPy, in the stepping dtype.
    """
    npdt = np.dtype(dtype or sim.dtypes["Ex"])
    dt = sim.dt
    comps = sim.eps_components() if eps_components is None else eps_components
    if sigma is None:
        sigma = sim.conductivity_map()
    ce, ca = {}, ({} if sigma is not None else None)
    for c, e in zip(("Ex", "Ey", "Ez"), comps):
        e = np.asarray(e, dtype=np.float64)
        base = dt / (e * EPS_0)
        if sigma is None:
            ce[c] = base.astype(npdt)
        else:
            sh = sigma * dt / (2.0 * e * EPS_0)
            ce[c] = (base / (1.0 + sh)).astype(npdt)
            ca[c] = ((1.0 - sh) / (1.0 + sh)).astype(npdt)
    return ce, ca


def run_jax(sim, jit: bool = True, ce=None, ca=None, modulation=None):
    """Run ``sim`` on JAX and return a Result (see module docstring).

    ``ce`` (optional) replaces the E-update coefficients ``{'Ex','Ey','Ez'}``
    (for example from :meth:`photonfdtd.active.ActiveRegion.coefficients`), and
    ``ca`` the lossy decay factors. ``modulation`` drives the permittivity in a
    box in time (see :func:`_device_simulate`); build it with
    :meth:`photonfdtd.active.ActiveRegion.modulation`.
    """
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

    # Per-component coefficients (equal unless smoothing or anisotropy). In
    # dispersive cells eps is eps_inf. The scalar one (from eps_r) is used for
    # moving-charge deposition.
    ce_scalar = (dt / (np.asarray(sim.eps_r, dtype=npdt) * EPS_0)).astype(npdt)
    if sim.per_component or sim._has_conductivity:
        ce_e_np, ca_np = update_coefficients(sim)
    else:
        ce_e_np, ca_np = {c: ce_scalar for c in ("Ex", "Ey", "Ez")}, None
    if ce is not None:
        ce_e_np = ce
    if ca is not None:
        ca_np = ca
    if modulation is not None and modulation["v"].shape[0] != n_steps:
        raise ValueError(f"modulation drive has {modulation['v'].shape[0]} steps; "
                         f"the simulation runs {n_steps}")

    # The modulation box is a static index window (closed over); its arrays
    # are traced inputs.
    box = None if modulation is None else tuple(modulation["box"])
    mod = None if modulation is None else {
        k: modulation[k] for k in ("eps_b", "deps", "v")}

    def fn(ce_e, ce_particle, ca_e, mod):
        return _device_simulate(
            sim, static, plans, ce_e, pplans, ade=ade, ce_particle=ce_particle,
            ca_e=ca_e, modulation=None if mod is None else dict(mod, box=box))
    fn = jax.jit(fn) if jit else fn
    ce_e = {c: jnp.asarray(v) for c, v in ce_e_np.items()}
    ca_e = None if ca_np is None else {c: jnp.asarray(v) for c, v in ca_np.items()}
    sf, sd, sfx = fn(ce_e, jnp.asarray(ce_scalar), ca_e, mod)

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

    Dispersive media are supported: in a dispersive cell ``eps_r`` is the
    medium's ``eps_inf``, and its gradient includes the polarization
    correction it scales. The poles themselves are held fixed.

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
    if sim._has_anisotropy or sim._has_conductivity:
        raise NotImplementedError(
            "value_and_grad_eps differentiates one scalar eps_r per cell; with "
            "anisotropic or conductive media use value_and_grad_params with a "
            "per-component coefficient builder.")
    if sim.subpixel:
        raise NotImplementedError(
            "value_and_grad_eps differentiates the scalar per-cell eps_r, which "
            "does not parameterize a subpixel-smoothed tensor. Use "
            "value_and_grad_params with a symbolic coefficient builder (see "
            "photonfdtd.design)."
        )
    _enable_x64_if_needed(sim)
    npdt = np.dtype(sim.dtypes["Ex"])
    static = _build_static(sim)
    plans = _monitor_plan(sim)
    pplans = _particle_plan(sim, static)
    ade = _build_ade_jax(sim, static)

    def loss_of_eps(eps_r):
        ce = sim.dt / (eps_r * EPS_0)
        ce_e = {c: ce for c in ("Ex", "Ey", "Ez")}
        sf, sd, sfx = _device_simulate(sim, static, plans, ce_e, pplans,
                                       ade=ade, ce_particle=ce, remat=remat,
                                       checkpoint_levels=checkpoint_levels)
        return loss(_out_dict(sf, sd, sfx, plans))

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


def _prepare(sim):
    """Validate ``sim`` and build its fixed-for-the-run plans once."""
    _validate(sim)
    _enable_x64_if_needed(sim)
    static = _build_static(sim)
    return (static, _monitor_plan(sim), _particle_plan(sim, static),
            _build_ade_jax(sim, static))


def _simulate_out(sim, prep, ce_e, remat="nested", checkpoint_levels=2):
    """Run ``sim`` with E coefficients ``ce_e``; return the loss-facing dict.

    ``ce_e`` may also carry ``'ca'``: a ``{'Ex','Ey','Ez'}`` dict of lossy decay
    factors, for conductive cells (see :func:`update_coefficients`)."""
    static, plans, pplans, ade = prep
    ca = ce_e.get("ca") if isinstance(ce_e, dict) else None
    ce_e = {c: ce_e[c] for c in ("Ex", "Ey", "Ez")}
    sf, sd, sfx = _device_simulate(
        sim, static, plans, ce_e, pplans, ade=ade,
        ce_particle=ce_e["Ex"], remat=remat,
        checkpoint_levels=checkpoint_levels, ca_e=ca)
    return _out_dict(sf, sd, sfx, plans)


_METHODS = ("checkpoint", "reversible", "reversible_pml", "tiled")


def _check_method(method):
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")


def _run_fn(sim, method="checkpoint", remat="nested", checkpoint_levels=2):
    """``ce_e -> out`` for ``sim``, differentiable with the chosen in-memory
    adjoint (``'checkpoint'``, ``'reversible'`` or ``'reversible_pml'``)."""
    _check_method(method)
    if method == "checkpoint":
        prep = _prepare(sim)
        return lambda ce_e: _simulate_out(sim, prep, ce_e, remat,
                                          checkpoint_levels)
    if sim._has_conductivity:
        raise NotImplementedError(
            f"the {method!r} adjoint does not support conductive media; use "
            "method='checkpoint'")
    if method == "reversible":
        from .reversible import reversible_run
        return reversible_run(sim)
    if method == "reversible_pml":
        from .reversible_pml import reversible_pml_run
        return reversible_pml_run(sim)
    raise ValueError("the tiled adjoint is not a JAX function; use "
                     "_tiled_value_and_grad")


def _tiled_value_and_grad(sims, ce_of_params, objective, tiled_options=None,
                          has_aux=False):
    """Design gradient through the disk-tiled adjoint, chained on the host.

    Returns ``vg(params, aux, *extra)``. The coefficient builder runs (and is
    pulled back) on the CPU, so the three coefficient arrays and their
    cotangents live in host RAM; every simulation in ``sims`` runs forward
    with them, ``objective(outs, *extra)`` is differentiated in JAX, and each
    simulation's output cotangent is pulled back through its tiled adjoint.
    """
    import jax
    import jax.numpy as jnp
    from .tiled import _TiledAdjoint
    opts = dict(tiled_options or {})
    cpu = jax.devices("cpu")[0]

    def vg(params, aux, *extra):
        with jax.default_device(cpu):
            p_cpu = jax.device_put(params, cpu)
            ce, pull_p = jax.vjp(lambda p: ce_of_params(p, **aux), p_cpu)
        ce_host = {c: np.asarray(ce[c]) for c in ("Ex", "Ey", "Ez")}
        adjs = []
        try:
            for s in sims:
                adjs.append(_TiledAdjoint(s, ce_host, **opts))
            outs = [a.forward() for a in adjs]
            if has_aux:
                val, pull_o, aux_out = jax.vjp(lambda o: objective(o, *extra),
                                               outs, has_aux=True)
            else:
                val, pull_o = jax.vjp(lambda o: objective(o, *extra), outs)
            (g_outs,) = pull_o(jnp.ones((), jnp.asarray(val).dtype))
            bar = {c: np.zeros_like(ce_host[c]) for c in ce_host}
            for a, g in zip(adjs, g_outs):
                b = a.backward(g, copy=False)
                for c in bar:
                    bar[c] += b[c]
        finally:
            for a in adjs:
                a.close()
        with jax.default_device(cpu):
            (grad,) = pull_p({c: jnp.asarray(bar[c].astype(ce_host[c].dtype))
                              for c in bar})
        grad = jax.tree_util.tree_map(lambda x: jnp.asarray(np.asarray(x)), grad)
        return ((val, aux_out), grad) if has_aux else (val, grad)
    return vg


def make_value_and_grad_params(sim, ce_of_params, loss, remat="nested",
                               checkpoint_levels=2, jit=True,
                               method="checkpoint", tiled_options=None):
    """Compile ``params -> (value, d loss / d params)`` once, for a loop.

    The same contract as :func:`value_and_grad_params`, but the setup and the
    XLA compile happen once and the returned ``vg(params, **aux)`` is reused
    every iteration. Keyword arguments ``aux`` (e.g. a projection sharpness
    ``beta``) are passed to ``ce_of_params(params, **aux)`` as *traced*
    values, so changing them between calls does not recompile; keep their
    shapes and dtypes fixed. ``vg`` returns ``(value, grad)`` as JAX arrays.

    ``method`` picks the adjoint:

    * ``'checkpoint'`` (default) - the in-memory run with two-level gradient
      checkpointing (``remat``, ``checkpoint_levels``).
    * ``'reversible'`` - O(1) memory in the number of timesteps by stepping
      the physics backward; needs no PML, lossless poles, no moving charges,
      DFT/Flux monitors only.
    * ``'reversible_pml'`` - the same with CPML, replaying a stored PML-shell
      tape; same restrictions apart from the PML.
    * ``'tiled'`` - the disk-tiled adjoint, for grids whose adjoint does not
      fit on the device; ``tiled_options`` go to it (``tile_cells``,
      ``steps_per_tile``, ``workdir``, ``budget_bytes``,
      ``checkpoint_every``). Not jitted as a whole (the time loop is a host
      loop over compiled slab updates); the coefficient builder runs on the
      CPU, so its three grid-sized arrays sit in host RAM.

    All four agree to rounding error.
    """
    import jax

    _check_method(method)
    if method == "tiled" and sim._has_conductivity:
        raise NotImplementedError("the tiled adjoint does not support conductive "
                                  "media; use method='checkpoint'")
    if method == "tiled":
        tvg = _tiled_value_and_grad([sim], ce_of_params,
                                    lambda outs: loss(outs[0]), tiled_options)

        def vg(params, **aux):
            return tvg(params, aux)
        return vg

    run = _run_fn(sim, method, remat, checkpoint_levels)

    def loss_of_params(params, aux):
        return loss(run(ce_of_params(params, **aux)))

    fn = jax.value_and_grad(loss_of_params)
    fn = jax.jit(fn) if jit else fn

    def vg(params, **aux):
        return fn(params, aux)
    return vg


def value_and_grad_params(sim, ce_of_params, params, loss,
                          remat="nested", checkpoint_levels=2,
                          method="checkpoint", tiled_options=None):
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
    ``Simulation`` (``subpixel=False``) is fine. Dispersive structures in
    ``sim`` keep their poles fixed; in their cells the coefficient from
    ``ce_of_params`` also scales the polarization correction (it plays the
    role of ``dt/(eps_inf*EPS_0)``), so map those cells to ``eps_inf``. For
    an optimization loop use :func:`make_value_and_grad_params`, which
    compiles once. ``method`` and ``tiled_options`` select the adjoint, as
    for :func:`make_value_and_grad_params`.
    """
    vg = make_value_and_grad_params(sim, lambda p: ce_of_params(p), loss,
                                    remat=remat,
                                    checkpoint_levels=checkpoint_levels,
                                    jit=False, method=method,
                                    tiled_options=tiled_options)
    val, grad = vg(params)
    return float(val), grad
