"""Adapters from third-party layout tools into a photonfdtd Simulation.

The single public function is :func:`from_gdsfactory`, which turns a
:class:`gdsfactory.Component` into a fully-built :class:`Simulation` (with
the grid sized to the layout bounding box plus a configurable padding,
polygons rasterised onto the Yee grid, and CPML on every transverse axis).

gdsfactory itself is an optional dependency: this module imports it lazily,
so a user who never touches the adapter will not pay an import cost.
"""
from __future__ import annotations
from typing import Dict, Iterable, Optional, Sequence, Tuple
import numpy as np

from .grid import Grid
from .materials import Medium
from .geometry import Box, PolySlab
from .simulation import Simulation


LayerSpec = int  # gdsfactory layer index, e.g. 1 for WG core
LayerMap = Dict[LayerSpec, Tuple[Medium, Tuple[float, float]]]
BackgroundSlabs = Iterable[Tuple[Medium, Tuple[float, float]]]


def _component_polygons_um(component) -> Dict[int, list]:
    """Return {layer_index: [Nx2 numpy arrays of (x, y) in microns, ...]}.

    Works against gdsfactory >=9.x (kfactory-based) Components.
    """
    raw = component.get_polygons()
    dbu = float(component.kcl.dbu)   # microns per database unit
    out: Dict[int, list] = {}
    for layer_key, plist in raw.items():
        polys_um = []
        for p in plist:
            pts = np.array([(pt.x * dbu, pt.y * dbu) for pt in p.each_point_hull()],
                           dtype=float)
            if pts.shape[0] >= 3:
                polys_um.append(pts)
        if polys_um:
            out[layer_key] = polys_um
    return out


def _layer_key(component, spec):
    """A ``layers`` / ``electrodes`` key as the kdb layer index that
    ``component.get_polygons()`` uses: an int is taken as that index already;
    a ``(gds_layer, gds_datatype)`` pair is looked up."""
    if isinstance(spec, tuple) and len(spec) == 2:
        return int(component.kcl.layout.layer(int(spec[0]), int(spec[1])))
    return spec


def from_gdsfactory(
    component,
    *,
    layers: LayerMap,
    background_slabs: BackgroundSlabs = (),
    cell_size: float,
    padding: Tuple[float, float, float] = (2e-6, 2e-6, 1e-6),
    pml_layers: Tuple[int, int, int] = (12, 12, 0),
    z_size: Optional[float] = None,
    run_time: float = 1e-12,
    extra_structures: Sequence = (),
    courant: float = 0.99,
    precision="float64",
    xy_bounds: Optional[Tuple[float, float, float, float]] = None,
    electrodes: Optional[Dict[str, tuple]] = None,
    heaters: Optional[Dict[str, tuple]] = None,
    electrode_medium=None,
    heater_medium=None,
    wavelength: float = 1.55e-6,
    **active_kwargs,
):
    """Build a 3D photonfdtd Simulation from a gdsfactory Component.

    Parameters
    ----------
    component : gdsfactory.Component
        Layout source. Polygons are read via ``component.get_polygons()`` and
        scaled from database units to microns via ``component.kcl.dbu``.
    layers : dict
        ``{layer: (Medium, (z_min_m, z_max_m))}``, with ``layer`` either the
        kdb layer index of ``component.get_polygons()`` or a
        ``(gds_layer, gds_datatype)`` pair. Each gdsfactory
        layer is rasterised as a :class:`PolySlab` filled with ``Medium``
        between ``z_min`` and ``z_max`` (both in metres). Layers not in this
        dict are ignored.
    background_slabs : iterable, optional
        ``[(Medium, (z_min_m, z_max_m)), ...]`` of unpatterned z-slabs that
        span the whole xy-extent (e.g. buried oxide, substrate, top
        cladding). Stamped before the layer polygons so the polygons win in
        regions of overlap.
    cell_size : float
        Yee cell size in metres. The same value is used on every axis.
    padding : (px, py, pz), in metres
        Extra extent added to the component bounding box on each axis. The
        z padding is symmetric around the z extent inferred from the layers.
    pml_layers : (nx, ny, nz)
        CPML thickness in cells per axis.
    z_size : float, optional
        Override the auto-computed z extent. If ``None``, z extent equals
        ``max_z - min_z + 2*padding[2]`` where ``min_z`` and ``max_z`` are
        taken from ``layers`` and ``background_slabs``.
    run_time : float
        FDTD run time in seconds. Passed through to the Simulation.
    extra_structures : sequence
        Additional Box/PolySlab structures stamped after the layer polygons
        (highest priority).
    courant : float
        Courant safety factor on the FDTD timestep.
    electrodes : dict, optional
        ``{name: (layer, (z_min_m, z_max_m), (x_um, y_um))}``: the polygon on
        ``layer`` containing the point ``(x_um, y_um)`` (layout coordinates, in
        microns) becomes an :class:`~photonfdtd.active.Electrode` called
        ``name``. Several electrodes may share a layer (signal and ground pads).
    heaters : dict, optional
        ``{name: (layer, (z_min_m, z_max_m), (x_um, y_um), power_W)}``, the
        same for :class:`~photonfdtd.active.Heater`.
    electrode_medium, heater_medium
        The metals, for the static solves (default gold and TiN from
        :func:`photonfdtd.active.metal_active`). The metal is not added to the
        optical layout; list its layer in ``layers`` too if the light should
        see it.
    wavelength, **active_kwargs
        Passed to :class:`~photonfdtd.active.ActiveRegion` (e.g. ``sinks``,
        ``boundary``, ``background``).

    Returns
    -------
    Simulation, or (Simulation, ActiveRegion)
        Built but not yet run. Add sources and monitors before calling
        ``sim.run()``. With ``electrodes`` or ``heaters`` the
        :class:`~photonfdtd.active.ActiveRegion` describing them is returned
        too; prepare it on the simulation (``region.prepare(sim,
        extrude="x")`` for straight electrodes along x) or on a ModeSolver of
        the cross-section.
    """
    polys_by_layer = _component_polygons_um(component)
    layers = {_layer_key(component, k): v for k, v in layers.items()}

    # XY domain in microns -> centre and extent in metres. Normally the whole
    # component bounding box (the kfactory-era DBox exposes .left/.right/
    # .bottom/.top in microns); xy_bounds (x_lo, x_hi, y_lo, y_hi in metres)
    # restricts it to a sub-region instead, so only that window is gridded.
    # Polygons outside the grid simply don't rasterise, so the structure is
    # clipped to the region automatically.
    if xy_bounds is not None:
        x_lo_um, x_hi_um = xy_bounds[0] * 1e6, xy_bounds[1] * 1e6
        y_lo_um, y_hi_um = xy_bounds[2] * 1e6, xy_bounds[3] * 1e6
    else:
        bb = component.bbox()
        x_lo_um, x_hi_um = float(bb.left), float(bb.right)
        y_lo_um, y_hi_um = float(bb.bottom), float(bb.top)
    x_center = 0.5 * (x_lo_um + x_hi_um) * 1e-6
    y_center = 0.5 * (y_lo_um + y_hi_um) * 1e-6
    x_extent = (x_hi_um - x_lo_um) * 1e-6 + 2.0 * padding[0]
    y_extent = (y_hi_um - y_lo_um) * 1e-6 + 2.0 * padding[1]

    # z extent
    z_mins, z_maxs = [], []
    for _med, (z0, z1) in layers.values():
        z_mins.append(min(z0, z1)); z_maxs.append(max(z0, z1))
    for _med, (z0, z1) in background_slabs:
        z_mins.append(min(z0, z1)); z_maxs.append(max(z0, z1))
    if z_mins:
        z_lo, z_hi = min(z_mins), max(z_maxs)
        z_extent_inner = z_hi - z_lo
        z_center = 0.5 * (z_lo + z_hi)
    else:
        z_extent_inner = 0.0
        z_center = 0.0
    z_extent_full = z_extent_inner + 2.0 * padding[2] if z_size is None else float(z_size)

    grid = Grid(
        size=(x_extent, y_extent, z_extent_full),
        cell_size=cell_size,
        pml_layers=pml_layers,
    )

    # Build structures: background slabs first, then per-layer polygons,
    # then any user extras. The Simulation rasteriser respects insertion
    # order (later overrides earlier).
    structures = []

    # Background slabs: stamped as Boxes spanning the entire xy domain. The
    # photonfdtd Grid coordinates are centred at the origin, so we shift the
    # slab centre by -z_center to land in grid space.
    for med, (z0, z1) in background_slabs:
        thickness = abs(z1 - z0)
        c_z = 0.5 * (z0 + z1) - z_center
        structures.append(Box(
            center=(0.0, 0.0, c_z),
            size=(1e9, 1e9, thickness),   # effectively infinite in xy
            medium=med,
        ))

    # gdsfactory polygons: shift vertices to metres and recentre on the
    # component bbox so the grid's origin lines up with the layout's.
    for layer_idx, (med, (z0, z1)) in layers.items():
        if layer_idx not in polys_by_layer:
            continue
        for poly_um in polys_by_layer[layer_idx]:
            verts_m = poly_um * 1e-6
            verts_m = [(float(x - x_center), float(y - y_center)) for x, y in verts_m]
            structures.append(PolySlab(
                vertices=tuple(verts_m),
                z_bounds=(float(z0) - z_center, float(z1) - z_center),
                medium=med,
            ))

    for s in extra_structures:
        structures.append(s)

    sim = Simulation(
        grid=grid,
        structures=structures,
        sources=[],
        monitors=[],
        run_time=run_time,
        courant=courant,
        precision=precision,
    )
    if not electrodes and not heaters:
        return sim

    from . import active as _active

    def pick(name, layer, zb, point, med):
        polys = polys_by_layer.get(_layer_key(component, layer), [])
        from matplotlib.path import Path
        hits = [p for p in polys if Path(p).contains_point(point)]
        if not hits:
            raise ValueError(f"{name!r}: no polygon on layer {layer} contains "
                             f"the point {point} (um)")
        verts = [(float(x * 1e-6 - x_center), float(y * 1e-6 - y_center)) for x, y in hits[0]]
        return PolySlab(vertices=tuple(verts),
                        z_bounds=(float(zb[0]) - z_center, float(zb[1]) - z_center),
                        medium=med)

    e_med = electrode_medium or _active.metal_active("Au")
    h_med = heater_medium or _active.metal_active("TiN")
    els, hts, metal = [], [], []
    for name, (layer, zb, point) in (electrodes or {}).items():
        g = pick(name, layer, zb, point, e_med)
        els.append(_active.Electrode(g, name)); metal.append(g)
    for name, (layer, zb, point, power) in (heaters or {}).items():
        g = pick(name, layer, zb, point, h_med)
        hts.append(_active.Heater(g, float(power), name)); metal.append(g)
    region = _active.ActiveRegion(list(structures) + metal, electrodes=els,
                                  heaters=hts, wavelength=wavelength, **active_kwargs)
    return sim, region
