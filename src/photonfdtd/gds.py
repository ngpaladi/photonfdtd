"""GDS export of optimized designs (optional dependency: ``gdstk``).

A density design is binarized at a threshold and contoured
(``contourpy``, which ships with matplotlib); nested contour loops become
holes by an even-odd XOR, and the result is clipped to the design box. A
parametric design is written from its polygons directly. Coordinates are
written in micrometres (GDS user unit 1 um, database unit 1 nm).

Install the extra with ``pip install photonfdtd[gds]``.
"""
from __future__ import annotations
from typing import Optional, Sequence

import numpy as np


def _gdstk():
    try:
        import gdstk
    except ImportError as e:            # pragma: no cover - exercised manually
        raise ImportError("GDS export needs gdstk: pip install photonfdtd[gds]"
                          ) from e
    return gdstk


def density_polygons(x, y, rho, threshold: float = 0.5):
    """Closed polygons (m) of ``rho >= threshold`` on cell centres ``x``, ``y``.

    The field is padded with one void cell so shapes touching the edge close;
    loops are returned as found (outer boundaries and holes alike), to be
    combined even-odd.
    """
    import contourpy
    rho = np.asarray(rho, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    dx = x[1] - x[0] if x.size > 1 else 1.0
    dy = y[1] - y[0] if y.size > 1 else dx
    xp = np.concatenate([[x[0] - dx], x, [x[-1] + dx]])
    yp = np.concatenate([[y[0] - dy], y, [y[-1] + dy]])
    zp = np.pad(rho, 1, constant_values=min(0.0, threshold - 1.0))
    gen = contourpy.contour_generator(xp, yp, zp.T, line_type="Separate")
    return [np.asarray(l) for l in gen.lines(threshold) if len(l) >= 3]


def export_gds(path: str, region=None, rho=None, grid=None,
               polygons: Optional[Sequence] = None, layer: int = 1,
               datatype: int = 0, threshold: float = 0.5,
               cell: str = "DESIGN"):
    """Write a design to ``path`` (GDSII) and return the gdstk library.

    Either ``region`` + ``rho`` + ``grid`` (a physical density on the
    region's cells, e.g. ``region.rho_phys(sim, rho)``) or ``polygons`` (a
    list of ``(n, 2)`` vertex arrays in metres, e.g.
    ``ShapeDesign.polygons(params)``).
    """
    gdstk = _gdstk()
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    c = lib.new_cell(cell)
    um = 1e6
    if polygons is not None:
        for p in polygons:
            c.add(gdstk.Polygon(np.asarray(p, float) * um, layer=layer,
                                datatype=datatype))
    else:
        if region is None or rho is None or grid is None:
            raise ValueError("pass polygons=..., or region, rho and grid")
        x, y = region.coords(grid)
        loops = density_polygons(x, y, rho, threshold)
        shape = []
        for l in loops:
            poly = [gdstk.Polygon(l * um)]
            shape = poly if not shape else gdstk.boolean(shape, poly, "xor")
        (cx, cy), (sx, sy) = region.center, region.size
        box = gdstk.rectangle(((cx - sx / 2) * um, (cy - sy / 2) * um),
                              ((cx + sx / 2) * um, (cy + sy / 2) * um))
        if shape:
            for p in gdstk.boolean(shape, box, "and", layer=layer,
                                   datatype=datatype):
                c.add(p)
    lib.write_gds(path)
    return lib
