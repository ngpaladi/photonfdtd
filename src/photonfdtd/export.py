"""Export grids, fields and structures for outside viewers.

* :func:`export_vti` writes arrays on the Yee grid (permittivity, field
  snapshots, DFT magnitudes) as a VTK ImageData file (``.vti``) for ParaView
  or VisIt, with the grid's origin and spacing.
* :func:`export_stl` writes the boundary of a boolean cell mask (a structure,
  an optimized design) as a binary STL surface, one square per exposed voxel
  face, so the mesh is closed and its faces align with the grid.
"""
from __future__ import annotations

import struct
from typing import Dict

import numpy as np

__all__ = ["export_vti", "export_stl"]

_VTK_TYPES = {np.dtype("float32"): "Float32", np.dtype("float64"): "Float64",
              np.dtype("int32"): "Int32", np.dtype("uint8"): "UInt8"}


def export_vti(path: str, grid, **arrays) -> None:
    """Write named arrays sampled on ``grid``'s nodes to a ``.vti`` file.

    Each array has the grid's shape (a leading size-1 axis is dropped, so a
    single snapshot ``result.fields[name][comp][k]`` works). Complex arrays
    are written as two arrays, ``<name>_re`` and ``<name>_im``; booleans as
    UInt8. The data is appended raw (little-endian), the compact form every
    VTK reader accepts.
    """
    shape = tuple(grid.shape)
    items: Dict[str, np.ndarray] = {}
    for name, a in arrays.items():
        a = np.asarray(a)
        while a.ndim > 3 and a.shape[0] == 1:
            a = a[0]
        if a.shape != shape:
            raise ValueError(f"{name}: shape {a.shape} is not the grid's {shape}")
        if np.iscomplexobj(a):
            items[name + "_re"] = a.real
            items[name + "_im"] = a.imag
        elif a.dtype == bool:
            items[name] = a.astype(np.uint8)
        else:
            items[name] = a
    origin = [float(grid.coords[a][0]) for a in range(3)]
    spacing = [float(grid.cell_size[a]) if shape[a] > 1 else 1.0 for a in range(3)]
    extent = " ".join(f"0 {n - 1}" for n in shape)
    head, blobs, offset = [], [], 0
    for name, a in items.items():
        a = np.asarray(a)
        if a.dtype not in _VTK_TYPES:
            a = a.astype(np.float64)
        # VTK images are x-fastest (Fortran order).
        raw = np.ascontiguousarray(a.transpose(2, 1, 0)).astype(
            a.dtype.newbyteorder("<")).tobytes()
        head.append(f'        <DataArray type="{_VTK_TYPES[a.dtype]}" Name="{name}" '
                    f'format="appended" offset="{offset}"/>')
        blob = struct.pack("<Q", len(raw)) + raw
        blobs.append(blob)
        offset += len(blob)
    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" '
        'header_type="UInt64">\n'
        f'  <ImageData WholeExtent="{extent}" '
        f'Origin="{origin[0]!r} {origin[1]!r} {origin[2]!r}" '
        f'Spacing="{spacing[0]!r} {spacing[1]!r} {spacing[2]!r}">\n'
        f'    <Piece Extent="{extent}">\n'
        '      <PointData>\n' + "\n".join(head) + '\n      </PointData>\n'
        '    </Piece>\n  </ImageData>\n'
        '  <AppendedData encoding="raw">\n   _')
    with open(path, "wb") as f:
        f.write(xml.encode())
        for b in blobs:
            f.write(b)
        f.write(b"\n  </AppendedData>\n</VTKFile>\n")


def export_stl(path: str, grid, mask, name: str = "photonfdtd") -> int:
    """Write the surface of the cells where ``mask`` is True as binary STL.

    Each cell is the box of side ``cell_size`` centred on its node; on a
    collapsed axis the box is the smallest cell size thick, so a 2-D design
    extrudes to a one-cell slab. Returns the triangle count. Coordinates are
    in metres.
    """
    m = np.asarray(mask, bool)
    if m.shape != tuple(grid.shape):
        raise ValueError(f"mask shape {m.shape} is not the grid's {tuple(grid.shape)}")
    d = [grid.cell_size[a] if grid.shape[a] > 1 else grid.min_cell for a in range(3)]
    lo = [np.asarray(grid.coords[a], float) - d[a] / 2 for a in range(3)]
    tris = []
    pad = np.pad(m, 1)
    for a in range(3):
        for sgn in (-1, 1):
            nb = np.roll(pad, -sgn, axis=a)[1:-1, 1:-1, 1:-1]
            face = m & ~nb
            ii, jj, kk = np.nonzero(face)
            if ii.size == 0:
                continue
            b, c = [x for x in range(3) if x != a]
            idx = (ii, jj, kk)
            base = np.stack([lo[x][idx[x]] for x in range(3)], axis=1)
            base[:, a] += d[a] if sgn > 0 else 0.0
            eb = np.zeros(3); eb[b] = d[b]
            ec = np.zeros(3); ec[c] = d[c]
            n = np.zeros(3); n[a] = sgn
            p0, p1, p2, p3 = base, base + eb, base + eb + ec, base + ec
            # Counter-clockwise seen from outside: orientation by sign and
            # the handedness of (a, b, c).
            right = (b - a) % 3 == 1
            if (sgn > 0) == right:
                quads = ((p0, p1, p2), (p0, p2, p3))
            else:
                quads = ((p0, p2, p1), (p0, p3, p2))
            for t in quads:
                tris.append((np.broadcast_to(n, base.shape), *t))
    count = sum(t[0].shape[0] for t in tris)
    with open(path, "wb") as f:
        f.write(name.encode()[:80].ljust(80, b"\0"))
        f.write(struct.pack("<I", count))
        for n, v0, v1, v2 in tris:
            rec = np.zeros(n.shape[0], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                              ("attr", "<u2")])
            rec["n"] = n
            rec["v"] = np.stack([v0, v1, v2], axis=1)
            f.write(rec.tobytes())
    return count
