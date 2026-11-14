"""VTI and STL export."""
import re
import struct
from collections import Counter

import numpy as np

import photonfdtd as pf


def _read_vti(path):
    raw = open(path, "rb").read()
    head = raw[:raw.index(b'<AppendedData')].decode()
    data = raw[raw.index(b'encoding="raw">') + len(b'encoding="raw">'):]
    data = data[data.index(b"_") + 1:]
    ext = [int(v) for v in re.search(r'WholeExtent="([^"]+)"', head).group(1).split()]
    shape = (ext[1] + 1, ext[3] + 1, ext[5] + 1)
    out = {}
    for typ, name, off in re.findall(
            r'type="(\w+)" Name="(\w+)" format="appended" offset="(\d+)"', head):
        off = int(off)
        n = struct.unpack("<Q", data[off:off + 8])[0]
        dt = {"Float64": "<f8", "Float32": "<f4", "UInt8": "u1", "Int32": "<i4"}[typ]
        a = np.frombuffer(data[off + 8:off + 8 + n], dtype=dt)
        out[name] = a.reshape(shape[::-1]).transpose(2, 1, 0)
    origin = [float(v) for v in re.search(r'Origin="([^"]+)"', head).group(1).split()]
    return out, origin


def test_vti_roundtrip(tmp_path):
    g = pf.Grid(size=(1e-6, 0.6e-6, 0.4e-6), cell_size=0.1e-6)
    rng = np.random.default_rng(1)
    eps = rng.random(g.shape)
    ez = rng.random(g.shape) + 1j * rng.random(g.shape)
    mask = eps > 0.5
    p = tmp_path / "f.vti"
    pf.export_vti(str(p), g, eps=eps, Ez=ez[None], solid=mask)
    out, origin = _read_vti(p)
    assert np.array_equal(out["eps"], eps)
    assert np.array_equal(out["Ez_re"], ez.real) and np.array_equal(out["Ez_im"], ez.imag)
    assert np.array_equal(out["solid"].astype(bool), mask)
    assert np.isclose(origin[0], g.coords[0][0])


def test_stl_is_closed_and_counts_faces(tmp_path):
    g = pf.Grid(size=(1e-6, 1e-6, 1e-6), cell_size=0.1e-6)
    m = np.zeros(g.shape, bool)
    m[2:5, 3:6, 4:7] = True                    # a 3x3x3 block
    m[7, 7, 7] = True                          # and one loose voxel
    p = tmp_path / "s.stl"
    n = pf.export_stl(str(p), g, m)
    assert n == 2 * (6 * 9 + 6)
    raw = open(p, "rb").read()
    assert struct.unpack("<I", raw[80:84])[0] == n
    rec = np.frombuffer(raw[84:], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                         ("a", "<u2")])
    # Closed, consistently oriented surface: every directed edge appears
    # exactly once and its reverse exactly once.
    edges = Counter()
    for v in rec["v"]:
        pts = [tuple(np.round(x / 1e-8).astype(int)) for x in v]
        for i in range(3):
            edges[(pts[i], pts[(i + 1) % 3])] += 1
    assert all(c == 1 for c in edges.values())
    assert all(edges[(b, a)] == 1 for (a, b) in edges)
    # Outward normals: the normal points away from the block centre.
    cen = rec["v"].mean(axis=1)
    blk = (cen[:, 0] < 0.2e-6) & (cen[:, 1] < 0.2e-6)
    c0 = np.array([g.coords[0][3], g.coords[1][4], g.coords[2][5]])
    assert (np.einsum("ij,ij->i", rec["n"][blk], cen[blk] - c0) > 0).all()
    # And the triangle winding agrees with the stored normal.
    v = rec["v"].astype(float)
    cr = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    assert (np.einsum("ij,ij->i", cr, rec["n"]) > 0).all()
