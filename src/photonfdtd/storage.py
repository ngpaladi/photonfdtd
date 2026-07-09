"""On-disk compressed storage for streamed FieldMonitor snapshots.

A :class:`CompressedFieldSeries` writes each recorded snapshot to a temporary
file as it is produced, so the simulation's RAM footprint stays flat (only a
small per-frame index lives in memory) regardless of how many frames a long,
large-volume run records. Each frame is stored lossily as

    scale  = max(|snapshot|)            # one float64 per frame
    q      = (snapshot / scale)         # quantised to `bits`, values in [-1, 1]

then compressed (zstd if the ``zstandard`` package is available, otherwise
stdlib zlib). The per-frame scale makes the quantisation robust to the field's
absolute magnitude - fields of order 1e-16 or 1e+6 are both stored with the
same *relative* precision, rather than under/overflowing a raw low-precision
cast.

Two quantisation depths are offered:

    bits=8  (int8, default)  : ~10-30x smaller than an uncompressed float64
                               monitor, ~4e-3 relative error. Meets a 10x
                               target with margin; ideal for field movies /
                               visualisation.
    bits=16 (float16)        : ~5-7x smaller, ~2e-4 relative error, when more
                               fidelity is wanted than the 8-bit mode gives.

The absorbing (PML) regions and smooth spatial structure make the quantised
frames highly compressible, which is why the 8-bit mode routinely lands well
past 10x rather than at the 8x its byte width alone would imply.

The result object behaves like a read-only array of snapshots: ``len(series)``,
``series.shape``, ``series[i]`` (decompress one frame), and ``np.asarray(series)``
or ``series[...]`` (materialise everything). Frame reads seek into the file and
decompress on demand, so browsing a few frames never loads the whole series.
"""
from __future__ import annotations
import os
import tempfile
import zlib
from typing import List, Tuple
import numpy as np

try:
    import zstandard as _zstd
    _HAVE_ZSTD = True
except ImportError:                       # pragma: no cover - env dependent
    _zstd = None
    _HAVE_ZSTD = False


class _ZlibCodec:
    name = "zlib"

    def __init__(self, level: int = 6):
        self._level = level

    def compress(self, data: bytes) -> bytes:
        return zlib.compress(data, self._level)

    def decompress(self, data: bytes) -> bytes:
        return zlib.decompress(data)


class _ZstdCodec:                          # pragma: no cover - exercised when zstd present
    name = "zstd"

    def __init__(self, level: int = 6):
        # zstd frames embed the content size, so one-shot decompress needs no
        # out-size hint. Fresh (de)compressor objects are cheap and thread-safe
        # to create per call; we keep one each for the series' lifetime.
        self._c = _zstd.ZstdCompressor(level=level)
        self._d = _zstd.ZstdDecompressor()

    def compress(self, data: bytes) -> bytes:
        return self._c.compress(data)

    def decompress(self, data: bytes) -> bytes:
        return self._d.decompress(data)


def get_codec(compression):
    """Resolve a ``compression`` spec to a codec instance.

    Accepts ``True``/``"auto"`` (zstd if installed, else zlib), ``"zstd"``
    (requires the ``zstandard`` package), or ``"zlib"`` (stdlib, always
    available).
    """
    if compression in (True, "auto"):
        return _ZstdCodec() if _HAVE_ZSTD else _ZlibCodec()
    if compression == "zstd":
        if not _HAVE_ZSTD:
            raise ImportError(
                "compression='zstd' requires the 'zstandard' package "
                "(pip install zstandard); use compression='zlib' for the "
                "stdlib fallback, or compression=True to auto-select."
            )
        return _ZstdCodec()
    if compression == "zlib":
        return _ZlibCodec()
    raise ValueError(
        f"unknown compression {compression!r}; use True/'auto', 'zstd', or 'zlib'"
    )


class CompressedFieldSeries:
    """Append-only, disk-backed, lossily-compressed stack of field snapshots.

    Snapshots are appended in time order via :meth:`append` during the run and
    read back lazily afterwards. Only the (offset, length, scale) index is held
    in RAM; the frame bytes live in a temporary file that is removed when the
    series is closed or garbage-collected.
    """

    def __init__(self, snap_shape, out_dtype, codec, bits=8, tmpdir=None):
        if bits not in (8, 16):
            raise ValueError(f"bits must be 8 or 16, got {bits!r}")
        self._snap_shape = tuple(int(s) for s in snap_shape)
        self._out_dtype = np.dtype(out_dtype)
        self._codec = codec
        self._bits = int(bits)
        # int8 for 8-bit (scale/127 quantiser), float16 for 16-bit.
        self._qdtype = np.int8 if bits == 8 else np.float16
        fd, self._path = tempfile.mkstemp(prefix="photonfdtd_mon_", suffix=".zfld",
                                          dir=tmpdir)
        self._fh = os.fdopen(fd, "w+b")
        self._index: List[Tuple[int, int, float]] = []  # (offset, length, scale)

    # -- write side -------------------------------------------------------- #
    def append(self, snap: np.ndarray) -> None:
        snap = np.ascontiguousarray(snap)
        scale = float(np.abs(snap).max())
        if scale == 0.0 or not np.isfinite(scale):
            scale = 1.0
        if self._bits == 8:
            # Symmetric int8: [-127, 127] maps to [-scale, scale].
            q = np.rint(snap / scale * 127.0).astype(np.int8)
        else:
            q = (snap / scale).astype(np.float16)
        blob = self._codec.compress(q.tobytes())
        off = self._fh.seek(0, os.SEEK_END)
        self._fh.write(blob)
        self._index.append((off, len(blob), scale))

    def finalize(self) -> None:
        """Flush buffered writes so frames are readable."""
        self._fh.flush()

    # -- read side --------------------------------------------------------- #
    def _read_frame(self, i: int) -> np.ndarray:
        off, length, scale = self._index[i]
        self._fh.seek(off)
        raw = self._codec.decompress(self._fh.read(length))
        q = np.frombuffer(raw, dtype=self._qdtype).reshape(self._snap_shape)
        rec = q.astype(self._out_dtype)
        if self._bits == 8:
            scale = scale / 127.0
        return rec * self._out_dtype.type(scale)

    def __len__(self) -> int:
        return len(self._index)

    @property
    def shape(self) -> Tuple[int, ...]:
        return (len(self._index),) + self._snap_shape

    @property
    def ndim(self) -> int:
        return 1 + len(self._snap_shape)

    @property
    def dtype(self) -> np.dtype:
        return self._out_dtype

    @property
    def nbytes(self) -> int:
        """Compressed on-disk size in bytes (index excluded)."""
        return sum(length for _, length, _ in self._index)

    @property
    def nbytes_uncompressed(self) -> int:
        return len(self._index) * int(np.prod(self._snap_shape)) * self._out_dtype.itemsize

    def __array__(self, dtype=None) -> np.ndarray:
        out = np.empty(self.shape, dtype=self._out_dtype)
        for i in range(len(self._index)):
            out[i] = self._read_frame(i)
        return out.astype(dtype) if dtype is not None else out

    def __getitem__(self, idx):
        # Integer index -> decompress just that frame (cheap, no full load).
        if isinstance(idx, (int, np.integer)):
            n = len(self._index)
            if idx < 0:
                idx += n
            if not 0 <= idx < n:
                raise IndexError(f"frame index {idx} out of range [0, {n})")
            return self._read_frame(int(idx))
        # Anything else (slices, ellipsis, fancy) -> materialise then index.
        return np.asarray(self)[idx]

    def __iter__(self):
        for i in range(len(self._index)):
            yield self._read_frame(i)

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None
                try:
                    os.unlink(self._path)
                except OSError:
                    pass

    def __del__(self):                     # best-effort cleanup
        try:
            self.close()
        except Exception:
            pass
