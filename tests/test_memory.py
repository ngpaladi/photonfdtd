"""Memory-efficiency features for large volumes:

- DFTMonitor: running frequency-domain accumulation (time-axis compression)
- FieldMonitor(compression=...): disk-streamed, lossily-compressed snapshots
- eps_r released during run() and regenerated on demand
- shared curl scratch (PHOTONFDTD_NO_SCRATCH) is bit-for-bit identical
"""
import os
import subprocess
import sys

import numpy as np
import pytest

import photonfdtd as pf


def _dipole_sim(monitors, run_time=60e-15, precision="float64"):
    freq0 = pf.C_0 / 1.0e-6
    grid = pf.Grid(size=(3e-6, 3e-6), cell_size=50e-9, pml_layers=(10, 10, 0))
    src = pf.PointDipole(position=(0.0, 0.0), component="Ez",
                         waveform=pf.GaussianPulse(freq0=freq0, fwhm=6e-15))
    return pf.Simulation(grid, sources=[src], monitors=monitors,
                         run_time=run_time, precision=precision)


# --------------------------------------------------------------------------- #
# DFTMonitor
# --------------------------------------------------------------------------- #
def test_dft_monitor_matches_manual_dft():
    """The running DFT equals a post-hoc DFT of the recorded time series to
    machine precision, using each component's own Yee half-step time."""
    freq0 = pf.C_0 / 1.0e-6
    freqs = [0.8 * freq0, freq0, 1.2 * freq0]
    td = pf.FieldMonitor(name="td", components=("Ez", "Hx"), interval=1)
    fd = pf.DFTMonitor(name="fd", components=("Ez", "Hx"), freqs=freqs, interval=1)
    sim = _dipole_sim([td, fd])
    res = sim.run()
    dt = sim.dt

    assert list(res.dft_freqs["fd"]) == pytest.approx(freqs)
    steps = np.round(res.monitor_times["td"] / dt).astype(int)
    for comp in ("Ez", "Hx"):
        series = res.fields["td"][comp]
        got = res.dft["fd"][comp]
        assert got.shape == (len(freqs),) + series.shape[1:]
        assert np.iscomplexobj(got)
        tvec = (steps + (1.0 if comp[0] == "E" else 0.5)) * dt
        manual = np.zeros_like(got)
        for fi, f in enumerate(freqs):
            phase = np.exp(1j * 2 * np.pi * f * tvec)
            manual[fi] = (series * phase[:, None, None, None]).sum(axis=0) * dt
        rel = np.abs(manual - got).max() / np.abs(got).max()
        assert rel < 1e-10, f"{comp}: DFT mismatch rel={rel:.2e}"


def test_dft_monitor_far_smaller_than_time_series():
    """DFT storage scales with n_freq, not n_steps: a big time-axis reduction."""
    freqs = [pf.C_0 / 1.0e-6]
    td = pf.FieldMonitor(name="td", components=("Ez",), interval=1)
    fd = pf.DFTMonitor(name="fd", components=("Ez",), freqs=freqs, interval=1)
    res = _dipole_sim([td, fd]).run()
    td_bytes = res.fields["td"]["Ez"].nbytes
    fd_bytes = res.dft["fd"]["Ez"].nbytes
    n_rec = res.fields["td"]["Ez"].shape[0]
    # complex128 (16 B) vs float64 (8 B) per cell => ratio ~ n_rec / 2.
    assert td_bytes / fd_bytes > n_rec / 4 > 10


def test_dft_monitor_requires_freqs():
    with pytest.raises(ValueError):
        pf.DFTMonitor(name="x", components=("Ez",), freqs=[])


# --------------------------------------------------------------------------- #
# Compressed / streamed FieldMonitor
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("codec", ["zlib", True])
def test_compressed_monitor_roundtrip(codec):
    plain = pf.FieldMonitor(name="p", components=("Ez", "Hx"), interval=2)
    comp = pf.FieldMonitor(name="c", components=("Ez", "Hx"), interval=2,
                           compression=codec)          # 8-bit default
    res = _dipole_sim([plain, comp]).run()
    for c in ("Ez", "Hx"):
        p = res.fields["p"][c]
        series = res.fields["c"][c]
        assert isinstance(series, pf.CompressedFieldSeries)
        assert series.shape == p.shape
        assert len(series) == p.shape[0]
        full = np.asarray(series)
        assert full.shape == p.shape
        rel = np.abs(full - p).max() / np.abs(p).max()
        assert rel < 1e-2, f"{c}: reconstruction rel err {rel:.2e}"
        # lazy single-frame read == materialised frame
        assert np.array_equal(series[3], full[3])
        # ndarray-style indexing used by examples still works
        _ = series[..., 0]
        _ = series[0]
        # 8-bit mode beats 10x versus the uncompressed float64 monitor
        assert p.nbytes / series.nbytes >= 10.0


def test_compressed_monitor_bits16_more_accurate():
    b8 = pf.FieldMonitor(name="b8", components=("Ez",), interval=2, compression=True)
    b16 = pf.FieldMonitor(name="b16", components=("Ez",), interval=2,
                          compression=True, compression_bits=16)
    plain = pf.FieldMonitor(name="p", components=("Ez",), interval=2)
    res = _dipole_sim([plain, b8, b16]).run()
    p = res.fields["p"]["Ez"]
    e8 = np.abs(np.asarray(res.fields["b8"]["Ez"]) - p).max() / np.abs(p).max()
    e16 = np.abs(np.asarray(res.fields["b16"]["Ez"]) - p).max() / np.abs(p).max()
    assert e16 < e8                      # float16 is more faithful than int8
    assert e16 < 1e-3


def test_compressed_monitor_ram_stays_flat():
    """Streaming keeps peak RAM roughly independent of the number of frames."""
    import tracemalloc

    def peak(compression):
        mon = pf.FieldMonitor(name="m", components=("Ez",), interval=1,
                              compression=compression)
        tracemalloc.start(); tracemalloc.reset_peak()
        _dipole_sim([mon], run_time=90e-15).run()
        _, pk = tracemalloc.get_traced_memory(); tracemalloc.stop()
        return pk

    pk_plain = peak(None)
    pk_comp = peak(True)
    assert pk_comp < pk_plain / 5, (pk_plain, pk_comp)


def test_compression_validation():
    with pytest.raises(ValueError):
        pf.FieldMonitor(name="x", compression="gzip")
    with pytest.raises(ValueError):
        pf.FieldMonitor(name="x", compression=True, compression_bits=4)


# --------------------------------------------------------------------------- #
# eps_r release / regeneration
# --------------------------------------------------------------------------- #
def test_eps_r_released_and_regenerated():
    from photonfdtd.geometry import Box
    from photonfdtd.materials import Medium
    grid = pf.Grid(size=(2e-6, 2e-6), cell_size=100e-9, pml_layers=(4, 4, 0))
    box = Box(center=(0, 0), size=(0.5e-6, 0.5e-6), medium=Medium(permittivity=4.0))
    sim = pf.Simulation(grid, structures=[box], run_time=5e-15)
    before = sim.eps_r.copy()
    sim.run()
    assert sim._eps_r is None                       # released during the run
    after = sim.eps_r                               # regenerated on access
    assert np.array_equal(before, after)
    assert after.max() == pytest.approx(4.0)


def test_custom_eps_r_preserved_across_run():
    grid = pf.Grid(size=(2e-6, 2e-6), cell_size=100e-9, pml_layers=(4, 4, 0))
    sim = pf.Simulation(grid, run_time=5e-15)
    custom = np.full(grid.shape, 2.25, dtype=np.float64)
    sim.eps_r = custom
    sim.run()
    assert sim._eps_r is custom                      # custom grid is never dropped


# --------------------------------------------------------------------------- #
# Curl scratch parity
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Out-of-core / domain-decomposition tiling
# --------------------------------------------------------------------------- #
def _ooc_case():
    freq0 = pf.C_0 / 1.0e-6
    grid = pf.Grid(size=(1.8e-6, 1.4e-6, 1.1e-6), cell_size=80e-9,
                   pml_layers=(4, 4, 4))
    wf = pf.GaussianPulse(freq0=freq0, fwhm=6e-15)
    # Two off-centre, differently-oriented sources so all six components are
    # genuinely excited (no symmetry zeros to hide behind).
    srcs = [pf.PointDipole(position=(0.2e-6, -0.1e-6, 0.15e-6), component="Ez",
                           waveform=wf),
            pf.PointDipole(position=(-0.3e-6, 0.25e-6, -0.1e-6), component="Ey",
                           waveform=wf)]
    comps = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    return grid, srcs, comps


def test_gpu_out_of_core_matches_in_core():
    """GPU/host/disk hierarchy: with use_gpu the disk-backed tiles are processed
    on the GPU (CuPy); the result matches the in-core CPU run to ~machine
    precision. Peak GPU memory is bounded by the tile, so a grid larger than GPU
    memory still runs. Skipped without a working CuPy/GPU."""
    cupy = pytest.importorskip("cupy")
    try:
        int((cupy.arange(3) + 1).sum())
    except Exception:                                  # cupy present but no GPU
        pytest.skip("no working CUDA GPU")
    grid, srcs, comps = _ooc_case()
    ref = pf.Simulation(grid, sources=srcs,
                        monitors=[pf.FieldMonitor(name="m", components=comps, interval=6)],
                        run_time=30e-15).run()
    got = pf.Simulation(grid, sources=srcs,
                        monitors=[pf.FieldMonitor(name="m", components=comps, interval=6)],
                        run_time=30e-15, use_gpu=True).run(out_of_core=True, tile_cells=4)
    escale = max(np.abs(ref.fields["m"][c]).max() for c in ("Ex", "Ey", "Ez"))
    for c in comps:
        rel = np.abs(ref.fields["m"][c] - got.fields["m"][c]).max() / escale
        assert rel < 1e-10, f"{c}: GPU-OOC vs in-core rel={rel:.2e}"


@pytest.mark.parametrize("tile_cells", [3, 1])
def test_out_of_core_matches_in_core(tile_cells):
    """Disk-backed tiled stepping reproduces the in-core result to ~machine
    precision for every component, at any tile size (halo correctness)."""
    grid, srcs, comps = _ooc_case()
    mon = pf.FieldMonitor(name="m", components=comps, interval=6)
    ref = pf.Simulation(grid, sources=srcs, monitors=[mon], run_time=36e-15).run()
    mon2 = pf.FieldMonitor(name="m", components=comps, interval=6)
    got = pf.Simulation(grid, sources=srcs, monitors=[mon2],
                        run_time=36e-15).run(out_of_core=True, tile_cells=tile_cells)
    escale = max(np.abs(ref.fields["m"][c]).max() for c in ("Ex", "Ey", "Ez"))
    hscale = max(np.abs(ref.fields["m"][c]).max() for c in ("Hx", "Hy", "Hz"))
    for c in comps:
        a = ref.fields["m"][c]
        b = got.fields["m"][c]
        scale = hscale if c[0] == "H" else escale
        rel = np.abs(a - b).max() / scale
        assert rel < 1e-11, f"{c}: OOC vs in-core rel={rel:.2e}"


def test_out_of_core_2d():
    """A 2D (nz=1) domain runs out-of-core and matches in-core."""
    freq0 = pf.C_0 / 1.0e-6
    grid = pf.Grid(size=(2e-6, 1.6e-6), cell_size=80e-9, pml_layers=(4, 4, 0))
    src = pf.PointDipole(position=(0.1e-6, -0.1e-6), component="Ez",
                         waveform=pf.GaussianPulse(freq0=freq0, fwhm=6e-15))
    mon = pf.FieldMonitor(name="m", components=("Ez", "Hx", "Hy"), interval=5)
    ref = pf.Simulation(grid, sources=[src], monitors=[mon], run_time=40e-15).run()
    mon2 = pf.FieldMonitor(name="m", components=("Ez", "Hx", "Hy"), interval=5)
    got = pf.Simulation(grid, sources=[src], monitors=[mon2],
                        run_time=40e-15).run(out_of_core=True, tile_cells=4)
    for c in ("Ez", "Hx", "Hy"):
        a = ref.fields["m"][c]
        b = got.fields["m"][c]
        rel = np.abs(a - b).max() / (np.abs(a).max() + 1e-300)
        assert rel < 1e-11, f"{c}: rel={rel:.2e}"


def test_out_of_core_with_compression():
    """Out-of-core composes with compressed monitors."""
    grid, srcs, _ = _ooc_case()
    plain = pf.FieldMonitor(name="p", components=("Ez",), interval=6)
    comp = pf.FieldMonitor(name="c", components=("Ez",), interval=6, compression=True)
    res = pf.Simulation(grid, sources=srcs, monitors=[plain, comp],
                        run_time=30e-15).run(out_of_core=True, tile_cells=3)
    p = res.fields["p"]["Ez"]
    series = res.fields["c"]["Ez"]
    assert isinstance(series, pf.CompressedFieldSeries)
    rel = np.abs(np.asarray(series) - p).max() / np.abs(p).max()
    assert rel < 1e-2


def test_out_of_core_bounds_ram():
    """Peak RAM under out-of-core is a fraction of the in-core footprint."""
    import tracemalloc
    freq0 = pf.C_0 / 1.0e-6
    grid = pf.Grid(size=(3.2e-6, 3.2e-6, 3.2e-6), cell_size=80e-9,
                   pml_layers=(6, 6, 6))
    src = pf.PointDipole(position=(0, 0, 0), component="Ez",
                         waveform=pf.GaussianPulse(freq0=freq0, fwhm=6e-15))

    def peak(ooc, tc=None):
        mon = pf.FieldMonitor(name="m", components=("Ez",), plane_z=0.0, interval=100)
        tracemalloc.start(); tracemalloc.reset_peak()
        pf.Simulation(grid, sources=[src], monitors=[mon],
                      run_time=10e-15).run(out_of_core=ooc, tile_cells=tc)
        _, pk = tracemalloc.get_traced_memory(); tracemalloc.stop()
        return pk

    assert peak(True, 4) < peak(False) / 3


def test_out_of_core_rejects_unsupported_monitor():
    grid, srcs, _ = _ooc_case()
    freq0 = pf.C_0 / 1.0e-6
    dft = pf.DFTMonitor(name="d", components=("Ez",), freqs=[freq0])
    with pytest.raises(NotImplementedError):
        pf.Simulation(grid, sources=srcs, monitors=[dft],
                      run_time=10e-15).run(out_of_core=True, tile_cells=3)


def test_curl_scratch_bit_identical_to_allocating_path():
    """The shared-scratch stepper must be byte-identical to the allocating one."""
    prog = (
        "import sys; sys.path.insert(0,'src'); import numpy as np, hashlib, photonfdtd as pf\n"
        "freq0=pf.C_0/1e-6\n"
        "g=pf.Grid(size=(3e-6,3e-6),cell_size=50e-9,pml_layers=(10,10,0))\n"
        "s=pf.PointDipole(position=(0,0),component='Ez',waveform=pf.GaussianPulse(freq0=freq0,fwhm=6e-15))\n"
        "m=pf.FieldMonitor(name='m',components=('Ez','Hx'),interval=3)\n"
        "r=pf.Simulation(g,sources=[s],monitors=[m],run_time=50e-15).run()\n"
        "print(hashlib.sha256(np.ascontiguousarray(r.fields['m']['Ez']).tobytes()).hexdigest())\n"
    )
    env_on = dict(os.environ); env_on.pop("PHOTONFDTD_NO_SCRATCH", None)
    env_off = dict(os.environ); env_off["PHOTONFDTD_NO_SCRATCH"] = "1"
    h_on = subprocess.check_output([sys.executable, "-c", prog], env=env_on).strip()
    h_off = subprocess.check_output([sys.executable, "-c", prog], env=env_off).strip()
    assert h_on == h_off, "curl scratch changed the result"
