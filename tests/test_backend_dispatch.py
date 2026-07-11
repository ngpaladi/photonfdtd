"""backend="auto" dispatch: pick JAX when it's installed and the run is big
enough, otherwise the NumPy core; explicit choices win; out-of-core and
JAX-incompatible sims stay on NumPy."""
import numpy as np
import pytest

import photonfdtd as pf
import photonfdtd.simulation as sm


def _small(backend="auto", compression=None):
    grid = pf.Grid(size=(2e-6, 2e-6), cell_size=1e-6 / 16, pml_layers=(6, 6, 0))
    src = pf.PointDipole(position=(0, 0), component="Ez",
                         waveform=pf.GaussianPulse(freq0=2e14, fwhm=8e-15))
    mon = pf.DFTMonitor(name="d", components=("Ez",), freqs=[2e14])
    mons = [mon]
    if compression is not None:
        mons.append(pf.FieldMonitor(name="f", components=("Ez",), interval=5,
                                    compression=compression))
    return pf.Simulation(grid, sources=[src], monitors=mons,
                         run_time=30e-15, backend=backend)


def test_explicit_and_auto_selection():
    # Explicit numpy never dispatches to JAX; out-of-core never does either.
    assert _small(backend="numpy")._use_jax_backend(False) is False
    assert _small(backend="auto")._use_jax_backend(True) is False
    # A tiny run stays on NumPy under auto (below the cell-step threshold, or no
    # JAX installed - either way, not JAX).
    assert _small(backend="auto")._use_jax_backend(False) is False
    # Invalid backend name is rejected.
    with pytest.raises(ValueError):
        _small(backend="cupy")


def test_auto_picks_jax_when_worth_it(monkeypatch):
    pytest.importorskip("jax")
    # Drop the threshold so even the tiny run counts as "big enough".
    monkeypatch.setattr(sm, "AUTO_JAX_MIN_CELL_STEPS", 1)
    assert _small(backend="auto")._use_jax_backend(False) is True
    assert _small(backend="jax")._use_jax_backend(False) is True
    # A JAX-incompatible feature (disk-side compressed FieldMonitor) makes auto
    # fall back to NumPy even above the threshold.
    assert _small(backend="auto", compression=True)._use_jax_backend(False) is False

    # And the auto-dispatched (JAX) result matches an explicit NumPy run.
    r_np = _small(backend="numpy").run()
    r_auto = _small(backend="auto").run()          # now routes through JAX
    a = np.asarray(r_np.dft["d"]["Ez"])
    b = np.asarray(r_auto.dft["d"]["Ez"])
    assert np.abs(a - b).max() / np.abs(a).max() < 1e-9
