"""Stopping a forward run once the fields have decayed."""
import numpy as np

import photonfdtd as pf

import jax
jax.config.update("jax_enable_x64", True)

LAM = 1e-6
F0 = pf.C_0 / LAM


def _sim(run_time=400e-15):
    dx = LAM / 20
    g = pf.Grid(size=(6e-6, 6e-6), cell_size=dx, pml_layers=(12, 12, 0))
    src = pf.PointDipole(position=(0, 0), component="Ez",
                         waveform=pf.GaussianPulse(freq0=F0, fwhm=5e-15))
    mons = [pf.DFTMonitor(name="d", components=("Ez",), freqs=[F0],
                          plane_axis="x", plane_position=1.5e-6),
            pf.FieldMonitor(name="f", components=("Ez",), interval=100),
            pf.FluxMonitor(name="p", plane_axis="x", plane_position=1.5e-6)]
    return pf.Simulation(g, sources=[src], monitors=mons, run_time=run_time)


def test_shutoff_stops_early_and_keeps_the_answer():
    full = _sim().run()
    sim = _sim()
    res = sim.run(shutoff=1e-6, shutoff_interval=25)
    n = res.times.size
    assert n < 0.5 * sim.n_steps and n % 25 == 0
    d0 = np.asarray(full.dft["d"]["Ez"])
    d1 = np.asarray(res.dft["d"]["Ez"])
    assert np.abs(d1 - d0).max() / np.abs(d0).max() < 1e-3
    assert abs(res.flux["p"] / full.flux["p"] - 1) < 1e-3
    # Snapshots stop at the step reached and match the full run's.
    k = res.fields["f"]["Ez"].shape[0]
    assert k == len([t for t in full.monitor_times["f"] if t < n * sim.dt])
    assert np.allclose(res.fields["f"]["Ez"], full.fields["f"]["Ez"][:k])


def test_shutoff_waits_for_the_sources():
    """A threshold that the quiet start would meet does not stop the run
    before the pulse has been injected."""
    sim = _sim(run_time=100e-15)
    res = sim.run(shutoff=0.5, shutoff_interval=5)
    t_src = sim.sources[0].waveform.t0 + 4 * sim.sources[0].waveform.sigma
    assert res.times[-1] >= t_src * 0.9
