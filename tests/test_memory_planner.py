"""Peak-memory estimator / mode planner (photonfdtd.memory)."""
import photonfdtd as pf


def _sim(run_time=200e-15, pml=(0, 0, 0)):
    src = pf.PointDipole(position=(0, 0, 0), component="Ez",
                         waveform=pf.GaussianPulse(freq0=3e14, fwhm=6e-15))
    mon = pf.DFTMonitor(name="d", components=("Ez",), freqs=[3e14])
    grid = pf.Grid(size=(3e-6, 3e-6, 2e-6), cell_size=1e-6 / 12, pml_layers=pml)
    return pf.Simulation(grid, sources=[src], monitors=[mon],
                         run_time=run_time, use_jax=True)


def test_mode_ordering():
    """Adjoint memory: reversible < nested < none; out-of-core < forward."""
    est = pf.estimate_memory(_sim())
    assert est["adjoint_reversible"] < est["adjoint_nested"] < est["adjoint_none"]
    assert est["out_of_core"] < est["forward"]
    assert est["forward"] == est["working_set"]


def test_step_scaling():
    """none scales with n_steps; reversible is independent of it."""
    short, long = _sim(run_time=100e-15), _sim(run_time=400e-15)
    assert long.n_steps > 2 * short.n_steps
    e_s, e_l = pf.estimate_memory(short), pf.estimate_memory(long)
    # trajectory storage grows ~ n_steps
    assert e_l["adjoint_none"] / e_s["adjoint_none"] > 2.0
    # reversible is O(1) in steps -> identical working sets
    assert e_l["adjoint_reversible"] == e_s["adjoint_reversible"]


def test_recommend_fits_budget():
    sim = _sim()
    est = pf.estimate_memory(sim)
    # A tiny budget forces the least-memory adjoint; a huge one allows the
    # fastest (no checkpointing).
    assert pf.recommend_mode(sim, est["adjoint_reversible"] + 1) == "adjoint_reversible"
    assert pf.recommend_mode(sim, est["adjoint_none"] + 1) == "adjoint_none"
    # Below even the reversible adjoint -> fall back to tiling.
    assert pf.recommend_mode(sim, est["out_of_core"] + 1) == "out_of_core"
    # Forward-only planning.
    assert pf.recommend_mode(sim, est["forward"] + 1, differentiable=False) == "forward"


def test_pml_increases_estimate():
    """PML adds psi state to the working set."""
    assert pf.estimate_memory(_sim(pml=(6, 6, 6)))["forward"] > \
        pf.estimate_memory(_sim(pml=(0, 0, 0)))["forward"]
