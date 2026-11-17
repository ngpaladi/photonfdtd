"""Domain decomposition over several devices (simulated on CPU)."""
import json
import os
import subprocess
import sys

import pytest


def test_sharded_run_matches_single_device():
    """A 3-D run with PML, a periodic axis, magnetic and dielectric media,
    split over 4 devices, gives the single-device answer (fields, DFT, flux
    and the checkpointed gradient), and the compiled program really exchanges
    slab faces between devices."""
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=4",
               JAX_PLATFORMS="cpu")
    here = os.path.dirname(__file__)
    out = subprocess.run([sys.executable, os.path.join(here, "_sharded_check.py")],
                         env=env, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["ndev"] == 4
    assert r["err"] < 1e-10, r
    assert r["gerr"] < 1e-10, r
    assert r["permutes"] > 0, r


def test_reversible_refuses_devices():
    import photonfdtd as pf
    import jax.numpy as jnp
    g = pf.Grid(size=(2e-6, 1e-6), cell_size=1e-7, periodic=(False, True, False))
    mon = pf.DFTMonitor(name="d", components=("Ez",), freqs=[3e14])
    sim = pf.Simulation(g, monitors=[mon], run_time=1e-15, devices=1)
    assert sim.devices is None                 # one device: plain run
    sim.devices = ["a", "b"]
    with pytest.raises(NotImplementedError, match="one device"):
        pf.jax_value_and_grad_eps_reversible(sim, lambda o: jnp.sum(jnp.abs(o["dft"]["d"]["Ez"])))
