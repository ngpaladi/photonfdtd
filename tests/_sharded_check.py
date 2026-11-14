"""Run in a subprocess with 4 simulated CPU devices (see test_sharding.py)."""
import json
import sys

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import photonfdtd as pf
from photonfdtd import jaxbackend

LAM = 1e-6
F0 = pf.C_0 / LAM


def build(devices):
    dx = LAM / 14
    g = pf.Grid(size=(3.1e-6, 1.3e-6, 0.9e-6), cell_size=dx, pml_layers=(6, 0, 4),
                periodic=(False, True, False))
    st = [pf.Box(center=(0.2e-6, 0, 0), size=(0.6e-6, 0.5e-6, 0.4e-6),
                 medium=pf.Medium.from_index(2.0)),
          pf.Sphere(center=(-0.5e-6, 0.2e-6, 0), radius=0.25e-6,
                    medium=pf.Medium(permittivity=3.0, permeability=1.5)),
          pf.Cylinder(center=(0.9e-6, 0, 0), radius=0.2e-6, length=1.0e-6,
                      axis="y", medium=pf.DispersiveMedium.lorentz(
                          2.0, [(0.5, 0.5 * F0, 0.05 * F0)]))]
    w = pf.GaussianPulse(freq0=F0, fwhm=5e-15)
    src = [pf.PointDipole(position=(-0.9e-6, 0.1e-6, 0.05e-6), component="Ez",
                          waveform=w),
           pf.PlaneWave(center=(-1.1e-6, 0, 0), direction="+x",
                        polarization="Ey", waveform=w, size=(0, 1, 0.3e-6)),
           pf.ChargedParticle(charge=1e-13, velocity=(0, 0.8 * pf.C_0, 0),
                              start=(0.4e-6, -0.6e-6, 0.0), cloud_cells=1.0)]
    mons = [pf.DFTMonitor(name="d", components=("Ez", "Hy"), freqs=[F0],
                          plane_axis="x", plane_position=0.9e-6),
            pf.FluxMonitor(name="f", plane_axis="x", plane_position=0.9e-6),
            pf.FieldMonitor(name="s", components=("Ex",), times=[20e-15])]
    return pf.Simulation(g, structures=st, sources=src, monitors=mons,
                         run_time=25e-15, devices=devices)


one, four = build(None), build(4)
assert four.devices is not None and len(four.devices) == 4
r1, r4 = one.run(), four.run()
err = max(
    float(np.abs(r4.dft["d"]["Ez"] - r1.dft["d"]["Ez"]).max() / np.abs(r1.dft["d"]["Ez"]).max()),
    float(np.abs(r4.fields["s"]["Ex"] - r1.fields["s"]["Ex"]).max()
          / np.abs(r1.fields["s"]["Ex"]).max()),
    abs(r4.flux["f"] / r1.flux["f"] - 1))

loss = lambda out: jnp.sum(jnp.abs(out["dft"]["d"]["Ez"]) ** 2)
# Gradients through the checkpointed adjoint (magnetic sphere: params path).
def vg(sim):
    ce, _ = jaxbackend.update_coefficients(sim)
    run = jaxbackend._run_fn(sim)
    def f(scale):
        return loss(run({c: v * scale for c, v in ce.items()}))
    return jax.value_and_grad(f)(jnp.asarray(1.0))
v1, g1 = vg(one)
v4, g4 = vg(four)
gerr = abs(float(g4) / float(g1) - 1)

# Evidence of a real partition: the compiled step exchanges slab faces.
static = jaxbackend._build_static(four)
txt = jax.jit(lambda ce: jaxbackend._device_simulate(
    four, static, jaxbackend._monitor_plan(four), ce, ade=None)).lower(
    {c: jnp.asarray(v) for c, v in jaxbackend.update_coefficients(four)[0].items()}
).compile().as_text()
print(json.dumps(dict(err=err, gerr=gerr,
                      permutes=txt.count("collective-permute"),
                      ndev=len(jax.devices()))))
