"""Example 12: simulating an optoelectronic chip as a circuit.

A small lithium niobate chip, laid out in gdsfactory: light enters an x-cut
MZI switch (photonfdtd.lnoi), whose signal electrode is wired to a bond pad;
each output runs to a waveguide photodetector biased from its own pads.
photonfdtd.circuit reads the layout's netlist, gives every instance a model
(the switch from its electroded cross-section, solved once and cached; the
waveguides from their length; the detectors from their responsivity), traces
the metal from the pads to the electrodes, and solves the whole chip.

1. DC: detector photocurrents against the pad voltage.
2. Wavelength: the chip's transmission at 0 V and at V_pi.
3. Time domain: a 10 Gb/s PRBS on the pad, through the modulator's 30 GHz
   electrode bandwidth and the detectors' 20 GHz bandwidth.

Needs gdsfactory. The first run solves the switch's cross-section (about a
minute); later runs read it from the cache.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gdsfactory as gf
from photonfdtd import circuit, lnoi
from photonfdtd.circuit.cells import photodetector

gf.gpdk.PDK.activate()

# ---- The layout.
chip_layout = gf.Component("eo_chip")
sw = chip_layout << lnoi.mzi_switch(arm_length=5000)
pad = chip_layout << gf.components.pad()
pad.move((-300, 250))
gf.routing.route_single(chip_layout, pad.ports["e3"], sw.ports["sig_w"],
                        cross_section="metal_routing", allow_width_mismatch=True)
for name, port in (("pd_cross", "o3"), ("pd_bar", "o4")):
    wg = chip_layout << gf.components.straight(length=200, cross_section=lnoi.rib())
    wg.connect("o1", sw.ports[port])
    pd = chip_layout << photodetector(bandwidth=20e9)
    pd.connect("o1", wg.ports["o2"])
    chip_layout.add_port(f"{name}_anode", port=pd.ports["anode_w"])
    chip_layout.add_port(f"{name}_cathode", port=pd.ports["cathode_w"])
chip_layout.add_port("laser", port=sw.ports["o1"])
chip_layout.add_port("gnd_top", port=sw.ports["gnd_top_w"])
chip_layout.add_port("gnd_bot", port=sw.ports["gnd_bot_w"])
chip_layout.write_gds("12_optoelectronic_chip.gds")

# ---- The circuit. Waveguides: the LN ridge's n_eff and n_g, 0.3 dB/cm.
chip = circuit.Chip.from_gdsfactory(chip_layout, bandwidth=30e9,
                                    waveguide=dict(neff=1.89, ng=2.3, loss_db_cm=0.3))
detectors = sorted(n for n, m in chip.models.items() if isinstance(m, circuit.Photodetector))
# Name each detector after the pad its anode is wired to.
label = {d: chip.net_names[chip.net((d, "anode"))].replace("_anode", "") for d in detectors}
print("instances:", ", ".join(f"{n} ({type(m).__name__})" for n, m in chip.models.items()))
print("nets:", chip.net_names)

bias = {"gnd_top": 0.0, "gnd_bot": 0.0,
        "pd_cross_anode": 0.0, "pd_bar_anode": 0.0,
        "pd_cross_cathode": 2.0, "pd_bar_cathode": 2.0}     # 2 V reverse bias
laser = {"laser": 1e-3}                                     # 1 mW

# Which detector sits on which output: follow the waveguide from o3 / o4.
st0 = chip.solve(1.55, drives={**bias, "pad": 0.0}, inputs=laser)
print("photocurrent at 0 V:",
      {label[d]: f"{st0.detector_current[d][0] * 1e3:.3f} mA" for d in detectors})

# ---- 1. DC sweep.
v = np.linspace(-10, 10, 81)
I = {d: [] for d in detectors}
for x in v:
    st = chip.solve(1.55, drives={**bias, "pad": x}, inputs=laser)
    for d in detectors:
        I[d].append(st.detector_current[d][0])
I = {d: np.array(i) for d, i in I.items()}
v_pi = lnoi.model(lnoi.mzi_switch(arm_length=5000)).info["vpi"]
print(f"V_pi = {v_pi:.2f} V")

# ---- 2. Wavelength.
wl = np.linspace(1.50, 1.60, 201)

# ---- 3. 10 Gb/s PRBS.
rate = 10e9
t = np.arange(0, 24 / rate, 1 / (rate * 64))
prbs = circuit.PRBS(low=0.0, high=v_pi, rate=rate, order=7)
tr = chip.transient(t, {**bias, "pad": prbs}, inputs=laser)

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for d in detectors:
    ax[0].plot(v, I[d] * 1e3, label=label[d])
ax[0].axvline(v_pi, color="gray", ls=":")
ax[0].set(xlabel="pad voltage (V)", ylabel="photocurrent (mA)", title="DC switching")
ax[0].legend(); ax[0].grid(alpha=0.3)
# The chip's only optical port is the laser input, so plot the light that
# reaches each detector.
st_wl0 = chip.solve(wl, drives={**bias, "pad": 0.0}, inputs=laser)
st_wl1 = chip.solve(wl, drives={**bias, "pad": v_pi}, inputs=laser)
for d, ls in zip(detectors, ("-", "--")):
    ax[1].plot(wl, 10 * np.log10(st_wl0.detector_power[d] / 1e-3 + 1e-12), ls, label=f"{label[d]}, 0 V")
    ax[1].plot(wl, 10 * np.log10(st_wl1.detector_power[d] / 1e-3 + 1e-12), ls,
               label=f"{label[d]}, V_pi")
ax[1].set(xlabel="wavelength (um)", ylabel="power at detector (dB re 1 mW)",
          title="wavelength response", ylim=(-50, 2))
ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
ax2 = ax[2].twinx()
ax2.plot(t * 1e9, tr.drive["pad"], color="lightgray", lw=0.8)
ax2.set_ylabel("pad voltage (V)")
for d in detectors:
    ax[2].plot(t * 1e9, tr.detector_current[d] * 1e3, label=label[d])
ax[2].set(xlabel="time (ns)", ylabel="photocurrent (mA)", title="10 Gb/s PRBS")
ax[2].set_zorder(ax2.get_zorder() + 1); ax[2].patch.set_visible(False)
ax[2].legend(loc="upper right"); ax[2].grid(alpha=0.3)
fig.tight_layout()
fig.savefig("12_optoelectronic_chip.png", dpi=130)
print("saved 12_optoelectronic_chip.png")
