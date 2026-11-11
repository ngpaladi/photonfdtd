"""gdsfactory cells the circuit models know: a waveguide photodetector."""
from __future__ import annotations

from typing import Optional, Tuple

import gdsfactory as gf

__all__ = ["photodetector"]


@gf.cell
def photodetector(length: float = 20.0, width: float = 1.2, absorber_width: float = 4.0,
                  pad_width: float = 10.0, gap: float = 2.0, responsivity: float = 0.8,
                  dark_current: float = 1e-9, bandwidth: Optional[float] = 40e9,
                  waveguide_layer: Tuple[int, int] = (1, 0),
                  absorber_layer: Tuple[int, int] = (40, 0),
                  metal_layer: Tuple[int, int] = (49, 0)) -> gf.Component:
    """A waveguide photodetector: light enters at ``o1`` into an absorber
    (``absorber_layer``, the generic PDK's GE) of ``length`` um between an
    anode contact (below, ports ``anode_w``/``anode_e``) and a cathode contact
    (above). ``responsivity`` (A/W), ``dark_current`` (A) and ``bandwidth``
    (Hz) go to the circuit model."""
    c = gf.Component()
    c.add_polygon([(0, -width / 2), (length, -width / 2), (length, width / 2), (0, width / 2)],
                  layer=waveguide_layer)
    a = absorber_width / 2
    c.add_polygon([(0, -a), (length, -a), (length, a), (0, a)], layer=absorber_layer)
    c.add_port("o1", center=(0, 0), width=width, orientation=180, layer=waveguide_layer,
               port_type="optical")
    for name, sgn in (("anode", -1), ("cathode", +1)):
        y0 = sgn * (a + gap)
        y1 = y0 + sgn * pad_width
        lo, hi = min(y0, y1), max(y0, y1)
        c.add_polygon([(0, lo), (length, lo), (length, hi), (0, hi)], layer=metal_layer)
        yc = 0.5 * (lo + hi)
        c.add_port(f"{name}_w", center=(0, yc), width=pad_width, orientation=180,
                   layer=metal_layer, port_type="electrical")
        c.add_port(f"{name}_e", center=(length, yc), width=pad_width, orientation=0,
                   layer=metal_layer, port_type="electrical")
    c.info["photodetector"] = dict(responsivity=responsivity, dark_current=dark_current,
                                   bandwidth=bandwidth, anode=["anode_w", "anode_e"],
                                   cathode=["cathode_w", "cathode_e"])
    return c
