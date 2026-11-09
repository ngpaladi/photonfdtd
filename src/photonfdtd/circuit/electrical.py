"""Electrical side of a chip: drive waveforms and the nodal (DC) solve.

Each net (a set of electrical ports joined by metal) is a node. Driven nets
are held at their source voltage; resistors (heaters, loads) join nets;
photodetectors inject their photocurrent. Nets with no path to a source float
and are held at 0 V through a negligible leak, with a warning when an
electrode or heater sits on one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = ["DC", "Square", "Sine", "Pulse", "PRBS", "PWL", "value_at", "nodal"]

#: Conductance (S) tying every net to ground, so floating nets are defined.
G_LEAK = 1e-12


@dataclass(frozen=True)
class DC:
    value: float = 0.0

    def __call__(self, t):
        return np.full_like(np.asarray(t, float), self.value)


@dataclass(frozen=True)
class Square:
    """A square wave between ``low`` and ``high`` at ``freq`` (Hz), high for
    the first ``duty`` of each period, with linear edges of ``rise`` (s)."""
    low: float = 0.0
    high: float = 1.0
    freq: float = 1e9
    duty: float = 0.5
    delay: float = 0.0
    rise: float = 0.0

    def __call__(self, t):
        T = 1.0 / self.freq
        x = np.mod(np.asarray(t, float) - self.delay, T)
        on = self.duty * T
        if self.rise > 0:
            r = self.rise
            up = np.clip(x / r, 0, 1)
            down = np.clip((x - on) / r, 0, 1)
            frac = up - down
        else:
            frac = (x < on).astype(float)
        return self.low + (self.high - self.low) * frac


@dataclass(frozen=True)
class Sine:
    offset: float = 0.0
    amplitude: float = 1.0
    freq: float = 1e9
    phase: float = 0.0

    def __call__(self, t):
        return self.offset + self.amplitude * np.sin(
            2 * np.pi * self.freq * np.asarray(t, float) + self.phase)


@dataclass(frozen=True)
class Pulse:
    """``high`` between ``start`` and ``start + width``, ``low`` otherwise."""
    low: float = 0.0
    high: float = 1.0
    start: float = 0.0
    width: float = 1e-9

    def __call__(self, t):
        t = np.asarray(t, float)
        return np.where((t >= self.start) & (t < self.start + self.width), self.high, self.low)


@dataclass(frozen=True)
class PRBS:
    """A pseudo-random bit sequence (PRBS-``order`` LFSR) at ``rate`` bits/s
    between ``low`` and ``high``."""
    low: float = 0.0
    high: float = 1.0
    rate: float = 1e9
    order: int = 7
    seed: int = 1

    def bits(self, n: int) -> np.ndarray:
        taps = {7: (7, 6), 9: (9, 5), 15: (15, 14), 23: (23, 18), 31: (31, 28)}[self.order]
        state = (self.seed or 1) & ((1 << self.order) - 1)
        out = np.empty(n, int)
        for i in range(n):
            bit = ((state >> (taps[0] - 1)) ^ (state >> (taps[1] - 1))) & 1
            state = ((state << 1) | bit) & ((1 << self.order) - 1)
            out[i] = bit
        return out

    def __call__(self, t):
        t = np.asarray(t, float)
        idx = np.floor(t * self.rate).astype(int)
        n = int(idx.max()) + 1 if idx.size else 1
        b = self.bits(max(n, 1))
        return np.where(b[np.clip(idx, 0, n - 1)] == 1, self.high, self.low)


@dataclass(frozen=True)
class PWL:
    """Piecewise linear through ``(times, values)``."""
    times: Tuple[float, ...]
    values: Tuple[float, ...]

    def __call__(self, t):
        return np.interp(np.asarray(t, float), self.times, self.values)


def value_at(drive, t) -> float:
    """A drive (number or waveform) at time ``t``."""
    if callable(drive):
        return float(np.asarray(drive(np.asarray(t, float))))
    return float(drive)


def nodal(n: int, fixed: Dict[int, float], resistors: Sequence[Tuple[int, int, float]],
          currents: Dict[int, float]):
    """Solve for the voltages of ``n`` nets.

    ``fixed``: ``{net: volts}`` from sources. ``resistors``:
    ``(net_a, net_b, ohms)``. ``currents``: ``{net: amps injected}``.
    Returns ``(V, I_source)``, with ``I_source[net]`` the current each source
    delivers into its net.
    """
    G = np.zeros((n, n))
    for a, b, R in resistors:
        g = 1.0 / R
        G[a, a] += g; G[b, b] += g
        G[a, b] -= g; G[b, a] -= g
    G += G_LEAK * np.eye(n)
    I = np.zeros(n)
    for k, i in currents.items():
        I[k] += i
    V = np.zeros(n)
    free = [k for k in range(n) if k not in fixed]
    for k, v in fixed.items():
        V[k] = v
    if free:
        fx = list(fixed)
        rhs = I[free] - (G[np.ix_(free, fx)] @ V[fx] if fx else 0.0)
        V[free] = np.linalg.solve(G[np.ix_(free, free)], rhs)
    # Source current = what flows out of its net through the network,
    # minus what is injected there.
    I_src = {k: float(G[k] @ V - I[k]) for k in fixed}
    return V, I_src


def lowpass_factor(dt: float, tau: Optional[float]) -> float:
    """Update weight of a first-order low-pass over a step ``dt`` (1:
    instantaneous)."""
    if not tau:
        return 1.0
    return float(1.0 - np.exp(-dt / tau))


def tau_of(bandwidth: Optional[float]) -> Optional[float]:
    return None if not bandwidth else 1.0 / (2 * np.pi * bandwidth)
