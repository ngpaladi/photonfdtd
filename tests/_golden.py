"""Golden fingerprints for regression-testing the JAX stepper.

The reference values in the tests were recorded from the retired NumPy
stepping core (photonfdtd v0.12.0, ``backend="numpy"``), which the JAX engine
matched to ~1e-9 relative before that core was removed. A fingerprint is the
L2 norm plus a position-weighted checksum of each (real / imaginary) part, so
it catches amplitude, sign, and placement errors without storing whole arrays.
"""
import numpy as np


def fingerprint(a):
    a = np.asarray(a)
    parts = (a.real, a.imag) if np.iscomplexobj(a) else (a,)
    out = []
    for p in parts:
        p = p.astype(np.float64).ravel()
        w = np.cos(0.7 * np.arange(p.size))
        out += [float(np.sqrt(np.dot(p, p))), float(np.dot(p, w))]
    return out


def assert_golden(a, ref, rtol, label=""):
    """``fingerprint(a)`` matches ``ref`` to ``rtol`` of the reference norm."""
    fp = fingerprint(a)
    assert len(fp) == len(ref), label
    scale = max(abs(v) for v in ref[0::2]) + 1e-300
    err = max(abs(x - y) for x, y in zip(fp, ref)) / scale
    assert err < rtol, f"{label}: golden mismatch rel={err:.2e}"
