"""2.5-D effective-index method (EIM) for planar photonic circuits.

A photonic-integrated-circuit layer is a thin core (Si / SiN / LiNbO3) etched
into an in-plane pattern on a substrate under a cladding. A full 3-D FDTD of a
whole device is enormous; the EIM collapses the vertical (z) dimension into an
*effective index* so the device is simulated as a 2-D problem in the chip plane
(x, y) - two to three orders of magnitude fewer cells and timesteps, and the
resulting 2-D :class:`~photonfdtd.Simulation` runs unchanged on every backend
(NumPy / JAX / Rust / Rust-CUDA).

The reduction is the textbook two-step EIM (e.g. Chuang, *Physics of
Photonic Devices*, or Okamoto, *Fundamentals of Optical Waveguides*):

1. Solve the 1-D vertical slab mode of the layer stack **where the core is
   present** -> ``n_core``, and of the stack **where the core is etched away**
   -> ``n_background``. These are exact 1-D guided-mode indices (validated in
   the tests against the analytic symmetric-slab dispersion relation).
2. Build a 2-D permittivity map in the chip plane whose waveguide pattern has
   index ``n_core`` and whose surround has ``n_background``, and run a 2-D FDTD
   (or mode solve) on it. :func:`effective_index_media` returns the two
   :class:`~photonfdtd.Medium` objects to stamp into ordinary ``Box`` /
   ``PolySlab`` geometry.

The EIM is an approximation: it assumes the vertical mode profile is separable
from the in-plane confinement, which is excellent for weakly-etched, weakly
in-plane-confining routing (waveguides, bends, directional couplers, MZIs) and
degrades for strongly 3-D features (deep sub-wavelength gaps, out-of-plane
grating diffraction). It is the standard way to get layout-scale answers
quickly, and it is exact in the slab limit.

Polarization convention: ``polarization="TE"`` selects the slab mode whose
electric field lies in the layer plane (dominant ``E_y`` for a stack grown
along z) - the quasi-TE device mode. ``"TM"`` selects the field with a dominant
out-of-plane ``E_z``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .constants import C_0
from .materials import Medium

IndexLike = Union[float, Medium]


def _index_value(x: IndexLike, wavelength: float) -> float:
    """Refractive index of a float / Medium / DispersiveMedium at a wavelength."""
    if isinstance(x, (int, float)):
        return float(x)
    if getattr(x, "is_dispersive", False):
        return float(np.real(x.index(C_0 / wavelength)))
    return float(np.sqrt(x.eps_r))


@dataclass
class Layer:
    """One layer of a vertical stack: a thickness (m) and a material.

    ``index`` may be a refractive index (float), a :class:`~photonfdtd.Medium`,
    or a :class:`~photonfdtd.DispersiveMedium` (evaluated at the solve
    wavelength).
    """
    thickness: float
    index: IndexLike


@dataclass
class SlabModes:
    """Result of a 1-D slab solve (see :func:`slab_modes`)."""
    wavelength: float
    polarization: str
    n_eff: np.ndarray            #: guided-mode effective indices, descending
    z: np.ndarray                #: vertical coordinate (m), 0 at stack bottom
    profiles: np.ndarray         #: (num_modes, nz) field profile, peak-normalised
    eps: np.ndarray              #: eps(z) sampled on ``z``


def _build_eps_z(
    layers: Sequence[Layer],
    substrate_index: float,
    cladding_index: float,
    wavelength: float,
    dz: float,
    pad: float,
    harmonic: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample eps(z) for the stack, padding semi-infinite substrate/cladding.

    The guided mode decays exponentially into substrate and cladding, so a pad
    of a few wavelengths with a Dirichlet wall is indistinguishable from a
    semi-infinite medium for a confined mode. Cells straddling a layer
    interface get the length-weighted subpixel average: arithmetic in eps for a
    tangential field (TE), harmonic (``harmonic=True``, i.e. arithmetic in
    1/eps) for the normal field (TM).
    """
    thicknesses = [max(float(l.thickness), 0.0) for l in layers]
    indices = [_index_value(l.index, wavelength) for l in layers]
    core_total = sum(thicknesses)
    z_lo = -pad
    z_hi = core_total + pad
    nz = max(int(round((z_hi - z_lo) / dz)), 16)
    edges_z = z_lo + np.arange(nz + 1) * (z_hi - z_lo) / nz   # cell boundaries
    z = 0.5 * (edges_z[:-1] + edges_z[1:])                    # cell centres

    # Piecewise-constant eps(z): substrate below 0, the layers, cladding above.
    # Each cell's eps is the length-weighted average of the segments it spans
    # (1-D subpixel smoothing) - this removes the interface-position staircase
    # jitter, so the default resolution converges smoothly.
    seg_edges = [(-np.inf, 0.0, substrate_index ** 2)]
    z_run = 0.0
    for t, n in zip(thicknesses, indices):
        seg_edges.append((z_run, z_run + t, n ** 2))
        z_run += t
    seg_edges.append((core_total, np.inf, cladding_index ** 2))

    eps = np.empty(nz)
    for i in range(nz):
        a, b = edges_z[i], edges_z[i + 1]
        acc = 0.0
        for lo, hi, e in seg_edges:
            ov = min(b, hi) - max(a, lo)
            if ov > 0:
                acc += ov * (1.0 / e if harmonic else e)
        mean = acc / (b - a)
        eps[i] = 1.0 / mean if harmonic else mean
    return z, eps


def slab_modes(
    layers: Sequence[Layer],
    wavelength: float,
    *,
    polarization: str = "TE",
    num_modes: int = 1,
    substrate_index: IndexLike = 1.0,
    cladding_index: IndexLike = 1.0,
    dz: Optional[float] = None,
    pad: Optional[float] = None,
) -> SlabModes:
    """Guided modes of a 1-D vertical layer stack (finite-difference).

    Solves the scalar slab eigenproblem for the effective index of a stack
    invariant in the plane. TE (``E_y``) obeys ``d2E/dz2 + (k0^2 eps -
    beta^2) E = 0``; TM (``H_y``) obeys the ``eps d/dz(1/eps dH/dz)`` form, so
    the discontinuous-``eps`` interface conditions are built in. Returns the
    modes with the largest effective index (most confined) first.

    Parameters
    ----------
    layers : sequence of :class:`Layer`
        The stack, ordered bottom (substrate side) to top (cladding side).
    wavelength : float
        Free-space wavelength (m).
    polarization : {"TE", "TM"}
        Slab polarization (see module docstring).
    num_modes : int
        Number of guided modes to return.
    substrate_index, cladding_index : float or Medium
        Semi-infinite media below and above the stack.
    dz : float, optional
        Vertical step (default ``wavelength / 60 / max_index``).
    pad : float, optional
        Substrate/cladding pad thickness (default ``0.8 * wavelength``).
    """
    pol = polarization.upper()
    if pol not in ("TE", "TM"):
        raise ValueError("polarization must be 'TE' or 'TM'")
    n_sub = _index_value(substrate_index, wavelength)
    n_clad = _index_value(cladding_index, wavelength)
    n_hi = max([_index_value(l.index, wavelength) for l in layers]
               + [n_sub, n_clad])
    if dz is None:
        dz = wavelength / 60.0 / max(n_hi, 1.0)
    if pad is None:
        pad = 0.8 * wavelength

    # Arithmetic subpixel eps for both polarisations: the TM operator already
    # staggers 1/eps at cell faces, so a second (harmonic) cell average
    # double-counts the interface correction. TE converges to ~1e-4 at the
    # default resolution; TM to ~5e-3 (finer dz if you need TM to 1e-3).
    z, eps = _build_eps_z(layers, n_sub, n_clad, wavelength, dz, pad)
    nz = z.size
    dz_eff = z[1] - z[0]
    k0 = 2.0 * np.pi / wavelength
    inv_dz2 = 1.0 / dz_eff ** 2

    if pol == "TE":
        # A = D2 + k0^2 diag(eps); eigenvalue beta^2, eigenvector E_y.
        main = -2.0 * inv_dz2 + k0 ** 2 * eps
        off = inv_dz2 * np.ones(nz - 1)
        w, v = np.linalg.eigh(
            np.diag(main) + np.diag(off, 1) + np.diag(off, -1))
    else:
        # TM: eps * d/dz( (1/eps) dH/dz ) + k0^2 eps H = beta^2 H, with 1/eps at
        # half-cell faces. The operator is symmetrisable; solve the dense
        # generalized-symmetric form directly (small nz).
        eps_face = 0.5 * (eps[:-1] + eps[1:])          # eps at i+1/2
        inv_face = 1.0 / eps_face
        A = np.zeros((nz, nz))
        for i in range(nz):
            fac = eps[i] * inv_dz2
            lo = inv_face[i - 1] if i > 0 else 0.0
            hi = inv_face[i] if i < nz - 1 else 0.0
            A[i, i] = -fac * (lo + hi) + k0 ** 2 * eps[i]
            if i > 0:
                A[i, i - 1] = fac * lo
            if i < nz - 1:
                A[i, i + 1] = fac * hi
        w, v = np.linalg.eig(A)
        w = np.real(w)
        v = np.real(v)

    beta2 = w
    order = np.argsort(-beta2)
    beta2 = beta2[order]
    v = v[:, order]
    n_eff_all = np.sqrt(np.clip(beta2, 0.0, None)) / k0

    # Keep genuinely guided modes: n_eff between the higher cladding index and
    # the peak stack index.
    n_floor = max(n_sub, n_clad)
    guided = np.flatnonzero((n_eff_all > n_floor + 1e-6) & (n_eff_all <= n_hi + 1e-6))
    if guided.size == 0:                    # fall back to the top eigenpairs
        guided = np.arange(min(num_modes, n_eff_all.size))
    guided = guided[:num_modes]

    n_eff = n_eff_all[guided]
    prof = v[:, guided].T.copy()
    for i in range(prof.shape[0]):
        mx = np.abs(prof[i]).max()
        if mx > 0:
            prof[i] /= prof[i][np.argmax(np.abs(prof[i]))]  # real, peak +1
    return SlabModes(wavelength=float(wavelength), polarization=pol,
                     n_eff=n_eff, z=z, profiles=prof, eps=eps)


def effective_index(
    layers: Sequence[Layer],
    wavelength: float,
    *,
    polarization: str = "TE",
    **kwargs,
) -> float:
    """Fundamental-mode effective index of a vertical stack (see
    :func:`slab_modes`). Convenience wrapper returning a single float."""
    modes = slab_modes(layers, wavelength, polarization=polarization,
                       num_modes=1, **kwargs)
    if modes.n_eff.size == 0:
        raise ValueError(
            "no guided slab mode found for this stack at this wavelength - the "
            "core may be too thin or the index contrast too low to guide.")
    return float(modes.n_eff[0])


@dataclass
class EffectiveIndex2D:
    """2.5-D reduction of a patterned layer: the two effective-index media.

    Build with :meth:`from_stacks` from the *core* stack (where the waveguide
    layer is present) and the *etched* stack (where it has been removed). Stamp
    :attr:`medium_core` into the in-plane waveguide geometry and
    :attr:`medium_background` into the surround, then run an ordinary 2-D
    :class:`~photonfdtd.Simulation` in the chip plane.
    """
    wavelength: float
    polarization: str
    n_core: float
    n_background: float

    @classmethod
    def from_stacks(
        cls,
        core_layers: Sequence[Layer],
        etched_layers: Sequence[Layer],
        wavelength: float,
        *,
        polarization: str = "TE",
        substrate_index: IndexLike = 1.0,
        cladding_index: IndexLike = 1.0,
        **kwargs,
    ) -> "EffectiveIndex2D":
        n_core = effective_index(
            core_layers, wavelength, polarization=polarization,
            substrate_index=substrate_index, cladding_index=cladding_index,
            **kwargs)
        # The etched region may not guide; fall back to its highest index
        # (the effective slab index of a leaky/unguided region is its
        # substrate/cladding continuum edge), which is the right background for
        # the in-plane index-contrast problem.
        try:
            n_bg = effective_index(
                etched_layers, wavelength, polarization=polarization,
                substrate_index=substrate_index, cladding_index=cladding_index,
                **kwargs)
        except ValueError:
            n_bg = max([_index_value(l.index, wavelength) for l in etched_layers]
                       + [_index_value(substrate_index, wavelength),
                          _index_value(cladding_index, wavelength)])
        return cls(wavelength=float(wavelength), polarization=polarization.upper(),
                   n_core=float(n_core), n_background=float(n_bg))

    @property
    def index_contrast(self) -> float:
        return self.n_core - self.n_background

    @property
    def medium_core(self) -> Medium:
        return Medium.from_index(self.n_core, name="EIM-core")

    @property
    def medium_background(self) -> Medium:
        return Medium.from_index(self.n_background, name="EIM-background")

    def effective_index_media(self) -> Tuple[Medium, Medium]:
        """``(core, background)`` :class:`Medium` pair for the 2-D layout."""
        return self.medium_core, self.medium_background
