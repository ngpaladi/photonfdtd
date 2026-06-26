"""Material definitions.

v0.1 supports isotropic, non-dispersive dielectrics. Each Medium is reduced to
a single relative permittivity that the grid rasteriser stamps into a per-cell
epsilon_r array. Magnetic permeability is fixed at mu_r = 1.

Dispersive media (Lorentz, Drude, Debye) are planned for a future release via
the auxiliary-differential-equation method; the API is left open so a Lorentz
class can slot in without breaking callers.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Medium:
    """Isotropic, non-dispersive dielectric.

    Parameters
    ----------
    permittivity : float
        Relative permittivity eps_r at the simulation's centre frequency.
        Pass eps_r = n^2 for a refractive index n.
    name : str, optional
        Identifier, used only for logging.
    """
    permittivity: float = 1.0
    name: str = ""

    @property
    def eps_r(self) -> float:
        return float(self.permittivity)

    @classmethod
    def from_index(cls, n: float, name: str = "") -> "Medium":
        return cls(permittivity=float(n) ** 2, name=name)


VACUUM = Medium(permittivity=1.0, name="vacuum")
