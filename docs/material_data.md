# Reference material dispersion data

These are the actual numbers behind the material library — the dispersion models
and their coefficients — with each one traced back to the paper it came from, so
you can check my work rather than take it on faith. Every Sellmeier equation is
written as **n² − 1 = Σ Bᵢ λ²/(λ² − Cᵢ)** with **λ in micrometres** and
**Cᵢ in µm²**; the metal models give the complex permittivity ε(ω) with photon
energy ℏω in eV.

## Fused silica (SiO₂) — Malitson 1965

```
n² − 1 = 0.6961663 λ²/(λ² − 0.0684043²)
       + 0.4079426 λ²/(λ² − 0.1162414²)
       + 0.8974794 λ²/(λ² − 9.896161²)
```
Validity 0.21–6.7 µm. Sanity: n(1550 nm) = 1.44402.
I. H. Malitson, *J. Opt. Soc. Am.* **55**(10), 1205–1209 (1965).
DOI: 10.1364/JOSA.55.001205

## Crystalline silicon (Si) — Salzberg & Villa 1957

```
n² − 1 = 10.6684293 λ²/(λ² − 0.301516485²)
       + 0.0030434748 λ²/(λ² − 1.13475115²)
       + 1.54133408  λ²/(λ² − 1104²)
```
Validity 1.36–11.04 µm. Sanity: n(1550 nm) = 3.4777.
C. D. Salzberg & J. J. Villa, *J. Opt. Soc. Am.* **47**(3), 244–246 (1957).
DOI: 10.1364/JOSA.47.000244. (Coefficients as tabulated by refractiveindex.info;
for critically evaluated primary data with T-dependence see H. H. Li,
*J. Phys. Chem. Ref. Data* **9**(3), 561–658 (1980), DOI: 10.1063/1.555624.)

## Silicon nitride (Si₃N₄, stoichiometric LPCVD) — Luke et al. 2015

```
n² − 1 = 3.0249 λ²/(λ² − 0.1353406²)
       + 40314  λ²/(λ² − 1239.842²)
```
Validity 0.310–5.504 µm. Sanity: n(1550 nm) = 1.9963.
K. Luke, Y. Okawachi, M. R. E. Lamont, A. L. Gaeta, M. Lipson,
*Opt. Lett.* **40**(21), 4823–4826 (2015). DOI: 10.1364/OL.40.004823

## Lithium niobate (LiNbO₃, congruent undoped) — Zelmon, Small & Jundt 1997

Ordinary:
```
n_o² − 1 = 2.6734 λ²/(λ² − 0.01764)
         + 1.2290 λ²/(λ² − 0.05914)
         + 12.614 λ²/(λ² − 474.60)
```
Extraordinary:
```
n_e² − 1 = 2.9804 λ²/(λ² − 0.02047)
         + 0.5981 λ²/(λ² − 0.0666)
         + 8.9543 λ²/(λ² − 416.08)
```
Validity 0.4–5.0 µm. Sanity: n_o(1550 nm) = 2.2111, n_e(1550 nm) = 2.1376.
Use the **undoped congruent** set (not the paper's 5 mol.% MgO-doped set).
D. E. Zelmon, D. L. Small, D. Jundt, *J. Opt. Soc. Am. B* **14**(12),
3319–3322 (1997). DOI: 10.1364/JOSAB.14.003319

## Gold (Au) & Silver (Ag) — Rakić et al. 1998 Lorentz–Drude

```
ε(ω) = 1 − f₀ ωₚ²/(ω² + i ω Γ₀)
         + Σ_{j=1..5} f_j ωₚ²/(ω_j² − ω² − i ω Γ_j)      (ℏω, ωₚ, ω_j, Γ_j in eV)
```

Gold: ωₚ = 9.03 eV.

| j | f_j | Γ_j (eV) | ω_j (eV) |
|---|-----|----------|----------|
| 0 | 0.760 | 0.053 | 0.000 |
| 1 | 0.024 | 0.241 | 0.415 |
| 2 | 0.010 | 0.345 | 0.830 |
| 3 | 0.071 | 0.870 | 2.969 |
| 4 | 0.601 | 2.494 | 4.304 |
| 5 | 4.384 | 2.214 | 13.32 |

Sanity (633 nm): ε = −9.81 + 1.96i → n = 0.31, k = 3.15.

Silver: ωₚ = 9.01 eV.

| j | f_j | Γ_j (eV) | ω_j (eV) |
|---|-----|----------|----------|
| 0 | 0.845 | 0.048 | 0.000 |
| 1 | 0.065 | 3.886 | 0.816 |
| 2 | 0.124 | 0.452 | 4.481 |
| 3 | 0.011 | 0.065 | 8.185 |
| 4 | 0.840 | 0.916 | 9.083 |
| 5 | 5.646 | 2.419 | 20.29 |

Sanity (633 nm): ε = −14.49 + 1.10i → n = 0.14, k = 3.81.

eV → angular frequency: multiply by e/ℏ ≈ 1.519×10¹⁵ rad·s⁻¹·eV⁻¹. Convention is
e^(−iωt) (loss ⇒ Im ε > 0); flip imaginary signs for e^(+iωt).
A. D. Rakić, A. B. Djurišić, J. M. Elazar, M. L. Majewski, *Appl. Opt.*
**37**(22), 5271–5283 (1998). DOI: 10.1364/AO.37.005271

## Electro-optic and thermal properties (photonfdtd.active)

| Material | Quantity | Value | Source |
|---|---|---|---|
| LiNbO₃ | r13, r22, r33, r51 (clamped, > ~10 MHz) | 8.6, 3.4, 31, 28 pm/V | Weis & Gaylord, *Appl. Phys. A* **37**, 191 (1985), via the Crystal Technology datasheet and PM Optics tables |
| LiNbO₃ | r13, r22, r33, r51 (unclamped, DC) | 10, 6.8, 32, 32 pm/V | as above (the datasheet gives r22 = 7, r33 = r51 = 33) |
| LiNbO₃ | ε₁₁, ε₃₃ clamped / unclamped | 44, 27.9 / 85, 28.7 | Weis & Gaylord 1985 |
| LiNbO₃ | dn_e/dT, dn_o/dT (room temp., near IR) | 3.34×10⁻⁵, ≈0 /K | arXiv:1806.04755 (others quote ≈4×10⁻⁵, e.g. arXiv:1804.03621) |
| LiNbO₃ | thermal conductivity | 4.6 W/(m K) | typical bulk value |
| LiTaO₃ | n_o, n_e (1.55 µm); r13, r33 | 2.119, 2.123; 8.4, 30.5 pm/V | approximate literature values, not checked against a primary source |
| BaTiO₃ | n_o, n_e; r13, r33, r42 (bulk, clamped) | 2.286, 2.263; 8, 28, 1300 pm/V | Zgonik et al., *PRB* **50**, 5941 (1994); thin films differ |
| Si | dn/dT (1.55 µm), k, ε_dc | 1.86×10⁻⁴ /K, 148 W/(m K), 11.7 | Komma et al., *APL* **101**, 041905 (2012) |
| SiN (LPCVD) | dn/dT, k | 2.45×10⁻⁵ /K, ~20 W/(m K) (films 10–30) | Arbabi & Goddard, *Opt. Lett.* **38**, 3878 (2013) |
| SiO₂ | dn/dT, k, ε_dc | 0.95×10⁻⁵ /K, 1.38 W/(m K), 3.9 | typical values |
| Au, Al, TiN, NiCr, Ti, W | σ (S/m), k (W/(m K)) | see `metal_active` | handbook values; thin films are usually lower |

Free-carrier dispersion in silicon, Soref & Bennett, *IEEE JQE* **23**, 123
(1987), as tabulated by Reed et al., *Nat. Photon.* **4**, 518 (2010), with
densities in cm⁻³ and absorption in cm⁻¹:

- 1550 nm: Δn = −(8.8×10⁻²² ΔN + 8.5×10⁻¹⁸ ΔP^0.8), Δα = 8.5×10⁻¹⁸ ΔN + 6.0×10⁻¹⁸ ΔP
- 1310 nm: Δn = −(6.2×10⁻²² ΔN + 6.0×10⁻¹⁸ ΔP^0.8), Δα = 6.0×10⁻¹⁸ ΔN + 4.0×10⁻¹⁸ ΔP

The electro-optic values come from secondary tabulations of the cited papers
(vendor datasheets and review tables), which disagree in the second digit for
r22 and r51; thin-film values depend on the film. Treat the library as a
starting point and override the media for a specific process.

## Provenance notes

SiO₂ (Malitson) is canonical. Si₃N₄, LiNbO₃, and the Au/Ag Rakić tables pass
independent sanity checks and are digit-consistent across multiple published
reimplementations, but the coefficient digits were transcribed via
refractiveindex.info / secondary reproductions rather than the original PDFs;
verify against the DOIs if bit-exact provenance is required. The silicon
Sellmeier fit is the standard refractiveindex.info transcription of
Salzberg & Villa, numerically correct but not confirmed verbatim against the
1957 paper text.
