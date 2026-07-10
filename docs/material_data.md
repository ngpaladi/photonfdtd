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

## Provenance notes

SiO₂ (Malitson) is canonical. Si₃N₄, LiNbO₃, and the Au/Ag Rakić tables pass
independent sanity checks and are digit-consistent across multiple published
reimplementations, but the coefficient digits were transcribed via
refractiveindex.info / secondary reproductions rather than the original PDFs;
verify against the DOIs if bit-exact provenance is required. The silicon
Sellmeier fit is the standard refractiveindex.info transcription of
Salzberg & Villa, numerically correct but not confirmed verbatim against the
1957 paper text.
