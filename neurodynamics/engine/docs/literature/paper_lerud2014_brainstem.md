# Lerud, Almonte, Kim, Large (2014) — Mode-locking neurodynamics predict human auditory brainstem responses to musical intervals

**Reference:** Hearing Research 308 (2014), 41–49
**DOI:** 10.1016/j.heares.2013.09.010
**Special issue:** *Music: A window into the hearing brain*

---

## Why this matters for factory-dame

This is **the first multi-layer GrFNN paper** that models the
auditory pathway: cochlea → cochlear nucleus (CN) → inferior
colliculus / lateral lemniscus (IC/LL). It gives concrete
parameter values for each layer and shows the mode-locking-driven
predictions match human FFR (frequency-following response) data
with 68% variance explained, R²=0.77 (consonant, nonmusicians).

This is the **canonical citation for our Phase 3 layered
architecture**, especially the cochlea + brainstem layers.

## Three-layer architecture (Section 3 / Appendix B)

```
acoustic input
    │
    ↓
[middle ear filter MEF]  →  x(t) (linearly forced cochlea input)
    │
    ↓
COCHLEA (gammatone-like + active OC dynamics)
    │  z_bm  +  z_oc
    ↓
COCHLEAR NUCLEUS (CN)  — 397 oscillators, 4 octaves, 99/octave
    │
    ↓
INFERIOR COLLICULUS / LATERAL LEMNISCUS (IC/LL) — 397 osc, same layout
    │
    ↓
weighted average:  cochlea·25% + CN·50% + IC/LL·25%
    │  (3rd-order Butterworth low-pass 450 Hz: skull/meninges/scalp)
    ↓
predicted FFR
```

## Generic weakly-coupled form (Eq. 1)

```
dx_i/dt = f_i(x_i, y_i, λ) + ε p_i(x_1, y_1, …, x_n, y_n, s(t), λ, ε)
dy_i/dt = g_i(x_i, y_i, λ) + ε q_i(x_1, y_1, …, x_n, y_n, s(t), λ, ε)
```

Excitatory `x_i` and inhibitory `y_i` populations per oscillator.
`ε` small = weak interaction.

## Canonical oscillator equation (Eq. 2)

The brainstem-layer canonical form, derived via normal-form theory
near a Hopf bifurcation:

```
τ_i · dz_i/dt = z_i (α + i·2π + β₁|z_i|² + εβ₂|z_i|⁴ + …)
              + c · P(ε, s(t)) · A(ε, z_i)
```

This is the **same canonical equation** as Large 2010 / Large 2015,
applied here with explicit layer-specific parameters.

## Resonant transformations (Appendix A, Eq. A.2)

```
P(ε, s(t)) = (s + √ε s² + ε s³ + …)(1 + √ε s + ε s² + …)
           = s / (1 - √ε s)

A(ε, z_i) = 1 + √ε z̄_i + ε z̄_i² + … = 1 / (1 - √ε z̄_i)
```

The full resonant-monomial expansion (Eq. A.1) generates resonance
"at harmonics, subharmonics, integer ratios, combination
frequencies." `ε` controls how high-order the captured ratios go.

## Cochlear model (Appendix B, Eq. B.1)

Two-state cochlea per place:
- `z_bm` — basilar membrane (passive)
- `z_oc` — organ of Corti (active OC nonlinearity)

```
τ · dz_bm/dt = z_bm (α_bm + i·2π) + x(t)
τ · dz_oc/dt = z_oc (α_oc + i·2π + β₁|z_oc|² + εβ₂|z_oc|⁴ + …) + z_bm
```

with parameters from macaque tuning-curve fits (Joris et al. 2011):

| Parameter | Value |
|---|---|
| α_bm | -1 (heavy damping, BM is passive) |
| α_oc | 0 (critical, sharp tuning) |
| β₁ | **-10000** (strong nonlinearity) |
| β₂ | -1 |
| ε | **0.0025** |

**The cochlear active-tuning β₁ is -10000, not -100.** This is
much stronger than the Kim & Large 2015 / patent value of β₁=-100.
It reflects fitting against macaque cochlear tuning data, not just
"sharp filter" intuition. Worth using -10000 (or near it) at the
cochlear stage, given this is published, fitted-to-data, and now in
the public domain via the expired US 7,376,562.

## Brainstem layer parameters (Section 3)

For both CN and IC/LL layers:

| Parameter | Value |
|---|---|
| α | **0** (critical Hopf — phase-locks to anything) |
| β₁ | **0** |
| β₂ | **-1** (saturating nonlinearity from quartic term only) |
| ε | varied per fit (0.07–0.48) |

**β₁ = 0 is notable.** The cubic term is zero; the quartic ε β₂|z|⁴
term plus the resonant expansion does ALL the nonlinear work. This
contrasts with Kim & Large 2015 / patent which uses β₁ = -10
(brainstem) and β₁ = -100 (cochlea). Different formulations, same
philosophy: critical-Hopf-like dynamics with saturating nonlinearity.

## Network layout

- 99 oscillators / octave (denser than the 60/octave of Large 2010
  rhythm or the 36/octave of factory-dame's pitch bank)
- 4 octaves spanning **64 Hz to 1024 Hz**, center frequency 256 Hz
- 397 oscillators per brainstem layer
- Logarithmic spacing
- Inputs: gammatone-filtered, half-Hilbert-rectified ear-filtered
  acoustic signal

## Layer-specific weighting in FFR fit

When combining layer outputs to predict the scalp-recorded FFR
(Section 3 / Methods):

| Source | Weight in FFR |
|---|---|
| Cochlear microphonic | 25% |
| Cochlear nuclei | 50% |
| Lateral lemniscus / IC | 25% |

This is from cat lesion data (Gardi et al. 1979). When using
factory-dame's three-layer architecture, this weighting could
inform how we combine layer outputs for downstream features.

## Mode-locking results (Fig. 4)

Predictions matched FFR at every peak — `f₁`, `f₂`, `2f₁`, `3f₁`,
`f₁+f₂`, `2f₁-f₂`, `2f₂-f₁`, etc. — with parsimonious model and
**only ε varied per fit.**

Variance explained: 77% (consonant, nonmusicians) to 52% (dissonant,
musicians). Larger ε in musicians → higher-order mode-locks
contribute more → "experience-based differences ... synaptic plasticity
that links learned representations to the neural encoding of acoustic
features."

This is the *experimental* validation that Hebbian-W modulating ε
(or coupling strengths) is what differentiates trained-vs-untrained
listening — directly motivating Phase 5.

## Polarity-reversal / odd-vs-even-order separation

Average of two opposite-polarity simulations cancels odd-order
nonlinearities (cochlear distortions), leaving only even-order
(quadratic, neural). This is how they separate cochlear from
brainstem contributions in the FFR data.

We don't currently use polarity reversal in factory-dame — but if
we ever want to validate against FFR data, we'd need to.

## Implications for our implementation

### Phase 3: layered architecture

This is the **canonical reference** for our cochlea+brainstem layers.
Concrete parameters:

| Layer | α | β₁ | β₂ | ε | Notes |
|---|---|---|---|---|---|
| Cochlea (BM) | -1 | 0 | 0 | — | Passive, gammatone-like |
| Cochlea (OC) | 0 | -10000 | -1 | 0.0025 | Active, sharp tuning |
| CN (brainstem 1) | 0 | 0 | -1 | ~0.4 | Critical, saturating |
| IC/LL (brainstem 2) | 0 | 0 | -1 | ~0.4 | Critical, saturating |

Note: this paper's β₁=0 brainstem differs from US 8,930,292's
β₁=-10/-1 brainstem/cortex. Both are documented in different
publications; pick what fits the application.

### Density: 99/octave

Higher density than factory-dame's 36/octave. For the cochlea/
brainstem layers, denser is closer to physiology and gives smoother
FFR-style predictions. For voice-extraction (cortex layer), lower
density is fine.

### ε variation

ε is the "experience" parameter — larger ε → higher-order resonances
captured → matches musicians' brainstem responses better. This
hints at *how Hebbian learning modulates listening*: by effectively
increasing ε in trained listeners. Phase 5 implication:
expansion-series order should grow with learning.

### Cochlear pre-filter

Middle-ear filter (MEF) + Hilbert + half-rectification before
oscillator drive. Our current input is much simpler. For
authenticity at the cochlear stage we'd want this preprocessing.
Probably not needed before Phase 3.

## Relation to other papers

- **Large 2010 canonical**: this paper *uses* the canonical equation
  on a multi-layer pathway.
- **Kim & Large 2015 signal processing**: the four-regime analysis
  (Critical/Supercritical Hopf and DLC) is the bifurcation-theory
  basis; this paper uses the **Critical Hopf** regime (α=0) for the
  brainstem layers.
- **Large 2015 beat perception**: same canonical form, different
  domain (rhythm 0.5–8 Hz vs pitch 64–1024 Hz here).
- **Patent US 8,930,292**: layered architecture with different
  β₁ ladder. *Not* directly compatible parameter-wise with this
  paper, but architecturally similar (cochlea + brainstem +
  higher).
- **Phase 5 (Hebbian)**: this paper *predicts* that Hebbian-driven
  parameter changes (ε in particular) produce the musician-vs-
  nonmusician differences. Direct experimental ground for Phase 5.
