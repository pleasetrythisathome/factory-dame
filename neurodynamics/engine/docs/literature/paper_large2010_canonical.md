# Large, Almonte, Velasco (2010) — A canonical model for gradient frequency neural networks

**Reference:** Physica D 239 (2010), 905-911
**DOI:** 10.1016/j.physd.2009.11.015

[PDF on Music Dynamics Lab site](https://musicdynamicslab.uconn.edu/wp-content/uploads/sites/433/2016/03/LargeAlmonteVelasco2010PubsAHEdits.pdf)

---

## Why this matters for factory-dame

This is **the foundational paper** for the canonical GrFNN form we're
implementing. Eq. 15 is the canonical equation; Eq. 19/20 demonstrate
it on a network. Read this before changing the integrator.

## The canonical equation (Eq. 15)

For a single neural oscillator under external input `x`:

```
ż = z(a + b|z|² + dε|z|⁴/(1 - ε|z|²)) + (x / (1 - √ε x)) · (1 / (1 - √ε z̄))
```

where:
- `a = α + iω` — linear damping/instability + natural rotation
- `b = β₁ + iδ₁` — cubic nonlinearity (amplitude squashing + frequency detuning)
- `d = β₂ + iδ₂` — higher-order nonlinearity (mode-locking)
- `ε` — controls range of nonlinearities
- Stability constraint: `|z| < 1/√ε` AND `|x| < 1/√ε`

**Key structural insight we are missing:** the input `x` enters
through a **nonlinear transformation** `x/(1-√ε x)` and is
multiplied by the receptor term `1/(1-√ε z̄)`. Our current
implementation applies `x` linearly: `ż = ... + x`. This is
incorrect per the canonical form.

## Network input form (Eq. 11)

For oscillator `i` in a network of `n` oscillators:

```
x_i(t) = s(t) + Σ_{j≠i} c_ij · z_j
```

The **internal coupling is part of the input** — coupling oscillators
contribute *linearly* to `x_i`, then the combined input goes through
the nonlinear transformation. Our Phase 1.1 implementation has
internal coupling as a *separate* additive term `coupling_gain · A_n
· (C @ P)`, which is structurally different.

## Polar form (Eqs. 16, 17)

```
ṙ = r(α + β₁ r² + β₂ r⁴ / (1 - ε r²))
    + (F (ε F r + cos(φ - θ)) - √ε(F cos φ + r cos θ)) /
      ((1 + ε F² - 2F √ε cos θ)(1 + ε r² - 2r √ε cos φ))

φ̇ = ω + δ₁ r² + δ₂ r⁴ / (1 - ε r²)
    + (F(sin(φ - θ) + √ε(F sin θ - r sin θ))) /
      ((1 + ε F² - 2F √ε cos θ)(1 + ε r² - 2r √ε cos φ)) · 1/r
```

where `x = F e^(iθ)`, `z = r e^(iφ)`.

Useful for analytical work. We don't directly use this form.

## Specific parameter values (Fig. 2 demo)

The paper compares a Wilson-Cowan network and a canonical GrFNN on a
sinusoidal input `s(t) = F sin(2πt)`, `F = 0.30`, `ω₀ = 2π`.

**Wilson-Cowan parameters (Eq. 19):**
- `a = 10, b = 10, c = 8.6095, d = -1.1429`
- `ρ_u = -2.3486, ρ_v = -4.2411`
- `ε = 0.4`

**Canonical GrFNN parameters (Eq. 20):**
- `a_i = 0 + i·2π` (so `α = 0`, `ω = 2π`)
- `b_i = -10 + i(-9)` (so `β₁ = -10`, `δ₁ = -9`)
- `d_i = -10 + i(-9)` (so `β₂ = -10`, `δ₂ = -9`)
- `ε = 0.4`
- Period range: `0.125 ≤ τ_i ≤ 8` (i.e., `0.125 ≤ f_i ≤ 8` Hz —
  this is a RHYTHM-range network, not pitch)

**Network: 360 oscillators, 6 octaves, 60 osc/octave** (i.e., 17
cents/bin). No internal coupling in this demo (`c_ij = 0`).

**Comparison with our config:**

| Parameter | Paper (Fig. 2) | Our config | Direction of difference |
|---|---|---|---|
| α | 0 (critical) | -0.05 (sub-critical) | Ours more damped |
| β₁ | -10 | -1 | Theirs 10× stronger nonlinearity |
| β₂ | -10 | -1 | Theirs 10× stronger |
| δ₁ | -9 | 0 | Theirs has detuning, ours doesn't |
| δ₂ | -9 | 0 | Same |
| ε | 0.4 | 1.0 | Theirs less max nonlinear |
| Frequency range | 0.125-8 Hz (rhythm) | 20-4186 Hz (pitch) | Different domain |
| Density | 60/octave | 36/octave | Theirs denser |

## Behaviors demonstrated

The canonical GrFNN with these parameters responds to a sinusoidal
input at:
- **Stimulus frequency** (1:1) — strongest response
- **2:1 harmonic** — present but weaker
- **3:1 harmonic** — present
- **1:2 subharmonic** (phantom-fundamental signature) — present

Frequency-response correlation between Wilson-Cowan and canonical:
**r² = 0.946** — the canonical model is a faithful approximation.

## Time scale convention

Eq. 19/20 use `τ_i ż = ...` form with `τ_i = 2π/ω_i`. So `ω_i = 2π/τ_i`,
meaning ω is in rad per "natural period" — but the equations have
been transformed so that ω_i = 2π for all oscillators (each is
analyzed in its own natural-period frame). Then the τ scaling
brings everything back to per-second.

In our code, ω is in rad/s directly (so ω_i = 2π · f_i with f_i
in Hz). When we enable `per_oscillator_tau`, we multiply (α, β₁, β₂,
drive) by `f_n / tau_reference_hz`. With `tau_reference_hz = 1.0`
this is functionally equivalent to the paper's convention.

## Resonant terms (RT) — Eq. 12

The general expansion of the canonical input transformation:

```
RT = (x + √ε x z̄ + ε x z̄² + ε^(3/2) x z̄³ + ...)
   × (1 + √ε z̄ + ε z̄² + ε^(3/2) z̄³ + ...)
```

Equivalent to the closed-form `(x / (1-√ε x)) · (1 / (1-√ε z̄))`.

This expansion produces resonance at every integer combination of
input and receptor frequencies — capturing the "harmonics,
subharmonics, and combination frequencies (e.g., 2:1, 3:1
harmonics, 1:2 subharmonic) of stimulus frequency" demonstrated in
Fig. 2.

## Notes for our implementation

**Critical changes needed for canonical compliance** (queued for
Phase 1.x refactor):

1. **Nonlinear input form**: replace `+ x` with
   `+ (x_total / (1 - √ε x_total)) · (1 / (1 - √ε z̄))` where
   `x_total = stimulus + Σ_j c_ij z_j`.
2. **Internal coupling integrated into x_total**: not a separate
   additive term. Linear sum of `c_ij z_j` THEN nonlinear transform.
3. **Parameter ladder**: try canonical values (α=0, β₁=-10) at least
   for one layer, compare to current.
4. **Detuning δ**: paper uses non-zero δ₁, δ₂. Ours has zero. Worth
   experimenting with δ₁ ≈ -9 (or scaled) to see if mode-locking
   becomes stronger.

The canonical form is what the patent (and subsequent papers) build
on. Implementing it correctly is **load-bearing** for everything
downstream — voice extraction, phantom fundamentals, harmonic
clustering, mode-locking.
