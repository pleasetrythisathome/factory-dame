# Large, Kim, Flaig, Bharucha, Krumhansl (2016) — A Neurodynamic Account of Musical Tonality

**Reference:** Music Perception 33(3), 319–331
**DOI:** 10.1525/MP.2016.33.03.319

---

## Why this matters for factory-dame

**This is the most concretely-specified Hebbian-learning paper from
the Music Dynamics Lab.** It gives:

1. **The exact Hebbian update rules** for both two-frequency (C_ij)
   and three-frequency (D_ijk) connections (Appendix Eq. A2).
2. **A closed-form approximation** for the stability of mode-locked
   states (Eq. 2): `stability ≈ ε^((k+m-2)/2)`.
3. **Concrete supercritical Hopf parameters** used in their network
   simulation (α=0.002, β₁=-2, β₂=-4).
4. **Validation that small-integer ratios dominate** what the network
   learns and remembers — the foundational claim for Hebbian-based
   voice extraction.

This paper bridges Large 2010 (canonical equation) and Kim & Large
2021 (Hebbian theoretical analysis) — but is more
implementation-friendly than either, because it specifies the
network simulation in concrete terms.

## Generic two-oscillator dynamics (Eq. 1)

Two oscillators at frequencies near integer ratio `k:m`:

```
τ_1 · dz_1/dt = z_1(α + i·2π + β_1|z_1|² + εβ_2|z_1|⁴ + …)
              + c · ε^((k+m-2)/2) · z_2^k · z̄_1^(m-1)

τ_2 · dz_2/dt = z_2(α + i·2π + β_1|z_2|² + εβ_2|z_2|⁴ + …)
              + c · ε^((k+m-2)/2) · z_1^m · z̄_2^(k-1)
```

The coupling-strength factor `ε^((k+m-2)/2)` is *automatic* — it
emerges from the resonant-monomial expansion at order k+m.

## Closed-form stability of mode-locked states (Eq. 2)

```
stability ≈ {  ε^((k+m-2)/2)   if k/m is in context
             {  0               if k/m not in context
```

**This is the single most useful formula in the paper.** Given ε,
it predicts how stable a mode-lock at ratio k:m will be:

| Ratio (k:m) | k+m | Exponent | Stability at ε=0.85 | At ε=0.5 |
|---|---|---|---|---|
| 1:1 | 2 | 0 | 1.000 | 1.000 |
| 2:1, 1:2 | 3 | 0.5 | 0.922 | 0.707 |
| 3:2, 2:3, 4:3, 3:4 | 5–7 | 1.5 | 0.783 | 0.354 |
| 5:3, 3:5 | 8 | 3 | 0.522 | 0.125 |
| 15:8 | 23 | 10.5 | 0.183 | 0.001 |

**Implication for voice extraction:** if two oscillators' coupling
through learned W reflects mode-lock stability, real voices'
harmonics (low k+m) will have orders-of-magnitude stronger
connections than spurious complex-ratio pairs. The W matrix is
*automatically sparse along voice-membership lines* in the
small-ε limit.

ε was fit per rāga, with `0 < ε < 1`. Best fits had ε ≈ 0.78 to
0.97 across rāgas (more on this below).

## Network equation with two- and three-freq monomials (Eq. A1)

```
τ_i · dz_i/dt = z_i (α + i·2π + β_1|z_i|² + ε β_2|z_i|⁴ / (1 - ε|z_i|²))
              + Σ_{j≠i} C_ij · ε^((k_ij + m_ij - 2)/2) · z_j^(k_ij) · z̄_i^(m_ij - 1)
              + Σ_{j≠i} Σ_{k≠i} D_ijk · ε^((p_ijk + q_ijk + r_ijk - 2)/2)
                                       · z_j^(p_ijk) · z_k^(q_ijk) · z̄_i^(r_ijk - 1)
              + x_i(t)
```

- `C_ij` = two-frequency connection weight (i,j)
- `D_ijk` = three-frequency connection weight (i,j,k) — captures
  combination frequencies like (G+D)→E
- `k_ij, m_ij` integer-ratio exponents for the two-freq monomial
- `p_ijk, q_ijk, r_ijk` integer exponents for the three-freq monomial
- `x_i(t)` external linear input at the oscillator's natural freq

**Importantly: 24 two-freq monomials + 275 three-freq monomials**
in their simulation (one each per resonant pairing/tripling). Each
monomial is the *lowest-order resonant monomial* for the relevant
frequency relationship.

## HEBBIAN LEARNING RULES (Eq. A2)

The full Hebbian-update rules for both connection types:

```
τ_i · dC_ij/dt = C_ij (λ + μ_1|C_ij|² + ε_C μ_2 |C_ij|⁴ / (1 - ε_C|C_ij|²))
               + ε_C^((k_ij + m_ij - 2)/2) · κ · z_i^(m_ij) · z̄_j^(k_ij)

τ_i · dD_ijk/dt = D_ijk (γ + ν_1|D_ijk|² + ε_D ν_2 |D_ijk|⁴ / (1 - ε_D|D_ijk|²))
                + ε_D^((p_ijk + q_ijk + r_ijk - 2)/2) · η · z_i^(r_ijk) · z_j^(p_ijk) · z̄_k^(q_ijk)

   where:  λ, γ > 0   (supercritical Hopf for memory)
           μ_1, μ_2, ν_1, ν_2 < 0   (saturating nonlinearity)
           κ, η > 0   (Hebbian gain)
```

**This is the Phase 5 recipe.** The connection-weight equations
have the same canonical form as the oscillator equation: Hopf-
bifurcation core + saturating nonlinear damping + multifrequency
forcing. The C_ij and D_ijk connections are themselves oscillators
in a higher-dimensional space.

`λ > 0` means **connections spontaneously persist after stimulation
ends.** They don't decay back to zero unless explicitly driven
toward decay. This is how memory is implemented.

## Concrete simulation parameters (Appendix)

```
α = 0.002          (supercritical Hopf — spontaneous oscillation,
                    so memory persists without stimulus)
β_1 = -2
β_2 = -4
ε   varied per rāga, 0.78 ≤ ε ≤ 0.97
```

```
Hebbian parameters:  λ, γ > 0    (signs only stated, no values)
                     μ_1, μ_2, ν_1, ν_2 < 0
                     κ, η > 0
```

**The signs are specified; the magnitudes aren't.** This is the
gap left for implementers — picking magnitudes that produce
stable-but-responsive learning. We'd need to scan parameter space
or contact authors for specific values.

```
Network: 25 oscillators, C3 to C5 (2 octaves, equal temperament
         spacing — i.e., chromatic, 12 osc/octave)
Training: C IV-V_7-I cadence, 35 cycles, "until connection
         strengths appeared to reach stable values"
Initial conditions: z_i = 0 for all i (Hebbian rule then builds
         up structure from nothing)
```

**35 training cycles to convergence** is a useful benchmark for
Phase 5 prototyping — we know roughly how long Hebbian needs to
settle.

## Validation results

**Closed-form predictions (Eq. 2) explain Western tonality:**
- C major tone profile fitted at r² = 0.95 with ε = 0.78
- C minor mode fitted at r² = 0.77 with ε = 0.85
- Generalizes to **10 Hindustani rāgas**, r² = 0.79 to 0.97 each
  (cross-cultural prediction with no culture-specific training)

**Full network simulations (with Hebbian) extend the predictions:**
- Combination frequencies (E = "combination of C and G + D") get
  non-zero amplitudes via the three-frequency D_ijk monomials.
- Tone E achieves *greater* amplitude than F in Twinkle Twinkle,
  matching listener data — closed-form misses this; full Hebbian
  simulation captures it.
- Synaptic plasticity (Eq. A2) is what turns "predicted stability"
  into "experienced stability."

**Cross-cultural invariance:** "The neurodynamic predictions
correlated significantly with the averaged perceptual ratings for
every rāga, mean r²(11) = 0.90, min = 0.84, max = 0.93."

## Critical insight: ε IS the learning state

ε was fit per-rāga and per-listener-group, with best values
between 0.78 and 0.97. **Higher ε = more high-order mode-locks
captured.**

This explains the consistent finding (Lerud 2014, Tal 2017, and
here) that musicians vs nonmusicians, Western vs Indian listeners,
etc., differ primarily in *how high-order* their resonance
capture is. A trained listener has effectively higher ε.

**The Hebbian rule of Eq. A2 doesn't change ε directly** — it
changes C_ij and D_ijk. But the *cumulative effect* of training is
equivalent to a higher effective ε at the network level. This is
worth deeper investigation in Phase 5.

## Implications for our implementation

### Phase 5 plan, refined

We now have concrete Hebbian equations and a training protocol:

1. Build C_ij and D_ijk matrices initialized to zero.
2. Use Eq. A2 to evolve C_ij and D_ijk during input.
3. Use Eq. A1 to evolve z_i with the *learned* C_ij and D_ijk.
4. ~35 cycles of training input → stable W.
5. Read voice clusters from C_ij (and D_ijk).

The key innovation beyond US 8,930,292 patent's two-freq rule is
**three-frequency D_ijk** — this is what allows combination
frequencies (like the missing-fundamental and the phantom E from
C+G+D) to emerge. The patent's two-freq rule alone wouldn't
produce them.

### Patent-IP read

This paper publishes the Hebbian rule in academic literature.
**Is it covered by US 8,930,292?** The patent's Claim 1 covers
"determining multi-frequency phase coherency between oscillations"
and "changing connection amplitude/phase as a function of
coherency." Eq. A2 here implements exactly this. So yes, the
academic publication doesn't grant a license — it just publishes
the equation. For commercial use, the patent license is still
required.

That said: **the three-frequency rule (D_ijk)** for combination
frequencies may not be claimed in US 8,930,292 (which focuses on
two-oscillator coherency). Worth a closer patent-claim read; if
D_ijk is genuinely not claimed, it's a path to commercial-safe
combination-frequency learning.

### Supercritical Hopf for memory

α = 0.002 is just barely supercritical — the network oscillates
spontaneously, but only weakly, so external input can still drive
it. This is the regime US 8,930,292 specifies for the *cortex*
layer. The math here uses α = 0.002, the patent uses "α > 0"
(qualitative). They're compatible.

For factory-dame: if Phase 5 builds learnable connections, the
oscillator layer hosting them needs to be in the supercritical
regime too (so that learned patterns can persist). This is a
**second motivation** for our Phase 3 multi-layer architecture —
the W-learning layer is necessarily cortex-like.

### Network density

25 oscillators across 2 octaves = 12/octave (semitones). Much
sparser than factory-dame's 36/octave. For pitch tracking, denser
is needed. For tonal-memory learning, this paper's 12/octave is
sufficient. **Phase 5 could use a coarser parallel layer** for the
learned-W computation, even if the dense pitch layer continues
underneath.

### Three-frequency learning is novel

This is the first time I've seen explicit D_ijk three-frequency
Hebbian. It's a meaningful extension. For factory-dame, it would
let voice clustering capture combination-frequency phantoms (e.g.,
the perceived 200 Hz from 600+900+1200 Hz partials) directly in
the learned W.
