# Kim & Large (2021) — Multifrequency Hebbian plasticity in coupled neural oscillators

**Reference:** Biological Cybernetics 115(1), 43–57
**DOI:** 10.1007/s00422-020-00854-6

---

## Why this matters for factory-dame

**The complete mathematical theory of multifrequency Hebbian
plasticity in GrFNNs.** Direct grounding for Phase 5. Key findings:

1. Hebbian plasticity in GrFNNs is **locally dominated by a single
   nonlinear resonance** — the lowest-order resonant monomial for
   each k:m ratio. So we can approximate the full canonical model
   (infinite series) by analyzing one monomial per ratio.
2. **Low-order multifrequency learning (small k, m) is stronger
   than high-order learning** — simple ratios produce wider
   resonance regions / faster learning. This is why factory-dame's
   voice clustering should weight low k+m ratios more heavily.
3. **Plastic connections are neutrally stable in absence of forcing**
   — they remember whatever phase difference was last imposed.
   When forcing is removed, connections drift.
4. **Stabilization strategies analyzed.** Two ways to prevent
   the simple single-frequency model from diverging at γ = κ.

## The single-frequency canonical model (Eq. 3)

For two coupled equal-frequency oscillators with Hebbian plasticity:

```
ż_i = z_i (α + iω - |z_i|²) + c_ij z_j
ċ_ij = -γ c_ij + κ z_i z̄_j      ,    (i,j) ∈ {(1,2), (2,1)}
```

- `α + iω` — bifurcation + natural frequency
- `-|z_i|²` — cubic stabilizing nonlinearity
- `c_ij ∈ ℂ` — complex synaptic connection
- `γ > 0` — forgetting rate
- `κ > 0` — Hebbian learning rate
- `z_i z̄_j` — coactivity product (drives c_ij to learn phase
  difference)

## Stability of zero, asymmetric, symmetric solutions (Sec 2.2)

**Zero solution** `z_i = c_ij = 0`:
- Stable for α < 0 (passive oscillators decay to zero)
- Unstable for α > 0 (active oscillators)

**Nonzero symmetric solution** (Eq. 7):

```
r* = √(γα / (γ - κ))
A* = κα / (γ - κ)
ψ* = 0
```

Exists if `α > 0 AND γ > κ` or `α < 0 AND γ < κ`. Diverges at
`γ = κ` — this is the **key instability** the paper addresses.

**Stability of nonzero symmetric:**
- `γ > κ`: nonzero solution is **stable node**, monotonic
  approach
- `γ < κ`: nonzero solution is **saddle**, system diverges to
  infinity
- `γ = κ`: degenerate (separatrix)

## Two stabilization strategies (Sec 2.3)

### Strategy 1: Add quintic term to oscillator (Eq. 8)

```
ż_i = z_i (α + iω + β|z_i|² - |z_i|⁴) + c_ij z_j
ċ_ij = -γ c_ij + κ z_i z̄_j
```

Adding `-|z_i|⁴` keeps amplitudes bounded. Now near a **double
limit cycle (DLC) bifurcation** when β > 0, or Hopf when β < 0.

### Strategy 2: Add cubic damping to connections (Eq. 9)

```
ż_i = z_i (α + iω - |z_i|²) + c_ij z_j
ċ_ij = c_ij (λ - |c_ij|²) + κ z_i z̄_j
```

Adding `-|c_ij|²` to the learning rule keeps connection amplitudes
bounded. **The connection itself becomes a Stuart-Landau oscillator**
— with its own bifurcation, fixed points, limit cycles. Connection
phase rotates if forced off-resonance.

(Strategy 1 is analyzed in detail in this paper; Strategy 2 is
the form used in Large 2016 tonality Eq. A2 and is mentioned but
deferred to future work here.)

## Single-frequency model with detuning (Eq. 10)

Extends the stabilized model (Eq. 8) by allowing `ω_1 ≠ ω_2`:

```
ż_i = z_i (α + iω_i + β|z_i|² - |z_i|⁴) + c_ij z_j
ċ_ij = -γ c_ij + κ z_i z̄_j         (still single-frequency form)
```

**Key result:** when `Ω = ω_1 - ω_2 ≠ 0`, plastic connection
**phases rotate at a constant rate** `θ̇*_ij` matching the
frequency difference of the oscillators' *instantaneous*
frequencies. The connection compensates for frequency mismatch.

For small Ω: plastic connections have **large amplitude and slow
oscillation**. For large Ω: connections have small amplitude and
oscillate at near-Ω rate.

**Implication:** in our voice-clustering, two oscillators near but
not exactly at integer ratios will still develop a connection —
the connection just rotates to track the phase drift. This is good:
real audio has imperfect harmonic ratios, and a Hebbian rule that
*only* works at exact ratios would miss real voices.

## Multifrequency Hebbian (Sec 3.1) — Eq. 13

For two oscillators near integer ratio k:m (i.e., `mω_1 ≈ kω_2`):

```
ż_1 = z_1 (a_1 + b_1|z_1|² + εd_1|z_1|⁴/(1-ε|z_1|²))
     + ε^((k+m-2)/2) · c_12 · z_2^k · z̄_1^(m-1)

ż_2 = z_2 (a_2 + b_2|z_2|² + εd_2|z_2|⁴/(1-ε|z_2|²))
     + ε^((k+m-2)/2) · c_21 · z_1^m · z̄_2^(k-1)
```

The coupling terms `z_2^k z̄_1^(m-1)` and `z_1^m z̄_2^(k-1)` are the
**lowest-order resonant monomials** for the k:m relationship. They
satisfy the resonance condition:

```
iω_1 = k·iω_2 - (m-1)·iω_1
iω_2 = m·iω_1 - (k-1)·iω_2
```

**The ε^((k+m-2)/2) prefactor is the load-bearing scaling.** Higher-
order ratios (large k+m) are attenuated by powers of ε. This is
the formal derivation of Large 2016's `stability ≈ ε^((k+m-2)/2)`.

## Multifrequency Hebbian learning rule (Eq. 14)

```
ċ_12 = -γ c_12 + ε^((k+m-2)/2) · κ_12 · z_1^m · z̄_2^k
ċ_21 = -γ c_21 + ε^((k+m-2)/2) · κ_21 · z_2^k · z̄_1^m
```

The coupling terms `z_1^m z̄_2^k` and `z_2^k z̄_1^m` become
**stationary** when oscillators are mode-locked at k:m (their
relative phase `mφ_1 - kφ_2` is constant). Plastic connections
"resonate" — they reach a steady value — only when the input
satisfies the resonance condition. Otherwise the connection
oscillates and time-averages to zero.

## Rescaled multifrequency model (Eq. 15)

After rescaling `z_i → z_i/√ε`, `β_1 → β_1/ε`, `β_2 → β_2/ε`,
`κ → κ/ε`, and constraining `|z_i| < 1`:

```
ż_1 = z_1 (α + iω_1 + β_1|z_1|² + β_2|z_1|⁴/(1-|z_1|²))
     + c_12 · z_2^k · z̄_1^(m-1)

ż_2 = z_2 (α + iω_2 + β_1|z_2|² + β_2|z_2|⁴/(1-|z_2|²))
     + c_21 · z_1^m · z̄_2^(k-1)

ċ_12 = -γ c_12 + κ · z_1^m · z̄_2^k
ċ_21 = -γ c_21 + κ · z_2^k · z̄_1^m
```

**This is the recommended Phase 5 form.** Bounded amplitudes,
single-monomial coupling per ratio, clean Hebbian rule.

## Strength of k:m learning (Eq. 17, 18)

For symmetric solutions on the manifold `r_1 = r_2 = r`,
`A_12 = A_21 = A`, `ψ_12 = -ψ_21 = ψ`:

At Ω = 0 (perfect frequency match), nonzero solutions are
intersections of:

```
y_1 = α + β_1 X + β_2 X² / (1-X)       (intrinsic dynamics)
y_2 = -(κ/γ) X^(k+m-1)                 (learning-driven amplitude)
```

where `X = r*²`. Note y_2 depends on k+m. **For larger k+m, y_2
intersects y_1 at higher X**, meaning low-order ratios reach the
steady-state at lower amplitudes.

**Critical learning rate** `κ_0`: minimum κ for which a nonzero
solution exists (for α ≤ 0).
- **Higher k+m requires higher κ_0** — complex ratios need
  proportionally faster learning to overcome decay.

## Frequency scaling for logarithmic networks (Sec 3.2)

For tonotopic / logarithmically-spaced GrFNNs, scale the equation
by natural frequency:

```
(1/f_i) ż_i = z_i (α + 2πi + β_1|z_i|² + β_2|z_i|⁴/(1-|z_i|²))
             + c_12 · z_2^k · z̄_1^(m-1)        (Eq. 20)
```

For learning rules, approximate scaling `f_ij ≈ (2 f_i f_j)/(f_i + f_j)`:

```
(1/f_ij) ċ_ij = -γ_ij c_ij + κ_ij · z_i^m · z̄_j^k        (Eq. 21)
```

with `f_ij = (kf_2 + mf_1)/(k+m)` (Eq. 22).

**Constant-Q property:** Arnold tongue widths become *constant on
log frequency scale*. Each k:m resonance covers the same
log-frequency interval regardless of where in the spectrum it sits
— exactly what we want for music processing.

## Full GrFNN with plastic connections (Eq. 24)

The complete canonical multifrequency GrFNN with Hebbian learning:

```
ż_i = z_i (a_i + b_i|z_i|² + d_i|z_i|⁴/(1-|z_i|²))
     + Σ_{j≠i} c_ij · z_j/(1-z_j) · 1/(1-z̄_i)

ċ_ij = -γ_ij c_ij + κ_ij · z_i/(1-z_i) · z̄_j/(1-z̄_j)
```

Note: the products `z_j/(1-z_j)·1/(1-z̄_i)` and
`z_i/(1-z_i)·z̄_j/(1-z̄_j)` are **infinite-series expansions**:

```
z_j/(1-z_j) · 1/(1-z̄_i) = Σ_{k≥1, m≥1} z_j^k · z̄_i^(m-1)
z_i/(1-z_i) · z̄_j/(1-z̄_j) = Σ_{k≥1, m≥1} z_i^m · z̄_j^k
```

Each (k,m) term in the sum is a resonance monomial for the k:m
ratio. **The infinite series automatically includes resonance at
every integer ratio.**

## Frequency-scaled GrFNN model (Eq. 26)

```
(1/f_i) ż_i = z_i (a_i' + b_i|z_i|² + d_i|z_i|⁴/(1-|z_i|²))
             + Σ_{j≠i} c_ij · z_j/(1-z_j) · 1/(1-z̄_i)

(1/f_ij) ċ_ij = -γ_ij c_ij + κ_ij · z_i/(1-z_i) · z̄_j/(1-z̄_j)
```

with `a_i' = α + 2πi`. **This is the canonical form for Phase 5.**

## Validation simulation (Fig. 11)

Numerical simulations of Eq. 26:
- 601 oscillators, log-spaced 1–4 Hz (2 octaves)
- α = 2 (supercritical Hopf)
- β_1 = β_2 = -1
- δ_1 = δ_2 = 0
- γ_ij = 0.5 (forgetting rate)
- κ_ij = 8.33×10⁻⁵ = 0.05/(n-1) (learning rate scaled by network size)

**Result:** time-averaged connection matrix has clear bright
diagonals at 1:1, 2:1, 1:2, 3:2, 2:3, 3:1, 1:3, 4:3, 5:3, 5:2, 7:2.
**Resonance strength decreases with k+m**, exactly matching the
analytical prediction.

This figure is the canonical evidence that **W community structure
encodes harmonic relationships.** Voice extraction reads this
structure.

## Key theoretical conclusions

1. **Hebbian plasticity in GrFNNs is locally dominated by the
   lowest-order resonant monomial for each k:m ratio.** Full
   infinite-series analysis is overkill — analyze one monomial per
   resonance peak.
2. **Low-order ratios (small k+m) produce stronger and more stable
   learning** — direct mathematical justification for restricting
   factory-dame's voice clustering to k+m ≤ ~5.
3. **Plastic connections are neutrally stable in absence of forcing.**
   They drift without input, locked when forced. This is biologically
   appropriate (no input → no change in synapse value, just slow
   decay via γ).
4. **The minimum learning rate κ_0 scales with k+m** — complex
   ratios need proportionally faster learning. Phase 5 could
   adapt κ per ratio or just set κ above the worst-case κ_0.
5. **Constant-Q via logarithmic frequency scaling** — tonotopic
   networks naturally produce equal-strength resonance per octave.

## Implications for factory-dame

### Phase 5 implementation recipe

Drop directly into our codebase using **Eq. 26 (frequency-scaled
GrFNN with plastic connections)**. Specific parameters:

```python
# Recommended starting values per Kim & Large 2021 Fig. 11
alpha = 2.0          # supercritical Hopf (for spontaneous memory)
beta_1 = -1.0
beta_2 = -1.0
delta_1 = 0.0        # no detuning initially
delta_2 = 0.0
epsilon = 1.0        # captured via z/(1-z) coupling form
gamma_ij = 0.5       # forgetting rate
kappa_ij = 0.05 / (n - 1)   # scale learning rate by network size
```

The `z/(1-z)` and `z̄/(1-z̄)` infinite-series forms can be computed
once per sample (cheap) and feed into the c_ij update.

### Patent IP

This paper is the academic publication of equations also in
**US 8,930,292** (Eqs. 5, 6 of that patent). The literal `ċ_ij =
-γ c_ij + κ z/(1-z)·z̄/(1-z̄)` is **patent-claimed** for commercial
use. Personal/research use is fine.

**Patent-safe alternatives** (for commercial path):
- Use the **single-monomial form per ratio** (Eq. 14) and choose
  ratios at PLV detection rather than full series expansion. This
  is mathematically a subset of the patent claim — *probably* still
  claimed (Claim 1 covers "multi-frequency phase coherency" broadly).
- Use **strategy 2 from this paper (Eq. 9): cubic damping on
  connections.** This stabilizes via different math than the
  patent's exact equations. Need legal review to confirm.

### Three-frequency D_ijk

This paper analyzes only TWO-frequency C_ij. Large 2016 introduced
**three-frequency D_ijk**. Three-freq learning isn't analyzed here
— deferred to future work. **The three-freq rule may be a
patent-safer extension** since US 8,930,292 focuses on
two-oscillator coherency.

### Phase 4 voice clustering threshold

Per this paper's finding that learning strength scales as
`ε^((k+m-2)/2)`, **only ratios with k+m ≤ ~5 produce strong stable
connections** at reasonable ε. Phase 4's clustering can safely
ignore ratios with k+m > 5 — they'd be noise, not real voices.

### Constant-Q via log-frequency scaling

Our pitch bank is already log-spaced. Using Eq. 26's scaling (with
`1/f_i` and `1/f_ij` prefactors) gives constant-Q resonance widths
automatically. **Phase 1.2's per-osc τ is exactly this scaling** —
already in place.
