# Kim & Large (2019) — Mode locking in periodically forced gradient frequency neural networks

**Reference:** Physical Review E 99, 022421
**DOI:** 10.1103/PhysRevE.99.022421

---

## Why this matters for factory-dame

**The full theoretical analysis of mode-locking in our canonical
oscillator.** Complete characterization of:

1. Arnold-tongue widths for k:m mode-locking across all four
   parameter regimes (Critical Hopf, Supercritical Hopf,
   Supercritical DLC, Subcritical DLC)
2. **Exact analytical formula for Arnold-tongue boundaries**
3. Bistability and SNIC bifurcations at locking boundaries
4. Hysteresis and zero-stability regimes

This grounds:
- **Phase 4 voice clustering** — Arnold-tongue widths tell us
  which mode-locks to *trust* given input strength
- **Phase 3 layer parameter selection** — each regime has
  characteristic mode-locking behavior
- **`modelock.py` validation** — analytical predictions our PLV
  detection should match

## Canonical model with single monomial input (Eq. 4)

For analysis, the canonical GrFNN reduces to one oscillator with
single-monomial input:

```
ż = z (α + iω + β_1|z|² + εβ_2|z|⁴/(1-ε|z|²))
   + ε^((k+m-2)/2) · x^k · z̄^(m-1)
```

with `x(t) = F e^(iω_0 t)` sinusoidal forcing.

Full canonical form includes the infinite-series RT (Eq. 3):

```
RT = x/(1-√ε x) · 1/(1-√ε z̄)
   = Σ_{k≥1, m≥1} ε^((k+m-2)/2) · x^k · z̄^(m-1)
```

Each `(k,m)` term mediates the corresponding k:m mode-lock.

## Polar-form mode-lock equation (Eq. 5)

```
ṙ = αr + β_1 r³ + εβ_2 r⁵/(1-εr²)
   + ε^((k+m-2)/2) · F^k · r^(m-1) · cos ψ

ψ̇ = Ω - m ε^((k+m-2)/2) · F^k · r^(m-2) · sin ψ
```

where:
- `ψ = mφ - kω_0 t` — relative phase
- `Ω = mω - kω_0` — frequency difference for k:m ratio

**Stable fixed point in (r, ψ)** ⇔ stable mode-lock.

## Mode-locking condition — Eq. 6

Eliminating ψ* from steady-state conditions:

```
(α + β_1 r*² + εβ_2 r*⁴/(1-εr*²))² + (Ω/m)² = ε^(k+m-2) F^(2k) r*^(2m-4)
```

This is a polynomial in r*² of order 8 (for m ≤ 4). Numerical
root-finding gives all fixed points; Jacobian evaluation gives
their stability.

## Four parameter regimes (Table I + Fig. 1)

`β_2 < 0` for all regimes. Each regime has distinct
autonomous-amplitude vector field shape:

| Regime | α | β₁ | Autonomous behavior | Notable |
|---|---|---|---|---|
| **Critical Hopf** | 0 | – | r=0 sole attractor | Sharp tuning |
| **Supercritical Hopf** | + | – | Nonzero spontaneous amp | Memory persists |
| **Supercritical DLC** | – | + | Bistable: 0 or nonzero | Hysteresis |
| **Subcritical DLC** | – | + | r=0 + local max in dr/dt | "Hump" regime |

(DLC = Double Limit Cycle bifurcation.)

## Closed-form Arnold tongue formulas

### For weak forcing (locking-region width)

The boundary of the Arnold tongue at zero forcing amplitude scales
linearly. Width is approximately (Eq. 14):

```
|Ω| ≤ m · (√ε F)^k · (√ε r_s)^(m-2) ≡ Γ'
```

where `r_s` is the spontaneous oscillator amplitude (≈ √(-α/β_1)
in supercritical regime).

Define `γ = Γ'/m = (√εF)^k (√εr_s)^(m-2)`. Then:
- **Unscaled width** = 2γ
- **Logarithmically-scaled width** = `log_b((2π+γ)/(2π-γ))`

### Arnold tongue widths by k:m

For γ << 2π (weak resonance):

```
width ≈ 2γ ≈ 2(√εF)^k (√εr_s)^(m-2)
```

**Strong dependence on k+m.** For F < 1 and r_s < 1 (typical
factory-dame regime), the width drops sharply as k+m grows:

| Ratio (k:m) | k+m | Width relative to 1:1 |
|---|---|---|
| 1:1 | 2 | 1.0 |
| 2:1 | 3 | √ε · F |
| 1:2 | 3 | √ε · r_s |
| 3:2 | 5 | ε² F² · r_s |
| 2:3 | 5 | ε² F · r_s² |
| 5:3, 3:5 | 8 | ε³.5 (F or r_s)^... |

**Practical:** at our typical |z| ~ 0.04 and ε = 1, simple 1:1 and
2:1 ratios have wide locking; complex ratios have negligible
widths.

## Phase locking (k=1, m=1) — basic case

Eq. 5 reduces to:

```
ṙ = αr - r³ + Fr · cos ψ        (after rescaling)
ψ̇ = Ω - F · sin ψ                (the Adler equation)
```

The **Adler equation** governs ψ. Stable fixed point ψ* exists if
`|Ω| ≤ F`. This is the classical 1:1 phase-locking condition.

## Second subharmonic (1:2) mode-locking — detailed analysis

For k=1, m=2 (Eq. 8-9):

```
ṙ = αr + β_1 r³ + εβ_2 r⁵/(1-εr²) + (√εF) r cos ψ
ψ̇ = Ω - 2(√εF) sin ψ
```

**Critical Hopf regime (α = 0, β_1 = -0.5, β_2 = -1, ε = 1):**
- Arnold tongue coincides with locking region of ψ
- Zero is unstable inside the Arnold tongue, stable outside

**Supercritical Hopf regime (α = 0.5, β_1 = -1, β_2 = -1):**
- Inside locking region: up to 2 nonzero fixed points
- SNIC bifurcation at locking boundary
- Outside locking region: stable limit cycle (oscillator rotates
  off-lock)

**Supercritical DLC regime (α = -0.5, β_1 = 2, β_2 = -0.5):**
- Up to 4 fixed points in locking region; only 1 stable
- Saddle-node bifurcation at locking boundary (not SNIC)
- For strong forcing: doesn't leave a stable limit cycle (zero
  becomes global attractor)

**Subcritical DLC regime (α = -0.5, β_1 = 1.1, β_2 = -0.5):**
- Arnold tongue **lifted off F=0** (does not coincide with locking
  region for ψ)
- Up to 2 nonzero fixed points inside Arnold tongue
- Zero is stable globally for weak forcing
- Zero is locally stable inside Arnold tongue when a saddle exists

## Third subharmonic (1:3) and higher

For m ≥ 3, phase dynamics depend on amplitude (not closed-form
Adler-like). Similar regime-dependent structure but quantitatively
narrower tongues. **All m ≥ 3 share the same dynamics**; m = 2 is a
special case due to phase-amplitude decoupling.

## Logarithmic frequency scaling (Sec IV A)

Frequency-scaled canonical oscillator (Eq. with τ = 1/f):

```
τ ż = z (α + 2πi + β_1|z|² + εβ_2|z|⁴/(1-ε|z|²))
     + ε^((k+m-2)/2) x^k z̄^(m-1)
```

In polar coords, the polar equations look identical to unscaled
case **except Ω is replaced by Ω/f**. So the locking range for
unscaled oscillators (`|Ω| ≤ Γ`) becomes `|Ω|/f ≤ Γ` for
frequency-scaled, i.e., the locking range scales with natural
frequency.

**Constant-Q result (Eq. 13):**

```
k/(m + 1/(2π)) ≤ ω/ω_0 ≤ k/(m - 1/(2π))    (if Γ < 2πm)
```

Width on log scale = `log_b((2π+γ)/(2π-γ))` — constant across
the spectrum.

## Full GrFNN with infinite-series input (Eq. 15)

```
τ_i ż_i = z_i (α + 2πi + β_1|z_i|² + εβ_2|z_i|⁴/(1-ε|z_i|²))
         + c · x/(1-√εx) · 1/(1-√εz̄_i)
```

**Fig. 12 result:** numerical Arnold tongues for the full
infinite-series GrFNN match the single-monomial analytical
tongues for **weak forcing**. They **deviate at high forcing
amplitudes** — multiple resonant monomials become simultaneously
active, distorting individual tongues.

**Practical:** at low |z| (factory-dame's regime), single-monomial
analysis is accurate. At strong amplitudes, expect mode-lock
contamination across nearby ratios.

## Specific simulation parameters (Fig. 12)

- 2001 oscillators
- Log-spaced 0.23 Hz to 4.4 Hz
- Forcing 1 Hz, amplitudes 0 to 0.3
- α = 0.9 (supercritical Hopf)
- β_1 = -3, β_2 = -3
- ε = 1
- c = 3

Shows Arnold tongues at 1:1, 2:1, 3:1, 4:1, 3:2, 2:3, 1:2, 1:3,
1:4 — exactly the ratios we expect for music.

## Implications for factory-dame

### Voice clustering — which ratios are "real"

Per Eq. 14 width formula, at our typical operating point (ε = 1,
F ~ 1 for strong input, r_s ~ 0.5 in critical regime), Arnold
tongue widths:

- 1:1, 2:1, 1:2 — wide, robust mode-locks
- 3:1, 1:3, 3:2, 2:3 — narrower but reliable
- 5:n, 7:n — narrow, marginal
- k+m > 8 — essentially noise

**Phase 4 voice clustering** should restrict to k+m ≤ 7 ratios
(roughly). Anything beyond is unreliable per this analysis.

### Phase 4 confidence weighting

For each pair of cluster members with proposed ratio k:m, weight
their contribution by **expected Arnold-tongue width** at the
current amplitude. A 1:1 lock with strong PLV is decisive; a 5:3
lock with weak PLV is suggestive only.

```python
def ratio_confidence(k, m, F, r_s, epsilon=1.0):
    """Expected Arnold tongue width for k:m lock"""
    gamma = (epsilon**0.5 * F)**k * (epsilon**0.5 * r_s)**(m-2)
    return min(gamma, 1.0)  # cap at 1
```

### Phase 3 layer regime selection

This paper's four-regime analysis maps onto our Phase 3 layer roles:

- **Cochlear layer**: Critical Hopf (α=0) — always phase-locks,
  no spontaneous activity
- **Brainstem layers**: Critical or Subcritical DLC — broad
  mode-locking, slight selectivity
- **Cortex layer**: Supercritical Hopf — spontaneous oscillation,
  memory persistence

The hysteresis in Supercritical DLC could be exploited for
"sticky" voice identity — once a voice is detected, it persists
even when the stimulus momentarily weakens.

### `modelock.py` validation tests

Our PLV detection should predict mode-locks consistent with this
paper's Arnold tongues. **Validation test:** synthesize a stimulus
at frequency 1.97×f_natural (just outside 2:1 lock) and at
2.00×f_natural (perfectly locked). Our PLV value should drop
sharply across the Arnold-tongue boundary as predicted by Eq. 14.

If our PLV detects mode-lock at 1.5×f_natural (a 3:2 ratio), check
that the input amplitude is strong enough per the width formula —
if not, we're probably finding spurious locks.

### Why Phase 1.1 internal coupling was a no-op at low amplitudes

This paper explains it precisely. The coupling term
`ε^((k+m-2)/2) x^k z̄^(m-1)` is the resonant monomial. At our
typical |z| ~ 0.04, the prefactor `(√ε z)^(m-1)` is **tiny** for
m ≥ 2 — `0.04^1 = 0.04` for 2:1 coupling, `0.04^2 = 0.0016` for
3:2. The internal kernel's *form* was right; the **operating
amplitude was wrong** for any multi-frequency effects to register.

**This argues for** running the layer hosting internal coupling in
**higher-amplitude regime** — likely supercritical Hopf or
Supercritical DLC, where r_s ~ 1 and the resonant monomials have
real magnitude.
