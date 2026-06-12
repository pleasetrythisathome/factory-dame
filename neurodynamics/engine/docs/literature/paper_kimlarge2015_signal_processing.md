# Kim & Large (2015) — Signal Processing in Periodically Forced Gradient Frequency Neural Networks

**Reference:** Frontiers in Computational Neuroscience 9:152
**DOI:** 10.3389/fncom.2015.00152

[Full text on Frontiers](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00152/full)

---

## Why this matters for factory-dame

Provides the **rigorous bifurcation analysis** of the canonical GrFNN
under sinusoidal forcing. Identifies four distinct parameter regimes
with characteristic dynamical behaviors. Lets us pick parameters
*intentionally* per layer-role rather than guessing.

## Single-oscillator canonical equation (Eq. 1)

Same canonical form as Large 2010, simplified to single oscillator:

```
ż = z (α + iω + β₁|z|² + β₂|z|⁴) + RT
```

with `RT` = resonant terms (external forcing + coupling).

**δ₁ and δ₂ set to zero** throughout the paper's analysis.

## The four parameter regimes

This is the central contribution. Each regime has characteristic
autonomous behavior, bifurcation structure, and driven response.

| Regime | α | β₁ | β₂ | Autonomous | Driven (sinusoidal) |
|---|---|---|---|---|---|
| **Critical Hopf** | 0 | -100 | 0 | Decays to zero | **Always phase-locks at any freq/amp** |
| **Supercritical Hopf** | 1 | -100 | 0 | Spontaneous oscillation at one amp | Phase-lock near natural freq; SNIC bifurcation |
| **Supercritical DLC** | -1 | 4 | -1 | Bistable (zero or oscillation) | Bistability between phase-lock and slip |
| **Subcritical DLC** | -1 | 2.5 | -1 | Decays (with local max) | Always phase-locks or freq-locks; no bistability |

**Critical Hopf is the regime for ideal audio analysis** — phase-locks
to whatever input arrives, regardless of frequency or amplitude.

## Specific simulation parameters

For each regime + forcing strength tested:

**Critical Hopf** (Fig. 3): α=0, β₁=-100, β₂=0, F=0.2

**Supercritical Hopf** (Figs. 5-6):
- Weak forcing: α=1, β₁=-100, β₂=0, F=0.02
- Strong forcing: α=1, β₁=-100, β₂=0, F=0.2

**Supercritical DLC** (Figs. 7-9): α=-1, β₁=4, β₂=-1, ε=1
- Weak: F=0.1; Intermediate: F=0.3; Strong: F=1.5

**Subcritical DLC** (Figs. 10-11): α=-1, β₁=2.5, β₂=-1, ε=1
- Weak/Intermediate: F=0.1, 0.2; Strong: F=0.5

**Frequency-scaled networks** (Fig. 12, scaling demo):
- α=1, β₁=-1, β₂=-1, ε=1, F=1
- Logarithmically equally-spaced natural frequencies

## Bifurcation boundaries (analytical)

For Supercritical Hopf under WEAK forcing, the SNIC bifurcation
boundary (transition from frequency-locking to phase-locking):

```
Γ_SN = ±√(2α|β₁|)
```

For STRONG forcing, the secondary Hopf bifurcation boundary:

```
Γ_H = ± √(4αβ₁/3)
```

These tell us how strong an input needs to be to induce phase-locking
in each regime. Useful for tuning the input gain so signals reliably
cross into the phase-locked state.

## Implications for our parameter choices

**Our config: α=-0.05, β₁=-1, β₂=-1, ε=1.**

This sits closest to **Subcritical DLC** but with a much weaker β₁
(-1 vs the paper's -2.5 to -100). Effectively, our regime is a
*mild* subcritical DLC. The paper's analysis suggests this regime
"always phase-locks or frequency-locks" but with significantly
weaker bifurcation pressure.

**The patent (US 8,930,292) layered architecture** maps to:

| Layer | Patent's β₁ | Likely regime |
|---|---|---|
| 1 (cochlea) | -100 | Critical Hopf (with α=0) — sharp frequency selectivity |
| 2 (DCN/brainstem) | -10 | Mid-strength critical |
| 3 (ICC/cortex) | -1 | Mild subcritical DLC — broad integration |

**Our single layer at β₁=-1 is doing cortical-style dynamics**, which
explains why fine-grained pitch tracking is weak: we don't have a
"sharp filter" stage. Phase 3 (multi-layer) addresses this.

## Architectural notes

Single-oscillator analysis only — paper explicitly defers network-
level analysis ("Since only one oscillator is analyzed, the subscript
i in Equation (1) is dropped"). Network coupling, coupling-matrix
structure, and ASA aren't covered here.

## Notes for our implementation

1. **Phase 3 layered architecture** should target Critical Hopf for
   the cochlear layer (α=0, β₁=-100, β₂=0) — this is the
   "frequency analyzer" layer.
2. **Cortical layer** can stay close to our current config (mild
   subcritical DLC) — this is the "voice / pattern integration" layer.
3. **Brainstem layer** (intermediate) maps to β₁=-10 with α near 0.
4. **Detuning (δ₁, δ₂)** isn't analyzed here. Paper sets to zero.
   Our config also has zero. The Large 2010 demo uses non-zero δ — so
   detuning is regime-relevant. Phase 3 worth experimenting.
5. **Numerical integration**: paper uses unspecified methods. We use
   RK4 with dt = 1/16000 s. Stable for our parameter range; would
   need shorter dt at β₁ = -100 (stiff) to maintain stability.
