# US 8,930,292 B2 — Learning and auditory scene analysis in gradient frequency nonlinear oscillator networks

**Inventor:** Edward W. Large
**Assignee:** Oscilloscape (originally Circular Logic Inc.)
**Filed:** 2011-01-28 (priority 2010-01-29) · **Issued:** 2015-01-06
**EXPIRES:** 2032-10-13 — **active**, requires license for commercial use

[Google Patents link](https://patents.google.com/patent/US8930292B2/en)

---

## Why this matters for factory-dame

The headline patent for **multi-frequency Hebbian learning** and
**auditory scene analysis** (= voice extraction). This is the area
of the deepest IP exposure for our voice-extraction work. Most
relevant to:

- Phase 5: multi-frequency Hebbian learning rule
- Phase 4: voice extraction by reading W community structure
- Phase 3: layered architecture with critical/supercritical layers

The *concept* of layering and learning isn't claimable; the *specific*
equations and procedures are.

## Layered architecture (key disclosure)

Three-layer GrFNN modeling the auditory pathway:

| Layer | Models | β₁ | α | Regime |
|---|---|---|---|---|
| 1 | Cochlea | **−100** | 0 | Critical |
| 2 | DCN (brainstem) | **−10** | 0 | Critical |
| 3 | ICC (cortex) | **−1** | **> 0** | **Super-critical** (spontaneous) |

Other parameters (Fig. 5 of spec):
- β₂ = -1 (uniform across layers)
- δ₁ = δ₂ = 0
- ε = 1
- Time scale: oscillator-natural-frequency-relative
- Temporal window for learning: final 10 ms of network output
- Temporal window for ASA: 12.5 ms averaging window

**The layer-specific β₁ values are the load-bearing piece.** The
cochlear layer needs strong nonlinearity (β₁=-100) for sharp
frequency selectivity; the cortical layer needs weak nonlinearity
(β₁=-1) for broad integration and pattern emergence. Our single
layer at β₁=-1 is doing cortical-style dynamics for cochlear
frequency tracking — a structural mismatch.

## Multi-frequency phase-coherency learning rule

Eq. 5 of spec:
```
ċ_ij = -δ_ij · c_ij + k_ij · (z_i + √ε z_i² + ε z_i³ + ...) · (z_j + √ε z_j² + ε z_j³ + ...)
```

Eq. 6 (closed form):
```
ċ_ij = -δ_ij · c_ij + k_ij · (z_i / (1 - ε|z_i|²)) · (z̄_j / (1 - ε|z_j|²))
```

The series expansion captures coherency at *all* integer ratios.
Compare to classical Hebbian (`Δw ∝ z_i · z̄_j`) which only learns
1:1 coupling. This is the key claim. **For Phase 5, we need a
different math form** (PLV-gated update, amp-and-phase-coherency
learning, etc.) to avoid the specific equation.

## Time scales: same algorithm, different rates

> "Auditory scene analysis uses an algorithm that is fundamentally the
> same as the learning algorithm, but operating on a different time
> scale."

- **Long-term learning** (tonality, instrument identity): hours/days
- **ASA / voice separation**: tens of milliseconds to seconds

Achieved by adjusting `δ_ij` and `k_ij` parameters in Eq. 5/6.

## ASA via reading W community structure

Demo (Fig. 8 of spec): two-tone harmonic complex
- Tone 1: 500, 1000, 1500, 2000, 2500 Hz
- Tone 2: 600, 1200, 1800, 2400, 3000 Hz

After learning settles, the **connection matrix** has two distinct
clusters of strong connections — one across the 500-multiples, one
across the 600-multiples. Voices = communities in the W graph.

This is the basis of Phase 4 (W-based voice extraction). The *concept*
of clustering on the learned graph isn't claimable broadly; the
*specific* learning rule + clustering procedure is.

## Independent claims (paraphrased)

- **Claim 1**: Method including providing nonlinear oscillators,
  detecting input at first/second oscillators, comparing oscillations,
  determining multi-frequency phase coherency, changing connection
  amplitude/phase as a function of coherency.
- **Claim 6**: Decreases connection amplitude when oscillations are
  multi-frequency phase-coherent at certain ratios.
- **Claim 10**: Auditory scene analysis method that increases
  connection amplitude when oscillations exhibit multi-frequency
  phase coherency.

## What's *not* claimed (room to operate)

- The **abstract concept** of layered networks for audio analysis
- The **abstract concept** of learning-based voice separation
- Specific patent-safe alternatives:
  - PLV-gated Hebbian (use `modelock.phase_locking_value` as gate)
  - Amplitude-correlation Hebbian with external phase verification
  - Community detection on a learned-by-some-other-rule W matrix
- Different parameter regimes (we don't have to use β₁ = -100)
- Different number of layers

## Notes for Phase 3 / Phase 5

For Phase 3 (multi-layer):
- The β₁ ladder (-100 / -10 / -1) is *physiologically motivated* and
  *not patent-claimed* (the layered architecture as such isn't a
  claim — only its combination with the specific learning rule is).
- We can use these values freely — they're the right numbers for the
  layer's role.

For Phase 5 (multi-frequency Hebbian):
- Patent-safe path: build on `modelock.phase_locking_value` (already
  in our codebase). Compute PLV per pair at small-integer ratios, use
  a moving-average to track learned connection strength. Different
  from the patent's expansion-series form.
- Decision needed at commercialization time: does the expansion-series
  form give measurably better learning? If equivalent, patent-safe
  version is fine. If significantly better, need a license.

## Filing dates and expiration

| Item | Date |
|---|---|
| Priority filing | 2010-01-29 |
| US application | 2011-01-28 |
| US issued | 2015-01-06 |
| **US expires** | **2032-10-13** |
| EP counterpart (EP 2529369 B1) | 2019-12-04 issued |
| JP counterpart (JP 5864441 B2) | 2016-02-17 issued |

China counterpart **lapsed**.

7+ years remaining on the US patent. License feasibility is a 2027-
2029 question for commercial paths.
