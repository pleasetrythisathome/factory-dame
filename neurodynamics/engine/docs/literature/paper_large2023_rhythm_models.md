# Large, Roman, Kim, Cannon, Pazdera, Trainor, Rinzel, Bose (2023) — Dynamic models for musical rhythm perception and coordination

**Reference:** Frontiers in Computational Neuroscience 17:1151895
**DOI:** 10.3389/fncom.2023.1151895

---

## Why this matters for factory-dame

The most recent **comprehensive review** of dynamic-systems models
for beat perception. Synthesizes ~30 years of rhythm-modeling work
across multiple frameworks (Wilson-Cowan, canonical Hopf, phase
oscillators, Bayesian inference, biophysical beat-generator
circuits). Useful for Phase 2 grounding and for choosing which
model family to extend.

Does NOT introduce a single new architecture — it's a survey of
existing approaches.

## Architecture themes

- **Auditory-motor integration**: beat perception relies on
  auditory + motor pathway interactions. Auditory STG, basal
  ganglia (especially putamen), supplementary motor areas.
- **Multi-level organization**: neuronal mechanisms ↔ mid-level
  dynamics ↔ behavioral inference. Bidirectional interactions.
- **Beat generator circuit**: Bose et al. frame uses a stimulus
  neuron (S), beat generator (BG), and gamma-count comparator
  implementing period + phase learning rules.

## Equations from the review

### Phase oscillator (Sine Circle Map — Eq. 1)

```
ϕ_{n+1} = ϕ_n + 2π(q/p) + (α/2π) F(ϕ_n)
```

F(ϕ_n) = sin(2π ϕ_n) for sine circle map.

### Period adaptation (Eq. 2)

```
ṗ = β(q − p)
```

Adjusts intrinsic period p toward stimulus IOI q.

### Attentional Pulse — Dynamic Attending Theory (Eq. 3)

```
P(t) = [1 / (2π I_0(κ))] · exp(κ cos(2π (t−φ)/T))
```

Von Mises distribution; κ = precision of temporal expectations.

### PIPPET — Bayesian inference (Eqs. 4–6)

Generative model:
```
dϕ_t = θ dt + σ dW_t
λ(ϕ) = λ_0 + Σ_j λ_j exp(−(ϕ − ϕ_j)² / (2 v_j))
```

Continuous inference (mean μ, variance V of phase belief):
```
dμ_t = θ dt + [λ'(μ_t) / λ(μ_t)] V_t dt
dV_t = (σ² − [λ'(μ_t)]² V_t² / λ(μ_t)) dt
```

### Hopf normal form (Eq. 5)

```
ż = α z + b_1 |z|² z + x(t)
```

Decomposed to amplitude + phase (Eq. 6):
```
ṙ = α r − β_1 r³ + Re[b_1 e^(iδ_1) e^(−2iϕ) x(t)]
ϕ̇ = ω + Im[b_1 e^(iδ_1) e^(−2iϕ) x(t)] / r
```

Note: this is the *uncoupled* canonical form. Compare to Large 2015
Eq. 2 which adds internal coupling. This paper simplifies for
exposition.

### Hopf parameters (their reference values)

- α: bifurcation parameter (oscillation onset)
- ω: natural frequency
- β_1: nonlinear damping
- δ_1: detuning
- **Frequency range**: 0.5–8.0 Hz (matching tap-along tempos)

## Tempo / beat tracking specifics

- People re-synchronize to tempo changes "within a few beats" (~2–5
  cycles). Period adaptation rule β(q − p) handles this.
- Bose model uses I_bias adjustments for similar re-syncing.
- **Negative Mean Asynchrony (NMA)**: taps precede beat onsets by
  10–50 ms. Bose model produces NMA via asymmetric phase correction
  `δ_ϕ q(ϕ) |1 − ϕ|`.
- **Mode-locking**: p:q locking at integer ratios. Arnold tongues
  showed stability regions ↔ coupling strength α.
- **Synchronization-Continuation**: after stimulus removal, BG
  maintains rhythm through continued oscillation — tests hysteresis
  in limit-cycle dynamics.
- **Multi-level meter**: coupled oscillators at hierarchical beat
  levels; relative phases determine attentional pulse width. Phrase
  boundaries marked by phase drift.

## New developments cited beyond Large 2015 beat-perception

- **Gamma-count comparator** (Bose et al. 2019): tracks pacemaker
  cycle counts between beat and stimulus spikes, not direct interval
  measurement.
- **Biophysical beat generator**: Type I neuron with frequency-input
  curve enabling 0.5–8 Hz range; hidden asymmetries explain NMA.
- **PIPPET Bayesian inference**: explicit phase uncertainty V_t
  alongside phase estimate μ_t. Bridges Bayesian + dynamical.
- **Multi-template learning** (PIPPET extension, Kaplan et al.
  2022): library of metrical patterns, listener selects via
  inference. Cultural/developmental specialization.
- **Continuous Kalman filtering**: PIPPET uses continuous-time
  Kalman for real-time phase/tempo without discrete event sampling.

## Implications for factory-dame

### Phase 2 family choice

Three viable model families for rhythm-bank tempo estimation:
1. **Canonical Hopf GrFNN** — what we currently use. Strong
   theoretical grounding; bank-of-oscillators output is rich.
2. **Bose beat generator** — discrete-event, simpler. Better for
   pure trigger-based tempo (drum hits), worse for envelope-
   continuous music.
3. **PIPPET** — Bayesian, gives confidence intervals "for free."
   Could be a wrapper around our GrFNN output: extract candidate
   tempos from the bank, run PIPPET inference for confidence.

Current direction (canonical Hopf) is well-motivated; PIPPET wrapper
is a future enhancement.

### Frequency range

This paper confirms 0.5–8 Hz as the rhythm-bank target range. Our
config probably matches; worth verifying.

### NMA + sensorimotor delay

If we ever build a tap-along feature, plan for negative mean
asynchrony from the start. The Bose mechanism is a clean way to
include it.

### Tonal *and* rhythm GrFNN unified

This paper covers rhythm specifically. The same canonical equation
is used for pitch (Lerud 2014) and tonality (Large 2016). Suggests
a **single canonical-engine codebase** with parameter sets per
domain — exactly factory-dame's architectural choice.
