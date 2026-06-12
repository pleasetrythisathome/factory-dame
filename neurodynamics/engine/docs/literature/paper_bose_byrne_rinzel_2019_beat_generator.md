# Bose, Byrne, Rinzel (2019) — A neuromechanistic model for rhythmic beat generation

**Reference:** PLoS Computational Biology 15(5): e1006450
**DOI:** 10.1371/journal.pcbi.1006450
**PMC:** [PMC6508617](https://pmc.ncbi.nlm.nih.gov/articles/PMC6508617/)

---

## Why this matters for factory-dame

The **specific computational model** behind Cannon & Patel's
theoretical proposal — and the alternative beat-generator that
Large 2023 lists as a viable rhythm-perception framework. Concrete
equations, learning rules, parameter values. Useful for Phase 2 if
we want to compare GrFNN rhythm bank against this approach.

## Core architecture

Four components:
- **S (Stimulus neuron):** faithful spike from each stimulus onset
- **BG (Beat Generator):** leaky integrate-and-fire neuron (or
  INaP biophysical variant), produces a periodic spike at the
  anticipated beat
- **γ_BG, γ_S (Gamma counters):** 36.06 Hz oscillators counting
  cycles between BG spikes and between stimulus spikes respectively
- **GCC (Gamma-Count Comparator):** compares γ_BG and γ_S, drives
  learning rules

## Beat-generator equation (LIF)

```
v' = (I_bias - v) / τ
T  = τ · log(I_bias / (I_bias - 1))
```

- `v` — membrane potential
- `I_bias` — drive parameter (sets the BG's frequency)
- `τ` — membrane time constant
- `T` — period (closed-form)

The BG fires when `v` reaches threshold, then resets.

For more realistic biophysics, the **INaP variant** uses persistent
sodium, T-type calcium, sag, and leak currents — same f-I behavior
but biologically detailed.

## Learning rules

**Period Learning Rule (LR_T):** applied at each BG spike.

```
I_bias → I_bias + δ_T · (γ_BG - γ_S)
```

If BG counted more gamma cycles than S (BG is slow), I_bias goes
up → BG speeds up. And vice versa.

**Phase Learning Rule (LR_φ):** applied at each S spike.

```
I_bias → I_bias + δ_φ · q(φ) · φ · |1 - φ|
```

Where:
- `φ = CC_BG / γ_S` — phase alignment (0–1)
- `q(φ) = sgn(φ - 0.5)` — sign of phase error
- `|1 - φ|` — asymmetric weighting (key for negative mean asynchrony)

This drives BG to fire synchronously with stimulus onsets.

## Parameter values

| Parameter | Value | Description |
|---|---|---|
| δ_T | 0.2 | Period learning step size |
| δ_φ | 2.5 | Phase learning step size |
| Gamma frequency | 36.06 Hz | Period 27.73 ms |
| Stimulus range | 1–6 Hz | 166 ms to 1 s intervals |
| Synchronization criterion | ±1 gamma cycle | ±27.73 ms |
| Resync criterion | 3 consecutive BG spikes within 1 gamma cycle of S | Robustness threshold |

**Notable:** the gamma counter operates at ~36 Hz, which is in the
biological gamma-frequency range. The model proposes gamma
oscillations as the *substrate* for counting — beat-period
discrimination = "how many gamma cycles fit in this interval."

## Behaviour

- **Tempo tracking** within a few cycles of tempo changes
- **Negative Mean Asynchrony (NMA)** naturally emerges from the
  asymmetric phase rule (`φ · |1 - φ|` is asymmetric around 0.5)
- **Synchronization-Continuation:** after stimulus ends, BG keeps
  firing at the learned period — tests *memory*, not just
  entrainment
- **Range:** 1–6 Hz is the typical tap-along range (matches behavioral
  data)

## Implications for factory-dame

### Phase 2: alternative tempo-estimation mechanism

For the rhythm bank, we currently use amp-weighted KDE-peak-in-
log-BPM. Bose et al.'s mechanism is a different family:
- *Discrete-event* (operates at stimulus onsets, not continuous
  envelope)
- *Single-tempo* (one BG, not a bank of oscillators at all tempos)
- *Memory-capable* (synchronization-continuation, not just
  entrainment)

For factory-dame's continuous-audio mode, the bank approach is
better. For an onset-detected drum-loop mode (where we already
have S-like onset spikes), Bose et al. is competitive.

Worth keeping in mind for Phase 6+ if we add drum-machine-style
inputs.

### Motor-layer implementation reference

This is the most concrete computational model of a *single*
beat-generator neuron. If we build a Cannon-Patel-style motor
layer in Phase 3, this paper gives concrete equations to start
from. We'd drop the gamma-counter (use direct continuous timing
instead) but keep the learning-rule structure.

### Patent-IP read

Bose, Byrne, Rinzel are **independent of Large lab** for this
specific work. Their model is in the public scientific literature,
not patented. **Safe to implement directly** if we choose this
beat-generator architecture for any commercial product.

### Negative Mean Asynchrony

The asymmetric `φ · |1 - φ|` term reproduces the empirically-
observed 10–50 ms tap-precedence. If factory-dame ever does
tap-along output, we should include this asymmetry from the start
— it's a known signature of human beat-tracking.
