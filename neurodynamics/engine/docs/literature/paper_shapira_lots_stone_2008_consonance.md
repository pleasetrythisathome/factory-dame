# Shapira Lots & Stone (2008) — Perception of musical consonance and dissonance: an outcome of neural synchronization

**Reference:** Journal of the Royal Society Interface 5, 1429–1434
**DOI:** 10.1098/rsif.2008.0143
**PMC:** [PMC2607353](https://pmc.ncbi.nlm.nih.gov/articles/PMC2607353/)

---

## Why this matters for factory-dame

A **non-Large-lab** paper that independently validates the GrFNN
mode-lock-stability claim. Uses simpler integrate-and-fire (IF)
neurons, not the canonical Hopf model, and still produces the same
result: **mode-lock stability widths predict consonance ordering.**

For factory-dame: this paper is the *external citation* for "voice
clustering by mode-lock stability isn't a Large-lab idiosyncrasy —
it's a property of coupled oscillators generally." Useful for
external positioning.

## Oscillator equations (integrate-and-fire)

Two coupled IF neurons:

```
dx_1/dt = -x_1/τ_1 + I_1 + ε · E_1(t)
dx_2/dt = -x_2/τ_2 + I_2 + ε · E_2(t)
```

- `x_1, x_2` — membrane potentials
- `τ_1, τ_2` — decay constants
- `I_1, I_2` — external drive
- `ε` — coupling strength
- `E_1, E_2` — coupling functions (post-synaptic potential pulses
  from the other neuron's spikes)

When `x_i` reaches threshold α, it fires and resets to 0.

## Parameter values

Stability-interval calculation: `ε=5, α=100, τ_1=τ_2=1`.
Simulations (highlighting mode-locked states): `ε=8`.

## Stability-interval result (Table 1 paraphrased)

The **width of the mode-lock interval ΔΩ** for each frequency ratio
predicts consonance ordering. Wider ΔΩ = more consonant.

| Interval | Ratio | ΔΩ (relative) | Western consonance |
|---|---|---|---|
| Unison | 1:1 | 0.075 (widest) | Most consonant |
| Octave | 1:2 | next-widest | Perfect consonance |
| Perfect fifth | 2:3 | wide | Perfect consonance |
| Perfect fourth | 3:4 | wide | Perfect consonance |
| Major third | 4:5 | narrower | Imperfect consonance |
| Minor third | 5:6 | narrower | Imperfect consonance |
| Major second | 8:9 | narrow | Mild dissonance |
| Minor second | 15:16 | narrowest | Strong dissonance |

**This ordering matches Western music theory's consonance
hierarchy.** Achieved without any music-theoretic input — purely
from the dynamics of coupled oscillators.

## What this paper proves (vs Large lab claims it)

Large 2016 uses the same logic but with the canonical Hopf model
and a closed-form formula `stability ≈ ε^((k+m-2)/2)` (Eq. 2).
This paper uses *different math* (IF neurons + Arnold tongue width)
and arrives at *the same conclusion*.

Independent confirmation that:

1. **Coupled oscillators naturally produce a consonance hierarchy.**
2. **The mechanism is mode-lock stability**, not anything
   psychoacoustic or culturally learned.
3. **Different model families converge on the same answer** — this
   is robust physics, not an artifact of one choice of equations.

## Implications for factory-dame

### Cross-validation for Phase 4 voice extraction

Our voice clustering relies on detecting harmonic structure via
small-integer-ratio relationships. This paper proves that's what
coupled oscillators naturally do *regardless of the specific
oscillator model.* So even if we change our oscillator equation
later (to a more biophysical form), voice clustering should still
work as long as the new equation produces stable mode-locks.

### Cross-validation for Phase 5 Hebbian

If learned W reflects mode-lock stability (Large 2016's Eq. A2),
then the W ordering will match Western consonance — which means
W community structure will correctly group harmonic partials of
the same source. The Hebbian rule isn't doing something arbitrary;
it's discovering the consonance hierarchy that this paper showed
is intrinsic to coupled oscillators.

### Model-agnostic mode-lock primitive

Our `modelock.phase_locking_value` detects 1:1 (and easily extends
to 1:2, 2:3, etc.). The detection is meaningful regardless of which
oscillator model we use — confirmed by this paper's IF-neuron
demonstration. PLV is a robust voice-membership feature.

### Citation utility

Use this paper when explaining factory-dame to skeptics: "voice
clustering by mode-lock stability is established physics — see
Shapira Lots & Stone 2008 for the cleanest independent demonstration
that doesn't depend on any music-specific assumptions."
