# Roman, Roman, Kim, Large (2023) — Hebbian learning with elasticity explains how the spontaneous motor tempo affects music performance synchronization (ASHLE)

**Reference:** PLoS Computational Biology 19(6), e1011154
**DOI:** 10.1371/journal.pcbi.1011154
**PMC:** [PMC10281589](https://pmc.ncbi.nlm.nih.gov/articles/PMC10281589/)
**Code:** https://github.com/iranroman/ASHLE

---

## Why this matters for factory-dame

A **focused, open-source Hebbian extension** that fixes a known
problem with pure GrFNN learning: oscillators that learn a stimulus
frequency drift away from their natural frequency permanently. The
**elasticity** mechanism adds a "pull-back" force toward the
natural frequency, allowing tempo adaptation that's both
responsive (learns the stimulus) and grounded (returns toward
natural rate when stimulus stops).

This is *exactly* the behavior factory-dame's rhythm bank should
have: track stimulus tempo when present, but each oscillator
remembers its own preferred rate. ASHLE is the published recipe.

## ASHLE architecture

**Two coupled oscillators**: a sensory oscillator (entrains to
stimulus) and a motor oscillator (entrains to sensory, with
elasticity toward natural rate).

Both are canonical Hopf oscillators with **adaptive natural
frequency**. The natural frequency `f` itself evolves with Hebbian
learning + elastic pull.

## Sensory oscillator equations

```
(1/f_s) · ż_s = z_s (α + i·2π + β|z_s|²) + x(t)

ḟ_s = f_s · (λ_1 sin(φ_x - φ_s) - γ · (exp((f_s - f_m)/f_m) - 1))
```

- `z_s` — sensory complex state
- `f_s` — sensory natural frequency (now time-varying!)
- `φ_x, φ_s` — phases of stimulus and sensory oscillator
- `λ_1 sin(φ_x - φ_s)` — Hebbian frequency-tracking (drag f_s toward
  stimulus rate)
- `-γ · (exp((f_s - f_m)/f_m) - 1)` — **elasticity**: pull f_s
  toward motor frequency f_m

## Motor oscillator equations

```
(1/f_m) · ż_m = z_m (α + i·2π + β|z_m|²) + exp(iφ_s)

ḟ_m = f_m · (λ_1 sin(φ_s - φ_m) - λ_2 · (exp((f_m - f_0)/f_0) - 1))
```

- `z_m, f_m` — motor state and natural frequency
- Input is `exp(iφ_s)` — unit-amplitude rotation at sensory phase
- `λ_2 · (exp(...) - 1)` — **stronger elasticity**: motor f_m pulls
  toward `f_0` (the individual's *spontaneous motor tempo*, SMT)

## Parameter values

| Parameter | Value | Function |
|---|---|---|
| α | 1 | Supercritical Hopf — spontaneous oscillation |
| β | -1 | Nonlinear damping |
| λ_1 | 4 | Hebbian tracking rate |
| λ_2 | 2 | Motor elastic pull to f_0 |
| γ | 0.02 | Sensory weak elastic pull to f_m |
| f_0 | individual-specific | Spontaneous motor tempo |

**Asymmetric pull strengths matter:** sensory has weak γ=0.02 pull
toward motor; motor has strong λ_2=2 pull toward f_0. So:
- Sensory tracks stimulus tempo accurately (only weak pull
  elsewhere)
- Motor follows sensory but with strong "homing" to its preferred
  rate

## Behavioural results

ASHLE successfully simulates three behavioural datasets:

1. **Solo with metronome** (Scheurich et al.): "asynchrony grows
   as a function of the difference between the metronome tempo and
   the musician's SMT" — performers play closer to *their own*
   preferred tempo than to the metronome's, especially at extreme
   tempos.

2. **Unpaced solo** (Zamm et al.): musicians "drifted away from the
   initial tempo toward the SMT" — when no external pulse,
   performers naturally return to their preferred rate.

3. **Duet performance** (Zamm et al.): "absolute asynchronies were
   smaller if musicians had matching SMTs" — two performers stay
   together better when their preferred rates align.

These all emerge from the elasticity mechanism. Pure Hebbian
(without elasticity) would just learn whatever frequency, with no
preference. Adding elasticity gives individual-specific tempo
biases.

## Implications for factory-dame

### Phase 2: tempo-bank natural frequencies should be sticky

For our rhythm bank, each oscillator currently has a fixed natural
frequency. ASHLE shows that *adaptive natural frequency with
elasticity* is meaningfully better than fixed natural frequency.

Implementation sketch:
- Each rhythm oscillator has `(f, z)` state where f is mildly
  adaptive
- Hebbian-like update on f when stimulus is present
- Elastic pull back to original f when no stimulus
- Result: each oscillator can drift slightly to track local tempo,
  but maintains its identity

This gives us **smooth tempo tracking** without needing a separate
KDE-peak-finding stage. The bank itself does it.

### Phase 3: motor layer with elasticity

For the predictive-beat motor layer:
- Sensory layer (rhythm bank): tracks acoustic tempo
- Motor layer: takes phase signal from sensory, has its own
  preferred rate (configurable as f_0)

This matches Large 2015 sensory+motor architecture but with the
critical addition of *motor-specific preferred rate*. For
factory-dame, f_0 could be:
- Default: a global "system tempo" parameter
- Adaptive: learned from user behaviour (if we add interaction)
- Per-voice: each extracted voice has its own preferred rate

### Open-source MATLAB → Python port path

The ASHLE GitHub repo has reference MATLAB. Porting to Python
(numba JIT same as our other code) is straightforward — only
4 equations + integrator. **Phase 2 implementation cost: low.**

### Confidence: high

This paper is **open access**, **published in 2023**, **code
available**, **validated against 3 behavioural datasets**. Among
the lowest-risk extensions to factory-dame's current code.

### Patent IP

PLoS Comp Biology is fully open access. The elasticity mechanism
*isn't* in US 8,930,292 (which covers two-frequency Hebbian
phase coherency). Adaptive natural frequencies + exponential pull
toward equilibrium isn't claimed in any active Large-lab patent
we've reviewed. **Safe to implement directly.**
