# US 8,583,442 B2 — Rhythm processing and frequency tracking in gradient frequency nonlinear oscillator networks

**Inventor:** Edward W. Large
**Assignee:** Oscilloscape (originally Circular Logic Inc.)
**Filed:** 2011-01-28 (priority 2010-01-29) · **Issued:** 2013-11-12
**EXPIRES:** 2031-04-02 — **active**, requires license for commercial use

[Google Patents link](https://patents.google.com/patent/US8583442B2/en)

---

## Why this matters for factory-dame

Covers the **rhythm/tempo extraction mechanism** we use. Most relevant to:

- Phase 2: network-wide tempo aggregation
- Phase 3: layered architecture (rhythm bank as a separate layer)
- Router: per-voice rhythm/tempo CV outputs

Active patent — the **specific equation forms** are claimed. We can
study them and implement equivalent behaviors via different math
(see Phase 2 plan), but direct lift requires a license.

## Independent claim 1 (paraphrased)

A network of m nonlinear oscillators with different natural
frequencies, each obeying coupled differential equations governing
amplitude rate, phase rate, and frequency rate. Input signal x(t) is
applied; frequency output describes the time-varying structure.

Discrete-time formulation (claim 2) computes oscillator state only at
discrete event onset times — fast computation for impulsive sequences.

## Frequency-tracking equation (Eq. 5 of spec)

```
ω̇ = -k · x(t) · sin(φ) / [ε·r² - 2εr·cos(φ) + 1]
```

This is the per-oscillator natural-frequency adaptation rule. In live
mode, "every oscillator in the network provides an estimate of how
frequency should change. Estimates are weighted by amplitude, and
combined to form a global estimate."

**This is the canonical patent-claimed network-wide tempo aggregation.**
Our Phase 2 implements the *concept* (amp-weighted aggregation of
per-osc tempo estimates) using different math (KDE-peak in log-BPM
space), avoiding the specific claim.

## Predictive beat tracking

Expected event time (Eq. 7 of spec):

```
tx_{n+1} = tx_n + T_{n+1} - [c · s_n / (2πf)] · (1 - √ε · r_n) · sin(φ_n) / [ε·r_n² - 2ε r_n cos(φ_n) + 1]
```

"Expected event time (tx) keeps pace with real time, and allows the
system to output perceived musical beats."

Predictive — oscillator state extrapolates forward to anticipate the
next beat. Our motor layer (test_two_layer_pulse) is the
factory-dame implementation; its mechanism is conceptually similar
but mathematically different.

## Mode-locking demonstrated

Fig. 5 of the spec shows the discrete-time system responding "at
higher order resonances" — explicitly demonstrates response at
frequency ratios `1:1, 2:1, 3:1, 1:2, 3:2`. "An important feature of
nonlinear oscillator networks."

## Layer architecture noted

"Layers of one-dimensional arrays of nonlinear oscillators" with
"feedback...between oscillator layers, inputs from oscillator layers
both above and below the subject oscillator." Two-layer feedback is
mentioned in the spec without being claimed in the rhythm-processing
context — meaning our multi-layer architecture (Phase 3) is on safer
ground if implemented as a general property and not specifically tied
to rhythm processing.

## Specific parameter notes

| Parameter | Value/range |
|---|---|
| α | analyzed at α=0; can be positive or negative |
| β1, β2 | β1 < 0, β2 > 0 (signs specified for stability) |
| δ1, δ2 | assumed 0 in derivation |
| ε | constraint 0 ≤ r < 1/√ε at each step |
| c, k | not numerically specified |

## What's *not* claimed

- The **concept** of "network-wide tempo aggregation" — claims
  specify the *equation*. Different aggregation math (median of
  per-osc votes, KDE peak, neural-net regression on rhythm-bank
  state, etc.) is fine.
- **Motor coupling** — explicitly NOT mentioned in this patent. Our
  motor layer (sensory→motor bidirectional coupling, predictive beat
  via super-critical motor oscillators) appears to be unencumbered
  ground.
- **Discrete-time vs continuous-time formulation** — discrete claimed
  in claim 2. Continuous-time, which we use, is in the broader claim 1.

## Notes for Phase 2

Our patent-safe network-wide tempo plan: amp-weighted KDE peak in
log-BPM space, recomputed every snapshot. Each rhythm-bank oscillator
votes for its natural frequency as the candidate tempo, weighted by
its amplitude. Smooth peak-tracking across snapshots. Different from
this patent's specific `ω̇ = -k·x·sin(φ)/...` equation.
