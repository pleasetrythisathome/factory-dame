# Cannon & Patel (2021) — How Beat Perception Co-opts Motor Neurophysiology

**Reference:** Trends in Cognitive Sciences 25(2), 137–150
**DOI:** 10.1016/j.tics.2020.11.002

---

## Why this matters for factory-dame

This is the **strongest alternative-framework paper for beat
perception** — it explicitly argues *against* homogeneous oscillator
models (like GrFNN's rhythm bank) and proposes a competing
neurophysiological mechanism. We need to understand it because:

1. It's the *currently dominant* explanation in cognitive neuroscience
   circles for beat-tracking (cited heavily in 2023+ work).
2. It accommodates non-isochronous beats (Balkan 2/2/3 meters etc.)
   that pure-oscillator models struggle with.
3. Our Phase 3 motor layer is the natural integration point — we
   could combine an oscillator sensory layer with a striatum-
   inspired motor layer.

## Proposed mechanism

**Supplementary Motor Area (SMA):**

> "Generates precisely patterned neural time-keeping activity that
> tracks progress through each beat interval by generating firing-
> rate dynamics tuned to arrive at a characteristic state at the
> next beat time."

SMA produces what Cannon & Patel call a "proto-action" — neural
processes providing temporal structure without specifying particular
movements. Beat anticipation = SMA reaching its characteristic
"end-of-interval" state at the predicted next beat.

**Dorsal Striatum:**

> "Functions to sequence the short proto-actions of SMA into longer
> repeating patterns, orchestrating transitions between consecutive
> beat cycles and supporting hierarchical metrical structures."

Striatal dopaminergic signals initiate each successive proto-action.
The striatum is the *clock-of-clocks* — choosing which next interval
to time, based on metrical context.

**Closed loop:**

> "SMA generates beat-anticipatory trajectories that inform auditory
> forward models for predicting upcoming sounds. Simultaneously,
> SMA activity signals dorsal striatum, which disinhibits thalamic
> populations that reinforce the next SMA trajectory."

This integrates with auditory cortex via forward-model predictions —
SMA tells auditory cortex "the next sound should be at time T."

## Distinction from oscillator models

Cannon & Patel explicitly contrast their model with dynamic-
attending / GrFNN frameworks:

> "Rather than homogeneous, autonomous cycles like pendulums...
> neurophysiology involves active input from the striatum shaping
> this activity cycle by cycle."

> "The sequential disinhibition model accommodates non-isochronous
> beat patterns more naturally than homogeneous oscillators,
> supporting rhythms like Balkan 2/2/3 meters where beats have
> different durations."

> "The framework aligns better with concatenated diffusion processes
> than oscillatory accounts."

**Key tension:** oscillators are *autonomous* — they want to keep
oscillating at their natural frequency, and external input
*entrains* them. Cannon & Patel's mechanism is *driven* — each
beat interval is set up cycle by cycle by striatal input. Their
model handles "deliberate" rhythm variation (5/8 + 3/8 = 8) better
than a pure oscillator which would have to mode-lock at the
combined ratio.

## Testable predictions (Section: testable predictions)

1. **fMRI:** metrical periodicities (slower than beat) should appear
   in striatal activation patterns.
2. **EEG:** beat-related activity should anticipate non-isochronous
   beats in enculturated listeners (Balkan musicians).
3. **Basal ganglia recordings:** human recordings should show
   activity at characteristic beat-cycle phases regardless of
   specific auditory events.
4. **Dopamine dynamics:** striatal dopamine pulses observable via
   voltammetry, elevated during strong internal beat
   representations.
5. **SMA trajectories:** ECoG should reveal firing-rate trajectories
   aligned to beat cycles; perturbations should disrupt beat
   maintenance.

## Implications for factory-dame

### Should we use Cannon-Patel instead of GrFNN?

No — but we should understand the trade-offs:

| Feature | GrFNN rhythm bank | Cannon-Patel motor |
|---|---|---|
| Isochronous beats | Excellent | Excellent |
| Tempo tracking | Excellent | Excellent |
| Non-isochronous meters | Weak | Excellent |
| Real-time inference | Streaming | Streaming |
| Mathematical tractability | Well-understood | Less formal |
| Validated experimentally | Yes (FFR, missing pulse) | Yes (fMRI, EEG) |
| Code availability | GrFNN Toolbox, factory-dame | None standard |

For factory-dame's domain (Western tonal music, mostly isochronous
or simple-meter), GrFNN is sufficient. For ethnomusicological work
or beat-variation tracking, Cannon-Patel ideas would help.

### Hybrid architecture possibility

Our Phase 3 motor layer could be informed by Cannon-Patel:

- **Sensory layer** (GrFNN rhythm bank, sub/critical) — detects
  acoustic periodicities, produces candidate-pulse signal.
- **Motor layer** (Cannon-Patel-inspired) — supercritical (or
  driven), takes sensory candidate-pulse signal as input, generates
  *next-beat anticipation* via a learned trajectory. Doesn't need
  to be an oscillator; could be a learned-trajectory recurrent
  network.

This matches Large 2015's sensory-motor coupled architecture but
with a more biophysically-motivated motor layer.

### Don't reject either framework

Both frameworks have strong experimental support. Pure GrFNN
predicts missing-pulse + FFR + mode-lock; Cannon-Patel predicts
striatal periodicities + Balkan-meter anticipation. **The truth is
probably both:** oscillator dynamics at the perceptual level
(auditory cortex), with striatal sequencing on top. Worth keeping
this in mind for Phase 6+ when we tackle non-Western rhythms.

### Reference for external positioning

When someone asks "isn't beat perception about basal ganglia and
SMA, not oscillators?" — Cannon-Patel is the paper they're citing.
Our response: "Yes, the motor-cortical mechanism does that. We
focus on the auditory-cortical resonance side. The two are
complementary; see Large 2015 sensory-motor architecture."
