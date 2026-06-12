# Tal, Large, Rabinovitch, Wei, Schroeder, Poeppel, Zion Golumbic (2017) — Neural Entrainment to the Beat: The "Missing-Pulse" Phenomenon

**Reference:** Journal of Neuroscience 37(26), 6331–6341
**DOI:** 10.1523/JNEUROSCI.2500-16.2017

---

## Why this matters for factory-dame

This is the **direct experimental validation** that the brain
generates oscillation at frequencies *physically absent* from the
stimulus envelope — the rhythm-domain analogue of the phantom
fundamental. MEG data shows pulse-frequency phase-locking even when
the stimulus has no acoustic energy at the pulse frequency.

Our phantom-fundamental + missing-pulse architecture goal traces
directly to this paper. It's the empirical answer to "is GrFNN
making things up, or is the human brain really doing this?" — the
human brain really does this.

## Stimulus design

Syncopated rhythm sequences ("MP1", "MP2") were constructed such
that **no acoustic energy was present at the pulse (beat) frequency**
(2 Hz), but listeners reliably perceived a 2 Hz pulse. An
isochronous control had energy at 2 Hz directly.

The stimulus is a Fourier-controlled rhythm: spectral energy is
distributed at off-pulse frequencies in the missing-pulse condition,
with the listener's perceptual pulse emerging *internally*.

## MEG findings

Listeners produced enhanced auditory-cortex MEG power at 2 Hz when
listening to the missing-pulse rhythms — even though 2 Hz was absent
from the acoustic stimulus.

Statistical results:
- Strong negative correlation between time-to-tap (TTT) and 2 Hz
  amplitude: `r_s = -0.82, p < 0.01` (faster tappers show stronger
  neural pulse).
- ~30% of MEG sensors showed significant phase-locking at the
  soundless strong-beat positions (FDR-corrected p<0.05).
- Right superior temporal gyrus showed significant missing-pulse
  response (p<0.05); left STG approached significance (p=0.071).
- Motor regions showed **no** significant pulse-frequency response
  (p > 0.18). The pulse generator they observed is in auditory
  cortex — sensory layer, not motor.

## Theoretical interpretation

> "Nonlinear interactions give rise to oscillatory activity not only
> at the frequencies present in the stimulus, but also at more
> complex combinations, including the pulse frequency."

Phase-locking at temporal positions where no acoustic stimulus
existed proves the activity is internally generated, not stimulus-
driven. This rules out simple linear envelope-following and matches
GrFNN predictions.

## Implications for our implementation

### Phase 2: rhythm bank

Validates that a rhythm-bank GrFNN should produce strong activity
at pulse frequencies even when the stimulus envelope lacks energy
there. This is *the* experimental criterion for "does our rhythm
bank work right?"

A specific test: build a syncopated stimulus à la MP1/MP2, run it
through the rhythm bank, and check that the bank settles to the
perceived-pulse oscillator. If the bank only follows envelope
energy, it's not GrFNN-correct.

### Architectural note: pulse is sensory, not motor

The motor regions did NOT show pulse-frequency entrainment in this
study (p > 0.18). The internally-generated pulse lives in auditory
cortex (sensory layer). This is consistent with Large 2015's
**sensory layer = subcritical/critical**, **motor layer = supercritical**:
the *pulse perception* happens in sensory; the motor layer just
*tracks* the sensory pulse to drive movement.

For factory-dame: the rhythm bank that generates the missing-pulse
response is the **sensory** rhythm bank, not the motor predictor.
Both can exist; they have different roles.

### Connection to phantom-fundamental in pitch

The pitch-domain phantom fundamental (e.g., 200+300+400+500 Hz →
perceived 100 Hz) and the rhythm-domain missing pulse are the *same
phenomenon at different timescales*. Same generative mechanism
(nonlinear resonance + canonical input transformation). Phase 1.x
canonical-form refactor should produce both.

### Trained vs untrained listeners

Time-to-tap correlated with 2 Hz amplitude — suggests individual
differences in nonlinear-coupling parameters (likely ε and/or
Hebbian-modulated coupling) explain why some people lock to a
syncopated beat instantly and others don't. Same finding as Lerud
2014 (musicians have higher ε for harmonic mode-locking).

## Relation to other papers

- **Large 2015 beat perception**: predicts the missing pulse
  computationally; this paper validates it experimentally.
- **Lerud 2014 brainstem**: the pitch-domain analogue —
  brainstem mode-locking generates response components not present
  in the acoustic stimulus, fitted at 68% variance explained.
- **Patent US 8,583,442**: covers the *machinery* for rhythm-bank
  pulse generation. This paper is the empirical validation of why
  that machinery is desirable.
