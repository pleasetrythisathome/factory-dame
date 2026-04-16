---
reviewers:
  - scarlet@aswritten.ai
---

# NRT vs DAW/MIR and Eurorack: why the neurodynamics engine differs from conventional harmonic/rhythmic analysis

## Why this is here

Scarlet asked, before starting on the perceptual feature extractors (task-004), how the NRT-based approach compares to how DAWs and eurorack modules do equivalent work. Her words:

> "out of curiosity before you do this, how do this whole approach compare to how most DAW or even eurorack modules might go about related harmonic and rhythmic analysis"

This is a positioning question — what's the *actual* novelty of building tempo/key/consonance extractors on top of a neural resonance oscillator bank vs. doing it the conventional way? The answer is load-bearing for how factory-dame relates to adjacent fields (MIR, DAW design, modular synthesis), and it frames why the "extractors" are thin rollups rather than signal processing pipelines.

## How DAWs / MIR pipelines usually do it

**Key detection**
- STFT → chromagram (12 pitch-class energy bins per frame)
- Correlate chromagram against fixed profiles — Krumhansl-Schmuckler (1982) or Temperley profiles — across 24 keys, take argmax
- Industrial examples: Ableton Scale, Mixed In Key, Logic key detection, Rekordbox

**Tempo / beat tracking**
- Onset detection: spectral flux or energy novelty function
- Autocorrelation or comb-filter bank on novelty function → tempo histogram
- Dynamic programming (Ellis 2007) picks a beat sequence consistent with tempo + onsets
- Industrial examples: Ableton Warp, Logic Smart Tempo, madmom, aubio

**Groove / feel**
- Deviation from a quantized grid in ms (swing ratio, microtiming)
- Ableton's Groove Pool literally extracts these as templates

All of this is **linear, grid-assuming, snapshot-based**. Fundamentally it is reading a spectrogram with statistical rules on top.

## How Eurorack usually does it

Very little of this exists in eurorack — most modules *generate* rather than analyze external audio:

- **Pitch analysis is rare.** A few modules (Ornament & Crime, Expert Sleepers) do autocorrelation/YIN. Mostly pitch comes in as 1V/oct CV, not extracted.
- **Rhythmic.** Envelope followers, peak detectors, trigger extraction from threshold crossings. Grid-free but statistics-free — just "loud enough to count as a hit."
- **Harmonic.** Filter banks, modal resonators. Mutable Instruments Rings is the closest spiritual cousin to NRT: a modal resonator excited by input that has its own internal resonance behavior. But Rings is synthesis, not analysis.
- **Sync almost always comes from clock** (DIN, MIDI, gate). The module doesn't figure out tempo — the user tells it.

There is no "NRT analyzer" module in eurorack. If it existed it would likely have to be custom firmware on something like a MI hardware platform or Noise Engineering Versio.

## Where NRT differs from both approaches

1. **No fixed profile.** DAWs apply Krumhansl-Schmuckler — an empirical pitch-importance ranking from a 1982 study on Western listeners. NRT's Hebbian pitch W *learns* the tonal hierarchy from whatever music is playing. Non-Western scales, microtonal pieces, implied-but-absent fundamentals all produce coherent output without retraining or template selection.

2. **Phase, not magnitude.** Chromagrams throw phase away. NRT's oscillators HAVE phase, and that's where mode-locking lives. Consonance in NRT is not a lookup against an interval table — it's a measured dynamical property: whether two oscillators settle into a rational-ratio phase relationship. You're reading stability, not computing pairwise intervals.

3. **Grid-free tempo.** Conventional beat trackers snap to integer BPM. NRT's rhythm GrFNN lives at whatever continuous frequency wins entrainment — 1.37 Hz is fine. Polyrhythms, accelerandi, non-4/4 meters emerge as multiple stable oscillators rather than as failures to fit a grid.

4. **Phantom / missing fundamental is emergent.** DAW analyzers model the missing fundamental explicitly (harmonic product spectrum, cepstrum). In NRT it falls out of the quintic nonlinearity in the oscillator dynamics — the system generates a 100Hz response from an input containing only 200/300/400 Hz, because that's what the ODEs do. This is closer to how the ear actually works.

5. **Analysis IS the drive signal.** This is the biggest one for factory-dame's thesis specifically. In a DAW pipeline, tempo and key are metadata — data *about* the audio, not coupled to anything downstream. In NRT, the same oscillator state that produces tempo=92 is also the thing driving the particle systems / 3D forms / synthesis. There is no frame boundary between analysis and synthesis — they're the same set of ODEs integrating forward. This is the bet behind `Narrative_NonlinearOscillatorNetworks` and `Narrative_ParticleSystemsEmergence`.

6. **Prediction is real entrainment, not extrapolation.** Beat trackers predict the next beat by extrapolating recent history. NRT's motor layer (two-layer pulse network, `test_two_layer_pulse.py`) has an oscillator that has been entrained forward and continues oscillating during sensory silence. That's not extrapolation — it's a model of the motor-cortex prediction substrate continuing to run. This is the "headline NRT claim" that the test suite was designed to validate.

## What this means for the extractors being built in task-004

The extractors are thin rollups — argmax, weighted sum, ratio histogram — because **the hard work has already happened at the dynamical level**. Key detection in a DAW is a 20-line correlation loop. Key detection in this engine is `np.argmax(np.diag(w_pitch))` plus a pitch-class mapping. The reason it works is that the Hebbian dynamics have already discovered the tonal hierarchy; the extractor just reads it off.

### Tradeoffs inherited from NRT

**Weaknesses:**
- Oscillators need time to entrain. Sub-second audio won't produce clean extractions. A DAW's chromagram works per-frame with no warm-up.
- Engine tuning parameters (coupling strength, learn_rate, decay) affect extraction quality. A DAW's pipeline is self-contained — no parameters to tune upstream of analysis.

**Strengths:**
- No training, no templates, no grid assumptions. Works on any music.
- The features aren't *about* the audio — they ARE part of the signal driving downstream generative output. Same data, no additional pipeline.
- Inherits NRT's phantom-fundamental behavior, learned tonal hierarchy, and motor-layer prediction for free.

## Key claims that came out of this

- NRT analysis is a dynamical-systems alternative to statistical MIR, not a refinement of it.
- The unified analysis/synthesis property is the core bet behind factory-dame — it's why NRT matters here specifically, vs. other generative-art engines.
- Eurorack has no real analog. MI Rings is the closest in spirit but is synthesis, not analysis.
- Conventional pipelines assume grids and fixed templates; NRT assumes neither. This is why it's suited to Scarlet's interest in coupled nonlinear oscillator networks driving forms in the frequency domain.
- The extractors being built are thin because the system has already done the heavy lifting.

## Scope note

This memory is positioning/domain knowledge, not engine architecture. It should inform how factory-dame is described externally (to Taylor, in any eventual public writing, to potential collaborators) and how I orient future decisions about extractors, downstream consumers, and the Clojure/Overtone integration (`task-007`) — the same analysis-is-synthesis property is what makes OSC consumers of the engine more than conventional MIDI mappers.
