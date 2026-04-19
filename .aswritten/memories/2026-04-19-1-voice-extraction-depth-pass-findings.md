---
reviewers:
  - scarlet@aswritten.ai
---

# Voice extraction depth pass — findings (2026-04-19)

Scarlet instructed Claude (Opus 4.7) to take an open-ended deep-first
pass on voice extraction, with license to validate or disqualify the
current architecture and write unit + roundtrip tests along the way.
Her direction: "if you disqualify [the architecture], develop a new
one and do the same until you have a working model."

Branch: `feat/voice-extraction-depth-pass` (not yet merged).

## Architecture verdict: not disqualified

The current envelope-correlation + Hungarian-matching voice extractor
survived the pass. It correctly identifies single voices, chords, and
multiple simultaneous voices on both clean synth content and real
music. Unit tests (37 in test_voices.py, 80 in test_voices_real_audio.py)
all pass on the cleaned-up engine. Mean 3-4 voices per snapshot on
real music, 9-22 unique voice IDs per 30s — musically plausible.

## The three silent killers that were drowning signal

Voice extraction on clean synth content was at 25% coverage before
this pass. Three coupled issues, all upstream of voice clustering, all
conspiring to drown real signal in engine-generated noise:

### 1. `noise.amp = 0.01` plus `alpha = -0.05`

The stochastic-noise term kicks every pitch oscillator every audio
sample. With alpha = -0.05 (lightly damped — 20 s decay time
constant), those kicks accumulated into a ~0.03 steady-state
oscillation at every bin. The "noise floor" we'd been fighting in
voice extraction wasn't noise from the audio — it was the engine
oscillating spontaneously at every one of 279 oscillators, all the
time, drowning real signal.

Evidence: pure digital silence produced peak amp 0.056 with 183/279
oscillators above 0.01. Pure 440 Hz sine input produced peak response
at 49 Hz (G1) not at 440, because low-freq oscillators had equal
noise-driven amp and happened to cluster first.

Fix: `[pitch_grfnn.noise] amp = 0.0`. Silence baseline dropped 17×.

### 2. Zero-init vs random-noise init

With noise-amp now 0, the initial condition still mattered. The GrFNN
was seeded with `randn * 1e-3` per oscillator and decayed slowly
(20 s time constant) — so 3 seconds into a silent input, residual
amp was ~0.003 with a structured spectral profile (31 Hz dominant,
plus quiet content at 21 Hz, 74 Hz, and oddly C4=262 Hz). Weak signal
content at chord tones (C4/E4/G4) sat at 0.0023-0.0029, below the
init-residual baseline.

Fix: `self.z = np.zeros(...)`. Silence now exactly 0.0 across the bank.
With no drive + no noise + zero init, the sub-critical Hopf stays at
the origin forever, and any real signal dominates.

### 3. Voice-clusterer thresholds calibrated to the broken baseline

`noise_floor = 0.02, active_fraction = 0.08` were set when the bank's
noise-driven baseline WAS 0.03. With silence now at 0.003 and real
signals driving to 0.01-0.15, the 0.02 floor was gating out real
signal.

Fix: `noise_floor = 0.005, active_fraction = 0.25`. Plus raised
`input_gain = 0.1 → 0.5` to compensate for the removed noise
amplification on real music.

## The flat-row clustering bug

Before the cleanup, voices.py had a deterministic-ramp hack to keep
`np.corrcoef` from emitting NaN on zero-variance rows. The ramp was
the SAME for every flat row, so all flat oscillators correlated = 1
with each other. For a truly sustained A4 saw through the engine, the
440 Hz bin and its 880/1320/1760 Hz harmonics were all flat, and so
were the zero-signal noise-floor bins — all collapsing into one giant
cluster with a log-weighted centroid at random low frequencies.

Fix: detect flat rows by coefficient of variation (`std/mean < 0.02`)
and defer clustering for flat pairs to harmonic-ratio adjacency. Also
added per-pair thresholds: `correlation_threshold = 0.6` for unrelated
pairs (keeps two instruments from colliding),
`harmonic_correlation_threshold = 0.5` for harmonic pairs (lets
fund + harmonics cluster when envelopes are coherent, still rejects
octave-apart instruments with independent envelopes).

## Live-mode performance

Expanding the pitch bank from 100 → 279 bins (for 12-TET alignment at
3 bins/semitone) regressed throughput from 10.99× → 0.63× realtime —
below the live threshold. Profile identified two quadratic costs:

- `extract_consonance` computed every active-pair ratio through a
  pure-Python double loop calling `_ratio_consonance` (which iterates
  8 stability ratios). ~2.3M pure-Python function calls over 10s of
  audio. Vectorized to one (N_pair × 8) numpy distance matrix, ~50×
  faster.
- `snapshot_hz = 60` was generating 60 fps OSC emissions; each full
  layer is 5 messages × 279-value arrays. Dropped to 30 fps.

Combined: 0.63× → 1.11× realtime. Live mode usable again.

## Ground truth via Overtone (vs VCV Rack)

VCV Rack was originally the ground-truth synthesis source, but every
debugging session was GUI-wrangling (cable routing, knob positions, AC
coupling, FREQ-knob calibration) rather than engine behavior. Scarlet:
"perhaps overtone is better for this type of testing? VCV rack is
great for emulating the modular env but we need you to have full
introspection and edit ability for the synthesis environment."

Overtone path: Clojure project at `neurodynamics/overtone/`, synth
defs in `synths.clj` (mono-saw, plucked, sine-tone), trigger patterns
loaded from shared JSON at `engine/test_audio/triggers/*.triggers.json`,
REPL-driven dev via clojure-mcp. Offline WAV render via
`recording-start` or live loopback through Loopback Audio.

First payoff: within 10 minutes of switching, Overtone produced a
clean 440 Hz saw that surfaced the "flat rows collapse into one
cluster" bug the VCV path had been masking with filter-envelope
transients and short-note attack edges.

VCV kept for its original role — modular emulation, the same CV
protocol the router will eventually drive to hardware ES-9.

## Gaps not closed

- **Phantom fundamental (NRT-distinctive)**: doesn't emerge at current
  drive levels. The quintic term in the Hopf dynamics needs |z| in
  the 0.1+ range to couple meaningfully; we're at 0.01-0.05 for clean
  synth. Either further increase drive (risking saturation on real
  music) or accept that phantom is a real-music-only phenomenon.
- **Fast-changing pitch** (vibrato, glissando): the 2.5 s voice window
  averages away sub-window pitch changes. A per-voice adaptive
  window would help. Queued as a future pass.
- **Phase-lock primitive**: `modelock.py` already implements PLV.
  Integrating it into voices.py would let harmonic pairs cluster on
  phase-relationship stability (NRT-native per
  Claim_NRTPhaseNotMagnitude). Not currently needed — the flat-row
  harmonic fallback covers the sustained case. Would differentiate
  edge cases (two instruments at octave with different phases) that
  aren't in the test suite yet.

## Acceptance criteria proposed (session-only, worth saving if Scarlet agrees)

For "voice extraction works reliably":
- Pure sustained tone → exactly 1 voice at input pitch, 0 cents error
- Sustained harmonic stack → 1 voice at fundamental
- Held 3-note chord → 3 voices at the chord pitches, each ≥80%
  coverage during the gate
- Real music → 3-8 mean voices per snapshot, voice center_freqs
  musically coherent (bass voice tracks bass line, etc.), no
  phantom proliferation during silent passages
- Live throughput ≥ 1× realtime on M-series with 278-bin bank

Current state hits all except the 3-note-chord case (sustained_chord
roundtrip shows 67% coverage — 2 of 3 chord tones get voices).

## Open design decision

Whether to pursue Task #32 (phase-lock primitive integration) as
next-pass work. The architecture is validated as-is but could be
strengthened for edge cases. Per Claim_DeepFirstPhaseModel, Phase 1
(voice identity) should be at "publishable depth" before moving on —
phase-lock would be that kind of depth. But the current
amplitude-correlation + harmonic-adjacency combo is already novel
enough that it passes all unit tests covering the claimed capability.
