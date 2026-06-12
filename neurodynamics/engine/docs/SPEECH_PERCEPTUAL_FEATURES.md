# Speech Perceptual Features — Extending the Engine Beneath Paralinguistics

**Status — provisional scoping document. May 2026.**
**Not a phase plan; a bridge document.**

*A second domain for the neurodynamics engine: the perceptual features of
speech. The layer that captures how a voice was heard — pitch contour, voice
quality, the rhythm of delivery — beneath, and feeding, the paralinguistic
interpretation that happens elsewhere.*

---

## Why this document exists

The neurodynamics engine was built as a music instrument: audio in,
phase-locking oscillator dynamics, voice decomposition, OSC out to a eurorack
modular. `PHASE_PLAN.md` is its roadmap, and that roadmap is about music — the
corpus is electronic tracks, the perceptual extractors are key, chord, and
consonance, the pitch bank is twelve-tone equal temperament.

This document scopes a *second domain* for the same engine: speech.
Specifically, the **perceptual features of speech** — the measurable,
time-varying signal of how a voice sounded.

It exists because a separate project — aswritten.ai — turned out to need
precisely what this engine already is: a computational model of how the
auditory system hears. The two looked unrelated for a long time. One is an art
instrument that drives modular synthesizers; the other is collective-memory
infrastructure for organizations. The connection is real, and this document is
the bridge — written so the speech extension can be picked up deliberately when
its time comes, and so nothing about it is misremembered as more settled than
it is.

It does not prescribe phases, parameters, or a dependency graph. `PHASE_PLAN.md`
does that for the music track; the speech track is not ready for that level of
commitment. This is background, intent, architecture, and the boundary.

---

## The boundary — what this is, and what it is not

The extension produces the **perceptual features of speech**: the substrate. It
does **not** label, and it does not interpret.

| The engine produces — this work | Produced elsewhere — not this work |
|---|---|
| Fundamental-frequency (F0) contour over time | "She sounded uncertain" |
| Loudness / intensity contour | "That was sarcastic" |
| Formant trajectories | "He was hedging" |
| Voice-quality measures — breathy, creaky, tense, rough | "Said in passing — low conviction" |
| Rhythm, pace, pause structure, voicing | Intent. Conviction. Feeling. |

The left column is *signal* — what a careful ear registers. The right column is
*interpretation* — what the sounding meant. The engine is a **witness**: it
records how a voice sounded. It does not conclude what that meant.

The boundary is load-bearing, for three reasons.

1. **It keeps the engine reusable.** A perceptual-feature layer with no opinions
   serves any number of interpreters. The moment the engine emits "sarcastic,"
   it is welded to one theory of what sarcasm sounds like — and stops being a
   substrate.
2. **It matches the architecture on the consuming side.** aswritten settled, in
   May 2026, that interpretation of how-something-was-said is *question-relative*
   and *derived* — read fresh, in context, by a language model anchored to a
   human, never stamped at capture. An engine that stamped interpretations would
   put them exactly where that decision says they must not be.
3. **It is the honest division of labor.** Signal processing produces signal;
   meaning is made by readers. Collapsing the two is how you get confident,
   wrong summaries — the failure this entire line of work exists to prevent.

The phrase to hold: this is the **perceptual layer *beneath* paralinguistics**,
not paralinguistics itself.

---

## Context — two projects, one seam

### The engine

neurodynamics implements Neural Resonance Theory (Harding, Kim & Large,
*Musical neurodynamics*, Nature Reviews Neuroscience, 2025) as a multi-layer
cascade of coupled nonlinear oscillators: a gammatone cochlea, a brainstem
GrFNN that generates harmonics and combination tones (the frequency-following
response), a pitch / cortex GrFNN with multi-frequency Hebbian learning, and
phase-coherence voice extraction; with a parallel onset → rhythm → motor path
for tempo and the anticipated, felt beat. It is mature — 236 tests,
JIT-compiled, faster than realtime — and literature-grounded, every equation
cited.

### aswritten.ai

aswritten builds collective narrative memory: a knowledge graph of *who said
what* that organizations and individuals use to install their way of thinking
onto AI. A design arc in May 2026 established two things that matter here: the
graph records the **witnessed signals** of how something was said, and the
**interpretation** of those signals — conviction, intent, feeling — is derived
downstream by a language model, never stamped at capture. It also surfaced the
idea of an agent that listens in a live conversation and annotates
how-it-was-said in real time.

### The seam

aswritten needs to capture *how* things were said, because meaning rides on
delivery — and ordinary AI transcription destroys it. Cleanup models launder
hedges into assertions and collapse ambiguity into false fluency. To capture
delivery faithfully, you need a model of what a listener *actually hears*. That
is what an auditory-perception engine is.

The *lexical* part of "how it was said" — word choice, hedges, qualifiers —
survives in text and is available today. The *acoustic* part — intonation,
voice quality, timing — does not, and recovering it well is not a
feature-extraction afterthought. It wants a model of hearing. This engine is a
model of hearing.

The theoretical footing is solid. Neural Resonance Theory and the brainstem
frequency-following response are not music-only constructs — the speech-evoked
FFR is one of the more heavily studied objects in auditory neuroscience.
Extending an NRT engine from music to speech moves it *toward* a second native
home of its own literature, not outside its domain.

---

## Incentives

**For aswritten.** The acoustic how-it-was-said layer is the part of the product
thesis the company cannot currently deliver. Text gives lexical delivery; the
acoustic channel — intonation, voice quality, timing — is where most of
conviction and affect actually live, and it is exactly what transcription
discards. A perceptual-feature engine is the missing front-end. It is also a
*differentiator*: anyone can run an off-the-shelf prosody feature extractor, but
a model grounded in the auditory pathway computes something closer to what a
listener *perceived* — which is the entire product promise, capturing the
nuance the medium loses. And the in-meeting agent — listening live, surfacing
ambiguity — needs a live perceptual-feature stream; the engine already streams
live.

**For neurodynamics.** Speech is a large and serious domain. Today the engine is
an art instrument; speech gives it a use beyond the studio and a plausible
commercial path. It also validates the engine: if the same cochlea and brainstem
core handle speech as well as music, that is real evidence the model is doing
what it claims — modeling hearing — rather than curve-fitting one genre. The
clinical angle is genuine: speech prosody and voice quality carry clinical
signal — affect, cognitive load, voice and language disorders all have acoustic
signatures — and HIPAA-grade speech analysis is a real market. Whether to pursue
that is a separate decision; the incentive is recorded here, not chosen.

**For the founder.** It unifies two lines of work that looked separate. The
engine stops being a side project and becomes load-bearing; the company gains a
differentiated capability it would otherwise build from nothing or buy.

**Incentive alignment, via the boundary.** The "what this is not" boundary is
itself an incentive structure. Keeping the engine a *neutral perceptual
substrate* — no labels, no interpretation — means it cannot be captured by any
one product's theory of meaning. It stays an asset that serves aswritten and
remains independently reusable, independently defensible, independently
sellable. The boundary is what keeps the engine from collapsing into a single
vertical.

---

## Actors

- **Scarlet Dame** — builds and owns both neurodynamics and aswritten; the
  connection between the two projects exists, for now, in one person.
- **The neurodynamics engine** — the perceptual substrate; the witness layer.
- **aswritten.ai** — the consumer. Concretely: its extraction pipeline (records
  the feature track as witnessed signal), its paralinguistic-interpretation
  layer (prompt-level, reads the track), its in-meeting agent (the live
  consumer).
- **A commodity speech-to-text layer** — a *dependency*, not built here.
  HIPAA-compliant by selection. Supplies words, timings, and speaker
  diarization. The engine runs in parallel to it, never replaces it.
- **The downstream language model** — performs interpretation (conviction,
  intent, feeling), in context, anchored to a human. Strictly downstream of the
  engine.
- **Speakers and meeting participants** — the people whose voices are analyzed.
  A consent-relevant actor: analyzing *how someone sounded* is more sensitive
  than transcribing *what they said*, and the design must treat it that way.
- **Clinical users** — therapists and clinicians, if the HIPAA path is pursued.
  Speech prosody is diagnostically meaningful; this actor is contingent on a
  decision not yet made.
- **The NRT research lineage** — Large, Kim, Lerud, Harding and others. The
  engine's theoretical foundation; the speech-FFR literature sits within it.

---

## Architecture

How speech support slots into the existing cascade. The principle: **the
auditory periphery does not distinguish speech from music — the ear hears a
voice the way it hears a violin — so the lower layers transfer; the
music-specific assumptions live higher up; and the one genuine gap is
formants.**

### Transfers essentially as-is

- **Gammatone cochlea.** A passive frequency analysis of the auditory
  periphery. Speech-agnostic by construction.
- **Brainstem GrFNN.** Models the frequency-following response. The
  speech-evoked FFR is a primary object of the same literature — the brainstem
  layer is, if anything, *more* at home with speech than with music. Its
  harmonic and combination-tone generation is exactly what carries
  voiced-speech structure.

### Reworks — music assumptions to retire

- **Pitch / cortex layer.** Speech pitch (F0) is continuous and gliding;
  intonation is a *contour*, not notes on a twelve-tone grid. The GrFNN
  pitch-tracking machinery — oscillators phase-locking to a fundamental — works
  for speech F0; what must go is the temperament alignment and the musical
  readout. Retune the bank to the speech F0 range (~80–400 Hz typical) and read
  the contour, not a pitch class.
- **Perceptual extractors.** `extract_key`, `extract_chord`,
  `extract_consonance` are musical and do not apply. They are replaced, not
  adapted, by the speech extractors in *Features* below.
- **Rhythm / motor path.** Speech has rhythm — syllable rate, stress timing,
  pause structure — but it is not isochronous musical beat. The rhythm and
  motor layers partly transfer: the motor layer's handling of silences and its
  anticipatory dynamics are relevant (conversational timing is entrained and
  anticipated), but "tempo" and "beat" become "speech rate" and "stress / pause
  structure."

### New — the genuine gap

- **Formant analysis.** Speech carries information in *formants* — resonances of
  the vocal tract, largely independent of F0 — that define vowel identity and a
  large part of voice quality. Music analysis does not need them; speech
  analysis fundamentally does. The engine currently has no representation of
  formants. This is the one component that is genuinely additive rather than a
  retune. *How* to add it is an open question (see below): a
  vocal-tract-resonance layer reading off the cochlea, or a separate tracker
  running alongside the cascade.
- **Voicing detection.** Speech alternates voiced (periodic) and unvoiced
  (turbulent — fricatives, stops) segments. Voiced / unvoiced segmentation is a
  prerequisite for most of the other features and has no music analogue worth
  reusing.

### Composition with speech-to-text

The engine **does not transcribe.** It runs in parallel with a commodity,
HIPAA-compliant STT layer. STT produces words, timings, and speaker diarization.
neurodynamics produces the perceptual-feature track — F0 contour, formants,
intensity, voice quality, timing — time-stamped so the two align. Fusion —
attaching each utterance's how-it-was-said features to its words — happens
*downstream*, in aswritten. The engine is one of two parallel analyses of one
audio stream. It is never in the transcription business; this keeps it clear of
speech recognition both as scope and as a competitive fight.

### Output

Extend the OSC schema (or add a dedicated speech-feature output) with
per-speaker, time-stamped: F0 contour, intensity contour, formants F1–F3,
voice-quality scalars, a voicing flag, and timing / pause features — aligned to
STT word timings for downstream fusion.

---

## Features

The perceptual-feature outputs, all as time series — substrate, not labels.

- **F0 / intonation contour** — the fundamental-frequency track of voicing; the
  acoustic basis of intonation, emphasis, and question / statement shape.
- **Intensity contour** — loudness over time; the acoustic basis of stress and
  emphasis.
- **Formant trajectories** — F1–F3 vocal-tract resonances; articulation and a
  major component of voice quality.
- **Voice-quality measures** — jitter (cycle-to-cycle F0 perturbation), shimmer
  (amplitude perturbation), harmonics-to-noise ratio, and detection of creak /
  vocal fry and breathiness. Several of these are reachable from GrFNN phase-
  and amplitude-stability states the engine already computes; some need new
  extractors. These are also the clinically meaningful measures.
- **Speech-rhythm features** — speech rate, syllable-scale rhythm, stress
  pattern, and pause structure (silence is a first-class prosodic signal, and
  the engine already handles silences gracefully).
- **Voicing track** — voiced / unvoiced / silence segmentation.
- **Per-speaker segmentation** — the feature track resolved per speaker, aligned
  to STT diarization (see open questions on whether the engine's own voice
  extraction contributes here).

---

## Requirements

- Retuned F0 tracking for continuous speech pitch — speech range, gliding
  contours, no temperament grid.
- A formant-analysis capability — the one genuinely new component.
- Voiced / unvoiced / silence detection.
- Voice-quality extractors — jitter, shimmer, HNR, creak, breathiness.
- Speech-rhythm features — rate, stress, pause structure — adapted from the
  rhythm / motor path.
- A **speech validation corpus.** The music corpus does not transfer. Labeled
  speech, ideally conversational rather than only read speech, with prosodic
  ground truth. Consent and provenance of that corpus is itself a requirement,
  not an afterthought.
- A defined **integration contract** with the commodity STT layer — a shared
  audio input, STT's word / time / speaker output, the engine's time-aligned
  feature track, and where fusion happens.
- **Privacy and compliance.** If the clinical path is pursued, the STT must be
  HIPAA-compliant (this is part of why it is commodity and externally selected)
  — and the engine's own processing and outputs are PHI-adjacent: a
  voice-quality track of a patient is health-relevant data. The engine's
  deployment for any clinical use inherits the same compliance posture. This is
  a real requirement; it is not free.
- **Real-time performance** for the live / in-meeting flow. The engine already
  runs faster than realtime; formant analysis and the speech extractors add cost
  that must be measured against the live budget.
- **Patent-safe variants** for any commercial speech path. `PHASE_PLAN.md`
  already carries a patent-IP map (the two-frequency Hebbian rule under an
  active patent to 2032, network-wide tempo aggregation under another, with
  patent-safe variants documented). A commercial speech product inherits that
  map and the same "legal review before commercial deployment" caveat.

---

## User flows

### 1 — Offline / post-hoc

A recording — a meeting, an interview, a session — is processed by the commodity
STT layer and the neurodynamics engine in parallel. The outputs fuse into a
*paratranscript*: the transcript with the perceptual-feature track attached.
aswritten's memory-creation flow then works from the paratranscript
interactively with the participants — surfacing the points where the signal
leaves interpretation ambiguous, and resolving them with a human.

### 2 — Live / in-meeting

Live audio feeds the STT layer and the engine, both streaming. The engine's
existing live mode — faster-than-realtime, OSC out — is the basis. The
in-meeting agent consumes the live perceptual-feature stream, annotates
how-it-was-said in real time, and surfaces ambiguity past a threshold for
clarification. The engine supplies the *signal*; the agent decides what to do
with it.

### 3 — The consumer contract

From aswritten's side: the extraction pipeline treats the perceptual-feature
track as the **witnessed how-it-was-said signal.** It records the signal
richly; it does not interpret it. Interpretation — conviction, intent, feeling
— is performed later, by a language model, in context, anchored to a human. The
engine witnesses; aswritten records; the model, downstream, interprets. This
flow is the boundary restated as a sequence.

---

## Non-goals

- **Paralinguistic labeling or interpretation.** The headline boundary. No
  "sarcastic," no "uncertain," no conviction.
- **Speech-to-text.** STT is commodity, external, HIPAA-compliant. The engine is
  not in the recognition business — neither as scope nor as a competitive fight.
- **The in-meeting agent itself.** That is an aswritten product; this work is
  the perceptual-feature engine it consumes.
- **Regressing the music capability.** Speech is an *added* domain. The music
  track and `PHASE_PLAN.md` continue unchanged.
- **A phase plan.** This document is scoping. Phasing comes later, deliberately,
  when the speech track is committed to.

---

## Open questions

- **Formant analysis approach.** Extend the GrFNN cascade with a
  vocal-tract-resonance layer, or run a separate formant tracker alongside it?
  Worth a literature pull — whether the Large / Lerud lineage, or the
  speech-FFR literature within it, treats formants in a resonance framework.
- **Speaker separation.** Rely entirely on STT diarization, or invest in the
  engine's phase-coherence voice extraction for it? `PHASE_PLAN.md` is honest
  that voice extraction over-clusters even on dense music; conversational speech
  with crosstalk is harder still. Provisional answer: STT diarization for the
  first version; engine voice extraction is a later question.
- **The speech corpus.** Which datasets; conversational vs. read; how consented
  and clinical data are handled.
- **The STT layer.** Which commodity provider; the HIPAA path; the exact
  integration contract.
- **Where fusion happens.** In the engine's output, or in aswritten?
  Provisionally aswritten — the engine emits a clean feature track and stays
  unaware of words.
- **Real-time cost.** Formants plus the speech extractors against the live
  budget — unmeasured.
- **Patent review timing** for a commercial speech path.

---

## Sequencing and status

This is a **later-stage track.** aswritten's near-term how-it-was-said work is
text-only — lexical features, hedges, no audio at all. The speech
perceptual-feature layer matters when aswritten reaches the audio stage, which
is likely post-fundraise. This document exists to **bank the design** so that
stage is inexpensive and unscary to begin — not to start it now.

Its relationship to `PHASE_PLAN.md`: the music plan continues independently and
is unaffected. The speech track shares the engine core — cochlea, brainstem, the
GrFNN machinery, the JIT and performance work — and some music-track work
(voice-extraction depth, performance) benefits speech as a side effect. But the
speech track has its own validation corpus, its own extractors, and the new
formant component. It is not a phase of the music plan; it is a sibling track
that shares an engine.

When the speech track is committed to, this document should be promoted into a
proper phase plan of its own, with the same rigor `PHASE_PLAN.md` applies to
music: literature pulled per component, parameters cited, validation tests
defined before implementation.
