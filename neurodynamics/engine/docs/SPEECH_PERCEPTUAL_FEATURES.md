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

## Prior Art and Research

This document scopes a perceptual-feature layer. It is not the first attempt to
name the perceptual features of voice — and the strongest prior art is not
academic.

### Operationalized perceptual-feature frameworks — trans voice pedagogy

Trans voice training has, out of practical necessity, produced the most
operationalized vocabulary of voice perception that exists: a teacher there must
give a learner *names* for perceptual features precise enough to consciously
reshape them. The anchor reference is **Zheanna Erose / TransVoiceLessons** — a
trained music composer who built her framework by applying acoustics and music
theory to voice.

The load-bearing contribution is her **three-lens model**: every voice feature
is described across three frames at once —

- **Sensory** — what it sounds like; the direct perceptual experience; the most
  actionable frame for a human
- **Acoustic** — what is measurable in the signal
- **Biomechanical** — what physically produces it

— and each feature is mapped across all three. *Weight*: sensory
"heavy / light / dense" ↔ acoustic spectral tilt ↔ biomechanical vocal-fold
mass. *Resonance / Size*: sensory "perceptual size" ↔ acoustic
resonance-frequency shift ↔ vocal-tract chamber size. *Pitch*: high / low ↔ f0
↔ vocal-fold oscillation rate.

That cross-walk is the bridge this engine layer needs. The cascade produces the
*acoustic* column; the paralinguistic interpretation downstream needs the
*sensory* column — the words a listener actually uses. The three-lens model is a
hand-built, pedagogically-validated mapping between them, for the voice domain.
It also independently arrives at source-filter theory, the harmonic series, and
resonance-as-eigenfrequency — it already expresses voice perception in the
music/acoustics vocabulary the GrFNN engine uses.

**Honest scope.** The framework is organized around *gender perception*. That
does not make it niche: vocal weight, resonance, and the gendered quality of a
voice are live conversational signals, modulated moment to moment, carrying
affect, stance, and emphasis — they belong in the feature set. But the framework
is *not an exhaustive map* of perceptual speech features; it under-develops
prosody, intonation contour, rhythm, and other perception types. Take the
**framework** — the three-lens cross-walk method — as fully generalizable; take
the **feature set** as a real, conversational, but partial slice. The rest of
the space is what the research brief below points at.

The concrete artifact this implies: a percept ↔ acoustic **translation table** —
Erose's terms (weight, size, R1) cross-walked to standard acoustic-phonetics
(spectral tilt, formant-frequency shift, F1). Small, bounded, and genuinely the
join between the engine's output and the interpretation layer's vocabulary.

### Research brief — digging deeper

Three literatures bear on this layer. This brief scopes them; a dedicated
research pass should produce the vetted bibliography.

1. **The trans-fem voice corpus.** Anchor: the TransVoiceLessons curriculum
   (free and structured — "The Art of Voice Feminization" series) and the
   Erose & Grigsby book-in-progress *The Art of Voice Alteration*. Also
   AcousticGender.Space, which operationalizes the framework as an explicit 2-D
   pitch × resonance perceptual space. Beyond Erose, the **SumianVoice Voice
   Resource Project** (`wiki.sumianvoice.com`, with a real-time spectrogram tool
   at `spec.sumianvoice.com`) carries the broadest operationalized perceptual
   taxonomy of the lot — pitch, resonance, weight, and a whole "clarity"
   cluster: creak, breathiness, nasality, onsets, subharmonics,
   false-vocal-fold control. SumianVoice is also the bridge to item 2: its
   author, Sumi Koshin, co-authored *Speech After Gender*. Value: the most
   operationalized percept-side feature vocabulary available anywhere — and
   SumianVoice's taxonomy is the most complete single map.

2. **Academic research contextualizing trans voice in speech science.** Anchor:
   *Speech After Gender: A Trans-Feminine Perspective on Next Steps for Speech
   Science and Technology* (arXiv 2407.07235, 2024) — which independently adopts
   the pitch / resonance / weight triad and notes that trans-feminine voice
   teachers "have unique perspectives on voice that confound current
   understandings." The academy reaching toward what the pedagogy already
   operationalized; its citations are the entry point into the formal
   literature.

3. **Prosody and paralinguistics research.** The doc's largest coverage gap, and
   where the deepest academic literature exists — the perception of intonation
   contour, prosodic rhythm, stress, and of affect and stance in speech. The
   auditory-mechanism side is already anchored by the engine's NRT / GrFNN
   lineage (see the `docs/literature/` folder); the *speech-prosody* extension
   of it is unwritten here. A research pass should pull prosody perception,
   affective / paralinguistic prosody, and computational paralinguistics.

The agenda for that pass: for each perceptual feature the interpretation layer
will need, fill all three lenses (sensory / acoustic / biomechanical), and flag
where the academic literature and the voice pedagogy carve the feature
differently.

---

## Test and Validation Corpus

The corpus that validates this layer should span the perceptual operating
envelope deliberately — the same discipline the engine's music-side Tier-A
harness applies with controlled synthetic ground truth.

### Controlled perceptual-sweep artifacts

The highest-value test artifact is a single speaker deliberately modulating one
voice across named perceptual parameters: content and speaker held constant,
features swept on purpose. The exemplar is **L's Voice Training Guide** (signed
"~L"; the top all-time post on r/transvoice), which opens with audio of one
speaker sweeping roughly eight perceptual parameters — vocal-tract length, pitch
and register, resonance, open quotient (the breathiness ↔ compression axis),
intonation contour, articulation, twang, throat closure — **both combined and in
isolation**. The isolation clips are the prize: each is near-ground-truth for a
single feature. This is the speech counterpart of `synth_ground_truth.py` — a
human parameter sweep in place of a synthetic tone. Trans voice pedagogy is a
rich source of these, because teaching a feature *requires* demonstrating it in
isolation.

### Genre spread

Beyond ordinary conversation, the corpus should span the prosodic dynamic range
and the music↔speech boundary: poetry and heightened recitation (high-control,
deliberate prosody), singing (the music↔speech bridge), instrumental music (the
engine's existing home), normal conversation (the product's actual use case),
and arguments (high-affect, fast, overlapping — the stress case). The spread is
itself a test sweep: it exercises the engine across its full operating envelope
rather than at one comfortable point.

### Transcription-artifact cases

Two classes, both wanted. (a) Audio that *breaks* transcription — overlapping
speech, crosstalk, fast affect, strong accents — the inputs that produce garbled
or lossy transcripts. (b) Audio paired with a transcript that *misrepresents*
what was said — the sarcasm-flattened, hedge-laundered cases — so the layer is
tested on recovering exactly what transcription loses. Class (b) is the
on-thesis one: it is the failure this whole line of work exists to catch.

### Starting sources

- **Controlled sweeps** — L's guide and comparable trans voice demonstrations
  (see *Prior Art and Research*); the genre is purpose-built for isolated
  feature sweeps.
- **Diverse general audio** — the audio-bank archive assembled for Tom
  Whitwell's *Radio Music* Eurorack module: a large, deliberately heterogeneous
  set (field recordings, music, spoken word, noise), already in the engine's
  modular world, and a fast way to get genre breadth without curating from
  scratch.
- **Speech and prosody datasets** — a dedicated pass (per the research brief)
  for prosody-labelled, conversational, and consented clinical corpora.

This section names the corpus's shape, not its contents. The validation effort,
when it starts, should build to this spread rather than to whatever is easiest
to collect.

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
