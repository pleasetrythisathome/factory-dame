---
reviewers:
  - scarlet@aswritten.ai
---

# Neurodynamics Engine — Project Overview and Feature Semantics

## What This Is

A working real-time audio-analysis engine implementing **Neural Resonance Theory** (NRT) as a driver for generative systems. It ingests audio, runs it through a coupled-oscillator-bank model of musical perception grounded in Harding, Kim, Demos, Roman, Tichko, Palmer, and Large's 2025 *Nature Reviews Neuroscience* Perspective ("Musical neurodynamics", *Nat. Rev. Neurosci.* 26: 293–307, May 2025), and emits a rich multi-channel state stream over OSC and a per-file parquet log. The state stream is designed to drive two target applications: **particle system visualization** and **generative music synthesis**.

Location: `neurodynamics/engine/` within factory-dame.

## Genesis: How This Started

On 2026-04-15, Scarlet dropped three papers into `./neurodynamics/` — Harding et al.'s 2025 Perspective, a critique by Stone & Buks ("Musical neurodynamics and the 'inner voice'", April 2026), and Large et al.'s reply (April 2026). In her words:

> "added three papers to ./neurodynamics - just saw there's been some developments in the research of e w large since i was into his work in 2019. can you take a look? some controversy with another scientist"

She'd previously followed Edward W. Large's work on nonlinear resonance in music perception in 2019 and wanted to catch up on both the state of the theory and the active debate. The review surfaced that NRT has split into a camp claiming musical perception emerges from physical brain-body oscillator dynamics (Large) versus a camp arguing that predictive-coding internal models are a more parsimonious explanation (Stone & Buks; Lenc & Bouwer).

Notable: Large is founder and stockholder of **Oscilloscape, Inc. (dba Oscillo Biosciences)** — the theory is commercial infrastructure, not just academia. The critique reads sharper than typical because the framework is being productized.

After the paper review she asked:

> "are the algorithms included that we'd need to setup our own oscillator networks that respond to either pre-recorded (or is live possible) audio input? or perhaps multiple different approaches we could try?"

This flipped the session from "read papers" to "build the thing." The engine is the result.

## Rationale: Why This Matters to aswritten and Scarlet

This project directly embodies `Narrative_NonlinearOscillatorNetworks` ("using coupled nonlinear oscillator networks (cochlear dynamics) to drive generative forms in the frequency domain") and resolves the previously open stake `Claim_OscillatorNetworkApplication` ("Oscillator networks with large parameter sets drive music visualization or textural/sculptural forms") — that claim is now a working implementation.

It is squarely in the cochlear-modeling branch of `Narrative_AIUnlockingInaccessibleWork`: the math of NRT (gradient-frequency neural networks, Hopf normal form with higher-order terms, Hebbian weight coupling, delay-coupled anticipation) is work Scarlet explicitly considered inaccessible before AI-assisted development. She said she "was into" Large's 2019 work — now, in 2026, she can actually build from his published equations in an afternoon.

Strategically it sits inside `Narrative_CreativeCounterweightToAswritten` — breathing room outside the "dumber and faster" pressure of building aswritten itself — and `Narrative_AIWorkflowSandbox`, where she deliberately uses projects like this to push Claude Code practice without risking aswritten's velocity. The test-driven build approach taken here (detailed below) was itself practice for robust AI-assisted development.

It also has potential intersection with `Narrative_TaylorCreativeIntersection`: the same engine's state stream could drive 3D form generation for Bambu X1 Carbon prints, giving Scarlet and Taylor a creative overlap that hasn't been formally scoped yet.

## Approach: Architecture and Philosophy

**Grounded in primary sources, not third-party code.** The engine ports the equations directly from Harding/Kim/Large 2025 Fig. 2 and the cited Large lab papers. Earlier in the session Scarlet asked whether I was starting from a Python port of Large's MATLAB GrFNN Toolbox. Her actual question was about provenance, and I was honest:

> [assistant] "no... I shouldn't lean on that without verifying it's real and current — and even if it is, it'd be more work to understand someone else's port than to write this fresh from the paper's equations."

The ~200 lines of core oscillator math are first-party and readable, using Large's MATLAB toolbox only as a correctness reference if something behaves weirdly. The upside: every line can be modified without inheriting someone else's architecture.

**Two-layer oscillator architecture.**
- Rhythm GrFNN: 50 oscillators log-spaced 0.5–10 Hz (30–600 BPM equivalent range)
- Pitch GrFNN: 100 oscillators log-spaced 30–4000 Hz

Each oscillator implements the canonical Hopf normal form from the paper's Fig. 2 caption:
```
τ ż = z(α + iω + (β₁ + iδ₁)|z|² + ε(β₂ + iδ₂)|z|⁴ / (1 − ε|z|²)) + x(t)
```

**FIR gammatone cochlear front-end**, normalized to unit peak response per channel. The initial IIR implementation (via scipy.signal.gammatone) turned out to produce an unstable filter at 30 Hz — pole magnitude 1.0036 outside the unit circle — which would have been a silent accuracy bug without tests catching it. Switched to FIR form for unconditional stability.

**Decoupled consumers via OSC.** The engine broadcasts UDP OSC on 127.0.0.1:57120. Any language can subscribe — SuperCollider, TouchDesigner, Three.js via osc.js, and critically for the future: Clojure/Overtone. Scarlet explicitly requested this:

> "i'd prefer something that plays well with clojure in the future (aero)"

OSC is the right bridge: future Overtone/Quil consumers subscribe to the same port with zero Python coupling. This aligns with `Narrative_ClojureFunctionalLineage` — the engine's in Python for fast iteration on the math, but downstream creative consumers can live in Clojure territory.

**Test-driven development throughout.** On 2026-04-15 Scarlet explicitly requested this when the complexity started stacking:

> "yes, i want all of that! as this is getting more complicated, can you build out a testing framework to confirm things work as you expect and that we don't add regressions?"

And then, before I continued implementing features:

> "after you're done with that, use the testing framework as backpressure, design new tests for the new features to build out the full core platform, and iterate - i'll be away for 30 min or so so see what get to but always confirm fidelity and focus on code quality, rigorous testing, usability, interface etc over running forward with new features. we want to build a stable and robust foundation for the future here"

This was a direct instruction to prioritize quality over velocity, and it shaped the rest of the build. Every feature went red → green: failing test written first, then implementation to make it pass. The test suite is 52 tests across nine files (unit / integration / regression). Regression baselines are committed under `tests/baselines/*.json` and regenerated via `uv run python -m tests.generate_baselines`.

Bugs the tests caught that would otherwise have been silent:
1. The IIR gammatone instability above
2. Pitch channel weighting used destructive phase interference (averaged adjacent bands whose audio-rate oscillations canceled) — fixed by bumping sharpness default from 4 to 15 so weights are near one-hot
3. Pitch `input_gain=0.0005` was tuned against the broken cochlea — retuned to 0.1 against the corrected filter

## Application: The Two Target Use Cases

Scarlet stated the creative goals explicitly:

> "i'm interested in two uses cases, using the output to drive the parameters in particle systems, and two as an generative music tool to be able to take current signal and output shifts based on missing pulse, harmonic relationships, etc"

### Particle system driver
- Per-band amplitude → emission rate, size, brightness
- Phase → rotation, angular velocity, orbital position
- Phantom mask → ghost emitters that pulse in silence (the "inner voice" signal)
- Prediction residual → turbulence (positive = scatter, negative = coherent flow)
- Mode-lock groups → visual flocking, locked oscillators cluster
- Hebbian W (post-run) → constellation graph showing learned structure

### Generative music driver
- Phantom-triggered percussion fills in beats the listener hears but the audio doesn't play
- Implied-harmony pads synthesized at ringing pitch frequencies
- Residual scalar modulates filter cutoff / distortion for tension-release dynamics
- Mode-lock detection drives polyrhythm generator (detect 3:2 rhythm lock → synthesize percussion at the detected ratio) and harmonic chord extractor (pitch-layer locks at 3:2, 5:4, etc. → build chord voicings)
- Hebbian W + random seed → generative accompaniment that matches the piece's internal logic
- Delay-coupled anticipation → "premonition" synth layer that leads the audio

## Current State: What's Built

As of 2026-04-15 the engine runs end-to-end on arbitrary audio files (wav/flac/aiff natively; mp3/m4a/opus via ffmpeg pipe). It produces:

- `output/<audio_stem>.parquet` — state log with fields per snapshot per layer (t, layer, z_real, z_imag, amp, phase, phantom, drive, residual)
- `output/<audio_stem>.weights.npz` — learned Hebbian weights when plasticity is enabled
- Live UDP OSC broadcast on 127.0.0.1:57120 with `/<layer>/{amp,phase,phantom,drive,residual}` messages at 60 Hz

It ships a scrollable Tk/matplotlib viewer with:
- Scrolling horizontal heatmaps (pitch + rhythm, ±6s window around playhead)
- Stacked-ridge waterfall panels showing last 0.5s of amplitude structure
- Phantom-activation strips showing inner-voice activity over the same window
- Time-synced audio playback via afplay (macOS) or sounddevice (other platforms)
- Scrollable Tk window container for tall figures on small screens

Implemented oscillator features, all opt-in via config:
1. **Amplitude `|z|`** — perceived spectrum with memory
2. **Phase `arg(z)`** — instantaneous cycle position
3. **Drive** — raw cochlear input magnitude per band
4. **Phantom mask** — "ringing without drive" = the inner voice / imagination regime
5. **Prediction residual** — `drive − |α|·|z|`, positive = surprise, negative = phantom regime; the single-scalar tension signal
6. **Hebbian plasticity** — complex weight matrix W grows under co-activation, encodes tonal hierarchy and rhythmic attractors; persisted across runs
7. **Delay coupling** — ring-buffered self-feedback produces phase-lead / strong anticipation
8. **Stochastic noise** — √dt-scaled Gaussian kicks keep oscillators alive in silence
9. **Mode-lock detection** — standalone `neurodynamics.modelock` module computing phase-locking value (PLV) and scanning oscillator pairs for small-integer ratio locks (1:1, 2:1, 3:2, 4:3, 5:4, 7:4)

## Feature Semantics: Detailed Signal → Audio → Visualization/Synthesis Reference

The following is the detailed semantic map the engine's outputs afford. Each feature's entry covers what it actually *means about the audio*, how it drives visualization, and how it drives synthesis. Preserved here as reference material.

### Amplitude `|z|` per oscillator
**What it says about the audio.** How strongly the network is resonating at each frequency *with memory*. Unlike an FFT (which measures instantaneous spectrum), oscillator amplitude builds up under sustained exposure and decays slowly. FFT answers "what frequencies are in the audio now?" Amplitude answers "what frequencies has the system been listening to recently?" — it's the *perceived* spectrum.
**Visualization.** Heatmap; particle emission rate per frequency band; size/brightness/density of a cloud; amp-gated effects.
**Synthesis.** Gate a synth voice when a given oscillator's amp crosses threshold; drive envelope amplitude / filter cutoff from amp continuously; amp-weighted granular synthesis — only granulate the "active" frequencies so synth shadows what the network hears.

### Phase `arg(z)` per oscillator
**What it says about the audio.** Where in the oscillation cycle each oscillator currently is. For rhythm, sub-beat position at high resolution. For pitch, intra-cycle timing carrying consonance information implicitly (phase-locked pairs sound consonant).
**Visualization.** Angular position — rotation, orbital motion, sweep; phase-coherent particle release (particles emitted only at phase 0 cluster on the beat); color cycling with phase; concentric rings rotating at natural frequency.
**Synthesis.** Rhythm-synced trigger: gate a drum machine on the 2 Hz oscillator's zero-crossings — synth follows the network's *felt* beat, not raw audio onsets; phase-locked sequencer; LFOs whose phase locks to network phase so they breathe with the music.

### Drive (input magnitude)
**What it says about the audio.** The raw cochlear energy currently hitting each oscillator's band — the "objective" side. Spikes at onsets, continuous during sustained tones.
**Visualization.** Transient flash/bloom; spectrogram-like literal readout; screen-shake on large peaks.
**Synthesis.** Side-chain triggering; transient-driven generative events (new arp note chosen when drive spikes past threshold). Pairs with residual for onset *quality*.

### Phantom mask
**What it says about the audio.** *The network is still hearing something that isn't there anymore.* High amp and no current drive = "ringing memory" state. Rhythm layer keeps pulsing at 2 Hz during missing beats — the imagined kick. Pitch oscillator ringing after a note ends — lingering pitch in the listener's head. **This is the NRT-specific signal you can't get from a filter bank.** It distinguishes "what the audio said" from "what the brain heard."
**Visualization.** **Ghost particles** — emitters that fire when phantom is active but drive is low. Makes rests feel populated. A parallel particle layer only rendering on phantom-active oscillators, visually separable from the audio-reactive layer. Transitions between "audio drives the scene" and "imagination drives the scene" become readable.
**Synthesis.** **Missing-pulse percussion generator** — trigger a kick every time a rhythm-layer phantom fires; literally plays the beats the listener is hearing in their head during rests. **Echo tail that decays with phantom duration** instead of fixed time constant — echoes last as long as the listener would still feel them. **Phantom-gated synth pad** that fades in when audio stops, playing implied notes.

### Prediction residual (`drive − |α|·|z|`)
**What it says about the audio.** How much the input deviated from what the oscillator needed to sustain its state.
- **Positive** → more energy than expected. Fresh onset, syncopated hit, surprise. "Something new arrived."
- **Near zero** → steady state, sustained drone, expected groove. "The world is as predicted."
- **Negative** → less energy than expected. A note stopped but the oscillator is still ringing; a beat is missing; a chord dropped out. "Expected energy didn't come."
A single scalar per oscillator per frame encoding the most musically relevant thing: tension.
**Visualization.** Diverging colormap strip (red positive, white zero, blue negative). Time-aligned with phantom strip to read both simultaneously. Particle turbulence — positive scatters, negative draws coherent flows. Visual shockwaves on large positive spikes.
**Synthesis.** Tension/release modulator. Leaky-integrated residual drives filter cutoff / distortion depth. Positive opens the filter, rough transients emphasized; negative closes it, smooths into pads. Synth layer breathes with the perceived surprise-to-expectation arc — which is what makes music feel *dynamic*. Also compositional: switch sections on large sustained positive residual ("drop"), or on sustained negative ("breakdown").

### Hebbian plasticity — learned weight matrix W
**What it says about the audio.** After listening to a piece, the network has *learned its relational structure*. W_ij is large when oscillators i and j were frequently co-active with stable phase — which encodes:
- **Pitch layer W**: the *tonal hierarchy*. Tonic oscillator ends up strongly linked to dominant, mediant, etc. The piece's harmonic vocabulary.
- **Rhythm layer W**: learned rhythmic attractors. Primary pulse connects to characteristic subdivisions. The "groove fingerprint."

W is the closest thing we have to the network's **memory of the piece**. Save it and feed the network random seeds later — it drifts toward replaying those attractors.

**Visualization.** Matrix heatmap (static diagnostic panel). Graph visualization: oscillators as nodes on a log-frequency circle, W_ij as edges with thickness = |W_ij|, hue = arg(W_ij). As the piece plays, edges light up — the learned constellation emerges. Particle systems can use W as an attractive force map — each active oscillator pulls on others proportional to W_ij, producing coherent flocking that reflects the music's structure.

**Synthesis.** **"Echo the attractors."** Silence the audio input after a run, inject random noise into oscillator state, let W attract the dynamics. Output is the network's *improvisation in the style of the piece* — it gravitates toward the tonic, the characteristic subdivisions, the learned rhythm. This is how you get **generative accompaniment matching the piece's internal logic**, not a random synth drone. Seed-based continuation: load trained W from one song and play it in another context for "this song's memory applied to silence."

### Delay coupling — strong anticipation
**What it says about the audio.** The network's response *leads* the driver. Rhythm-layer phase can be tens of milliseconds ahead of the audio. Prediction without any explicit predictive model — emergent from delayed self-coupling.
**Visualization.** Visuals anticipate the audio. Particles burst slightly before the kick lands. Creates the impression of visuals "breathing with" the music rather than reacting — the difference between watching a VU meter and watching a dancer. Can expose the lead explicitly: two playheads, one at audio time, one at phase-leading time.
**Synthesis.** **Premonition layer.** Synth notes triggered from the delay-coupled network fire slightly before main audio onsets. Works especially well for percussion layers — adds "humanized push" or urgency. Or: compare anticipation vs lag per onset and use asymmetry as an "urgency" signal driving filter brightness.

### Stochastic noise
**What it says about the audio.** Nothing directly — it says the *network is alive* in the sense neurons always have background firing. Without noise, silence means literal zero; with noise, silence means gentle oscillator flicker.
**Visualization.** Ambient motion during rests. Particles don't freeze — they breathe, drift, wander into phantom-active regions. Much more organic/alive feel; never a dead scene.
**Synthesis.** Micro-modulation on synthesized voices — slight detuning, amplitude jitter, attack-time wobble. Makes synthesis feel less perfect, more *performed*.

### Mode-lock detection (PLV, integer ratios)
**What it says about the audio.** Which oscillator pairs are phase-locked at small-integer ratios — *where consonance, polyrhythm, and harmonic structure actually live* in the signal.
- Rhythm 2:1 → octave subdivision (16ths over 8ths)
- Rhythm 3:2 → dotted / swing / triplet against straight time
- Rhythm 7:4 → exotic polyrhythm (Afro-Cuban clave vs rock beat)
- Pitch 2:1 → octave
- Pitch 3:2 → perfect fifth
- Pitch 4:3 → perfect fourth
- Pitch 5:4 → major third
- Pitch 7:4 → subminor seventh (just intonation)

PLV value tells you how *strongly* the ratio is present.
**Visualization.** **Constellation overlay** — lines between phase-locked oscillator pairs on the heatmaps, colored by ratio (1:1 green, 2:1 blue, 3:2 purple, 7:4 red). Constellation forms and dissolves as the music moves. Locked oscillators cluster — consonant intervals cluster visually, dissonant don't.
**Synthesis.** **Automatic polyrhythm generator** — when a stable 3:2 rhythm lock is detected, synthesize percussion at the detected ratio. **Harmonic extractor** — detect pitch-layer locks at musical ratios, build chord voicings from active locks. Real-time consonance/dissonance score → timbral morph (high consonance opens sweet voices; high dissonance engages grittier timbres). **Improvisation guide:** detect most stable current lock, constrain generative melody to its ratios.

## Future: Deferred Features and Their Semantics

### Two-layer pulse network (motor cortex model) — Tier 1
A second rhythm-scale GrFNN coupled bidirectionally to the first, modeling motor cortex predicting and influencing auditory perception. This is what the paper claims produces the *full* missing-pulse effect — not just single-oscillator phantom ringing, but an embodied prediction feeding back into what you perceive. Closer to how you actually feel a beat in your body.
- **Viz**: A second "body" layer alongside perception — parallel stream that leads or lags subtly. Two dancers, slightly out of phase. Motor-layer phantom can fire even when auditory-layer phantom doesn't → ghost beats only the body feels.
- **Synth**: Groove enhancement layer. Motor network "wants" to move at the expected pulse; fires percussion in the "unheard" slot when audio is syncopated against expectation. Dubstep drops work by abruptly silencing the motor prediction — this would let you synthesize that dynamic explicitly.

### Live mic input — Tier 4
Real-time streaming instead of offline file processing. Opens up performance, improvisation, feedback loops.
- **Viz**: Live performance visualizer reacting to room audio / singing / conversation.
- **Synth**: Real-time improvisation partner. Play an instrument, let the engine resonate, feed back resonant frequencies as a synth layer that follows what you're doing.

### Brainstem / frequency-following layer — Tier 4
A high-resolution third GrFNN at cochlear-nucleus timescales resonating at *sum and difference frequencies* of active pitches. Two tones at 400 and 600 Hz produce a 200 Hz tone in the brainstem that isn't in the stimulus. Major chords sound "bright" and minor chords "warm" at a physiological level because the difference tones differ.
- **Viz**: Ghost pitches — secondary pitch layer showing sum/difference tones. Reveals structure that isn't in the audio but *is* in perception.
- **Synth**: **Auto-harmonization from real psychophysics.** Instead of diatonic voice-leading rules, synthesize at brainstem-derived difference/summation frequencies. Harmonization that sounds right because it replicates what the brain is already hearing. Subharmonic bass synthesis matching the implied difference tone between any two melody notes.

### Perceptual feature extractors — Tier 3
Higher-level rollups derived from state — interpretations, not raw state:
- Key detection (from learned pitch W)
- Beat tracking (peak rhythm oscillator + phase, smoothed)
- Implied harmony (weighted pitch oscillators against learned tonal hierarchy)
- Meter detection (dominant stable ratio in rhythm network)
- Groove index (syncopation × coherence)
- Tempo confidence (peak rhythm oscillator sharpness)
- Consonance/dissonance scalar (sum of stability contributions from active pitch pairs)

- **Viz**: Text overlays ("Key: G minor · 98 BPM · 4/4 · high consonance"). Key-signature wheel highlighting tonic. Meter display. Harmonic regions color-shifting warm/cool with consonance. Beat prediction indicator ("next beat in 340ms") as countdown ring.
- **Synth**: **Smart harmonization and sequencing**. Key-aware arpeggiators. Beat-aligned sequencers adapting to detected tempo changes in real time. Chord progression generators following detected implied harmony. Consonance-weighted voice-leading constraints on generative melody so it doesn't drift into dissonance unless told to.

### Downstream integration layers — not yet scoped
- **Clojure consumer** — Overtone (audio) + Quil (viz) subscribing to the same OSC stream. Natural fit given `Narrative_ClojureFunctionalLineage`.
- **TouchDesigner / Notch / Three.js** — particle system demos
- **3D form generation for Bambu X1 Carbon** — drive generative geometry from oscillator state; potential overlap with `Narrative_TaylorCreativeIntersection`

## Controversy Context: NRT in Debate

Important context preserved from the paper review:

1. **The theory**: Large's NRT claims musical structure (pitch, rhythm, tonality, groove) arises from physical brain-body oscillator dynamics, not from predictive/internal models. Stability hierarchy follows integer ratios (1:1 > 2:1 > 3:2 > 7:4…). Pulse perception from nonlinear resonance explains "missing pulse" rhythms. Strong anticipation emerges from delay-coupled oscillators.

2. **The critique** (Stone & Buks; Lenc & Bouwer, April 2026): NRT overclaims. Stability-ordering predictions reduce to "more stable / less stable" binaries; integer ratios are trivially derivable from basic signal-processing principles; musicians rehearse/imagine music without any external sound — an "inner voice" — which is hard to reconcile with NRT's insistence that perception is grounded in *physical* oscillatory entrainment. They argue predictive coding is more parsimonious.

3. **Large's reply** (April 2026): The inner voice is real but NRT handles it — oscillator networks with Hebbian plasticity sustain activity without external drive (learned attractors); neurodynamical models already reproduce imagery findings (May et al. 2022, Quiroga-Martinez 2024). Pushes back that linear/predictive models can't explain nonlinear phenomena like missing-pulse or polyrhythm perception.

4. **Commercial subtext**: Large is founder and stockholder of Oscilloscape, Inc.; J.C. Kim is a paid employee. NRT isn't just a theory — it's productized. The critique reads sharper than typical academic disagreement because the framework is being commercialized.

5. **Open question**: Whether NRT makes predictions a predictive-coding model *can't*. The engine we've built doesn't resolve this debate, but it does give Scarlet a platform to experimentally probe the distinctive NRT predictions (phantom activation, mode-locking at non-trivial ratios, Hebbian attunement) against real audio.

## Testing Philosophy (Validated)

- pytest-based, 52 tests across unit (`test_grfnn`, `test_cochlea`, `test_hebbian`, `test_delay`, `test_noise`, `test_modelock`, `test_residual`), integration (`test_pipeline`), and regression (`test_regression`) tiers
- All features developed red → green (test first, then implementation)
- No mocking of the engine — unit tests verify math against analytically-derivable properties; integration tests feed synthetic audio with known properties and assert the expected oscillators activate
- Regression baselines committed; `uv run python -m tests.generate_baselines` regenerates on intentional behavior changes
- Deterministic output asserted as a precondition for regression comparison

This testing rigor came from Scarlet's explicit request and is documented in the engine README as foundational, not incidental.

## Tooling Stack

- Python 3.11, managed by `uv` (per Scarlet's preference for minimal setup friction)
- `hatchling` build backend, editable install
- Core deps: numpy, scipy, soundfile, sounddevice, python-osc, pyarrow, matplotlib
- Dev deps: pytest, pytest-xdist
- External: ffmpeg (for mp3/m4a/opus decode); afplay (macOS playback)

## Source Material: Direct Quotes Preserved

**Scarlet on the project initiation** (2026-04-15):
> "added three papers to ./neurodynamics - just saw there's been some developments in the research of e w large since i was into his work in 2019. can you take a look? some controversy with another scientist"

**Scarlet on target use cases**:
> "i'm interested in two uses cases, using the output to drive the parameters in particle systems, and two as an generative music tool to be able to take current signal and output shifts based on missing pulse, harmonic relationships, etc"

**Scarlet on the Clojure future**:
> "i'd prefer something that plays well with clojure in the future (aero)"

**Scarlet on testing as backpressure** (the decisive instruction that shaped the build):
> "yes, i want all of that! as this is getting more complicated, can you build out a testing framework to confirm things work as you expect and that we don't add regressions?"

**Scarlet on quality over velocity** (before a 30-min absence during active development):
> "after you're done with that, use the testing framework as backpressure, design new tests for the new features to build out the full core platform, and iterate - i'll be away for 30 min or so so see what get to but always confirm fidelity and focus on code quality, rigorous testing, usability, interface etc over running forward with new features. we want to build a stable and robust foundation for the future here"

**Scarlet acknowledging a stall during debugging**:
> (after the viewer heatmap wasn't updating) "top two heatmap sections aren't changing, but bottom row now looks live although it's rather hard to really understand"

This led to the scrolling-horizontal-with-waterfalls redesign — an example of her willingness to flag UX confusion directly so the tool matures into something *useful* rather than just *technically correct*.

## Connections to the Perspective

- **Resolves**: `Claim_OscillatorNetworkApplication` moves from stake to working platform
- **Embodies**: `Narrative_NonlinearOscillatorNetworks`, `Narrative_ParticleSystemsEmergence`
- **Executes**: cochlear-modeling branch of `Narrative_AIUnlockingInaccessibleWork`
- **Fits**: `Narrative_CreativeCounterweightToAswritten` (creative space outside business velocity), `Narrative_AIWorkflowSandbox` (Claude Code practice)
- **Future bridge to**: `Narrative_ClojureFunctionalLineage` via OSC decoupling enabling Overtone/Quil consumers; potential `Narrative_TaylorCreativeIntersection` via 3D-form generation for the Bambu printer

The neurodynamics engine is the most concrete realization to date of the oscillator-network thread in Scarlet's creative work.
