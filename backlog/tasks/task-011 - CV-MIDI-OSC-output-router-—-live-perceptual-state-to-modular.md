---
id: task-011
title: CV/MIDI/OSC output router — live perceptual state to modular
status: To Do
assignee: []
created_date: '2026-04-16 20:02'
updated_date: '2026-04-16 21:48'
labels:
  - engine
  - live
  - hardware
  - product
dependencies:
  - task-004
  - task-005
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A subscriber that consumes the live perceptual extractor stream (tempo, tonic, beat phase, consonance, chord) and routes it out as control signals for hardware and software consumers.

## Why this exists

The product thesis: carry the modular to a jazz open mic, or play it alongside arbitrary tracks/CDJs, and stay in key + in sync with whatever audio is in the room. NRT specifically makes this worth doing — conventional beat trackers snap to a grid and can't represent rubato smoothly. The rhythm GrFNN's peak oscillator drifts continuously with the music, and that continuous signal is what should reach the CV.

This is the spine that ties task-004 (extractors), task-005 (live audio), task-006 (TouchDesigner), task-007 (Overtone) together — they are not separate ideas, they are facets of one product vision.

## Scope

### Outputs

- **CV** via DC-coupled USB interface. Near-term target: Expert Sleepers ES-9 (14 DC outs). Pipe numeric CV channels as "audio" to the device.
  - `cv/tonic` — 1V/oct of the currently detected tonic (drives an external quantizer)
  - `cv/beat_phase` — sawtooth 0-5V following the peak rhythm oscillator's phase
  - `cv/beat_trigger` — gate pulse on beat-phase wraparound
  - `cv/tempo_confidence` — continuous 0-5V
  - `cv/consonance` — continuous 0-5V
  - `cv/chord_root`, `cv/chord_quality` — discrete levels per chord type
- **MIDI** via standard output device.
  - clock messages at detected tempo (24 ppq)
  - note-on / note-off for implied chord on a chosen channel (MPE-compatible)
  - CCs mirroring CV values for devices without CV input
- **OSC** — same message set as existing `osc_out.py` plus the `/features/*` namespace from task-004. Unified schema so TouchDesigner, Overtone, or any other consumer gets a consistent stream.

### Architecture

- New module `neurodynamics/router.py` that subscribes to the live state stream (state queue from task-005). Pure subscriber — no engine internals, no feedback path into the GrFNN.
- Pluggable backends: CVBackend (sounddevice OutputStream to ES-9), MIDIBackend (mido), OSCBackend (existing).
- Config-driven mapping table: user writes `[router.cv.channel_0]` etc. to wire extractor outputs to hardware channels. This is the creative configuration surface — it matters.
- Command: `uv run nd-route --config ...` — runs alongside `nd-live`.

### Calibration

- 1V/oct tonic CV needs calibration against the user's quantizer. First-time setup routine: user tunes the quantizer to a reference C, engine measures output voltage, stores offset.
- Same for clock phase-align.

## Hardware targets (near → far)

- **Near:** laptop + ES-9 (or any DC-coupled USB audio IO — MOTU UltraLite also works). No custom hardware.
- **Mid:** Raspberry Pi 5 / LattePanda with USB audio IO, running the engine headlessly. Boot-to-modular box.
- **Far:** custom embedded board with onboard DACs for CV. Daisy Seed at reduced network sizes (maybe), or STM32MP1-class SBC for full bank.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 #1 `nd-route` command runs alongside `nd-live`, subscribing to the live state stream
- [ ] #2 #2 Configurable routing: user maps extractor outputs (e.g. `tonic`) to output channels (e.g. `cv/0`, `midi/note`, `osc/...`)
- [ ] #3 #3 1V/oct tonic CV verified with a calibrated quantizer
- [ ] #4 #4 Analog clock trigger is jitter-free under steady-tempo input (< 5 ms drift over 30s)
- [ ] #5 #5 MIDI clock messages sync a second device at detected tempo
- [ ] #6 #6 OSC message schema matches the documented schema (task-010)
- [ ] #7 #7 Latency budget: audio-in → CV-out < 60 ms on a modest laptop
- [ ] #8 #8 Integration test with synthetic audio verifies CV values fall in expected voltage range
- [ ] #9 #9 Documentation: wiring diagram for ES-9 use case, example modular patch demonstrating tonic-quantizer sync
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
## Deep-first phased plan (per 2026-04-16 feedback)

Each phase is a **standalone novel capability built to publishable depth**. Not "shallow first, deepen later" — that failure mode is explicitly documented in the aswritten memory `2026-04-16-5-feedback-novelty-and-deep-first-engineering`. Each phase is shippable as its own novel artifact.

### Phase 1 — Voice identity to publishable depth

The primitive: decompose the NRT pitch GrFNN state into a dynamic set of coherent voice clusters, each with a stable identity tracked across time. No fixed voice count, no band-based shortcuts — real phase-coherence clustering on real music.

**In scope:**
- Amplitude-envelope correlation across active pitch oscillators
- Integer-ratio PLV for harmonic relationships (fundamental + harmonics in one voice)
- Connected-components clustering → dynamic voice count
- Hungarian matching across frames for voice identity persistence
- Edge cases: silence, unison, voice split, voice merge, transients, noise floor
- Real-audio validation (Flights + at least one other track)
- Viewer: colored voice overlay on pitch heatmap
- OSC `/voice/*` namespace
- Comprehensive test suite covering all edge cases

**Out of scope for P1:**
- Per-voice rhythm association (that's P2)
- Per-voice motor coupling (that's P3)
- CV/MIDI trigger emission (router-level, orthogonal)

### Phase 2 — Per-voice rhythm association

For each pitch voice from P1, identify the rhythm oscillator(s) entrained to it. Expose per-voice clock rate, beat phase, subdivision ratio relative to master beat. Tested on multiple tracks; handles polyrhythmic cases.

### Phase 3 — Per-voice motor coupling

Read existing motor GrFNN state as multi-voice (associate motor oscillator clusters with pitch voices). OR if reading proves insufficient, introduce per-voice motor GrFNN pool. Phase decision after P2 empirical findings.

### Phase 4 — Router (the original task-011 scope)

CV/MIDI/OSC router consuming P1-P3 state, producing analog clock triggers, 1V/oct tonic, CV lanes per voice, MIDI notes with MPE per voice. Existing task-011 description still applies; now built on top of the voice primitives rather than a flat feature stream.
<!-- SECTION:PLAN:END -->

<!-- AC:END -->
