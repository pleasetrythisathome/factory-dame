---
reviewers:
  - scarlet@aswritten.ai
---

# Vision: live NRT-to-modular bridge — improvising with arbitrary audio

## What this is

A new product thesis that emerged during the task-004 implementation session. Scarlet asked:

> "so might we be able to create a piece of hardware (or software running on hardware rigged to do this, like a laptop connected to an expert sleeps es-9, etc) that listens to live audio and outputs osc, midi and CV for all of these parameters? so you could improve live along with arbitrary audio. like i could bring the modular to a jazz open mic and be able to track even the subtle push pull of tempo, or just play along with tracks i like for fun and be in key, or use the modular next to turntables/CDJs."

This is not a small feature. It is a **product-level unification of the existing backlog** into a single coherent use case — the same thesis that has been implicit across `task-005` (live audio), `task-006` (TouchDesigner OSC), `task-007` (Overtone), but never stated.

## The concrete use cases

1. **Jazz open mic.** Bring the modular to a venue. Put a mic on the room or DI the band. The modular follows the drummer's push/pull of tempo, the pianist's modulations. You improvise inside the music rather than on top of a fixed click.

2. **Play along with arbitrary tracks.** Spotify, bandcamp, whatever. Modular is in key, synced, following the track's dynamics. Low-stakes fun, or practice.

3. **Modular alongside DJ hardware.** Turntables, CDJs, anything without MIDI sync. The NRT box extracts clock, tonic, and key from the audio output — the modular syncs without a dedicated sync protocol.

## Why NRT makes this worth doing (not just a DAW plugin)

Conventional beat trackers and key detectors are grid-based and template-based. They assume 12-TET Western tuning, quantized tempi, and fixed profiles (Krumhansl-Schmuckler). NRT doesn't:

- **Continuous tempo.** The rhythm GrFNN peak oscillator drifts smoothly with the music. Rubato, accelerandi, and the subtle push/pull of live jazz show up as a continuous CV signal. A beat tracker snaps — NRT flows.
- **Learned tonal hierarchy.** The Hebbian pitch W re-weights as the piece modulates. Works on non-Western scales, microtonal music, and ambiguous tonal pieces (dub, drone, ambient) where chromagram+Krumhansl collapses.
- **Phantom fundamentals.** Bass lines with missing fundamentals still produce clean tonic detection because the nonlinear dynamics generate the fundamental internally.
- **Analysis IS the CV.** The same oscillator state driving the particle systems and 3D forms is the state driving the CV to the modular. No separate analysis pipeline, no metadata handoff, no "analysis frame" vs. "synthesis frame" boundary.

This is the thesis from the memory `2026-04-16-3-nrt-vs-daw-eurorack-analysis-comparison.md` (`Narrative_AnalysisIsSynthesis`, `Claim_NRTGridFreeTempo`, `Claim_NRTNoFixedProfile`) put to work as a product.

## Architecture — what it would take

### Near-term (weeks, no custom hardware)

- Laptop running the engine in live mode (task-005 prereq)
- Expert Sleepers ES-9 (or any DC-coupled USB audio IO — MOTU UltraLite works too) as the CV bridge: 14 DC-coupled outs, sounddevice pipes numeric signals as "audio," they become CV on the modular
- Standard MIDI interface for clock + MPE note output
- New routing module: subscribes to the live perceptual state stream, maps extractor outputs to CV channels, MIDI messages, and OSC
- Config-driven mapping — user writes `[router.cv.channel_0] source = "tonic"` etc. The mapping is the creative surface

This is captured as **task-011** (CV/MIDI/OSC output router) with `task-004` and `task-005` as dependencies.

### Medium-term (months, product packaging)

- Raspberry Pi 5 or LattePanda with USB audio IO, running the engine headlessly
- Boot-to-modular box: plug in audio-in, CV-out, power; no screen needed
- Small OLED + encoder for basic status / routing preset selection

### Long-term (if the product lands)

- Custom embedded board with onboard DACs for CV. Daisy Seed at reduced network sizes, or STM32MP1-class SBC for the full engine
- Module-form-factor version for Eurorack — though the main engine probably won't fit on a 14HP panel; a "control module" that talks OSC to a host computer might be more realistic

## Dependency chain

The existing backlog already encodes this:

1. `task-004` — perceptual feature extractors (in progress)
2. `task-005` — live audio mode (nd-live)
3. `task-011` — CV/MIDI/OSC router (new)
4. `task-010` — OSC schema documentation becomes load-bearing (consumer contract)
5. `task-007` — Overtone subscriber is one flavor of consumer
6. `task-006` — TouchDesigner is another flavor of consumer

They are not separate ideas — they are facets of this one vision. The product is "live NRT-to-modular bridge," and the tickets are the implementation path to get there.

## Product positioning

If this gets packaged and sold, the positioning would be something like:

> "The first Eurorack-compatible bridge that lets your modular follow arbitrary live audio — jazz, DJ sets, streamed tracks, field recordings. Continuous tempo, learned key, phantom-aware fundamental detection. Not a beat tracker snapping to a grid; a neural resonator that entrains like an ear."

Adjacent products for context:
- **Mutable Instruments Rings** — modal resonator, closest in spirit but synthesis not analysis
- **Expert Sleepers FH-2/ES-9** — CV bridge infrastructure, not analysis
- **Ableton Link** — tempo sync but only for Link-compatible software, not arbitrary audio
- **No direct competitor** in the "listen to arbitrary audio, output continuous musical CV" space

## Why this fits the factory-dame project strategy

Factory-dame is the creative counterweight to aswritten (`Narrative_CreativeCounterweightToAswritten`). A product pivot here doesn't threaten that — the process minimalism stance still applies (light tickets, no velocity tracking) — but the vision clarifies that this isn't a dead-end art project. It's an art project that might also be a real product if it keeps going.

Taylor intersection (`Narrative_TaylorCreativeIntersection`): if the hardware packaging happens, this becomes a physical object that could be designed/CAD'd, creating natural overlap with Taylor's generative 3D interests. The modular context is already Scarlet's world; the CAD for a custom enclosure or module faceplate would be a shared surface.

## Where this is in the perspective

Net-new narrative — belongs under `Narrative_NonlinearOscillatorNetworks` as a specific application, and connected to `Narrative_AnalysisIsSynthesis` as the architectural foundation. Creates a new product-level thesis distinct from the engine-as-generative-art-platform framing.

## Status

- Memory saved 2026-04-16 during task-004 implementation session
- Backlog ticket `task-011` created with `task-004` + `task-005` as dependencies
- No implementation commitment yet beyond continuing task-004

Scarlet's framing was exploratory ("might we be able to...") — this is a product direction worth being oriented toward, not a commitment to ship. The correct near-term action is to continue task-004, finish task-005, and then revisit whether task-011 is the right next step or whether the engine needs to prove itself more first.
