---
reviewers:
  - scarlet@aswritten.ai
---

# Milestone: the live-NRT-to-modular product loop is functionally complete

## What happened

Between 2026-04-16 and 2026-04-17, a deep-first autonomous run closed
out the four core layers of the modular bridge vision
(`Narrative_LiveNRTModularBridge`) and landed the engine optimization
needed to make the bridge actually live.

The system now has four CLIs that compose into the full product loop:

    nd-run   — offline engine: audio file → parquet state + OSC
    nd-live  — live engine: mic/Loopback audio → OSC (no parquet)
    nd-view  — offline viewer: parquet + voice overlay + banner
    nd-route — CV/MIDI/OSC router: OSC → ES-9, MIDI, OSC forwards

Combined with the Phase 1-4 primitives (voice identity, per-voice
rhythm, per-voice motor, router) the pipeline runs end-to-end
without a DAW: point a mic or Loopback at the laptop, start
``nd-live`` + ``nd-route``, and the modular gets per-voice CV
streams derived from whatever's in the room.

## Commits landed in the run (newest first)

- `2514135` feat: nd-live — live audio mode, real-time engine from mic/Loopback
- `a656245` perf: numba-JIT the GrFNN hot path, engine now above 1x realtime
- `6bfd0d5` feat: CV/MIDI/OSC router — Phase 4 of task-011
- `4ae3449` feat: per-voice motor coupling — Phase 3 of task-011
- `fc47897` feat: per-voice rhythm association — Phase 2 of task-011
- `ff72c3e` feat: voice identity extraction — phase-coherence clustering with tracking
- `3bfd2d6` feat: perceptual extractors — key, chord, consonance, multi-clock rhythm

## The numba story

The engine ran at ~0.38× realtime pre-optimization — a single second
of input took 2.6 seconds of CPU on an M-series laptop. Live mode
was impossible. The benchmark pinpointed the pitch GrFNN's audio-
rate RK4 loop at 95% of wall time.

Three compounding optimizations brought it to 10.99× realtime
on the batched API (``GrFNN.step_many``):

1. ``@njit`` on the Hopf RHS and RK4 integrator (fastmath=True,
   explicit per-oscillator loop, no intermediate allocations)
2. ``(W @ z) / n`` computed once per RK4 step instead of per stage
3. ``step_many`` — an entire batch of samples processed in one
   JIT call, eliminating Python↔C boundary overhead

The per-sample ``step`` API is still available at ~1.2× realtime
for compatibility with offline ``run.py``. Live mode uses
``step_many`` over each audio chunk.

## Hardware path finalized

Per the earlier memory `2026-04-17-1-hardware-path-m1-mini-not-pi`,
the headless form factor is:

- MacBook in a bag / drawer for the "laptop out of sight" case
- M1 Mac mini 8GB (Scarlet owns one, sitting idle) as the
  dedicated boot-to-modular box
- Pi-class ARM is YAGNI — the numba optimization got us to >1x
  realtime on M-series, and Scarlet has the hardware she needs

task-015 (engine optimization) is effectively done for the
current product path; re-targeting for Pi would require more work
(Cython/Rust inner loop, smaller networks) but only unlocks form
factors Scarlet doesn't care about.

## What still requires hardware to validate

The software side is complete. Hardware-in-the-loop testing, all
Scarlet's to do when she plugs in the ES-9:

- 1V/oct tonic CV drives a quantizer correctly (calibration routine
  is a one-time tune against her specific quantizer)
- Analog clock trigger jitter < 5 ms over 30s of steady tempo
- MIDI clock messages sync a downstream device at detected tempo
- Full end-to-end: live mic → nd-live → nd-route → ES-9 → modular
  actually responds musically

## What's left in the backlog

Looking ahead, in rough priority order (all non-blocking for the
product loop itself):

- `task-012` per-voice motor GrFNN pool — replaces the current
  shared-bank Phase 3 read with genuine per-voice motor networks.
  Real depth upgrade; the per-voice motor BPMs will diverge from
  per-voice rhythm BPMs in interesting ways once spawned motors
  entrain to each voice's onset pattern independently.
- `task-003` brainstem / sum-difference tone layer — deeper NRT
  fidelity. Adds harmonic combination tones as an emergent layer.
- `task-002` pitch-class folding viewer mode — quick viewer polish.
- `task-013` meter detection — 4/4 vs 3/4 vs 6/8 from companion
  ratios. Small additive feature.
- `task-014` groove index — research + implementation. Unclear
  scope until prototype.
- `task-010` OSC schema documentation — becomes load-bearing when
  third-party consumers (task-006 TouchDesigner, task-007 Overtone)
  arrive.
- `task-001` mode-lock arcs, `task-008` Bambu 3D forms, `task-009`
  W matrix downsampling — low-priority polish / consumer / infra.

## Scope note on "complete"

"Functionally complete" specifically means: the software pipeline
does what the product thesis says it does — listen to arbitrary
audio, decompose into voices, emit per-voice CV streams. It does
not mean "shipped," "polished," or "validated on hardware." Those
are separate surfaces; this memory is about the code, not the
product.

## Attribution to perspective

This milestone embodies what `Narrative_NoveltyAsNorthStar` and
`Narrative_DeepFirstNotShallowFirst` asked for: each phase landed
as a publishable, novel capability (per-voice rhythm extraction,
per-voice motor coupling, per-voice CV output) rather than a
shallow horizontal sweep. Each capability exists at depth you
could explain to a peer without hedging "we'll improve that
later."

The engine optimization is also the first place this run chose
"enable the novel capability" (live mode) over "stay pure numpy
for ideological consistency" — which is exactly the call
`Narrative_DeepFirstNotShallowFirst` framed when a decision
comes up.
