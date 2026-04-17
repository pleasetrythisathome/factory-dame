---
id: task-012
title: Per-voice motor GrFNN networks (upgrade Phase 3 from read-state)
status: To Do
assignee: []
created_date: '2026-04-17 02:17'
labels:
  - engine
  - voices
  - motor
dependencies:
  - task-011
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Phase 3 motor coupling currently reads the existing (shared, mono-voice) motor GrFNN state and DFT-matches each voice's envelope against it. Because motor and rhythm banks share the same frequencies, motor BPM equals rhythm BPM by construction; only the phase differs.

The true "per-voice motor" version would spawn a motor GrFNN network per voice, each driven by the voice's own onset signal (derivative of the voice's amplitude envelope), producing genuinely independent anticipatory prediction per voice. This is a substantial engine change — not a post-hoc read of existing state.

## Why this matters

Per Narrative_LiveNRTModularBridge and the novelty-and-deep-first feedback: the goal is "per-voice dynamic phase-coherent motor networks that generate polyrhythmic CV for arbitrary live audio." Reading existing motor state is the shortcut; this ticket is the real thing.

## Scope

- Motor pool architecture: K motor GrFNNs (K=8 initial), dynamically assigned to tracked voices
- Allocation: when a voice becomes active, grab an unused motor network and start driving it with the voice's onset signal; when voice retires, release the network for reuse (LRU or first-free)
- Per-voice onset signal: derivative of the voice's amplitude envelope, rectified and normalized — similar to how the engine already computes the global rhythm drive
- Integration with run.py's time-stepped loop: each motor network advances per step, coupled to its assigned voice
- Extend VoiceMotor to track whether the motor is from shared-bank read (Phase 3) or voice-local network (this ticket)

## Success criteria

- Per-voice motor BPM can genuinely differ from per-voice rhythm BPM (e.g. a voice's motor locks at 2x for anticipation while rhythm locks at 1x)
- Per-voice motor phase still has "sustains through silence" property (the NRT Fig. 4 headline claim)
- No regression in existing motor tests (test_two_layer_pulse)
- Real-audio validation across the 10-track corpus

## Acceptance Criteria

- [ ] #1 Motor pool implementation with N-voice allocation/deallocation
- [ ] #2 Per-voice onset signal computation from voice envelope derivative
- [ ] #3 Per-voice motor GrFNN driven by that onset signal
- [ ] #4 Synthetic test: voice with unique rhythm → motor entrains independently of global rhythm
- [ ] #5 Real-audio test: motor and rhythm BPMs genuinely diverge for at least some voices on the corpus
- [ ] #6 test_two_layer_pulse still passes (motor-sustain-through-silence preserved)
<!-- SECTION:DESCRIPTION:END -->
