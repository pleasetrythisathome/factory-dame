---
reviewers:
  - scarlet@aswritten.ai
---

# Feedback: novelty is the whole point, and build deep-first not shallow-first

## The two claims, in Scarlet's words

Two interlocking pieces of feedback surfaced during the voice-extraction architecture discussion (2026-04-16). Both are project-strategy-level, not implementation tactics.

### 1. Novelty is the whole point

> "I have no goal with this project unless it does something truly novel and incredible."

Factory-dame exists specifically to do something that couldn't exist before AI-enabled complex-systems development unlocked it. Conventional beat trackers, chromagrams, and filter banks already exist. If the project ends up replicating what DAW plugins and eurorack modules already do, there's no reason to have built it.

Concretely this means:
- "Pragmatic" is not a north star. "Does this produce a capability that genuinely didn't exist before?" is.
- When choosing between a proven-but-boring approach and a harder-but-novel approach, default to the novel one — even if it's more work, even if the first version is rough.
- The NRT analysis-is-synthesis property (`Narrative_AnalysisIsSynthesis`) is the specific well of novelty. Every architectural decision should be checked: is this choice leveraging NRT's distinctive properties, or is this choice one that could have been made in a conventional DSP pipeline?
- Shipping a "competent NRT-powered beat tracker" is a failure mode. Shipping "per-voice dynamic phase-coherent motor networks that generate polyrhythmic CV for arbitrary live audio" is the goal.

### 2. Build deep-first, not shallow-first

> "I see you consistently trying to make decisions about doing things in small, progressive components and then coming back to it later. That's correct in theory, but we want to change the focus there and do things where we build the fully baked, full-depth novel capability of single components first and then assemble those to create the full system. Instead of doing the deep or the shallow version of a component, building the full shallow system and then making it deep. What happens when you do that is you end up with a fully shallow system that you then go back and try to make deep, and you never arrive at it. We want to go deep on these things first and then slowly build out."

This is a direct correction to my default engineering pattern. I kept proposing phased plans that looked like:

- P1: simple version of component A
- P2: simple version of component B
- P3: simple version of component C
- P4: go back and make everything deep

The failure mode Scarlet is naming: you ship the shallow system, it works, the pressure to go deep evaporates because the shallow version is "fine," and the novel capability never materializes. You end up with a competent implementation of a conventional approach.

The alternative:

- P1: component A built to publishable/novel depth. Not an MVP, the real thing.
- P2: component B built the same way.
- P3: component C same.
- Each phase is a standalone novel capability you could ship.

This is slower per-phase but arrives at genuinely novel ground. The horizontal shallow sweep never does.

## How to apply this going forward

- When I propose a phased plan, each phase should be a complete, deep, publishable component — not a "minimal viable" version pending later deepening.
- When I'm tempted to say "we can improve this later," stop and ask: am I deferring depth that will never get added?
- Default to the more-ambitious architectural choice over the pragmatic one, unless there's a specific reason (time, budget, external constraint) the ambitious one is unworkable in the session.
- Check every scope decision against the novelty criterion: does this actually leverage NRT's distinctive properties, or is this a choice that makes the project more conventional?
- Volume-over-perfection still applies to *content and artifacts* (from earlier feedback) — but NOT to core technical capabilities. Content goes wide; capabilities go deep.

## Scope

Both claims are factory-dame-specific. Aswritten and business projects operate on different logic (velocity, GTM, etc.) and have their own strategies documented elsewhere.

## Ripple through existing work and memory

- `Narrative_LiveNRTModularBridge` — still the product thesis, but reframes from "ship something functional" to "ship something that couldn't exist before."
- `Narrative_AnalysisIsSynthesis` — reinforces. The novelty IS in the analysis-is-synthesis property; conventional approaches can't get there.
- `Narrative_CreativeCounterweightToAswritten` — now includes the constraint that the creative project exists to be novel, not merely to be creative. A competent replica of existing tech doesn't satisfy the counterweight role.
- Voice extraction architecture decisions (dynamic count, phase-coherence clustering, per-voice motor networks) were chosen per this rubric. The "fixed 4 bands" option I initially proposed was the shallow-first trap.

## Immediate applications

The voice-extraction work queued for `task-011` (CV/MIDI/OSC router) should be built deep-first:

- Phase 1 is **voice identity done to publishable depth** — phase-coherence clustering, voice tracking across frames, tested on multiple real tracks, visually legible. Not "simple threshold clustering we'll improve later."
- Phase 2 adds per-voice rhythm association to the same depth.
- Phase 3 adds per-voice motor coupling (reading existing motor state as multi-voice; or per-voice motor network pool if reading proves insufficient).
- Each phase you could ship as a standalone novel capability.
