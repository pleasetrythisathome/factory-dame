---
reviewers:
  - scarlet@aswritten.ai
---

# Hardware path for the live-NRT-to-modular vision: M1 Mac mini, not Raspberry Pi

## Correction (2026-04-16)

The earlier vision memory (`2026-04-16-4-vision-live-nrt-to-modular-bridge`) positioned the mid-term hardware path as "Raspberry Pi 5 / LattePanda / similar SBC running the engine headlessly." That was premature. Scarlet clarified mid-session:

> "i have a mac mini m1 8gb ram that's sitting unused. but my laptop is fine for now and maybe ever"

## Updated hardware path

- **Short-term:** MacBook in a bag, USB to ES-9, lid closed. Caffeinate keeps audio running. This is the "laptop out of sight" use case and it works today. No changes needed.
- **Mid-term:** M1 Mac mini 8GB — already owned, sitting idle, roughly equivalent single-thread performance to the MacBook. Headless via SSH. No DAW, no custom hardware, no engineering investment.
- **Pi / custom SBC:** not needed for Scarlet's use case. Shelved unless and until external demand materializes for a low-cost form factor.

## Ripple through task-015 (engine optimization)

The ticket previously framed optimization as "enable Pi deployment." That framing is obsolete. The *actual* reason to optimize is enabling **live mode (task-005)**: the engine currently runs at 3× realtime on M-series, so streaming audio chunks at 1× realtime is impossible without speedup. Even on the M1 mini, live needs ~3× engine speedup.

numba JIT on the Hopf RK4 integrator is the remaining live-mode gate, not a hardware-compatibility concern.

## Why this matters beyond the hardware choice

Re-reading this against `Narrative_NoveltyAsNorthStar` and the deep-first principle: the hardware path was a distraction. The actual novelty lives in the engine + voice primitives + router, not in where they run. Deploying on whatever hardware Scarlet already owns keeps attention on the novel capability rather than on build-vs-buy engineering for a speculative SBC port.

The deep-first check that applies here: when hardware-compatibility optimization competes with novel-capability investment, choose the latter. Scarlet owns adequate hardware; use it.

## Scope

Factory-dame specific. The M1 mini is the household machine Scarlet chose for this; the reasoning doesn't generalize to aswritten or business projects (which have different deployment pressures).