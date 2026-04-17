---
id: task-004
title: Perceptual feature extractors
status: Done
assignee: []
created_date: '2026-04-16 17:52'
updated_date: '2026-04-17 02:17'
labels:
  - engine
  - features
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Higher-level rollups derived from oscillator state. Not raw state but *interpretations* that are actionable for downstream consumers:
- key detection (from learned pitch W — the implied tonic)
- beat tracking (peak rhythm oscillator + phase, smoothed)
- tempo confidence (sharpness of peak rhythm oscillator)
- implied harmony (active pitch oscillators weighted against learned tonal hierarchy)
- meter detection (dominant stable ratio in rhythm network)
- groove index (syncopation × phase coherence)
- consonance/dissonance scalar (sum of stability contributions from active pitch-pair locks)

Design as a state-slice-agnostic module so it works on both offline parquet reads AND live in-memory state (unblocks live mode later).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 New neurodynamics.perceptual module
- [x] #2 Each extractor takes a state slice, returns a scalar or dict
- [x] #3 Unit tests with synthetic audio of known key/tempo/etc.
- [x] #4 Viewer shows text overlays: key, tempo, consonance bar
- [x] #5 Results emitted over OSC too (e.g. /features/key, /features/tempo)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed 2026-04-16. Tier A (key + tempo + consonance) and the chord portion of Tier B landed in commits ff72c3e, fc47897. Meter detection and groove index from Tier B/C deferred to dedicated tickets (task-012 meter, task-013 groove).
<!-- SECTION:NOTES:END -->
