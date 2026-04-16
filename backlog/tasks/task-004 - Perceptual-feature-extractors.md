---
id: task-004
title: Perceptual feature extractors
status: To Do
assignee: []
created_date: '2026-04-16 17:52'
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
- [ ] #1 New neurodynamics.perceptual module
- [ ] #2 Each extractor takes a state slice, returns a scalar or dict
- [ ] #3 Unit tests with synthetic audio of known key/tempo/etc.
- [ ] #4 Viewer shows text overlays: key, tempo, consonance bar
- [ ] #5 Results emitted over OSC too (e.g. /features/key, /features/tempo)
<!-- AC:END -->
