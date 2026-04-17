---
id: task-005
title: Live audio mode (nd-live)
status: Done
assignee: []
created_date: '2026-04-16 17:52'
updated_date: '2026-04-17 03:06'
labels:
  - engine
  - live
  - architecture
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Real-time streaming from Loopback (or mic). User has Loopback installed for system audio capture.

Architecture requirements:
- new nd-live entry point using sounddevice.InputStream with a small chunk size (~20–50 ms)
- engine restructured to process incoming audio chunks rather than full arrays (the run() loop becomes a callback that advances oscillators by one chunk at a time)
- viewer reads state from an in-memory rolling buffer, not parquet
- mode-lock scan becomes rolling/online (can't precompute) — may need to use a shorter window or lower refresh rate to stay within frame budget
- W snapshot logging still works (cadence-triggered in the callback)

Important: keep offline nd-run working. Live mode is additive, not replacing.

Key risk: matplotlib TkAgg is single-threaded. The audio callback runs on a separate thread; viewer updates happen on main. Need a thread-safe rolling buffer (numpy array + threading.Lock, or a single-producer-single-consumer ring).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 uv run nd-live subscribes to default input device
- [ ] #2 Viewer reads live state and scrolls in real time
- [ ] #3 Offline nd-run path unchanged
- [ ] #4 No audio dropouts on a modest laptop
- [ ] #5 Tests: can instantiate + stop without hanging; offline parity test ensures streaming same audio matches file processing within tolerance
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed 2026-04-17 via nd-live in commits a656245 (numba JIT making the engine faster than realtime) and 2514135 (LiveEngine + sounddevice callback + snapshot-aware chunked processing). Hardware verification (ES-9 + live input) is the operator's side.
<!-- SECTION:NOTES:END -->
