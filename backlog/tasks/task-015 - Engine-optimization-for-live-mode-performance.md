---
id: task-015
title: Engine optimization for live-mode performance
status: To Do
assignee: []
created_date: '2026-04-17 02:30'
updated_date: '2026-04-17 02:31'
labels:
  - engine
  - performance
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Current engine runs at ~3x realtime on Apple M-series. Live audio mode (task-005) requires ≤1x realtime — streaming from mic/Loopback can't tolerate the engine falling behind. This ticket is specifically about enabling live on hardware Scarlet already has: MacBook + an M1 Mac mini sitting unused. Pi-class deployment is YAGNI unless someone else wants a Pi.

## Approach

1. numba @jit on GrFNN.step — probably 5-10x speedup from eliminating Python overhead per sample. Primary lever.
2. Vectorize per-oscillator operations across the bank where feasible (RK4 slopes for all oscillators at once).
3. If still insufficient: Cython or Rust for the RK4 inner loop.
4. Reduce network sizes as an opt-in config preset (not a default) if live is still infeasible at full resolution.

## Why this gates task-005

Live mode streams audio chunks from sounddevice InputStream. If the engine consumes a chunk slower than realtime, the input buffer backlogs and eventually overruns. At 3x realtime today, every second of audio takes 3 seconds of engine time — immediate failure.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 #1 Benchmark current engine on a representative track (90s) — wall time, output per-phase breakdown (pitch RK4, rhythm RK4, cochlea, Hebbian)
- [ ] #2 #2 numba JIT integration on the identified hot loops
- [ ] #3 #3 Regression: engine produces bit-identical output after JIT (synthetic test + parquet diff)
- [ ] #4 #4 Benchmark after JIT — target ≤1x realtime on Apple M-series
- [ ] #5 #5 Benchmark on Pi 5 or equivalent ARM — target ≤2x realtime (acceptable for headless offline, not live)
<!-- SECTION:DESCRIPTION:END -->

- [ ] #6 #6 If still insufficient on Pi: Cython/Rust fallback or smaller-network config preset
<!-- AC:END -->
