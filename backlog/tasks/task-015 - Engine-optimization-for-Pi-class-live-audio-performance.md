---
id: task-015
title: Engine optimization for Pi-class / live audio performance
status: To Do
assignee: []
created_date: '2026-04-17 02:30'
labels:
  - engine
  - performance
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Current engine runs at ~3x realtime on Apple M-series. Live audio (task-005) requires ≤1x realtime; Pi-class deployment requires ≤0.5x. The gap is mostly Python/numpy overhead in the Hopf RK4 integrator at audio rate (pitch GrFNN at 16 kHz).

## Approach

1. numba @jit on GrFNN.step — probably 5-10x speedup from eliminating Python overhead per sample
2. Vectorize per-oscillator operations across the bank where feasible (the RK4 slopes can be computed for all oscillators at once)
3. If still insufficient: Cython or Rust for the RK4 inner loop (another 2-3x)
4. Reduce network sizes (100 pitch → 50, 50 rhythm → 30) — cuts compute 2x but loses frequency resolution. Consider as a config preset for headless-box deployment rather than default.

## Why this blocks live

Live mode (task-005) needs streaming from Loopback / mic at 16 kHz. The engine must consume audio chunks and produce state at ≤1x realtime or the input buffer overruns. Currently impossible at 3x realtime.

## Why this blocks headless-box

Raspberry Pi 5 is ~25% of M2 single-thread → ~12x realtime without optimization. x86 mini PC (NUC13, LattePanda) works today, but Pi is the most compelling form factor for the "boot to modular" box.

## Acceptance Criteria

- [ ] #1 Benchmark current engine on a representative track (90s) — wall time, output per-phase breakdown (pitch RK4, rhythm RK4, cochlea, Hebbian)
- [ ] #2 numba JIT integration on the identified hot loops
- [ ] #3 Regression: engine produces bit-identical output after JIT (synthetic test + parquet diff)
- [ ] #4 Benchmark after JIT — target ≤1x realtime on Apple M-series
- [ ] #5 Benchmark on Pi 5 or equivalent ARM — target ≤2x realtime (acceptable for headless offline, not live)
- [ ] #6 If still insufficient on Pi: Cython/Rust fallback or smaller-network config preset
<!-- SECTION:DESCRIPTION:END -->
