# Neurodynamics Engine — Session Handoff (2026-04-16)

This document is a comprehensive handoff for anyone (or any future AI session) picking up the neurodynamics engine project. Read this before touching code.

## What This Is

A working real-time audio-analysis engine implementing **Neural Resonance Theory** (Harding/Kim/Large 2025, Nat. Rev. Neurosci.) as the driver for generative particle systems and music synthesis. The engine converts audio into a rich multi-channel state stream (amplitude, phase, phantom activation, prediction residual, learned Hebbian weights) over OSC + parquet, and there's a scrollable time-synced viewer for introspection.

Full overview in the aswritten memory: `.aswritten/memories/2026-04-15-1-neurodynamics-engine-project.md`.

Read papers: `neurodynamics/papers/` — three Nat. Rev. Neurosci. items (Large's 2025 Perspective, Stone & Buks critique, Large's reply).

## Where You Are Right Now

**Project lives at** `neurodynamics/engine/` inside the factory-dame repo.

**Entry points:**
- `uv run nd-run --audio PATH` — process audio → `output/<stem>.parquet` + `output/<stem>.weights.npz`
- `uv run nd-view --audio PATH` — scrollable Tk viewer with time-synced audio playback
- `uv run pytest` — the test suite (59 tests currently, all green)
- `uv run python -m tests.generate_baselines` — regenerate regression baselines

**What's implemented:**
1. Canonical Hopf GrFNN oscillator bank (RK4 integrator, mean-field normalized Hebbian coupling)
2. FIR gammatone cochlear front-end (unit-gain normalized, avoids IIR instability at low fc)
3. Dual-timescale networks: rhythm (0.5–10 Hz) + pitch (30–4000 Hz)
4. Hebbian plasticity with weight decay
5. Delay-coupled oscillators for strong anticipation
6. Stochastic noise per step (√dt Wiener scaling)
7. Mode-lock detection (vectorized PLV over integer ratios)
8. Per-oscillator prediction residual (surprise signal)
9. Two-layer pulse network (motor-cortex sensory↔motor bidirectional coupling)
10. Weight matrix history snapshotting + replay
11. OSC broadcaster + parquet state logger
12. Scrollable Tk viewer with ink-on-paper Feltron aesthetic

**Viewer panels (top → bottom):**
- Pitch heatmap (navy, PowerNorm γ=0.35 for low-amp detail)
- Rhythm heatmap (terracotta, γ=0.6)
- Motor heatmap (brown, earthy) — two-layer pulse
- Pitch / rhythm scope traces (oscilloscope-style amplitude profiles + history)
- Phase coherence strips (per-oscillator phase stability)
- Phantom + residual composite strips (red luminance)
- Mode-lock constellation overlay on heatmaps (colored vertical segments, precomputed at startup)
- Optional animated W matrix panels (bottom, when Hebbian is enabled)

## Current Tunings (What's On by Default)

All feature toggles live in `neurodynamics/engine/config.toml`. Current state:

| Feature | Status | Config block | Notes |
|---------|--------|--------------|-------|
| Rhythm Hebbian | ON | `[rhythm_grfnn.hebbian]` | learn_rate=0.5, decay=0.05 |
| Rhythm delay coupling | ON | `[rhythm_grfnn.delay]` | tau=0.1s, gain=0.2 |
| Rhythm noise | ON | `[rhythm_grfnn.noise]` | amp=0.02 |
| Pitch Hebbian | ON | `[pitch_grfnn.hebbian]` | learn_rate=0.2, decay=0.02 |
| Pitch noise | ON | `[pitch_grfnn.noise]` | amp=0.01 |
| Motor layer | ON | `[motor_grfnn]` | alpha=-0.05, forward=0.3, backward=0.03 |
| W history snapshotting | ON | `[state_log]` | w_snapshot_hz=2 |
| OSC broadcast | ON | `[osc]` | 127.0.0.1:57120 |

These aren't "canonical NRT params" — they're empirically tuned to avoid integrator saturation on mastered music. Expect to retune if you change network sizes or audio characteristics.

## The Hard-Won Lessons

Things the test suite or user testing surfaced that you should not re-learn:

1. **`scipy.signal.gammatone('iir')` is unstable at low fc/fs ratios** (pole magnitude > 1). We use FIR form exclusively. See `src/neurodynamics/cochlea.py`.

2. **Pitch channel weighting must be sharp (~one-hot)** — default sharpness=15. Broad weighting causes destructive phase interference across bands at the driven frequency.

3. **Hebbian coupling requires mean-field normalization** — `(W @ z) / n`. Without it, network size affects dynamics scale and Hebbian self-reinforces into runaway. See `grfnn.py::_deriv`.

4. **The `|z|^4 / (1 - eps*|z|^2)` quintic term has a pole** at `|z| = 1/√ε`. We saturate `abs2` at 0.95/ε and post-clamp `|z|` at 0.98/√ε. Without these guards the integrator blows to NaN under heavy drive.

5. **Matplotlib blit requires explicitly returning every animated artist** from the update function. Forgetting one means it disappears at rest and only reappears while scrolling.

6. **Mode-lock PLV scan is O(n² × ratios × window)** — too slow per-frame. We precompute all locks at viewer startup (~2s for a 6-minute track) and the update loop just indexes in. This is fine for offline but will need re-architecting for live mode.

7. **tk CPU usage can hit 100%** even on idle because the animation runs every 33ms. Not a bug, but don't be alarmed.

8. **Weight file size scales linearly with duration** — 138MB for a 6-minute track at 2 Hz W snapshotting with 100×100 pitch weights. Will need downsampling or sparse storage for long sessions.

9. **Motor layer needs relatively strong damping (alpha=-0.05)** on mastered audio — lighter damping saturates the integrator. The original NRT-style `alpha=-0.005` only works for sparse test inputs.

10. **User strongly prefers ink-on-paper Feltron aesthetic with Futura** — cream background `#F5F0E6`, narrow monochrome palettes (navy/terracotta/brown), clean editorial labels in ALL CAPS. Do not revert to matplotlib defaults.

## The Test Suite Is Non-Negotiable

**59 tests across 9 files. All green.** Run `uv run pytest` before and after any change.

TDD workflow: write a failing test first, then the implementation to make it pass. Every feature added this session followed this pattern. The test suite caught multiple real bugs (the IIR gammatone instability, the channel-weighting interference, pitch gain mistuning) that would have been silent accuracy problems otherwise.

The regression baselines in `tests/baselines/*.json` lock deterministic output against drift from dependency updates. When you intentionally change engine defaults, regenerate with:
```
uv run python -m tests.generate_baselines
```

## Backlog of Next Work

Tickets are in `backlog/tasks/`. Prioritized roughly in the order I'd tackle them if I were starting fresh:

**Viz polish (fast follows):**
- `task-001`: mode-lock constellation as arcs instead of vertical lanes
- `task-002`: pitch-class folding option for pitch heatmap (octave-equivalent view)

**Deeper NRT fidelity:**
- `task-003`: brainstem / frequency-following layer (sum/difference tones)
- `task-004`: perceptual feature extractors (key, tempo, implied harmony, groove index, consonance scalar)

**Architectural pivot:**
- `task-005`: live audio mode — `nd-live` streaming from Loopback (user has Loopback installed)

**Consumer integrations (last):**
- `task-006`: TouchDesigner / Three.js particle demo consuming OSC
- `task-007`: Clojure/Overtone OSC subscriber
- `task-008`: Bambu X1 Carbon 3D form generation from oscillator state

**Infrastructure:**
- `task-009`: W matrix file-size downsampling strategy for long audio
- `task-010`: OSC message schema doc for downstream consumers

## On Working with Scarlet

Read the `CLAUDE.md` files at both the repo root and user-global level first — they document Scarlet's working preferences in detail. Key points for this project specifically:

- **This is a creative side project**, not aswritten work. The tone is playful discovery, not velocity tracking. Avoid process overhead.
- **Show don't tell**: run the viewer to verify visual changes. Describe what should be on screen and then ask "what do you see?" — iterate from there.
- **Scarlet values aesthetic judgment**: if the user says something looks noisy or wrong, redesign rather than defend. The ink-on-paper + Futura direction came from exactly such feedback.
- **Test-driven**: Scarlet explicitly asked for tests as backpressure on feature velocity. Honor that.
- **Honest about provenance**: when you don't know something, say so rather than guess. Scarlet spotted and appreciated my honest "I shouldn't lean on that without verifying" about the GrFNN Toolbox port.

## Quickstart Commands for Your First Session

```bash
cd neurodynamics/engine

# Sanity check — all tests should pass
uv run pytest

# Process a test audio file
uv run nd-run --audio test.wav

# View it
uv run nd-view --audio test.wav

# Run on real audio (Scarlet's reference piece)
uv run nd-view --audio "/Users/scarletdame/Dropbox/music/masters/Tea With Someone Dangerous/Flights.wav"
```

The engine's deterministic on fixed audio (noise uses seeded RNG), so you can reproduce any behavior exactly.

## Orient Yourself Before Acting

1. Read the aswritten project memory (`2026-04-15-1-neurodynamics-engine-project.md`) for the full project context.
2. Read `neurodynamics/engine/README.md` for the engine's architectural layout.
3. Skim `config.toml` to see what's wired on.
4. Run the test suite. Confirm green.
5. Run the viewer on Flights.wav to see the current state visually.
6. Check `backlog/tasks/` for the next ticket.

Good luck. Build carefully. The foundation is solid — don't regress it.
