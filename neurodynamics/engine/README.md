# Neurodynamics engine

An oscillator-bank engine implementing Neural Resonance Theory (Large et al. 2025, "Musical neurodynamics", *Nat. Rev. Neurosci.*). Audio in → phase-locking oscillator dynamics out → state log + OSC stream → whatever downstream consumer you point at it (particle systems, synthesis, viz).

## What's in the box

```
engine/
├── pyproject.toml           uv project, hatchling-built wheel
├── config.toml              all tunable parameters (well-commented)
├── src/neurodynamics/
│   ├── cochlea.py           FIR gammatone filterbank + Hilbert envelope
│   ├── grfnn.py             canonical-form oscillator bank (Hopf normal form)
│   ├── modelock.py          phase-locking value + mode-lock detection
│   ├── osc_out.py           UDP OSC broadcaster
│   ├── run.py               audio → cochlea → dual GrFNN → state + OSC
│   ├── state_log.py         parquet snapshot logger
│   └── viewer.py            Tk/matplotlib scrollable viewer with waterfalls
├── tests/                   pytest suite (52 tests)
│   ├── conftest.py          fixtures: synthetic audio, minimal configs
│   ├── test_cochlea.py      filterbank unit tests
│   ├── test_grfnn.py        oscillator math unit tests
│   ├── test_hebbian.py      plasticity tests
│   ├── test_delay.py        delay coupling tests
│   ├── test_noise.py        stochastic noise tests
│   ├── test_modelock.py     mode-lock detection tests
│   ├── test_residual.py     prediction residual tests
│   ├── test_pipeline.py     end-to-end integration tests
│   ├── test_regression.py   baseline-comparison regression tests
│   ├── generate_baselines.py   regenerates regression baselines
│   └── baselines/*.json     committed regression baselines
└── output/                  per-audio-file state.parquet + .weights.npz
```

## How to run

```bash
cd neurodynamics/engine
uv sync --extra dev                          # install deps
uv run nd-run  --audio path/to/song.wav      # process → output/<stem>.parquet
uv run nd-view --audio path/to/song.wav      # scrollable time-synced viewer
uv run pytest                                # all 52 tests (~20 s)
```

Supported audio formats: wav, flac, aiff/aif, ogg, au natively; mp3, m4a, opus, webm via an ffmpeg pipe (requires `brew install ffmpeg`).

## Signals exposed per layer (rhythm + pitch)

Every snapshot (60 Hz default) writes these to `state.parquet` and broadcasts them over OSC on `127.0.0.1:57120`:

| Field       | Shape       | OSC path               | Meaning |
|-------------|-------------|------------------------|---------|
| `z_real`, `z_imag` | n_osc | —                | Complex oscillator state |
| `amp`       | n_osc       | `/<layer>/amp`         | \|z\| — resonance strength |
| `phase`     | n_osc       | `/<layer>/phase`       | arg(z) — instantaneous phase |
| `phantom`   | n_osc bool  | `/<layer>/phantom`     | "Ringing without drive" — the inner-voice signal |
| `drive`     | n_osc       | `/<layer>/drive`       | Input magnitude received this step |
| `residual`  | n_osc       | `/<layer>/residual`    | Prediction residual: `drive − \|alpha\|·\|z\|`. Positive = surprise/onset; negative = phantom regime |

Frequency vectors `layer.<name>.f` are stored in the parquet metadata.

If Hebbian plasticity is enabled on a layer, the final learned weight matrix is saved alongside the state file as `output/<stem>.weights.npz` with entries `rhythm_W`, `rhythm_f`, `pitch_W`, `pitch_f`.

## Features enabled via config

Every feature is off by default (no behavior change) and gated in `config.toml`. Each layer has its own subsections.

### Hebbian plasticity (`[rhythm_grfnn.hebbian]`, `[pitch_grfnn.hebbian]`)
Attunement mechanism. Complex weights between co-active oscillators grow when they phase-lock and decay otherwise. Produces learned tonal/rhythmic attractors, implied harmony, familiar-groove memory.
```toml
enabled = true
learn_rate = 0.5       # kappa in dW/dt = -lambda·W + kappa·z·conj(z')
weight_decay = 0.05    # lambda
```
Final `W` persisted to `<stem>.weights.npz`.

### Delay coupling (`[rhythm_grfnn.delay]`)
Strong anticipation. Ring-buffered self-coupling from `tau` seconds ago lets the network phase-lead a periodic driver.
```toml
tau = 0.1              # seconds of delay
gain = 0.2             # feedback strength
```

### Stochastic noise (`[rhythm_grfnn.noise]`, `[pitch_grfnn.noise]`)
Small complex Gaussian kicks per step with √dt Wiener scaling. Breaks symmetry, keeps phantom activity alive during silence, models intrinsic neural fluctuation.
```toml
amp = 0.02
seed = 0               # reproducible when >0
```

## Derived-signal utilities

### Mode-lock detection (`neurodynamics.modelock`)
Phase-locking value (PLV) between any two phase trajectories, and an O(n²) pair-scanner that reports all pairs locked at small-integer ratios.
```python
from neurodynamics.modelock import phase_locking_value, detect_mode_locks
locks = detect_mode_locks(phases, threshold=0.85,
                          ratios=[(1,1),(2,1),(3,2),(7,4)])
```
Use on rolling windows of `phase` from the state log to emit a stream of active polyrhythm / consonance detections.

## Testing philosophy

All features go red → green: test first, then implement. Running `uv run pytest` takes ~20 s.

**Unit tests** (`test_grfnn`, `test_cochlea`, `test_hebbian`, `test_delay`, `test_noise`, `test_modelock`, `test_residual`) verify core math and invariants against analytically-derivable properties, not mocks.

**Integration tests** (`test_pipeline`) run the whole pipeline on synthetic audio with known properties (2 Hz pulse train, 440 Hz sine) and assert the expected oscillators activate, the output schema is correct, and the engine is deterministic.

**Regression tests** (`test_regression`) compare a run's summary statistics against committed baselines in `tests/baselines/*.json`. Any behavioral drift surfaces as a test failure. When changing defaults or adding features intentionally, regenerate baselines:
```bash
uv run python -m tests.generate_baselines
```

## Downstream consumer cookbook

### Particle system driver
Subscribe to OSC on `127.0.0.1:57120`:
- `/rhythm/amp` → emission rate per band (slow oscillators = sparse bursts)
- `/rhythm/phase` → angular velocity / orbital position
- `/pitch/amp` → color hue (12 pitch classes → color wheel)
- `/rhythm/phantom` → ghost emitters (active without audio)
- `/rhythm/residual` → turbulence (positive = scattered, negative = coherent flow)

### Generative synth driver
- `/rhythm/phantom` → trigger synthesized "missing pulse" percussion
- `/pitch/phantom` → implied-harmony synth pad at ringing pitch frequencies
- `/*/residual` → tension/release (positive residual = filter opens, negative = closes)
- Hebbian `W` (post-run) → play back the learned attractors as echoes

## What's intentionally not here yet

- **Two-layer pulse network** (Tier 1) — second coupled rhythm GrFNN to reproduce NRT's missing-pulse motor-cortex prediction. Current rhythm layer shows *modes* but not the full prediction.
- **Live mic input** — current pipeline is offline (audio file → parquet). Real-time streaming from `sounddevice.InputStream` would need a different step-loop structure.
- **Brainstem / frequency-following layer** — third GrFNN producing summation/difference tones at cochlear-nucleus resolution.
- **Perceptual feature extractors** — key detection, beat tracking, implied harmony rollups. Layer cleanly on top of the state log + mode-lock module.
