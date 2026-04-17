# Neurodynamics engine

An oscillator-bank engine implementing Neural Resonance Theory (Large et al. 2025, *Musical neurodynamics*, Nat. Rev. Neurosci.) with a live-audio-to-modular bridge built on top. Audio in → phase-locking oscillator dynamics → voice decomposition → OSC stream → CV/MIDI to a eurorack modular. No DAW required at any point.

## The product loop

Four CLI commands compose into the full pipeline. Any combination works; each is a standalone OSC publisher or subscriber.

| Command     | Role                  | Input         | Output                          |
|-------------|-----------------------|---------------|---------------------------------|
| `nd-run`    | Offline engine        | audio file    | parquet state log + OSC         |
| `nd-live`   | Live engine           | mic/Loopback  | OSC (no parquet)                |
| `nd-view`   | Offline viewer        | parquet       | Tk/matplotlib window            |
| `nd-view --live` | Live viewer      | OSC           | Tk/matplotlib window            |
| `nd-route`  | CV/MIDI router        | OSC           | ES-9 CV + MIDI + OSC forward    |

Typical workflows:

```bash
# Offline analysis + visualization of a recording
uv run nd-run  --audio ~/Music/Flights.wav
uv run nd-view --audio ~/Music/Flights.wav

# Live: mic or system-audio capture → modular
uv run nd-live --device "Loopback Audio"  # listen to system audio
uv run nd-view --live                      # watch what the engine sees
uv run nd-route                            # send CV to ES-9
```

## Quickstart

```bash
cd neurodynamics/engine
uv sync --extra dev                          # install deps
uv run pytest                                # 236 tests (~5 min with JIT warmup)
uv run nd-run  --audio path/to/song.wav      # offline: process → output/<stem>.parquet
uv run nd-view --audio path/to/song.wav      # offline viewer
```

Supported audio formats: wav, flac, aiff/aif, ogg, au natively; mp3, m4a, opus, webm via an ffmpeg pipe (requires `brew install ffmpeg`).

## CLI reference

All commands read `config.toml` in the current directory by default. Override with `--config`.

### `nd-run` — offline engine

Processes an audio file end-to-end. Writes per-layer state snapshots to `output/<stem>.parquet` and broadcasts OSC in real time as it runs.

```bash
uv run nd-run [--config config.toml] [--audio PATH] [--output PATH]
```

| Flag         | Default               | Notes                                       |
|--------------|-----------------------|---------------------------------------------|
| `--config`   | `config.toml`         | engine + router config (this file)          |
| `--audio`    | `config.audio.input_file` | any format ffmpeg reads                 |
| `--output`   | `output/<stem>.parquet` | state-log destination                     |

### `nd-live` — live engine

Streams audio from a CoreAudio input device (mic, Loopback, any USB interface), advances the engine through each chunk, emits OSC. No parquet written — live-mode state is consumed by `nd-view --live` and `nd-route` over OSC.

```bash
uv run nd-live [--config config.toml] [--device NAME] [--block-size N]
```

| Flag           | Default       | Notes                                               |
|----------------|---------------|-----------------------------------------------------|
| `--device`     | system default | sounddevice input name; `"Loopback Audio"` captures system audio on macOS |
| `--block-size` | `512`         | samples per audio callback; ~32 ms at 16 kHz       |

List available input devices:

```bash
uv run python -c "import sounddevice as sd; print(sd.query_devices())"
```

### `nd-view` — offline viewer

Tk/matplotlib scrollable viewer with time-synced audio playback. Reads `output/<audio_stem>.parquet` plus the optional `<audio_stem>.weights.npz` for Hebbian W history.

```bash
uv run nd-view [--config config.toml] [--audio PATH] [--state PATH] \
               [--screenshot-at T --screenshot-path OUT]  # headless snapshot
               [--live] [--port PORT]                     # live mode
```

Offline panels: pitch/rhythm heatmaps (scrollable, ±6 s around playhead), oscilloscope-style amplitude profiles, phase-coherence strips, phantom+residual composite, mode-lock constellation overlay, Hebbian W animation, voice ticks on the right edge of the pitch heatmap, banner with key / chord / BPM / consonance / voice count.

`--screenshot-at T` renders a single PNG at time `T` and exits — useful for headless verification without a display.

### `nd-view --live` — live viewer

Same rendering, different data source. Subscribes to an OSC endpoint populated by `nd-live` (or `nd-run`) and animates from a rolling 12 s ring buffer. Scrolls right-to-left in real time; can't scroll into the future by design. Drops W matrix panels (the engine doesn't broadcast W over OSC).

```bash
uv run nd-view --live [--port 57121]
```

Defaults to the first configured endpoint (57121 per default config).

### `nd-route` — CV/MIDI/OSC router

Subscribes to the engine's OSC stream on `[router.osc].listen_port` (57120 by default), applies per-channel scaling from the `[[router.mapping]]` table, and dispatches to CV (sounddevice → ES-9), MIDI (mido → IAC / USB), and OSC forward targets (TouchDesigner, Overtone, chained routers).

```bash
uv run nd-route [--config config.toml]
```

Hardware falls back to mock mode when a device or port isn't present — the router runs fine for OSC-only development without an ES-9 attached.

## Config

`config.toml` is fully commented; the important blocks:

- `[audio]`, `[cochlea]` — sample rate + cochlear filterbank
- `[rhythm_grfnn]`, `[pitch_grfnn]` — main oscillator banks (Hopf params, Hebbian, delay, noise)
- `[motor_grfnn]` — optional predictive motor layer, bidirectionally coupled to rhythm
- `[phantom]` — amplitude/drive thresholds that distinguish phantom from driven regime
- `[state_log]` — snapshot rate + output directory
- `[osc]` — broadcast endpoints (engine fans out to all listed)
- `[router.*]` — CV + MIDI + OSC forward config and the `[[router.mapping]]` table

### OSC fan-out

The engine broadcasts every message to **every** configured endpoint — UDP doesn't share ports across receivers, so router + live viewer + TouchDesigner each bind their own port.

```toml
[osc]
enabled = true
host = "127.0.0.1"
port = 57120              # router default
[[osc.endpoints]]
host = "127.0.0.1"
port = 57121              # nd-view --live
```

## Signals — OSC schema

Every snapshot (60 Hz default) broadcasts the following. Layers: `pitch`, `rhythm`, `motor` (if enabled).

| Path                              | Type           | Meaning |
|-----------------------------------|----------------|---------|
| `/<layer>/amp`                    | float array    | \|z\| — resonance strength per oscillator |
| `/<layer>/phase`                  | float array    | arg(z) — instantaneous phase |
| `/<layer>/phantom`                | int array      | "ringing without drive" per oscillator |
| `/<layer>/drive`                  | float array    | input magnitude received |
| `/<layer>/residual`               | float array    | drive − \|alpha\|·\|z\| — surprise signal |
| `/rhythm/peak/{freq,bpm,phase,idx}` | scalar       | main beat's oscillator + phase |
| `/rhythm/companion/count`         | int            | active phase-locked companions |
| `/rhythm/companion/<i>/{ratio,freq,phase,plv}` | mixed | i'th companion (subdivision clock) |
| `/features/{tempo,key,mode,chord,consonance,...}` | mixed | perceptual rollups |
| `/voice/active_count`             | int            | number of currently-tracked voices |
| `/voice/active_ids`               | int array      | list of active voice ids |
| `/voice/<id>/{active,center_freq,amp,phase,confidence,age_frames}` | mixed | per-voice state |
| `/voice/<id>/rhythm/{freq,bpm,phase,osc_idx,confidence}` | mixed | voice's own rhythm entrainment |
| `/voice/<id>/motor/{freq,bpm,phase,osc_idx,confidence}`  | mixed | voice's motor (anticipated beat) |

Frequency vectors for each layer are stored as parquet metadata (`layer.<name>.f`) — not broadcast over OSC.

## Feature toggles

Every feature is off by default (no behavior change without explicit opt-in). Each layer has its own subsections.

### Hebbian plasticity (`[<layer>.hebbian]`)
Attunement. Weights between co-active oscillators grow with phase-lock, decay otherwise. Produces learned tonal/rhythmic attractors.
```toml
enabled = true
learn_rate = 0.5
weight_decay = 0.05
```
Final `W` persisted to `<stem>.weights.npz`.

### Delay coupling (`[<layer>.delay]`)
Strong anticipation. Self-coupling from `tau` seconds ago lets the network phase-lead a periodic driver.
```toml
tau = 0.1
gain = 0.2
```

### Stochastic noise (`[<layer>.noise]`)
Complex Gaussian kicks per step with √dt Wiener scaling. Breaks symmetry, keeps phantom activity alive.
```toml
amp = 0.02
seed = 0
```

### Motor layer (`[motor_grfnn]`)
Two-layer pulse network. Second rhythm-scale GrFNN bidirectionally coupled to the sensory rhythm network. Sustains pulse activity across audio silences (the "felt beat" prediction).
```toml
enabled = true
forward_gain = 0.1
backward_gain = 0.03
```

## Voice extraction (Phase 1-3 of `task-011`)

`neurodynamics.voices`:

- `extract_voices(window, prev_state)` — phase-coherence clustering of pitch oscillators into dynamic voice identities. Hungarian matching across frames keeps voice IDs stable through silences and splits.
- `extract_voice_rhythms(window, voice_state)` — DFT each voice's amplitude envelope against the rhythm GrFNN bank. Each voice gets its own tempo, beat phase, oscillator index.
- `extract_voice_motor(window, voice_state)` — same DFT against the motor GrFNN bank. Each voice gets an anticipated-beat phase (motor sustains through silences via bidirectional coupling).

All three compose orthogonally. Engine + live path runs all three on a rolling 2.5 s window; voices broadcast at 20 Hz. See `task-011` and `task-012` for per-voice motor network upgrade path.

## Derived-signal utilities

### Mode-lock detection (`neurodynamics.modelock`)
PLV between two phase trajectories, O(n²) pair-scanner for locks at small-integer ratios.

### Perceptual extractors (`neurodynamics.perceptual`)
- `extract_tempo` — peak rhythm oscillator + confidence
- `extract_key` — Krumhansl-Schmuckler template matching on learned pitch W
- `extract_chord` — 8 templates (maj/min/dom7/maj7/min7/dim/sus2/sus4)
- `extract_consonance` — amplitude-weighted NRT stability ratio mean
- `extract_rhythm_structure` — peak + phase-locked companions (multi-clock)

## Router — CV/MIDI/OSC output

`neurodynamics.router`:

- `CVBackend` — sounddevice `OutputStream` to a DC-coupled interface (ES-9). Per-channel voltage vector; sounddevice samples into the output as DC.
- `MIDIBackend` — mido wraps clock messages (24 ppq), note-on/off, CC.
- `OSCForwardBackend` — re-broadcast to additional endpoints.

Pure scaling functions:

- `scale_tonic_to_1v_per_oct(pc, octave)` — pitch-class name → volts
- `scale_phase_to_sawtooth(phase)` — ±π → 0-5 V with clean wrap
- `scale_unit_to_5v(unit)` — [0, 1] → [0, 5] V clipped
- `phase_wrapped(current, prev)` — trigger detector for "beat just landed"

Mapping table in `config.toml`:

```toml
[[router.mapping]]
source = "/features/key"
output = "cv/0"
scale  = "1V_per_oct"

[[router.mapping]]
source = "/voice/0/rhythm/phase"
output = "cv/2"
scale  = "sawtooth_0_5V"
```

### Expert Sleepers ES-9 setup

1. Put the ES-9 in **class-compliant mode** via its config tool (usually the ship default).
2. Plug in USB. macOS treats it as a 14-channel CoreAudio output with no driver install.
3. In `config.toml`, enable the CV backend:
   ```toml
   [router.cv]
   enabled = true
   # device = "ES-9"      # pin to the device; omit for system default
   channels = 14
   sample_rate = 48000
   ```
4. `uv run nd-route`.

No DAW, no Max/M4L, no Ableton. The router writes directly to CoreAudio via sounddevice. Works headless on any Mac — laptop in a bag, M1 mini in a closet, whatever.

## Testing

```bash
uv run pytest                                # 236 tests
uv run python -m tests.benchmark_engine      # per-stage profile
uv run python -m tests.rip_corpus            # rebuild real-audio corpus
uv run python -m tests.generate_baselines    # after intentional changes
```

**Test suites:**
- `test_grfnn`, `test_cochlea`, `test_hebbian`, `test_delay`, `test_noise`, `test_modelock`, `test_residual` — unit tests on core math
- `test_pipeline`, `test_regression`, `test_two_layer_pulse`, `test_w_snapshots` — integration + regression
- `test_perceptual` — key/chord/consonance/rhythm extractors on synthetic state
- `test_voices` — voice identity clustering + tracking + per-voice rhythm/motor on synthetic state
- `test_voices_real_audio` — 10-track electronic-music corpus (Four Tet, Fred again.., Chemical Brothers, Underworld, Function, Sandwell District, Burial, Disclosure, KPop Demon Hunters, Daft Punk). Gitignored; regenerate via `rip_corpus`.
- `test_router` — scale functions, OSC dispatch, end-to-end UDP round-trip with mock backends
- `test_live` — `LiveEngine.process()` on synthetic chunks; snapshot timing; faster-than-realtime guard
- `test_live_view` — ring buffer + OSC handler integration

All features follow red → green: test first, then implement. Regression baselines in `tests/baselines/` lock numeric behavior; regenerate with `generate_baselines` after intentional changes.

## Performance

Engine hot path (pitch GrFNN RK4 at audio rate) is numba-JIT'd. On an Apple M-series laptop:

- Per-sample `step()` API: ~1.2× realtime
- Batched `step_many()` API: ~11× realtime

Live mode uses `step_many` on each audio chunk so the engine consumes input faster than it arrives.

## Downstream consumer cookbook

### TouchDesigner / Overtone / custom
Add an `[[osc.endpoints]]` entry pointing at your consumer's port. Subscribe to whichever subset of the namespace you want — the engine broadcasts the whole stream regardless.

### Modular via ES-9 (see above)
The `[[router.mapping]]` table is the creative surface. Wire `/features/key` to a quantizer, `/voice/0/rhythm/phase` to a clock input, `/voice/<id>/amp` to a VCA, etc.

### Particle system
- `/rhythm/amp` → emission rate per band
- `/rhythm/phase` → angular velocity / orbital position
- `/pitch/amp` → color hue per pitch class
- `/voice/*/center_freq` → particle pitch / size
- `/rhythm/phantom` → ghost emitters
- `/*/residual` → turbulence

### Generative synth
- `/voice/*/rhythm/phase` → per-voice gate triggers
- `/voice/*/motor/phase` → anticipated-beat envelope pre-attack
- `/*/phantom` → synthesized "missing pulse" percussion
- Hebbian W (post-run) → play back learned attractors as echoes

## Where the work is

See `backlog/tasks/`. High-leverage pending:

- `task-012` per-voice motor GrFNN pool (deep upgrade of Phase 3 from shared-bank read)
- `task-003` brainstem / sum-difference tone layer
- `task-013` meter detection, `task-014` groove index
- `task-002` pitch-class folding viewer mode
- `task-006` TouchDesigner demo, `task-007` Overtone subscriber

Closed: `task-001` (partial), `task-004` (perceptual extractors), `task-005` (live audio mode), `task-011` (router through all 4 phases), `task-015` (engine optimization).
