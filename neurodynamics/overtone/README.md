# Overtone ground-truth harness

Programmatic synthesis for voice-extraction testing. Same trigger
patterns as the Python/VCV path — `<pattern>.triggers.json` in
`../engine/test_audio/triggers/` — played through SuperCollider
synth defs that live as code, not GUI patches.

Why this exists alongside the VCV bridge:

- **VCV path**: modular-emulation. Router sends CV through a virtual
  audio device, VCV synthesizes, audio comes back. Same architecture
  as the eventual ES-9 chain.
- **Overtone path**: ground truth. Every synthesis parameter is in
  Clojure code. No knobs, no cable routing, no AC-coupling surprises
  — change `src/neurodynamics/synths.clj`, re-render, diff.

## Setup (one-time)

```sh
brew install --cask supercollider
# Overtone needs scsynth on PATH:
ln -s /Applications/SuperCollider.app/Contents/Resources/scsynth \
      /opt/homebrew/bin/scsynth
```

Clojure tooling (`clj`/`clojure`/`lein`) assumed present; `deps.edn`
handles all JVM deps.

## Offline render (→ WAV)

```sh
# Render one pattern
clojure -M:render --pattern single --synth mono-saw

# Render all patterns
clojure -M:render --pattern all --synth mono-saw

# Different synth character
clojure -M:render --pattern bassline --synth plucked
```

Output goes to `../engine/test_audio/triggers/<pattern>.overtone.wav`
— same directory as `<pattern>.vcv.wav` from the VCV path, so the
comparison harness can diff both against the trigger schedule.

Then on the Python side:

```sh
cd ../engine
uv run nd-run --audio test_audio/triggers/single.overtone.wav \
              --output output/overtone_single.parquet
```

Or use the existing `trigger_roundtrip` compare pipeline (point it at
the `.overtone.wav` file).

## Live (→ loopback → nd-live)

Configure SuperCollider to send its output to a virtual device (e.g.
`Loopback Audio`) that `nd-live` reads:

```sh
# Either set Loopback Audio as the macOS default output first, or:
SC_HW_DEVICE_NAME="Loopback Audio" clojure -M:live --pattern single
```

In another terminal, `nd-live` listens on the same device:

```sh
cd ../engine
uv run nd-live --input-device "Loopback Audio"
```

## Synths

Defined in `src/neurodynamics/synths.clj`. Currently:

| Name | What | When to use |
|---|---|---|
| `mono-saw` | saw + RLPF with envelope on cutoff | default subtractive monosynth — broadband, covers the engine's typical input |
| `plucked` | fast percussive envelope, no sustain | short-note patterns where long sustain smears voice extraction |
| `sine-tone` | pure sine + amp envelope | reference / debugging capture path |

Add new synths by writing another `defsynth` and adding to
`synth-registry` at the bottom of the file. No GUI, no cables.

## Trigger patterns

Loaded from `../engine/test_audio/triggers/<name>.triggers.json`.
Generate them by running the Python harness once:

```sh
cd ../engine && uv run python -m tests.trigger_roundtrip --pattern all
```

Patterns currently defined: `single`, `quarter_notes`, `bassline`,
`chord_progression`, `polyrhythm`.
