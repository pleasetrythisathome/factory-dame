"""Trigger → audio → engine round trip. The ground-truth mechanism.

A trigger pattern specifies what the router would emit to the modular
(or to VCV Rack, same CV protocol). We synthesize the audio that the
modular would produce in response to those triggers — VCO at the
pitch CV, shaped by an ADSR on each gate — then feed it through the
engine and verify the voice extractor recovers the original triggers.

Why this works as ground truth: the trigger pattern IS the expected
voice timeline. If the engine detects a voice at A4 active from
t=1.0s to t=1.3s, that should correspond to a trigger at A4 with
duration 0.3s at t=1.0. One-to-one, no hand-labeling.

Two synthesis paths:

1. **Python synth (default).** VCO (sine, or sum-of-harmonics for
   piano-ish timbre) × ADSR envelope. Fast, zero dependencies,
   testable in CI.

2. **VCV Rack bridge (opt-in).** Render the same trigger pattern as
   a pair of CV audio files (gate + pitch), route them into a VCV
   Rack patch via Loopback, record VCV's audio output. Higher
   fidelity, tests the exact CV protocol the router will use. Scarlet
   sets up the VCV patch once; this script only has to produce the
   CV WAVs and consume the rendered audio.

Usage:
    uv run python -m tests.trigger_roundtrip                      # default pattern
    uv run python -m tests.trigger_roundtrip --pattern bassline
    uv run python -m tests.trigger_roundtrip --timbre piano       # sum-of-harmonics
    uv run python -m tests.trigger_roundtrip --vcv                # write CV WAVs
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from neurodynamics.perceptual import StateWindow
from neurodynamics.voices import VoiceState, extract_voices

ENGINE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = ENGINE_DIR / "test_audio" / "triggers"

FS = 16000
CV_FS = 48000   # VCV Rack + ES-9 operate at 48 kHz


# ── Trigger pattern ────────────────────────────────────────────────

@dataclass
class Trigger:
    """One note event. Freq is the pitch (Hz); start_s/duration_s
    define the gate; velocity scales the envelope peak."""

    start_s: float
    duration_s: float
    freq_hz: float
    velocity: float = 1.0


@dataclass
class TriggerPattern:
    name: str
    duration_s: float
    triggers: list[Trigger] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "duration_s": self.duration_s,
            "triggers": [asdict(t) for t in self.triggers],
        }


def _hz_from_semitones_above_A4(semitones: float) -> float:
    return 440.0 * 2 ** (semitones / 12.0)


# ── Built-in patterns ──────────────────────────────────────────────

def pattern_single() -> TriggerPattern:
    """One sustained A4 — simplest possible case."""
    return TriggerPattern(
        name="single",
        duration_s=4.0,
        triggers=[Trigger(start_s=0.5, duration_s=3.0, freq_hz=440.0)],
    )


def pattern_quarter_notes() -> TriggerPattern:
    """A4 at 120 BPM quarter notes for 8 seconds. Ground truth for
    'engine sees one voice at 440 Hz that pulses at 2 Hz'."""
    triggers = []
    period = 0.5
    for i in range(16):
        triggers.append(Trigger(
            start_s=0.25 + i * period,
            duration_s=period * 0.8,
            freq_hz=440.0,
        ))
    return TriggerPattern(name="quarter_notes", duration_s=8.5,
                           triggers=triggers)


def pattern_bassline() -> TriggerPattern:
    """Walking bassline: C2-E2-G2-C3 arpeggio, repeating. Low-register
    voice that the bass should track."""
    notes = [-33, -29, -26, -21]   # semitones from A4 (C2 E2 G2 C3)
    triggers = []
    beat = 0.4
    for bar in range(4):
        for i, n in enumerate(notes):
            triggers.append(Trigger(
                start_s=bar * len(notes) * beat + i * beat,
                duration_s=beat * 0.9,
                freq_hz=_hz_from_semitones_above_A4(n),
            ))
    return TriggerPattern(name="bassline", duration_s=beat * len(notes) * 4 + 0.5,
                           triggers=triggers)


def pattern_chord_progression() -> TriggerPattern:
    """C-major chord (C4 E4 G4) held for 2 s, then F-major, then G,
    then C again. Three simultaneous voices per chord — voice
    extractor should see 3 voices at the chord notes."""
    chords = [
        ("C", [-9, -5, -2]),    # C4 E4 G4
        ("F", [-4, 0, 3]),      # F4 A4 C5
        ("G", [-2, 2, 5]),      # G4 B4 D5
        ("C", [-9, -5, -2]),
    ]
    triggers = []
    for i, (_name, semis) in enumerate(chords):
        for s in semis:
            triggers.append(Trigger(
                start_s=i * 2.0, duration_s=1.8,
                freq_hz=_hz_from_semitones_above_A4(s),
            ))
    return TriggerPattern(name="chord_progression",
                           duration_s=len(chords) * 2.0,
                           triggers=triggers)


def pattern_polyrhythm() -> TriggerPattern:
    """Two voices at distinct pitches with distinct periods — a 120
    BPM quarter at low freq, a 180 BPM triplet at high freq. Engine
    should see two voices with different per-voice rhythm BPMs."""
    triggers = []
    # Bass: 120 BPM quarters at A3
    for i in range(12):
        triggers.append(Trigger(start_s=i * 0.5, duration_s=0.4,
                                  freq_hz=220.0))
    # Melody: 180 BPM (equivalent) triplets at E5
    period = 60.0 / 180.0
    for i in range(18):
        triggers.append(Trigger(start_s=i * period,
                                  duration_s=period * 0.8,
                                  freq_hz=659.25))
    return TriggerPattern(name="polyrhythm", duration_s=6.5,
                           triggers=triggers)


ALL_PATTERNS = {
    "single": pattern_single,
    "quarter_notes": pattern_quarter_notes,
    "bassline": pattern_bassline,
    "chord_progression": pattern_chord_progression,
    "polyrhythm": pattern_polyrhythm,
}


# ── Python synth: VCO + ADSR + VCA ─────────────────────────────────

def _adsr(n: int, attack_s: float, decay_s: float, sustain: float,
           release_s: float, fs: int = FS) -> np.ndarray:
    env = np.zeros(n, dtype=np.float32)
    a = int(attack_s * fs); d = int(decay_s * fs); r = int(release_s * fs)
    s = max(n - a - d - r, 0)
    if a > 0: env[:a] = np.linspace(0, 1, a)
    if d > 0: env[a:a + d] = np.linspace(1, sustain, d)
    if s > 0: env[a + d:a + d + s] = sustain
    if r > 0: env[-r:] = np.linspace(sustain, 0, r)
    return env


def _voice_wave(freq_hz: float, duration_s: float, timbre: str,
                 fs: int = FS) -> np.ndarray:
    """One note's waveform, unmodulated. `timbre`: 'sine' or 'piano'
    (sum of 6 harmonics with 1/n weights, gives a richer spectrum
    closer to real instrument content)."""
    t = np.arange(int(duration_s * fs)) / fs
    if timbre == "piano":
        wave = np.zeros(len(t), dtype=np.float32)
        for h in range(1, 7):
            weight = 1.0 / h
            wave = wave + weight * np.sin(2 * np.pi * freq_hz * h * t).astype(np.float32)
        wave = wave / np.max(np.abs(wave))
    else:
        wave = np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
    return wave


def synthesize(pattern: TriggerPattern, timbre: str = "sine",
                peak_amp: float = 0.85) -> np.ndarray:
    """Render the trigger pattern to audio as if a simple modular
    patch (VCO → ADSR → VCA) ran it."""
    n = int(pattern.duration_s * FS)
    audio = np.zeros(n, dtype=np.float32)
    for trig in pattern.triggers:
        # ADSR shaped by the trigger's duration: short attack + decay,
        # sustain at 0.7, quick release at note-off.
        attack_s = 0.005
        decay_s = min(0.04, trig.duration_s * 0.2)
        release_s = 0.08
        note_len = trig.duration_s + release_s
        env = _adsr(int(note_len * FS), attack_s, decay_s,
                     0.7, release_s)
        wave = _voice_wave(trig.freq_hz, note_len, timbre)
        # Align lengths (floor errors)
        L = min(len(env), len(wave))
        note = (env[:L] * wave[:L] * trig.velocity).astype(np.float32)
        start_idx = int(trig.start_s * FS)
        end_idx = min(start_idx + L, n)
        if start_idx < n:
            audio[start_idx:end_idx] += note[:end_idx - start_idx]
    # Normalize to avoid clipping while pushing peak to PEAK_AMP so
    # the engine's noise floor doesn't dominate (see
    # tests/synth_ground_truth.py for the rationale).
    m = float(np.max(np.abs(audio)))
    if m > 0:
        audio = audio / m * peak_amp
    return audio.astype(np.float32)


# ── Voice extraction + comparison ──────────────────────────────────

def extract_voice_timeline(parquet_path: Path) -> list[dict]:
    t = pq.read_table(parquet_path)
    mask = [x == "pitch" for x in t.column("layer").to_pylist()]
    pitch = t.filter(mask)
    times = np.array(pitch.column("t").to_pylist(), dtype=np.float64)
    amps = np.array(pitch.column("amp").to_pylist(), dtype=np.float32)
    phases = np.array(pitch.column("phase").to_pylist(), dtype=np.float32)
    meta = t.schema.metadata or {}
    freq_str = meta.get(b"layer.pitch.f", b"").decode()
    freqs = np.array([float(x) for x in freq_str.split(",") if x])
    if len(times) < 2:
        return []
    snap_hz = 1.0 / (times[1] - times[0])
    feat_stride = max(1, int(round(snap_hz / 5.0)))
    feat_half = max(1, int(snap_hz * 1.25))
    state = VoiceState()
    timeline = []
    for i in range(0, len(times), feat_stride):
        lo = max(0, i - feat_half)
        hi = min(len(times), i + feat_half + 1)
        pitch_z = (amps[lo:hi].astype(np.complex128)
                   * np.exp(1j * phases[lo:hi].astype(np.complex128)))
        sw = StateWindow(pitch_z=pitch_z, pitch_freqs=freqs,
                          rhythm_z=np.zeros(1, dtype=np.complex128),
                          rhythm_freqs=np.array([1.0]),
                          frame_hz=float(snap_hz))
        state = extract_voices(sw, prev_state=state)
        timeline.append({
            "t": float(times[i]),
            "voices": [(v.id, float(v.center_freq), float(v.amp))
                        for v in state.active_voices],
        })
    return timeline


def compare(pattern: TriggerPattern, timeline: list[dict]) -> dict:
    """For each trigger, find the best-matching voice in the engine
    output — a voice active during the trigger's gate whose
    center_freq matches within a tolerance. Report coverage +
    phantom-voice rate."""
    matched = 0
    phantom_frames = 0
    per_trigger: list[dict] = []
    for trig in pattern.triggers:
        best: tuple | None = None
        # Look at snapshots inside the gate, sorted by amp desc.
        for snap in timeline:
            if not (trig.start_s <= snap["t"] <= trig.start_s + trig.duration_s):
                continue
            for vid, freq, amp in snap["voices"]:
                # Frequency match within a semitone.
                if freq <= 0:
                    continue
                cents = 1200 * abs(np.log2(freq / trig.freq_hz))
                if cents < 200:   # within 2 semitones
                    if best is None or amp > best[2]:
                        best = (vid, freq, amp, cents)
        if best:
            matched += 1
            per_trigger.append({
                "trigger_hz": trig.freq_hz,
                "matched_voice_id": best[0],
                "matched_freq": best[1],
                "cents_off": best[3],
            })
        else:
            per_trigger.append({
                "trigger_hz": trig.freq_hz,
                "matched_voice_id": None,
            })

    # Count "phantom" voices: per snapshot, voices whose freq doesn't
    # match any trigger's expected freq (± 200 cents) count as phantoms.
    active_ranges = [(t.start_s, t.start_s + t.duration_s, t.freq_hz)
                     for t in pattern.triggers]
    total_frames = 0
    total_voices = 0
    for snap in timeline:
        for _, freq, _ in snap["voices"]:
            total_voices += 1
            any_match = False
            for start, end, f_target in active_ranges:
                if start <= snap["t"] <= end:
                    cents = 1200 * abs(np.log2(freq / f_target + 1e-9))
                    if cents < 200:
                        any_match = True
                        break
            if not any_match:
                phantom_frames += 1
        total_frames += 1

    return {
        "triggers_total": len(pattern.triggers),
        "triggers_matched": matched,
        "coverage_pct": 100 * matched / max(len(pattern.triggers), 1),
        "phantom_voice_frames": phantom_frames,
        "total_voice_frames": total_voices,
        "phantom_pct": 100 * phantom_frames / max(total_voices, 1),
        "per_trigger": per_trigger,
    }


# ── VCV Rack bridge ────────────────────────────────────────────────

def write_cv_wav(pattern: TriggerPattern, out_dir: Path) -> Path:
    """Render the trigger pattern as one stereo DC-coupled CV WAV
    at 48 kHz — L=gate, R=pitch — that VCV Rack plays back via an
    Audio File module. VCV treats audio and CV identically (both are
    ±10 V floats internally, mapped from ±1.0 in WAV), so a WAV is a
    perfectly valid CV source.

    Channel mapping (same CV protocol as the Expert Sleepers ES-9):
    - Gate (L): 5 V on, 0 V off → WAV value 0.5 / 0.0
    - Pitch (R): 1 V/octave, C0 at 0 V. A4 = 4.75 V → WAV 0.475.
      Highest note in the default patterns (C5) = 5.75 V → 0.575.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(pattern.duration_s * CV_FS)
    stereo = np.zeros((n, 2), dtype=np.float32)
    for trig in pattern.triggers:
        start = int(trig.start_s * CV_FS)
        end = min(int((trig.start_s + trig.duration_s) * CV_FS), n)
        stereo[start:end, 0] = 0.5 * trig.velocity            # 5 V gate
        volts = float(np.log2(trig.freq_hz / 16.3516))         # C0 ref
        stereo[start:end, 1] = volts / 10.0                    # 1 V/oct
    cv_path = out_dir / f"{pattern.name}.cv.wav"
    sf.write(cv_path, stereo, CV_FS, subtype="FLOAT")
    return cv_path


VCV_SETUP_INSTRUCTIONS = """\
VCV Rack setup (one-time, ~5 min):

  1. Install VCV Rack 2 — free at https://vcvrack.com
  2. File → New, then add these modules:
       • VCV Audio File  (or VCV Recorder in playback mode)
       • Fundamental VCO-1
       • Fundamental ADSR
       • Fundamental VCA
       • Fundamental VCF  (optional, for more character)
       • VCV Audio-2      (output interface)
       • VCV Recorder     (to capture rendered audio)
  3. Patching:
       Audio File L out  →  ADSR GATE
       Audio File R out  →  VCO-1 V/OCT
       ADSR ENV out      →  VCA CV in
       VCO-1 SAW/SQR out →  VCF IN  (or straight to VCA if no VCF)
       VCF OUT           →  VCA IN
       VCA OUT           →  Audio-2 L+R in
       Audio-2 L/R out   →  Recorder L/R in
  4. Audio File module → right-click → load file → pick
     {cv_path}
  5. Recorder → set output folder, format WAV 48k 32-bit float,
     arm it, then press play on the Audio File.
  6. When playback ends, stop recording. The rendered audio file
     will be at the folder you picked. Rename/move to:
       {expected_output}
  7. Run the comparison:
       uv run python -m tests.trigger_roundtrip --compare-vcv {name}
"""


def run_vcv_comparison(pattern_name: str, config_path: Path) -> None:
    """After VCV has rendered a WAV, load it, run nd-run, and diff
    against the original trigger schedule."""
    if pattern_name not in ALL_PATTERNS:
        print(f"unknown pattern: {pattern_name}")
        return
    pattern = ALL_PATTERNS[pattern_name]()
    vcv_audio = OUT_DIR / f"{pattern.name}.vcv.wav"
    if not vcv_audio.exists():
        print(f"missing VCV-rendered audio: {vcv_audio}")
        print("Render it in VCV Rack first (see --vcv flag output).")
        return
    parquet_path = ENGINE_DIR / "output" / f"trigger_vcv_{pattern.name}.parquet"
    print(f"\n=== VCV comparison: {pattern.name} ===")
    run_engine(vcv_audio, parquet_path, config_path)
    timeline = extract_voice_timeline(parquet_path)
    report = compare(pattern, timeline)
    print(f"  coverage:       {report['coverage_pct']:5.1f}% "
          f"({report['triggers_matched']}/{report['triggers_total']})")
    print(f"  phantom voices: {report['phantom_pct']:5.1f}% "
          f"({report['phantom_voice_frames']}/{report['total_voice_frames']})")
    cents = [r["cents_off"] for r in report["per_trigger"]
             if r.get("cents_off") is not None]
    if cents:
        print(f"  pitch error:    median {float(np.median(cents)):.0f} cents, "
              f"max {max(cents):.0f} cents")


# ── CLI ────────────────────────────────────────────────────────────

def run_engine(wav_path: Path, parquet_path: Path,
                config_path: Path) -> None:
    print(f"  [engine] processing {wav_path.name}")
    subprocess.run([
        "uv", "run", "nd-run",
        "--config", str(config_path),
        "--audio", str(wav_path),
        "--output", str(parquet_path),
    ], cwd=ENGINE_DIR, check=True,
       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", type=str, default="quarter_notes",
                    help=f"one of {', '.join(ALL_PATTERNS)} or 'all'")
    ap.add_argument("--timbre", type=str, default="piano",
                    choices=["sine", "piano"])
    ap.add_argument("--vcv", action="store_true",
                    help="Write a stereo CV WAV (L=gate, R=pitch) for "
                         "VCV Rack to render, then print setup instructions")
    ap.add_argument("--compare-vcv", type=str, default=None,
                    help="After VCV renders <pattern>.vcv.wav, run it "
                         "through nd-run and compare to the trigger schedule")
    ap.add_argument("--config", type=Path,
                    default=ENGINE_DIR / "config.toml")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.compare_vcv:
        run_vcv_comparison(args.compare_vcv, args.config)
        return

    if args.pattern == "all":
        patterns = list(ALL_PATTERNS.values())
    else:
        if args.pattern not in ALL_PATTERNS:
            print(f"unknown pattern: {args.pattern}")
            return
        patterns = [ALL_PATTERNS[args.pattern]]

    for factory in patterns:
        pattern = factory()
        print(f"\n=== {pattern.name} === "
              f"({len(pattern.triggers)} triggers, "
              f"{pattern.duration_s:.1f}s)")
        wav_path = OUT_DIR / f"{pattern.name}.wav"
        json_path = OUT_DIR / f"{pattern.name}.triggers.json"
        parquet_path = ENGINE_DIR / "output" / f"trigger_{pattern.name}.parquet"

        audio = synthesize(pattern, timbre=args.timbre)
        sf.write(wav_path, audio, FS, subtype="FLOAT")
        json_path.write_text(json.dumps(pattern.to_dict(), indent=2))

        if args.vcv:
            cv_path = write_cv_wav(pattern, OUT_DIR / "cv")
            expected_out = OUT_DIR / f"{pattern.name}.vcv.wav"
            print(f"\n  CV WAV → {cv_path}")
            print(VCV_SETUP_INSTRUCTIONS.format(
                cv_path=cv_path, expected_output=expected_out,
                name=pattern.name,
            ))
            continue  # skip Python-synth run when setting up VCV

        run_engine(wav_path, parquet_path, args.config)
        timeline = extract_voice_timeline(parquet_path)
        report = compare(pattern, timeline)

        print(f"  coverage:       {report['coverage_pct']:5.1f}% "
              f"({report['triggers_matched']}/{report['triggers_total']} "
              f"triggers matched)")
        print(f"  phantom voices: {report['phantom_pct']:5.1f}% "
              f"({report['phantom_voice_frames']} phantom / "
              f"{report['total_voice_frames']} total voice-frames)")
        matched_freqs = [r["matched_freq"] for r in report["per_trigger"]
                         if r.get("matched_freq")]
        if matched_freqs:
            cents = [r["cents_off"] for r in report["per_trigger"]
                     if r.get("cents_off") is not None]
            print(f"  pitch error:    median {float(np.median(cents)):.0f} "
                  f"cents, max {max(cents):.0f} cents")


if __name__ == "__main__":
    main()
