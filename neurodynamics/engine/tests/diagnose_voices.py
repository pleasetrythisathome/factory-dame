"""Diagnose why voice extraction produces the results we see.

Answers two specific questions from Scarlet's Apr 17 review:

1. Are harmonics producing multiple voices instead of being treated as
   part of the single voice that produced them? (For `single` — one
   sustained A4 with piano-timbre synth — show every voice's center
   freq and the harmonic relationship to 440 Hz.)

2. Why is bass coverage so low on the `bassline` fixture? (Show the
   pitch bank's frequency layout — how many oscillators below 200 Hz
   — plus what voices actually get detected during bass notes.)

Usage:
    uv run python -m tests.diagnose_voices
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from neurodynamics.perceptual import StateWindow
from neurodynamics.voices import VoiceState, extract_voices

ENGINE_DIR = Path(__file__).resolve().parent.parent
TRIGGER_OUT = ENGINE_DIR / "output"


def _load_pitch(parquet_path: Path):
    t = pq.read_table(parquet_path)
    mask = [x == "pitch" for x in t.column("layer").to_pylist()]
    pitch = t.filter(mask)
    times = np.array(pitch.column("t").to_pylist(), dtype=np.float64)
    amps = np.array(pitch.column("amp").to_pylist(), dtype=np.float32)
    phases = np.array(pitch.column("phase").to_pylist(), dtype=np.float32)
    meta = t.schema.metadata or {}
    freq_str = meta.get(b"layer.pitch.f", b"").decode()
    freqs = np.array([float(x) for x in freq_str.split(",") if x])
    return times, amps, phases, freqs


def _cents(f: float, ref: float) -> float:
    if f <= 0 or ref <= 0:
        return float("inf")
    return 1200 * np.log2(f / ref)


def _harmonic_label(f: float, fundamental: float) -> str:
    """Return a label like '1f', '2f', '3f', '0.5f' for f relative to
    fundamental, or '?' if it's not a small-integer ratio."""
    if f <= 0 or fundamental <= 0:
        return "?"
    ratio = f / fundamental
    candidates = [0.25, 0.333, 0.5, 0.667, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    best = min(candidates, key=lambda c: abs(np.log2(ratio / c)))
    cents_off = 1200 * abs(np.log2(ratio / best))
    if cents_off < 100:
        if best == 1.0:
            return "1f (fund)"
        elif best < 1:
            return f"{best:.3f}f"
        else:
            return f"{best:.1f}f"
    return "?"


def diagnose(parquet_path: Path, expected_fundamental_hz: float,
              snapshot_times_s: list[float], label: str) -> None:
    """Print, for each requested time, the full voice list with
    harmonic labels relative to the expected fundamental."""
    print(f"\n{'=' * 76}")
    print(f"{label}: expected fundamental {expected_fundamental_hz:.1f} Hz")
    print(f"{'=' * 76}")

    times, amps, phases, freqs = _load_pitch(parquet_path)
    if len(times) < 2:
        print("  no data")
        return

    # Pitch bank layout
    print(f"\nPitch bank: {len(freqs)} oscillators, "
          f"{freqs[0]:.1f}–{freqs[-1]:.1f} Hz "
          f"(spacing: {np.log2(freqs[-1]/freqs[0]) * 12 / (len(freqs)-1):.2f} semitones/bin)")
    low = np.sum(freqs < 200)
    mid = np.sum((freqs >= 200) & (freqs < 1000))
    high = np.sum(freqs >= 1000)
    print(f"  oscillators below 200 Hz: {low}   "
          f"200-1000 Hz: {mid}   "
          f"above 1000 Hz: {high}")

    snap_hz = 1.0 / (times[1] - times[0])
    feat_half = max(1, int(snap_hz * 1.25))

    # Replay extract_voices frame-by-frame up to each snapshot time so
    # voice identities are realistic (threading prev_state through).
    state = VoiceState()
    snapshot_set = {int(ts * snap_hz) for ts in snapshot_times_s}
    printed_for: set[int] = set()

    # Stride at 5 Hz for the replay (same as trigger_roundtrip uses).
    stride = max(1, int(round(snap_hz / 5.0)))
    target_idxs = sorted(snapshot_set)

    for i in range(0, len(times), stride):
        lo = max(0, i - feat_half)
        hi = min(len(times), i + feat_half + 1)
        pitch_z = (amps[lo:hi].astype(np.complex128)
                   * np.exp(1j * phases[lo:hi].astype(np.complex128)))
        sw = StateWindow(pitch_z=pitch_z, pitch_freqs=freqs,
                          rhythm_z=np.zeros(1, dtype=np.complex128),
                          rhythm_freqs=np.array([1.0]),
                          frame_hz=float(snap_hz))
        state = extract_voices(sw, prev_state=state)
        for ti in target_idxs:
            if ti in printed_for:
                continue
            if i >= ti:
                t_actual = times[i]
                print(f"\n--- snapshot @ t={t_actual:.2f}s "
                      f"({len(state.active_voices)} active voices) ---")
                print(f"{'id':>4}  {'cfreq':>7}  {'cents':>6}  "
                      f"{'harmonic':>10}  {'amp':>5}  {'n_osc':>5}  "
                      f"{'osc_idxs (sample)'}")
                for v in sorted(state.active_voices, key=lambda v: v.center_freq):
                    cents = _cents(v.center_freq, expected_fundamental_hz)
                    harm = _harmonic_label(v.center_freq, expected_fundamental_hz)
                    osc_sample = list(v.oscillator_indices)[:6]
                    osc_freqs = [f"{float(freqs[i]):.0f}" for i in osc_sample]
                    print(f"{v.id:>4}  {v.center_freq:7.1f}  {cents:+6.0f}  "
                          f"{harm:>10}  {v.amp:5.3f}  "
                          f"{len(v.oscillator_indices):>5}  "
                          f"[{','.join(osc_freqs)}]")
                printed_for.add(ti)


def main() -> None:
    # The `single` fixture has one A4 sustained from 0.5s–3.5s.
    single_parquet = TRIGGER_OUT / "trigger_single.parquet"
    if single_parquet.exists():
        diagnose(
            single_parquet,
            expected_fundamental_hz=440.0,
            snapshot_times_s=[1.0, 2.0, 3.0],
            label="single (sustained A4, piano timbre)",
        )
    else:
        print(f"missing {single_parquet} — run trigger_roundtrip first")

    # `bassline` fixture: C2(65.4) E2(82.4) G2(98.0) C3(130.8), each
    # ~0.36s. First note starts at t=0. Look at each note's mid-point.
    bassline_parquet = TRIGGER_OUT / "trigger_bassline.parquet"
    if bassline_parquet.exists():
        diagnose(
            bassline_parquet,
            expected_fundamental_hz=65.4,  # C2
            snapshot_times_s=[0.2, 0.6, 1.0, 1.4],
            label="bassline (C2-E2-G2-C3 @ 0.4s each, piano timbre)",
        )

    # `chord_progression` at 1.0s should be C major chord (C4 E4 G4)
    chord_parquet = TRIGGER_OUT / "trigger_chord_progression.parquet"
    if chord_parquet.exists():
        diagnose(
            chord_parquet,
            expected_fundamental_hz=261.6,  # C4
            snapshot_times_s=[1.0, 3.0, 5.0],
            label="chord_progression (C4 E4 G4 chord, piano timbre)",
        )


if __name__ == "__main__":
    main()
