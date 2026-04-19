"""Check what pitch VCV actually produced vs what the trigger asked
for. Takes the captured VCV audio, FFTs the sustain portion, and
reports the dominant spectral peaks.

If the VCO is tracking 1V/oct correctly with C0=0V reference, the
fundamental of `single.vcv.wav` should be at 440 Hz (A4). Any
systematic offset tells us the VCO's internal reference.

Usage:
    uv run python -m tests.check_tuning --pattern single
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

from tests.trigger_roundtrip import ALL_PATTERNS, OUT_DIR


def analyze(wav_path: Path, expected_hz: float, sustain_start_s: float,
             sustain_end_s: float) -> None:
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    seg = audio[int(sr * sustain_start_s):int(sr * sustain_end_s)]
    if len(seg) < 256:
        print("  sustain window too short")
        return
    # Hann window to reduce spectral leakage
    window = np.hanning(len(seg))
    F = np.fft.rfft(seg * window)
    freqs = np.fft.rfftfreq(len(seg), 1 / sr)
    mag = np.abs(F)
    peaks, _ = find_peaks(mag, height=mag.max() * 0.03, distance=20)
    top = sorted(peaks, key=lambda p: -mag[p])[:10]
    print(f"\n{wav_path.name}:  captured {len(audio) / sr:.2f}s @ {sr}Hz, "
          f"peak amp {float(np.max(np.abs(audio))):.3f}")
    print(f"  expected fundamental: {expected_hz:.1f} Hz")
    print(f"  top spectral peaks in sustain [{sustain_start_s}s–{sustain_end_s}s]:")
    print(f"  {'Hz':>8}  {'cents(A4)':>10}  {'ratio':>7}  {'interp':>18}")
    for p in top:
        hz = freqs[p]
        if hz <= 0:
            continue
        cents = 1200 * np.log2(hz / expected_hz)
        ratio = hz / expected_hz
        # Interpret the ratio as a simple integer harmonic if close.
        harmonics = {
            0.25: "1/4f (–2oct)", 0.333: "1/3f", 0.5: "1/2f (–1oct)",
            1.0: "1f fundamental", 1.5: "3/2f (fifth up)",
            2.0: "2f (+1oct)", 3.0: "3f", 4.0: "4f", 5.0: "5f",
            6.0: "6f", 8.0: "8f",
        }
        best = min(harmonics, key=lambda h: abs(np.log2(ratio / h)))
        off = 1200 * abs(np.log2(ratio / best))
        label = harmonics[best] if off < 80 else "(unrelated)"
        print(f"  {hz:8.1f}  {cents:+10.0f}  {ratio:7.3f}  {label:>18}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", type=str, default="single")
    args = ap.parse_args()

    # For single, trigger is 0.5-3.5s but with lead_s=0.5, in the
    # captured file it sits at 1.0-4.0s. Analyze 1.5-3.5 to skip
    # attack and release.
    if args.pattern == "single":
        wav = OUT_DIR / "single.vcv.wav"
        analyze(wav, expected_hz=440.0,
                sustain_start_s=1.5, sustain_end_s=3.5)
    else:
        pattern = ALL_PATTERNS[args.pattern]()
        wav = OUT_DIR / f"{pattern.name}.vcv.wav"
        if not pattern.triggers:
            return
        t0 = pattern.triggers[0]
        # Offset by lead_s=0.5 used in vcv_live_roundtrip
        analyze(wav, expected_hz=t0.freq_hz,
                sustain_start_s=0.5 + t0.start_s + 0.05,
                sustain_end_s=0.5 + t0.start_s + t0.duration_s - 0.05)


if __name__ == "__main__":
    main()
