"""Qualitative voice-extraction check on the real-audio corpus.

Runs the engine on 30-second clips from a handful of diverse tracks,
then prints per-track voice statistics. This is a regression-detection
tool, not a unit test — acceptable voice counts and frequency ranges
depend on the music's content.

Prior baselines (captured 2026-04-19 after the noise.amp=0 / gain=0.5
cleanup + zero-init):

    track                       mean  max  ids  peak   median   range
    four_tet_angel_echoes       3.0    6    15  0.04   0.0007   38-466
    fred_again_marea            3.7    7     9  0.13   0.002    34-117
    burial_archangel            3.6   11    15  0.06   0.001    38-139
    disclosure_latch            3.1   10    22  0.07   0.002    40-349
    daft_punk_da_funk           3.7    9    20  0.05   0.001    22-196

These are musically plausible: each track's frequency range matches
its musical character (fred_again is bass-heavy, disclosure has
vocals up to 349 Hz, etc.). 3-4 voices per snapshot with 9-22 unique
IDs over 30 seconds feels right for mastered electronic music.

Usage:
    cd neurodynamics/engine
    uv run python -m tests.real_audio_summary
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from neurodynamics.perceptual import StateWindow
from neurodynamics.tuning import note_name
from neurodynamics.voices import VoiceState, extract_voices

ENGINE_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = ENGINE_DIR / "test_audio"
OUT_DIR = ENGINE_DIR / "output"

TRACKS = [
    "four_tet_angel_echoes",
    "fred_again_marea",
    "burial_archangel",
    "disclosure_latch",
    "daft_punk_da_funk",
]


def _ensure_clip(name: str, duration_s: float = 30.0) -> Path:
    """Trim the full track to a 30s clip if not already done."""
    import soundfile as sf
    clip_path = AUDIO_DIR / f"{name}_30s.wav"
    if clip_path.exists():
        return clip_path
    full = AUDIO_DIR / f"{name}.wav"
    if not full.exists():
        raise FileNotFoundError(f"no track at {full}")
    a, sr = sf.read(str(full))
    if a.ndim > 1:
        a = a.mean(axis=1)
    trim = a[:int(sr * duration_s)].astype(np.float32)
    sf.write(str(clip_path), trim, sr)
    return clip_path


def _run_engine(clip_path: Path, parquet_path: Path) -> None:
    subprocess.run(
        ["uv", "run", "nd-run",
         "--audio", str(clip_path),
         "--output", str(parquet_path)],
        cwd=ENGINE_DIR, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _analyze(parquet_path: Path) -> dict:
    t = pq.read_table(parquet_path)
    pitch = t.filter([x == "pitch" for x in t.column("layer").to_pylist()])
    times = np.array(pitch.column("t").to_pylist())
    amps = np.array(pitch.column("amp").to_pylist(), dtype=np.float32)
    phases = np.array(pitch.column("phase").to_pylist(), dtype=np.float32)
    meta = t.schema.metadata or {}
    freqs = np.array([float(x) for x in meta[b"layer.pitch.f"].decode().split(",") if x])
    snap_hz = 1.0 / (times[1] - times[0])
    feat_half = max(1, int(snap_hz * 1.25))
    amps_2d = amps.reshape(len(times), len(freqs))

    peak = amps_2d.max(axis=1)
    med = np.median(amps_2d, axis=1)

    state = VoiceState()
    stride = max(1, int(snap_hz * 0.2))
    counts = []
    unique_ids: set[int] = set()
    all_freqs: list[float] = []
    for i in range(feat_half, len(times) - feat_half, stride):
        lo, hi = max(0, i - feat_half), min(len(times), i + feat_half + 1)
        pz = (amps[lo:hi].astype(np.complex128)
              * np.exp(1j * phases[lo:hi].astype(np.complex128)))
        sw = StateWindow(
            pitch_z=pz, pitch_freqs=freqs,
            rhythm_z=np.zeros(1, dtype=np.complex128),
            rhythm_freqs=np.array([1.0]),
            frame_hz=float(snap_hz),
        )
        state = extract_voices(sw, prev_state=state)
        counts.append(len(state.active_voices))
        for v in state.active_voices:
            unique_ids.add(v.id)
            all_freqs.append(v.center_freq)
    return {
        "mean_voices": float(np.mean(counts)),
        "max_voices": max(counts) if counts else 0,
        "unique_ids": len(unique_ids),
        "mean_peak_amp": float(peak.mean()),
        "mean_median_amp": float(med.mean()),
        "freq_lo": min(all_freqs) if all_freqs else 0.0,
        "freq_hi": max(all_freqs) if all_freqs else 0.0,
    }


def main() -> None:
    print(f"{'track':<25}  {'mean':>5}  {'max':>4}  {'ids':>4}  "
          f"{'peak':>5}  {'median':>7}  {'range (Hz)':>17}")
    print("-" * 80)
    for track in TRACKS:
        clip = _ensure_clip(track)
        parquet = OUT_DIR / f"real_{track}.parquet"
        if not parquet.exists():
            _run_engine(clip, parquet)
        r = _analyze(parquet)
        print(f"{track:<25}  "
              f"{r['mean_voices']:5.1f}  "
              f"{r['max_voices']:>4}  "
              f"{r['unique_ids']:>4}  "
              f"{r['mean_peak_amp']:5.3f}  "
              f"{r['mean_median_amp']:7.4f}  "
              f"{r['freq_lo']:6.1f}-{r['freq_hi']:6.1f}")


if __name__ == "__main__":
    main()
