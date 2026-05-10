"""Capture the three test tiers as a single JSON snapshot.

This is the framework for the depth-pass phased plan. Before each
substantive engine change, run this to lock in the current state.
After the change, run again — diff against the prior snapshot to see
what improved, regressed, or stayed flat.

Tier A — Overtone roundtrip across all 11 patterns. Per-pattern
         coverage, pitch error, phantom rate. Aggregated mean.
Tier B — Real-audio corpus across 5 tracks. Per-track voice stats.
Tier C — LiveEngine throughput on Four Tet 30s clip. Realtime factor
         + per-frame voice count distribution.

Usage:
    cd neurodynamics/engine
    uv run python -m tests.capture_baselines --label phase0_baseline

Snapshots land at tests/baselines/<label>.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import tomllib

ENGINE_DIR = Path(__file__).resolve().parent.parent
BASELINES_DIR = ENGINE_DIR / "tests" / "baselines"
TRIGGERS_DIR = ENGINE_DIR / "test_audio" / "triggers"
AUDIO_DIR = ENGINE_DIR / "test_audio"
OUTPUT_DIR = ENGINE_DIR / "output"

TIER_A_PATTERNS = [
    "single", "quarter_notes", "bassline", "chord_progression",
    "polyrhythm", "sustained_chord", "polyphonic_duet",
    "harmonic_stack", "phantom_fundamental", "vibrato", "glissando",
]

TIER_B_TRACKS = [
    "four_tet_angel_echoes",
    "fred_again_marea",
    "burial_archangel",
    "disclosure_latch",
    "daft_punk_da_funk",
]


def _run_engine_offline(audio_path: Path, parquet_path: Path) -> None:
    subprocess.run(
        ["uv", "run", "nd-run",
         "--audio", str(audio_path),
         "--output", str(parquet_path)],
        cwd=ENGINE_DIR, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _parquet_pitch(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (times, amps_2d, phases_2d, freqs, snap_hz) from a state log."""
    t = pq.read_table(path)
    pitch = t.filter([x == "pitch" for x in t.column("layer").to_pylist()])
    times = np.array(pitch.column("t").to_pylist(), dtype=np.float64)
    amps = np.array(pitch.column("amp").to_pylist(), dtype=np.float32)
    phases = np.array(pitch.column("phase").to_pylist(), dtype=np.float32)
    meta = t.schema.metadata or {}
    freqs = np.array(
        [float(x) for x in meta[b"layer.pitch.f"].decode().split(",") if x]
    )
    if len(times) < 2:
        return times, amps, phases, freqs, 0.0
    snap_hz = 1.0 / (times[1] - times[0])
    return times, amps.reshape(-1, len(freqs)), phases.reshape(-1, len(freqs)), freqs, snap_hz


def tier_a() -> dict:
    """Run Overtone-rendered patterns through the engine, compare to triggers."""
    print("\n=== Tier A: Overtone roundtrip ===")
    from tests.trigger_roundtrip import (
        ALL_PATTERNS, extract_voice_timeline, compare,
    )
    results = {}
    for name in TIER_A_PATTERNS:
        wav = TRIGGERS_DIR / f"{name}.overtone.wav"
        if not wav.exists():
            print(f"  skip {name}: no overtone wav at {wav}")
            results[name] = None
            continue
        parquet = OUTPUT_DIR / f"baseline_{name}.parquet"
        _run_engine_offline(wav, parquet)
        timeline = extract_voice_timeline(parquet)
        pat = ALL_PATTERNS[name]()
        rep = compare(pat, timeline)
        cents = [r["cents_off"] for r in rep["per_trigger"]
                 if r.get("cents_off") is not None]
        results[name] = {
            "coverage_pct": rep["coverage_pct"],
            "phantom_pct": rep["phantom_pct"],
            "median_cents_err": float(np.median(cents)) if cents else None,
            "max_cents_err": float(max(cents)) if cents else None,
            "triggers_total": rep["triggers_total"],
            "triggers_matched": rep["triggers_matched"],
        }
        print(f"  {name:<22}  cov={results[name]['coverage_pct']:5.1f}%  "
              f"phant={results[name]['phantom_pct']:5.1f}%")
    covs = [r["coverage_pct"] for r in results.values() if r is not None]
    phants = [r["phantom_pct"] for r in results.values() if r is not None]
    aggregate = {
        "mean_coverage_pct": float(np.mean(covs)) if covs else 0.0,
        "mean_phantom_pct": float(np.mean(phants)) if phants else 0.0,
    }
    print(f"  AGGREGATE  cov={aggregate['mean_coverage_pct']:.1f}%  "
          f"phant={aggregate['mean_phantom_pct']:.1f}%")
    return {"per_pattern": results, "aggregate": aggregate}


def tier_b() -> dict:
    """Run engine on 30s clips of corpus tracks. Per-track voice stats."""
    print("\n=== Tier B: Real-audio corpus ===")
    from neurodynamics.perceptual import StateWindow
    from neurodynamics.voices import VoiceState, extract_voices
    results = {}
    for track in TIER_B_TRACKS:
        clip = AUDIO_DIR / f"{track}_30s.wav"
        if not clip.exists():
            full = AUDIO_DIR / f"{track}.wav"
            if not full.exists():
                print(f"  skip {track}: no audio")
                results[track] = None
                continue
            a, sr = sf.read(str(full))
            if a.ndim > 1:
                a = a.mean(axis=1)
            sf.write(str(clip), a[:sr * 30].astype(np.float32), sr)
        parquet = OUTPUT_DIR / f"baseline_{track}.parquet"
        _run_engine_offline(clip, parquet)
        times, amps_2d, phases_2d, freqs, snap_hz = _parquet_pitch(parquet)
        if len(times) < 2:
            results[track] = None
            continue
        feat_half = max(1, int(snap_hz * 1.25))
        peak = amps_2d.max(axis=1)
        med = np.median(amps_2d, axis=1)
        state = VoiceState()
        stride = max(1, int(snap_hz * 0.2))
        counts = []
        unique_ids: set[int] = set()
        all_freqs: list[float] = []
        voice_lifespans: dict[int, int] = {}
        for i in range(feat_half, len(times) - feat_half, stride):
            lo, hi = max(0, i - feat_half), min(len(times), i + feat_half + 1)
            pz = (amps_2d[lo:hi].astype(np.complex128).flatten()
                  * np.exp(1j * phases_2d[lo:hi].astype(np.complex128).flatten()))
            pz = pz.reshape(hi - lo, len(freqs))
            sw = StateWindow(
                pitch_z=pz, pitch_freqs=freqs,
                rhythm_z=np.zeros(1, dtype=np.complex128),
                rhythm_freqs=np.array([1.0]), frame_hz=float(snap_hz),
            )
            state = extract_voices(sw, prev_state=state)
            counts.append(len(state.active_voices))
            for v in state.active_voices:
                unique_ids.add(v.id)
                all_freqs.append(v.center_freq)
                voice_lifespans[v.id] = voice_lifespans.get(v.id, 0) + 1
        spans = sorted(voice_lifespans.values())
        median_lifespan_frames = spans[len(spans) // 2] if spans else 0
        results[track] = {
            "mean_voices": float(np.mean(counts)) if counts else 0.0,
            "max_voices": max(counts) if counts else 0,
            "unique_ids": len(unique_ids),
            "mean_peak_amp": float(peak.mean()),
            "mean_median_amp": float(med.mean()),
            "freq_lo": min(all_freqs) if all_freqs else 0.0,
            "freq_hi": max(all_freqs) if all_freqs else 0.0,
            "median_voice_lifespan_frames": median_lifespan_frames,
            "median_voice_lifespan_s": median_lifespan_frames * 0.2,
        }
        r = results[track]
        print(f"  {track:<26}  mean_v={r['mean_voices']:.2f}  "
              f"ids={r['unique_ids']}  span={r['median_voice_lifespan_s']:.1f}s")
    return {"per_track": results}


def tier_c() -> dict:
    """LiveEngine throughput + per-frame voice counts on Four Tet 30s."""
    print("\n=== Tier C: Live throughput ===")
    from neurodynamics.live import LiveEngine
    config = ENGINE_DIR / "config.toml"
    with open(config, "rb") as f:
        cfg = tomllib.load(f)
    clip = AUDIO_DIR / "four_tet_30s.wav"
    if not clip.exists():
        print(f"  skip: no {clip}")
        return {}
    a, sr = sf.read(str(clip))
    if a.ndim > 1:
        a = a.mean(axis=1)
    a = a.astype(np.float32)
    engine = LiveEngine(cfg)
    engine.process(a[:4096])  # JIT warm-up
    chunk_size = 1024
    voice_counts = []
    t0 = time.perf_counter()
    for i in range(0, len(a) - chunk_size, chunk_size):
        engine.process(a[i:i + chunk_size])
        voice_counts.append(len(engine._voice_state.active_voices))
    elapsed = time.perf_counter() - t0
    audio_dur = len(a) / sr
    rt_factor = audio_dur / elapsed
    counts_arr = np.array(voice_counts) if voice_counts else np.array([0])
    result = {
        "audio_duration_s": float(audio_dur),
        "processing_time_s": float(elapsed),
        "realtime_factor": float(rt_factor),
        "voice_count_mean": float(counts_arr.mean()),
        "voice_count_max": int(counts_arr.max()),
        "voice_count_min": int(counts_arr.min()),
        "voice_count_p50": float(np.percentile(counts_arr, 50)),
        "voice_count_p95": float(np.percentile(counts_arr, 95)),
    }
    print(f"  realtime: {result['realtime_factor']:.2f}x  "
          f"voices: mean={result['voice_count_mean']:.2f} "
          f"p95={result['voice_count_p95']:.0f}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", type=str, required=True,
                    help="Snapshot filename (without .json), e.g. phase0_baseline")
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--skip-c", action="store_true")
    args = ap.parse_args()

    BASELINES_DIR.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "label": args.label,
        "captured_at": datetime.now().isoformat(),
        "engine_dir": str(ENGINE_DIR),
    }
    # Capture key config values for reproducibility
    config = ENGINE_DIR / "config.toml"
    with open(config, "rb") as f:
        cfg = tomllib.load(f)
    snapshot["config_subset"] = {
        "pitch_alpha": cfg["pitch_grfnn"]["alpha"],
        "pitch_beta1": cfg["pitch_grfnn"]["beta1"],
        "pitch_input_gain": cfg["pitch_grfnn"]["input_gain"],
        "pitch_noise_amp": cfg["pitch_grfnn"]["noise"]["amp"],
        "pitch_n_oscillators": cfg["pitch_grfnn"]["n_oscillators"],
        "snapshot_hz": cfg["state_log"]["snapshot_hz"],
    }
    if not args.skip_a:
        snapshot["tier_a"] = tier_a()
    if not args.skip_b:
        snapshot["tier_b"] = tier_b()
    if not args.skip_c:
        snapshot["tier_c"] = tier_c()
    out = BASELINES_DIR / f"{args.label}.json"
    out.write_text(json.dumps(snapshot, indent=2))
    print(f"\nWrote baseline snapshot to {out}")


if __name__ == "__main__":
    main()
