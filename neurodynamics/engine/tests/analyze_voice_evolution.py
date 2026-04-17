"""Analyze how voices evolve across corpus tracks.

For each parquet, extracts voices at every feature frame and
computes:
- mean + max simultaneous voice count
- number of distinct voice IDs across the track
- voice lifespan distribution (frames alive per id)
- center-frequency drift per voice (spread from min to max in octaves)
- new-voices-per-second rate
- how often the "top 3 by current frequency sort" changes identity

The last one is the key question for sort stability — if frequency
sort produces constant reordering, we want a stable slot-based sort.

Usage:
    uv run python -m tests.analyze_voice_evolution
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from neurodynamics.perceptual import StateWindow
from neurodynamics.voices import VoiceState, extract_voices

ENGINE_DIR = Path(__file__).resolve().parent.parent
TEST_AUDIO = ENGINE_DIR / "test_audio"
STATE_DIR = TEST_AUDIO / "state"
TRACKLIST = TEST_AUDIO / "tracklist.toml"


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


def analyze_track(parquet_path: Path, track_name: str,
                   window_s: float = 2.5, feature_hz: float = 5.0) -> dict:
    times, amps, phases, freqs = _load_pitch(parquet_path)
    if len(times) < 2:
        return {}
    snap_hz = 1.0 / (times[1] - times[0])
    feat_stride = max(1, int(round(snap_hz / feature_hz)))
    feat_half = max(1, int(snap_hz * window_s / 2))
    state = VoiceState()

    # Per-frame voice list, each entry = list of (id, center_freq, amp)
    frames: list[list[tuple]] = []
    id_lifespan: dict[int, int] = {}  # id → frames alive
    id_freq_range: dict[int, tuple[float, float]] = {}

    for i in range(0, len(times), feat_stride):
        lo = max(0, i - feat_half)
        hi = min(len(times), i + feat_half + 1)
        pitch_z = (amps[lo:hi].astype(np.complex128)
                   * np.exp(1j * phases[lo:hi].astype(np.complex128)))
        sw = StateWindow(
            pitch_z=pitch_z, pitch_freqs=freqs,
            rhythm_z=np.zeros(1, dtype=np.complex128),
            rhythm_freqs=np.array([1.0]),
            frame_hz=float(snap_hz),
        )
        state = extract_voices(sw, prev_state=state)
        row = [(v.id, v.center_freq, v.amp) for v in state.active_voices]
        frames.append(row)
        for vid, freq, _amp in row:
            id_lifespan[vid] = id_lifespan.get(vid, 0) + 1
            if vid in id_freq_range:
                lo_f, hi_f = id_freq_range[vid]
                id_freq_range[vid] = (min(lo_f, freq), max(hi_f, freq))
            else:
                id_freq_range[vid] = (freq, freq)

    # Voice-count summary
    counts = [len(f) for f in frames]
    mean_count = float(np.mean(counts)) if counts else 0.0
    max_count = max(counts) if counts else 0
    n_distinct = len(id_lifespan)
    total_frames = len(frames)
    duration = total_frames / feature_hz

    # Lifespan histogram (in frames).
    spans_s = sorted((v / feature_hz for v in id_lifespan.values()))
    median_span = spans_s[len(spans_s) // 2] if spans_s else 0.0
    long_lived = sum(1 for s in spans_s if s >= 5.0)   # ≥ 5 seconds
    short_lived = sum(1 for s in spans_s if s < 1.0)   # < 1 second

    # Frequency drift per voice (max/min in octaves).
    drifts = []
    for vid, (lo_f, hi_f) in id_freq_range.items():
        if id_lifespan[vid] < 3:
            continue
        if lo_f > 0:
            drifts.append(np.log2(hi_f / lo_f))
    median_drift = float(np.median(drifts)) if drifts else 0.0
    max_drift = float(np.max(drifts)) if drifts else 0.0

    # Sort-stability metric: if we sort active voices by center
    # frequency at each frame, how often does the top row's voice
    # id change between consecutive frames? (Lower is more stable.)
    top_changes_freq = 0
    top_changes_id = 0
    top_changes_age = 0
    prev_top_freq = None
    prev_top_id = None
    prev_top_age = None
    # Age tracked as first-seen frame index.
    first_seen: dict[int, int] = {}
    for idx, row in enumerate(frames):
        for vid, _f, _a in row:
            if vid not in first_seen:
                first_seen[vid] = idx
        if not row:
            prev_top_freq = None
            prev_top_id = None
            prev_top_age = None
            continue
        # Sort by frequency (ascending).
        top_by_freq = min(row, key=lambda r: r[1])[0]
        # Sort by id (ascending).
        top_by_id = min(row, key=lambda r: r[0])[0]
        # Sort by age (oldest = smallest first_seen).
        top_by_age = min(row, key=lambda r: first_seen[r[0]])[0]
        if prev_top_freq is not None and top_by_freq != prev_top_freq:
            top_changes_freq += 1
        if prev_top_id is not None and top_by_id != prev_top_id:
            top_changes_id += 1
        if prev_top_age is not None and top_by_age != prev_top_age:
            top_changes_age += 1
        prev_top_freq = top_by_freq
        prev_top_id = top_by_id
        prev_top_age = top_by_age

    # New voices per second (birth rate).
    new_voices = len(id_lifespan)
    birth_rate = new_voices / duration if duration else 0.0

    return {
        "track": track_name,
        "duration_s": duration,
        "mean_active": mean_count,
        "max_active": max_count,
        "n_distinct_ids": n_distinct,
        "birth_rate_per_s": birth_rate,
        "median_lifespan_s": median_span,
        "long_lived_count": long_lived,
        "short_lived_count": short_lived,
        "median_drift_octaves": median_drift,
        "max_drift_octaves": max_drift,
        "top_changes_freq": top_changes_freq,
        "top_changes_id": top_changes_id,
        "top_changes_age": top_changes_age,
        "total_frames": total_frames,
    }


def main() -> None:
    if not TRACKLIST.exists():
        print("no tracklist.toml; run rip_corpus first")
        return
    with open(TRACKLIST, "rb") as f:
        tracks = tomllib.load(f)["track"]
    # Add Flights if present.
    flights = ENGINE_DIR / "output" / "Flights.parquet"
    results = []
    for track in tracks:
        slug = track["slug"]
        parquet = STATE_DIR / f"{slug}.parquet"
        if not parquet.exists():
            continue
        print(f"[analyze] {slug} …")
        r = analyze_track(parquet, slug)
        if r:
            results.append(r)
    if flights.exists():
        print("[analyze] Flights …")
        r = analyze_track(flights, "Flights")
        if r:
            results.append(r)

    if not results:
        print("No parquets found. Run tests.rip_corpus.")
        return

    # Pretty-print the summary.
    print()
    header = (
        f"{'track':<40}  {'dur':>5}  {'mean':>5}  {'max':>4}  "
        f"{'IDs':>5}  {'births/s':>8}  {'med_life':>8}  "
        f"{'long':>5}  {'short':>6}  {'drift':>6}  "
        f"{'Δtop_f':>7}  {'Δtop_i':>7}  {'Δtop_a':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['track']:<40}  "
            f"{r['duration_s']:5.1f}  "
            f"{r['mean_active']:5.1f}  "
            f"{r['max_active']:4d}  "
            f"{r['n_distinct_ids']:5d}  "
            f"{r['birth_rate_per_s']:8.2f}  "
            f"{r['median_lifespan_s']:8.2f}  "
            f"{r['long_lived_count']:5d}  "
            f"{r['short_lived_count']:6d}  "
            f"{r['median_drift_octaves']:6.2f}  "
            f"{r['top_changes_freq']:7d}  "
            f"{r['top_changes_id']:7d}  "
            f"{r['top_changes_age']:7d}"
        )
    print()
    print(
        "Δtop_f = times per track the top-row voice changes under "
        "frequency sort\n"
        "Δtop_i = same under id-ascending sort\n"
        "Δtop_a = same under age (first-seen) sort\n"
        "long  = voices with ≥5s lifespan  ·  short = voices <1s"
    )


if __name__ == "__main__":
    main()
