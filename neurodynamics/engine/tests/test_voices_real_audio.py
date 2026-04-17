"""Real-audio validation tests for voice identity extraction.

Runs the extractor across every track in
``test_audio/tracklist.toml`` for which a precomputed parquet exists
(``test_audio/state/<slug>.parquet``). Validates:

- Voice count across the track lands within the
  ``expected_voices_min``/``max`` bounds from the tracklist (loose
  bounds, eyeballed from actual music content).
- Voice IDs persist — the number of distinct IDs observed across the
  track is not absurdly larger than the simultaneous voice count,
  meaning the Hungarian matcher is actually preserving identity
  rather than allocating a new ID every frame.
- No frame has an impossible-seeming voice count (> 20).
- Active voices have positive amplitude.

Tests are gracefully skipped when a parquet doesn't exist — a fresh
clone of the repo runs all unit tests immediately; corpus tests come
online after ``uv run python -m tests.rip_corpus``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from neurodynamics.perceptual import StateWindow
from neurodynamics.voices import (
    VoiceClusteringConfig,
    VoiceState,
    extract_voice_rhythms,
    extract_voices,
)


ENGINE_DIR = Path(__file__).resolve().parent.parent
TEST_AUDIO = ENGINE_DIR / "test_audio"
STATE_DIR = TEST_AUDIO / "state"
TRACKLIST = TEST_AUDIO / "tracklist.toml"


def _load_tracklist() -> list[dict]:
    if not TRACKLIST.exists():
        return []
    with open(TRACKLIST, "rb") as f:
        return tomllib.load(f).get("track", [])


def _load_pitch_layer(parquet_path: Path):
    """Return (times, amps, phases, freqs) for the pitch layer."""
    t = pq.read_table(parquet_path)
    layer_col = t.column("layer").to_pylist()
    mask = [x == "pitch" for x in layer_col]
    pitch = t.filter(mask)
    times = np.array(pitch.column("t").to_pylist(), dtype=np.float64)
    amps = np.array(pitch.column("amp").to_pylist(), dtype=np.float32)
    phases = np.array(pitch.column("phase").to_pylist(), dtype=np.float32)
    meta = t.schema.metadata or {}
    freq_str = meta.get(b"layer.pitch.f", b"").decode()
    freqs = np.array([float(x) for x in freq_str.split(",") if x])
    return times, amps, phases, freqs


def _load_rhythm_layer(parquet_path: Path):
    """Return (times, amps, phases, freqs) for the rhythm layer."""
    t = pq.read_table(parquet_path)
    layer_col = t.column("layer").to_pylist()
    mask = [x == "rhythm" for x in layer_col]
    rhythm = t.filter(mask)
    times = np.array(rhythm.column("t").to_pylist(), dtype=np.float64)
    amps = np.array(rhythm.column("amp").to_pylist(), dtype=np.float32)
    phases = np.array(rhythm.column("phase").to_pylist(), dtype=np.float32)
    meta = t.schema.metadata or {}
    freq_str = meta.get(b"layer.rhythm.f", b"").decode()
    freqs = np.array([float(x) for x in freq_str.split(",") if x])
    return times, amps, phases, freqs


def _extract_voices_over_track(
    parquet_path: Path,
    *,
    window_seconds: float = 2.5,
    feature_hz: float = 5.0,
) -> list[list[dict]]:
    """Run extract_voices + extract_voice_rhythms over every feature
    frame in a parquet and return per-frame voice snapshots."""
    p_times, p_amps, p_phases, p_freqs = _load_pitch_layer(parquet_path)
    r_times, r_amps, r_phases, r_freqs = _load_rhythm_layer(parquet_path)
    if len(p_times) < 2:
        return []
    # Align pitch and rhythm time series. If they have different
    # lengths (they usually don't), use whichever is shorter.
    n = min(len(p_times), len(r_times))
    p_amps, p_phases = p_amps[:n], p_phases[:n]
    r_amps, r_phases = r_amps[:n], r_phases[:n]
    snap_hz = 1.0 / (p_times[1] - p_times[0]) if len(p_times) > 1 else 60.0
    feat_stride = max(1, int(round(snap_hz / feature_hz)))
    feat_half = max(1, int(snap_hz * window_seconds / 2))
    voice_state = VoiceState()
    voices_per_step: list[list[dict]] = []
    for i in range(0, n, feat_stride):
        lo = max(0, i - feat_half)
        hi = min(n, i + feat_half + 1)
        pitch_z = (p_amps[lo:hi].astype(np.complex128)
                   * np.exp(1j * p_phases[lo:hi].astype(np.complex128)))
        rhythm_z = (r_amps[lo:hi].astype(np.complex128)
                    * np.exp(1j * r_phases[lo:hi].astype(np.complex128)))
        sw = StateWindow(
            pitch_z=pitch_z,
            pitch_freqs=p_freqs,
            rhythm_z=rhythm_z,
            rhythm_freqs=r_freqs,
            frame_hz=float(snap_hz),
        )
        voice_state = extract_voices(sw, prev_state=voice_state)
        voice_state = extract_voice_rhythms(sw, voice_state)
        voices_per_step.append([
            {
                "id": int(v.id),
                "osc_indices": tuple(int(o) for o in v.oscillator_indices),
                "center_freq": float(v.center_freq),
                "amp": float(v.amp),
                "confidence": float(v.confidence),
                "rhythm_bpm": (float(v.rhythm.bpm)
                               if v.rhythm is not None else None),
                "rhythm_confidence": (float(v.rhythm.confidence)
                                      if v.rhythm is not None else None),
            }
            for v in voice_state.active_voices
        ])
    return voices_per_step


# Generate one test per track in the tracklist. Tests skip cleanly
# when the parquet isn't present.

def _tracklist_ids() -> list[tuple]:
    return [(t["slug"], t) for t in _load_tracklist()]


@pytest.mark.parametrize("slug,track", _tracklist_ids())
def test_voice_count_within_expected_bounds(slug, track):
    """Average voice count across the track should sit within the
    loose bounds specified in tracklist.toml."""
    parquet = STATE_DIR / f"{slug}.parquet"
    if not parquet.exists():
        pytest.skip(f"{parquet} not present — run rip_corpus")
    voices_per_step = _extract_voices_over_track(parquet)
    if not voices_per_step:
        pytest.skip(f"{slug} parquet has no pitch data")
    counts = [len(v) for v in voices_per_step]
    mean_count = float(np.mean(counts))
    lo = track["expected_voices_min"]
    hi = track["expected_voices_max"]
    # Loose bound on mean — the extractor can fluctuate; asking mean
    # to fit in the expected window catches gross under- or
    # over-clustering without requiring pinpoint accuracy.
    assert lo - 1 <= mean_count <= hi + 1, (
        f"{slug}: expected mean voices in [{lo}, {hi}], got {mean_count:.1f}; "
        f"distribution: min={min(counts)} max={max(counts)}"
    )


@pytest.mark.parametrize("slug,track", _tracklist_ids())
def test_voice_average_lifetime_is_meaningful(slug, track):
    """Average voice lifetime should be at least a few feature
    frames. Low mean lifetime means voices are appearing and
    disappearing with no persistence — the Hungarian matcher is
    failing. Catches regressions in the matching cost gating."""
    parquet = STATE_DIR / f"{slug}.parquet"
    if not parquet.exists():
        pytest.skip(f"{parquet} not present — run rip_corpus")
    voices_per_step = _extract_voices_over_track(parquet)
    if not voices_per_step:
        pytest.skip(f"{slug} parquet has no pitch data")
    # Count how many frames each distinct voice ID was active.
    id_frames: dict[int, int] = {}
    total_voice_frames = 0
    for frame in voices_per_step:
        for v in frame:
            id_frames[v["id"]] = id_frames.get(v["id"], 0) + 1
            total_voice_frames += 1
    if not id_frames:
        pytest.skip(f"{slug} had no active voices across the track")
    # At 5 Hz feature rate, 3 frames = 0.6 s. A voice that doesn't
    # persist at least that long is essentially a transient, not an
    # identified entity.
    mean_lifetime = total_voice_frames / len(id_frames)
    assert mean_lifetime >= 3.0, (
        f"{slug}: mean voice lifetime {mean_lifetime:.1f} frames — "
        f"voices are not persisting, matcher is mis-assigning"
    )


@pytest.mark.parametrize("slug,track", _tracklist_ids())
def test_voice_count_bounded_below_twenty(slug, track):
    """No frame should report > 20 voices — that's pathological
    over-clustering. Catches regressions in cluster threshold or
    noise floor tuning."""
    parquet = STATE_DIR / f"{slug}.parquet"
    if not parquet.exists():
        pytest.skip(f"{parquet} not present — run rip_corpus")
    voices_per_step = _extract_voices_over_track(parquet)
    if not voices_per_step:
        pytest.skip(f"{slug} parquet has no pitch data")
    max_count = max((len(v) for v in voices_per_step), default=0)
    assert max_count <= 20, (
        f"{slug}: max simultaneous voices {max_count} — likely the "
        f"noise floor or correlation threshold is too loose"
    )


@pytest.mark.parametrize("slug,track", _tracklist_ids())
def test_voice_rhythms_in_musical_bpm_range(slug, track):
    """Every per-voice rhythm BPM should fall within 30-240 BPM
    (the musical tempo window). Values outside this range signal
    that the voice envelope is dominated by non-rhythmic
    modulation (drift, fades) rather than a beat pattern — or
    that extract_voice_rhythms is picking the wrong oscillator."""
    parquet = STATE_DIR / f"{slug}.parquet"
    if not parquet.exists():
        pytest.skip(f"{parquet} not present — run rip_corpus")
    voices_per_step = _extract_voices_over_track(parquet)
    if not voices_per_step:
        pytest.skip(f"{slug} parquet has no pitch data")
    out_of_range = 0
    total = 0
    for frame in voices_per_step:
        for v in frame:
            if v.get("rhythm_bpm") is None:
                continue
            total += 1
            if not (30.0 <= v["rhythm_bpm"] <= 240.0):
                out_of_range += 1
    if total == 0:
        pytest.skip(f"{slug} had no voice-rhythm assignments")
    # Allow up to 5% of voices to fall outside musical range
    # (edge-case envelopes at the min/max of the DFT eligibility
    # window); the vast majority should be inside it.
    ratio = out_of_range / total
    assert ratio < 0.05, (
        f"{slug}: {out_of_range}/{total} voices with BPM outside "
        f"[30, 240] ({ratio:.1%})"
    )


@pytest.mark.parametrize("slug,track", _tracklist_ids())
def test_dominant_voice_rhythm_is_musically_plausible(slug, track):
    """The highest-amplitude active voice at each frame carries the
    main musical entity — its rhythm BPM should roughly match
    conventional beat-tracker expectations. We don't require
    pinpoint accuracy (NRT may lock to a subdivision or multiple),
    just that the distribution peaks somewhere in the 50-200 BPM
    range where human music lives."""
    parquet = STATE_DIR / f"{slug}.parquet"
    if not parquet.exists():
        pytest.skip(f"{parquet} not present — run rip_corpus")
    voices_per_step = _extract_voices_over_track(parquet)
    if not voices_per_step:
        pytest.skip(f"{slug} parquet has no pitch data")
    dominant_bpms: list[float] = []
    for frame in voices_per_step:
        if not frame:
            continue
        top = max(frame, key=lambda v: v["amp"])
        if top.get("rhythm_bpm") is not None:
            dominant_bpms.append(top["rhythm_bpm"])
    if not dominant_bpms:
        pytest.skip(f"{slug} had no dominant-voice rhythm assignments")
    median = float(np.median(dominant_bpms))
    assert 50.0 <= median <= 200.0, (
        f"{slug}: dominant voice median BPM {median:.0f} outside "
        f"50-200 BPM musical range"
    )


@pytest.mark.parametrize("slug,track", _tracklist_ids())
def test_active_voices_have_positive_amplitude(slug, track):
    """Every active voice should have nonzero amplitude (the active
    filter is on amplitude, so this is a sanity check)."""
    parquet = STATE_DIR / f"{slug}.parquet"
    if not parquet.exists():
        pytest.skip(f"{parquet} not present — run rip_corpus")
    voices_per_step = _extract_voices_over_track(parquet)
    if not voices_per_step:
        pytest.skip(f"{slug} parquet has no pitch data")
    for frame in voices_per_step:
        for v in frame:
            assert v["amp"] > 0.0, f"{slug}: active voice with zero amp"
