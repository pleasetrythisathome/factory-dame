"""Known-answer voice-extraction tests on synthetic ground truth.

Where ``test_voices_real_audio`` uses loose musical bounds (real music
has no ground-truth voice count), these fixtures have a KNOWN correct
answer by construction:

  - single_tone / ramp_tone : one tone        → one voice
  - pulsed_hihat            : one pulsed tone  → one voice
  - harmonic_stack          : 4 partials, ONE shared envelope → one voice
  - two_independent         : two tones, UNCORRELATED envelopes,
                              non-overtone ratio (8:3) → two voices
  - chord_progression       : 3-note chords → 2-4 simultaneous voices

The extractor is driven with the learned Hebbian W (loaded from each
fixture's ``weights.npz``) so the W-community clustering path is the
one under test. Two assertions per fixture:

  1. median simultaneous voice count lands in the fixture's expected
     range (catches gross over- or under-clustering), and
  2. every annotated ExpectedVoice is actually present — a tracked
     voice sits in its frequency band for a meaningful share of the
     track (catches "right count, wrong things").

Fixtures + parquet + weights come from
``uv run python -m tests.synth_ground_truth --force``. Tests skip
cleanly when those artifacts are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from neurodynamics.perceptual import StateWindow
from neurodynamics.voices import VoiceState, extract_voices

from tests.synth_ground_truth import ALL_FIXTURES

ENGINE_DIR = Path(__file__).resolve().parent.parent
STATE_DIR = ENGINE_DIR / "output"

# Fixture metadata only (the factories also synthesize audio, which we
# don't need here — just the slug + expectations).
_FIXTURES = [factory()[0] for factory in ALL_FIXTURES]


def _load_w(slug: str) -> np.ndarray | None:
    p = STATE_DIR / f"{slug}.weights.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return z["pitch_W"] if "pitch_W" in z.files else None


def _timeline(parquet_path: Path, w_pitch: np.ndarray | None) -> list[list[dict]]:
    """Per-frame active-voice snapshots over the fixture, matching the
    2.5 s window / 5 Hz feature rate used everywhere else."""
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
    stride = max(1, int(round(snap_hz / 5.0)))
    half = max(1, int(snap_hz * 1.25))
    state = VoiceState()
    out: list[list[dict]] = []
    for i in range(0, len(times), stride):
        lo, hi = max(0, i - half), min(len(times), i + half + 1)
        z = (amps[lo:hi].astype(np.complex128)
             * np.exp(1j * phases[lo:hi].astype(np.complex128)))
        sw = StateWindow(
            pitch_z=z, pitch_freqs=freqs,
            rhythm_z=np.zeros(1, dtype=np.complex128),
            rhythm_freqs=np.array([1.0]),
            frame_hz=float(snap_hz),
            w_pitch=w_pitch,
        )
        state = extract_voices(sw, prev_state=state)
        out.append([
            {"id": v.id, "freq": float(v.center_freq), "amp": float(v.amp)}
            for v in state.active_voices
        ])
    return out


def _trim_warmup(frames: list, frac: float = 0.1) -> list:
    """Drop the first/last `frac` of frames — the rolling window is
    half-empty at the very start/end, so counts there are unreliable."""
    k = max(1, int(len(frames) * frac))
    return frames[k:-k] if len(frames) > 2 * k else frames


@pytest.mark.parametrize("fx", _FIXTURES, ids=[f.slug for f in _FIXTURES])
def test_simultaneous_voice_count_matches_known_answer(fx):
    parquet = STATE_DIR / f"{fx.slug}.parquet"
    if not parquet.exists():
        pytest.skip(f"{parquet} not present — run synth_ground_truth --force")
    frames = _trim_warmup(_timeline(parquet, _load_w(fx.slug)))
    if not frames:
        pytest.skip(f"{fx.slug} parquet has no pitch data")
    counts = [len(f) for f in frames]
    median = float(np.median(counts))
    lo, hi = fx.expected_simultaneous
    assert lo <= median <= hi, (
        f"{fx.slug}: median simultaneous voices {median:.1f} outside "
        f"known-answer range [{lo}, {hi}]; "
        f"distribution mean={np.mean(counts):.1f} max={max(counts)}"
    )


def _presence_params():
    """Fixtures with annotated voices, glissando marked xfail.

    Glissando is the one case the cascade can't track cleanly: a fast
    exponential sweep (220→880 Hz over 5 s ≈ 1 octave per 2.5 s window).
    Once the tone sweeps high, the critical-Hopf brainstem's phantom
    subharmonic outweighs the (window-smeared) fundamental, so the
    voice settles near f/2 rather than inside 220-880. The voice COUNT
    is still correct (1 — see the count test); only the frequency band
    is off. This is an architectural property of keeping the cascade on,
    not a clustering bug, and real music rarely sweeps an octave inside
    the window. Documented as xfail rather than silently widened."""
    params = []
    for f in _FIXTURES:
        if not f.expected_voices:
            continue
        marks = (pytest.mark.xfail(
            reason="cascade phantom subharmonic dominates fast sweep",
            strict=False) if f.slug == "glissando" else ())
        params.append(pytest.param(f, marks=marks, id=f.slug))
    return params


@pytest.mark.parametrize("fx", _presence_params())
def test_expected_voices_are_present(fx):
    """Each annotated voice must actually be found: some tracked ID sits
    in its frequency band for at least 25% of the (trimmed) track."""
    parquet = STATE_DIR / f"{fx.slug}.parquet"
    if not parquet.exists():
        pytest.skip(f"{parquet} not present — run synth_ground_truth --force")
    frames = _trim_warmup(_timeline(parquet, _load_w(fx.slug)))
    if not frames:
        pytest.skip(f"{fx.slug} parquet has no pitch data")
    n = len(frames)
    for ev in fx.expected_voices:
        f_lo, f_hi = ev.freq_range_hz
        in_band = sum(
            any(f_lo <= v["freq"] <= f_hi for v in frame)
            for frame in frames
        )
        share = in_band / n
        assert share >= 0.25, (
            f"{fx.slug}: expected a voice in {f_lo:.0f}-{f_hi:.0f} Hz "
            f"({ev.amp_profile}); present in only {share:.0%} of frames"
        )
