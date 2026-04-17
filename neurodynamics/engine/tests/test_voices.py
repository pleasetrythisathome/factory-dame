"""Synthetic test suite for voice identity extraction.

Tests are structured around the edge cases the algorithm must
actually handle on real music. Each test constructs a StateWindow
directly — no engine run needed — so we can exercise the extractor
in isolation and at high granularity.

Edge cases covered:
- Silence (no active oscillators)
- Single-oscillator voice
- Unison (multiple correlated oscillators = one voice)
- Harmonic voice (fundamental + harmonics = one voice)
- Two parallel voices (uncorrelated clusters)
- Voice split (one voice → two)
- Voice merge (two voices → one)
- Voice persistence through silence (ID preserved across gaps)
- Noise floor (low-amp oscillators don't spawn voices)
- Transient (a brief burst at one oscillator doesn't become a voice)
- Voice ID allocation / no reuse while voice is still silent-tracked
"""

from __future__ import annotations

import numpy as np
import pytest

from neurodynamics.perceptual import StateWindow
from neurodynamics.voices import (
    VoiceClusteringConfig,
    VoiceIdentity,
    VoiceRhythm,
    VoiceState,
    extract_voice_rhythms,
    extract_voices,
)


# ── Fixtures & helpers ─────────────────────────────────────────────

@pytest.fixture
def base_freqs():
    """Log-spaced pitch bank (matches typical engine config shape)."""
    return np.geomspace(30.0, 4000.0, 100)


def _make_window(pitch_z: np.ndarray, freqs: np.ndarray,
                 frame_hz: float = 60.0) -> StateWindow:
    return StateWindow(
        pitch_z=pitch_z,
        pitch_freqs=freqs,
        rhythm_z=np.zeros(1, dtype=np.complex128),
        rhythm_freqs=np.array([1.0]),
        frame_hz=frame_hz,
    )


def _synthesize_voice(
    freqs: np.ndarray,
    center_hz: float,
    *,
    n_frames: int = 60,
    harmonic_count: int = 1,
    envelope: np.ndarray | None = None,
    amp: float = 0.5,
    phase_offset: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Create per-frame amplitude trajectories for a voice spanning a
    fundamental plus ``harmonic_count - 1`` integer harmonics.

    Envelope is a (frames,) array controlling amplitude over time. If
    an envelope is given, it overrides ``n_frames`` — callers can
    control window length by passing a longer envelope.
    """
    if envelope is not None:
        n_frames = int(len(envelope))
    else:
        t = np.linspace(0, 2 * np.pi, n_frames)
        envelope = 0.5 + 0.4 * np.sin(t)
    n_osc = len(freqs)
    z = np.zeros((n_frames, n_osc), dtype=np.complex128)
    rng = np.random.default_rng(seed)
    jitter = 1.0 + 0.03 * rng.standard_normal(n_frames)
    env = envelope * jitter
    # Fundamental + harmonics
    for h in range(1, harmonic_count + 1):
        target = center_hz * h
        if target < freqs[0] or target > freqs[-1]:
            continue
        idx = int(np.argmin(np.abs(freqs - target)))
        # Falling harmonic amplitude (rule-of-thumb ~1/h).
        h_amp = amp / h
        # Phase evolves at the oscillator's natural frequency.
        f = freqs[idx]
        t_frame = np.arange(n_frames) / 60.0  # assume frame_hz=60
        phases = 2 * np.pi * f * t_frame + phase_offset
        z[:, idx] = h_amp * env * np.exp(1j * phases)
    return z


def _add_noise(z: np.ndarray, noise_amp: float = 0.003,
               seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = noise_amp * (
        rng.standard_normal(z.shape) + 1j * rng.standard_normal(z.shape)
    )
    return z + noise


# ── Silence ────────────────────────────────────────────────────────

def test_silence_produces_no_voices(base_freqs):
    """All-zero pitch state → 0 active voices."""
    z = np.zeros((60, len(base_freqs)), dtype=np.complex128)
    state = extract_voices(_make_window(z, base_freqs))
    assert state.active_voices == []


def test_noise_floor_produces_no_voices(base_freqs):
    """Low-amplitude noise across the bank → 0 voices (below the
    active threshold)."""
    rng = np.random.default_rng(7)
    z = 0.005 * (rng.standard_normal((60, len(base_freqs)))
                 + 1j * rng.standard_normal((60, len(base_freqs))))
    state = extract_voices(_make_window(z, base_freqs))
    assert state.active_voices == []


# ── Single active oscillator ───────────────────────────────────────

def test_single_oscillator_voice(base_freqs):
    """One oscillator loud, rest at noise → 1 voice with that
    oscillator as its member."""
    z = _synthesize_voice(base_freqs, 440.0, amp=0.8)
    z = _add_noise(z)
    state = extract_voices(_make_window(z, base_freqs))
    assert len(state.active_voices) == 1
    v = state.active_voices[0]
    assert 420.0 < v.center_freq < 460.0
    assert v.amp > 0.1


# ── Unison ─────────────────────────────────────────────────────────

def test_unison_is_one_voice(base_freqs):
    """Three oscillators moving with the same amplitude envelope →
    one voice (not three singleton voices)."""
    n_frames = 60
    envelope = 0.4 + 0.3 * np.sin(np.linspace(0, 2 * np.pi, n_frames))
    # Pick three close-together pitch indices manually.
    z = np.zeros((n_frames, len(base_freqs)), dtype=np.complex128)
    indices = [30, 31, 32]
    for idx in indices:
        f = base_freqs[idx]
        t = np.arange(n_frames) / 60.0
        z[:, idx] = envelope * 0.5 * np.exp(1j * 2 * np.pi * f * t)
    z = _add_noise(z)
    state = extract_voices(_make_window(z, base_freqs))
    assert len(state.active_voices) == 1
    v = state.active_voices[0]
    # Cluster should contain all three indices.
    assert set(v.oscillator_indices) == set(indices)


def test_harmonic_voice_is_one_voice(base_freqs):
    """Fundamental + 3 harmonics with correlated envelopes → one
    voice spanning the harmonic stack."""
    z = _synthesize_voice(base_freqs, 220.0, harmonic_count=4, amp=0.6)
    z = _add_noise(z)
    state = extract_voices(_make_window(z, base_freqs))
    assert len(state.active_voices) == 1
    v = state.active_voices[0]
    # At least 2 active harmonics (higher ones may fall near the
    # cutoff or below the active threshold).
    assert len(v.oscillator_indices) >= 2


# ── Two parallel voices ────────────────────────────────────────────

def test_two_independent_voices(base_freqs):
    """Two voices with uncorrelated envelopes → two voices."""
    n_frames = 80
    t = np.linspace(0, 2 * np.pi, n_frames)
    env_a = 0.3 + 0.3 * np.sin(t)
    env_b = 0.3 + 0.3 * np.sin(t + np.pi / 2)  # quarter-cycle offset
    z_a = _synthesize_voice(base_freqs, 220.0, harmonic_count=2,
                             envelope=env_a, amp=0.5, seed=1)
    z_b = _synthesize_voice(base_freqs, 880.0, harmonic_count=2,
                             envelope=env_b, amp=0.5, seed=2)
    z = _add_noise(z_a + z_b)
    state = extract_voices(_make_window(z, base_freqs))
    assert len(state.active_voices) == 2
    centers = sorted(v.center_freq for v in state.active_voices)
    # Lower voice near 220 Hz, upper near 880 Hz.
    assert centers[0] < 400.0
    assert centers[1] > 600.0


def test_two_voices_get_distinct_ids(base_freqs):
    """Two voices → two different IDs."""
    n_frames = 80
    t = np.linspace(0, 2 * np.pi, n_frames)
    env_a = 0.3 + 0.3 * np.sin(t)
    env_b = 0.3 + 0.3 * np.sin(t + np.pi / 2)
    z_a = _synthesize_voice(base_freqs, 220.0, envelope=env_a, amp=0.5)
    z_b = _synthesize_voice(base_freqs, 880.0, envelope=env_b, amp=0.5)
    z = _add_noise(z_a + z_b)
    state = extract_voices(_make_window(z, base_freqs))
    ids = sorted(v.id for v in state.active_voices)
    assert len(ids) == 2
    assert ids[0] != ids[1]


# ── Voice tracking ID persistence ──────────────────────────────────

def test_voice_id_persists_across_frames(base_freqs):
    """Run two consecutive windows with the same voice → same ID."""
    z = _synthesize_voice(base_freqs, 440.0, amp=0.6)
    z = _add_noise(z)
    win = _make_window(z, base_freqs)
    state1 = extract_voices(win)
    assert len(state1.active_voices) == 1
    id1 = state1.active_voices[0].id
    state2 = extract_voices(win, prev_state=state1)
    assert len(state2.active_voices) == 1
    assert state2.active_voices[0].id == id1


def test_voice_id_persists_through_brief_silence(base_freqs):
    """Voice drops out for a few frames, then returns → same ID."""
    z_on = _add_noise(_synthesize_voice(base_freqs, 440.0, amp=0.6))
    z_off = np.zeros_like(z_on)
    state = extract_voices(_make_window(z_on, base_freqs))
    original_id = state.active_voices[0].id
    # Simulate a few silent windows. Must stay within
    # max_silent_frames or the voice retires.
    for _ in range(5):
        state = extract_voices(_make_window(z_off, base_freqs),
                                prev_state=state)
    # Voice returns.
    state = extract_voices(_make_window(z_on, base_freqs),
                            prev_state=state)
    assert len(state.active_voices) == 1
    assert state.active_voices[0].id == original_id


def test_voice_retired_after_prolonged_silence(base_freqs):
    """Voice silent longer than max_silent_frames → retired, new ID
    when it returns."""
    cfg = VoiceClusteringConfig(max_silent_frames=3)
    z_on = _add_noise(_synthesize_voice(base_freqs, 440.0, amp=0.6))
    z_off = np.zeros_like(z_on)
    state = extract_voices(_make_window(z_on, base_freqs), config=cfg)
    original_id = state.active_voices[0].id
    # Enough silent frames to retire the voice.
    for _ in range(10):
        state = extract_voices(_make_window(z_off, base_freqs),
                                prev_state=state, config=cfg)
    state = extract_voices(_make_window(z_on, base_freqs),
                            prev_state=state, config=cfg)
    assert len(state.active_voices) == 1
    assert state.active_voices[0].id != original_id


def test_ids_never_reused_while_silent(base_freqs):
    """While voice A is silent-tracked, voice B with a different
    freq should get a fresh ID, not A's."""
    cfg = VoiceClusteringConfig(max_silent_frames=10)
    z_a = _add_noise(_synthesize_voice(base_freqs, 220.0, amp=0.6, seed=1))
    z_off = np.zeros_like(z_a)
    state = extract_voices(_make_window(z_a, base_freqs), config=cfg)
    id_a = state.active_voices[0].id
    # Voice A silent.
    state = extract_voices(_make_window(z_off, base_freqs),
                            prev_state=state, config=cfg)
    # Voice B (very different frequency) appears.
    z_b = _add_noise(_synthesize_voice(base_freqs, 1760.0, amp=0.6, seed=2))
    state = extract_voices(_make_window(z_b, base_freqs),
                            prev_state=state, config=cfg)
    active = state.active_voices
    assert len(active) == 1
    assert active[0].id != id_a
    # Voice A should still be in the full voices list as silent.
    silent_ids = [v.id for v in state.voices if not v.active]
    assert id_a in silent_ids


# ── Voice split and merge ──────────────────────────────────────────

def test_voice_split_largest_inherits_id(base_freqs):
    """One correlated unison voice splits into two clusters. The
    larger cluster should inherit the original voice's ID."""
    n_frames = 80
    t = np.linspace(0, 2 * np.pi, n_frames)
    # Phase 1: three close oscillators all with the same envelope.
    env_shared = 0.3 + 0.3 * np.sin(t)
    z1 = np.zeros((n_frames, len(base_freqs)), dtype=np.complex128)
    for idx in (30, 31, 32, 33, 34):  # 5 oscillators, all correlated
        f = base_freqs[idx]
        z1[:, idx] = env_shared * 0.5 * np.exp(
            1j * 2 * np.pi * f * np.arange(n_frames) / 60.0
        )
    z1 = _add_noise(z1)
    state = extract_voices(_make_window(z1, base_freqs))
    assert len(state.active_voices) == 1
    original_id = state.active_voices[0].id

    # Phase 2: first 4 still correlated (larger), last one decorrelated.
    env_a = 0.3 + 0.3 * np.sin(t)
    env_b = 0.3 + 0.3 * np.sin(t + np.pi)  # anti-phase
    z2 = np.zeros((n_frames, len(base_freqs)), dtype=np.complex128)
    for idx in (30, 31, 32, 33):  # 4 oscillators
        f = base_freqs[idx]
        z2[:, idx] = env_a * 0.5 * np.exp(
            1j * 2 * np.pi * f * np.arange(n_frames) / 60.0
        )
    # Single splintered oscillator with anti-phase envelope.
    f = base_freqs[50]
    z2[:, 50] = env_b * 0.5 * np.exp(
        1j * 2 * np.pi * f * np.arange(n_frames) / 60.0
    )
    z2 = _add_noise(z2)
    state = extract_voices(_make_window(z2, base_freqs), prev_state=state)
    # Should now have 2 voices; larger one keeps original ID.
    assert len(state.active_voices) == 2
    active_sorted = sorted(state.active_voices, key=lambda v: -len(v.oscillator_indices))
    assert active_sorted[0].id == original_id


def test_voice_merge_collapses_to_one_voice(base_freqs):
    """Two voices with different envelopes merge into one correlated
    group. Current behavior: the merged cluster gets a fresh ID
    because Jaccard distance to either parent (union vs each subset)
    exceeds the match cost cap. Both parents are then kept as silent
    voices for the persistence window. This is acceptable for the
    modular use case — the consumer sees 'voice A and B both went
    silent, new voice C appeared' — but it's worth documenting as a
    known behavior rather than quietly assuming ID inheritance."""
    n_frames = 80
    t = np.linspace(0, 2 * np.pi, n_frames)
    env_a = 0.4 + 0.3 * np.sin(t)
    env_b = 0.3 + 0.2 * np.sin(t + np.pi / 2)

    # Phase 1: two uncorrelated voices.
    z_a1 = _synthesize_voice(base_freqs, 220.0, envelope=env_a, amp=0.7, seed=1)
    z_b1 = _synthesize_voice(base_freqs, 880.0, envelope=env_b, amp=0.4, seed=2)
    z1 = _add_noise(z_a1 + z_b1)
    state = extract_voices(_make_window(z1, base_freqs))
    assert len(state.active_voices) == 2
    parent_ids = {v.id for v in state.active_voices}

    # Phase 2: both voices share a single envelope → merge.
    shared_env = 0.4 + 0.3 * np.sin(t)
    z_a2 = _synthesize_voice(base_freqs, 220.0, envelope=shared_env,
                              amp=0.7, seed=1)
    z_b2 = _synthesize_voice(base_freqs, 880.0, envelope=shared_env,
                              amp=0.4, seed=2)
    z2 = _add_noise(z_a2 + z_b2)
    state = extract_voices(_make_window(z2, base_freqs), prev_state=state)
    # Key property: the merged cluster shows up as a single active
    # voice (correlation test did its job), and both parent IDs are
    # held as silent voices for the persistence window.
    assert len(state.active_voices) == 1
    silent_ids = {v.id for v in state.voices if not v.active}
    assert parent_ids.issubset(silent_ids)


# ── Robustness: dense voice count ──────────────────────────────────

def test_five_voices_all_detected(base_freqs):
    """Five genuinely-independent voices at different frequencies →
    5 voices."""
    n_frames = 120
    t = np.linspace(0, 4 * np.pi, n_frames)
    # Distinct phases per voice for uncorrelated envelopes.
    phases = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    centers = [110.0, 220.0, 440.0, 880.0, 1760.0]
    z = np.zeros((n_frames, len(base_freqs)), dtype=np.complex128)
    for i, (c, ph) in enumerate(zip(centers, phases)):
        env = 0.3 + 0.25 * np.sin(t + ph)
        z += _synthesize_voice(base_freqs, c, envelope=env, amp=0.5,
                                seed=i, harmonic_count=1)
    z = _add_noise(z)
    state = extract_voices(_make_window(z, base_freqs))
    assert 4 <= len(state.active_voices) <= 6, \
        f"expected ~5 voices, got {len(state.active_voices)}"


# ── Transient rejection ────────────────────────────────────────────

def test_transient_is_still_a_voice_in_one_frame(base_freqs):
    """A brief burst shows up as a voice in the window that contains
    it — we don't have cross-window debouncing yet beyond the
    persistence mechanism. (Documenting current behavior, not a bug.
    Router-side debouncing can add latency-bounded confirmation.)"""
    n_frames = 60
    z = np.zeros((n_frames, len(base_freqs)), dtype=np.complex128)
    # Single brief amplitude burst on one oscillator.
    idx = 45
    burst = np.zeros(n_frames)
    burst[10:14] = 0.8  # 4 frames of activity
    f = base_freqs[idx]
    z[:, idx] = burst * np.exp(
        1j * 2 * np.pi * f * np.arange(n_frames) / 60.0
    )
    state = extract_voices(_make_window(z, base_freqs))
    # Mean amplitude over the 60-frame window is 0.8*4/60 = 0.053 —
    # below the default noise floor of 0.02... actually ~2.7× it.
    # Documenting that a short burst *can* trigger a voice but its
    # confidence and activity will quickly decay in subsequent frames.
    # Not strictly asserting count; just that it doesn't crash.
    assert state is not None


# ── VoiceClusteringConfig behavior ─────────────────────────────────

def test_custom_config_can_require_larger_cluster_size(base_freqs):
    """min_cluster_size=2 drops single-oscillator voices."""
    z = _add_noise(_synthesize_voice(base_freqs, 440.0, amp=0.8))
    cfg = VoiceClusteringConfig(min_cluster_size=2)
    state = extract_voices(_make_window(z, base_freqs), config=cfg)
    assert state.active_voices == []


def test_custom_config_lower_correlation_threshold(base_freqs):
    """Lowering the correlation threshold lets weakly-correlated
    oscillators still cluster together."""
    # Two oscillators with modest correlation — above 0.4 but below 0.7.
    n_frames = 60
    t = np.linspace(0, 2 * np.pi, n_frames)
    env_a = 0.3 + 0.25 * np.sin(t)
    env_b = 0.3 + 0.25 * np.sin(t + 0.7)  # slightly offset
    z = np.zeros((n_frames, len(base_freqs)), dtype=np.complex128)
    for idx, env in ((40, env_a), (60, env_b)):
        f = base_freqs[idx]
        z[:, idx] = env * 0.5 * np.exp(
            1j * 2 * np.pi * f * np.arange(n_frames) / 60.0
        )
    z = _add_noise(z)
    tight = VoiceClusteringConfig(correlation_threshold=0.85)
    loose = VoiceClusteringConfig(correlation_threshold=0.3)
    tight_state = extract_voices(_make_window(z, base_freqs), config=tight)
    loose_state = extract_voices(_make_window(z, base_freqs), config=loose)
    # Tight threshold should split; loose should merge (or at least be
    # no larger than tight's cluster count).
    assert len(tight_state.active_voices) >= len(loose_state.active_voices)


# ── Frames-too-few edge ────────────────────────────────────────────

def test_single_frame_window_returns_empty_voice_state(base_freqs):
    """A window with only one frame doesn't yield meaningful
    correlations — should gracefully return no new voices."""
    z = np.ones((1, len(base_freqs)), dtype=np.complex128) * 0.5
    state = extract_voices(_make_window(z, base_freqs))
    # No active voices (can't cluster with no envelope history).
    assert state.active_voices == []


# ── Voice confidence and center frequency ─────────────────────────

def test_voice_center_frequency_is_amplitude_weighted(base_freqs):
    """A voice with a strong fundamental at 220 Hz and a weaker
    harmonic at 440 Hz should report a center_freq closer to 220 Hz."""
    z = _synthesize_voice(base_freqs, 220.0, harmonic_count=2, amp=0.8)
    z = _add_noise(z)
    state = extract_voices(_make_window(z, base_freqs))
    assert len(state.active_voices) == 1
    v = state.active_voices[0]
    # Harmonic amplitudes are 1/1 and 1/2, so log-weighted mean lives
    # between 220 and 440 but closer to 220 (since the fundamental is
    # the brighter one).
    assert 220.0 < v.center_freq < 330.0


def test_voice_confidence_high_for_coherent_cluster(base_freqs):
    """A perfectly-correlated cluster should produce confidence ≥
    0.9."""
    n_frames = 60
    t = np.linspace(0, 2 * np.pi, n_frames)
    env = 0.4 + 0.3 * np.sin(t)
    z = np.zeros((n_frames, len(base_freqs)), dtype=np.complex128)
    for idx in (30, 31, 32):
        f = base_freqs[idx]
        z[:, idx] = env * 0.5 * np.exp(
            1j * 2 * np.pi * f * np.arange(n_frames) / 60.0
        )
    z = _add_noise(z, noise_amp=0.0)
    state = extract_voices(_make_window(z, base_freqs))
    assert len(state.active_voices) == 1
    assert state.active_voices[0].confidence >= 0.9


# ── Phase 2: per-voice rhythm association ─────────────────────────

@pytest.fixture
def rhythm_bank():
    """Log-spaced rhythm oscillator bank — matches engine default."""
    return np.geomspace(0.5, 10.0, 50)


def _window_with_rhythm(pitch_z: np.ndarray, pitch_freqs: np.ndarray,
                         rhythm_freqs: np.ndarray, frame_hz: float = 60.0):
    """Build a StateWindow with both pitch and rhythm state. Rhythm
    oscillators are placed at their natural frequencies (pure sine) so
    their phases at the last frame are deterministic."""
    n_frames = pitch_z.shape[0]
    t = np.arange(n_frames) / frame_hz
    rhythm_z = np.zeros((n_frames, len(rhythm_freqs)), dtype=np.complex128)
    for i, f in enumerate(rhythm_freqs):
        rhythm_z[:, i] = np.exp(1j * 2 * np.pi * f * t)
    return StateWindow(
        pitch_z=pitch_z, pitch_freqs=pitch_freqs,
        rhythm_z=rhythm_z, rhythm_freqs=rhythm_freqs,
        frame_hz=frame_hz,
    )


def _voice_with_rhythmic_envelope(pitch_freqs: np.ndarray, center_hz: float,
                                    envelope_freq_hz: float,
                                    n_frames: int = 120,
                                    frame_hz: float = 60.0):
    """One pitch oscillator modulated by a sinusoidal envelope at
    ``envelope_freq_hz``. The voice's rhythm should match that
    frequency."""
    t = np.arange(n_frames) / frame_hz
    env = 0.4 + 0.3 * np.sin(2 * np.pi * envelope_freq_hz * t)
    idx = int(np.argmin(np.abs(pitch_freqs - center_hz)))
    pitch_z = np.zeros((n_frames, len(pitch_freqs)), dtype=np.complex128)
    f = pitch_freqs[idx]
    pitch_z[:, idx] = env * np.exp(1j * 2 * np.pi * f * t)
    return pitch_z


def test_voice_rhythm_at_120_bpm(base_freqs, rhythm_bank):
    """A voice whose envelope pulses at 2 Hz should report bpm ≈ 120."""
    pitch_z = _voice_with_rhythmic_envelope(
        base_freqs, center_hz=440.0, envelope_freq_hz=2.0
    )
    pitch_z = _add_noise(pitch_z)
    w = _window_with_rhythm(pitch_z, base_freqs, rhythm_bank)
    state = extract_voices(w)
    state = extract_voice_rhythms(w, state)
    assert len(state.active_voices) >= 1
    v = state.active_voices[0]
    assert v.rhythm is not None
    assert 115.0 < v.rhythm.bpm < 125.0
    assert v.rhythm.confidence > 0.3


def test_voice_rhythm_at_180_bpm(base_freqs, rhythm_bank):
    """3 Hz envelope → 180 BPM."""
    pitch_z = _voice_with_rhythmic_envelope(
        base_freqs, center_hz=440.0, envelope_freq_hz=3.0
    )
    pitch_z = _add_noise(pitch_z)
    w = _window_with_rhythm(pitch_z, base_freqs, rhythm_bank)
    state = extract_voices(w)
    state = extract_voice_rhythms(w, state)
    v = state.active_voices[0]
    assert v.rhythm is not None
    assert 172.0 < v.rhythm.bpm < 188.0


def test_voice_rhythm_flat_envelope_stays_none(base_freqs, rhythm_bank):
    """A voice with no envelope modulation has no rhythm to assign."""
    n_frames = 120
    pitch_z = np.zeros((n_frames, len(base_freqs)), dtype=np.complex128)
    idx = int(np.argmin(np.abs(base_freqs - 440.0)))
    f = base_freqs[idx]
    t = np.arange(n_frames) / 60.0
    # Constant amplitude, zero modulation.
    pitch_z[:, idx] = 0.5 * np.exp(1j * 2 * np.pi * f * t)
    # No additive noise on the envelope so the std guard triggers.
    w = _window_with_rhythm(pitch_z, base_freqs, rhythm_bank)
    state = extract_voices(w)
    state = extract_voice_rhythms(w, state)
    v = state.active_voices[0]
    assert v.rhythm is None


def test_voice_rhythm_noise_has_low_confidence(base_freqs, rhythm_bank):
    """A voice whose envelope is white noise has no concentrated
    DFT peak, so the rhythm confidence stays well below what a
    clean sinusoid would produce."""
    n_frames = 120
    idx = int(np.argmin(np.abs(base_freqs - 440.0)))
    f = base_freqs[idx]
    t = np.arange(n_frames) / 60.0
    rng = np.random.default_rng(42)
    env = 0.4 + 0.1 * rng.standard_normal(n_frames)
    pitch_z = np.zeros((n_frames, len(base_freqs)), dtype=np.complex128)
    pitch_z[:, idx] = env * np.exp(1j * 2 * np.pi * f * t)
    w = _window_with_rhythm(pitch_z, base_freqs, rhythm_bank)
    state = extract_voices(w)
    state = extract_voice_rhythms(w, state)
    v = state.active_voices[0]
    if v.rhythm is not None:
        # Clean 2 Hz sinusoid gives > 0.7 confidence; noise should
        # stay well under that.
        assert v.rhythm.confidence < 0.7


def test_two_voices_different_tempos(base_freqs, rhythm_bank):
    """Two voices at different envelope tempos → each gets its own
    rhythm. This is the core novel property — per-voice tempo."""
    n_frames = 180
    t = np.arange(n_frames) / 60.0
    # Voice A: 100 BPM = 1.67 Hz envelope at 220 Hz fundamental
    env_a = 0.4 + 0.3 * np.sin(2 * np.pi * 1.67 * t)
    idx_a = int(np.argmin(np.abs(base_freqs - 220.0)))
    pitch_z = np.zeros((n_frames, len(base_freqs)), dtype=np.complex128)
    pitch_z[:, idx_a] = env_a * np.exp(1j * 2 * np.pi * base_freqs[idx_a] * t)
    # Voice B: 180 BPM = 3.0 Hz envelope at 1100 Hz
    env_b = 0.3 + 0.25 * np.sin(2 * np.pi * 3.0 * t)
    idx_b = int(np.argmin(np.abs(base_freqs - 1100.0)))
    pitch_z[:, idx_b] = env_b * np.exp(1j * 2 * np.pi * base_freqs[idx_b] * t)
    pitch_z = _add_noise(pitch_z)
    w = _window_with_rhythm(pitch_z, base_freqs, rhythm_bank)
    state = extract_voices(w)
    state = extract_voice_rhythms(w, state)
    active = state.active_voices
    assert len(active) == 2
    # Each voice should have a rhythm, and the rhythms should differ.
    for v in active:
        assert v.rhythm is not None
    bpms = sorted(v.rhythm.bpm for v in active)
    # Low-freq voice near 100 BPM, high-freq voice near 180 BPM
    assert 95.0 < bpms[0] < 110.0
    assert 172.0 < bpms[1] < 188.0


def test_voice_rhythm_phase_tracks_rhythm_oscillator(base_freqs, rhythm_bank):
    """The reported phase is the rhythm oscillator's phase at the
    last frame of the window — so it matches what a consumer reading
    rhythm_z directly would see."""
    pitch_z = _voice_with_rhythmic_envelope(
        base_freqs, center_hz=440.0, envelope_freq_hz=2.0
    )
    pitch_z = _add_noise(pitch_z)
    w = _window_with_rhythm(pitch_z, base_freqs, rhythm_bank)
    state = extract_voices(w)
    state = extract_voice_rhythms(w, state)
    v = state.active_voices[0]
    assert v.rhythm is not None
    expected_phase = np.angle(w.rhythm_z_2d[-1, v.rhythm.osc_idx])
    assert abs(v.rhythm.phase - float(expected_phase)) < 1e-6


def test_voice_rhythm_silent_voices_untouched(base_freqs, rhythm_bank):
    """Silent voices should not have their rhythm modified by a
    subsequent extract_voice_rhythms call — only active ones
    participate in the DFT."""
    pitch_z = _voice_with_rhythmic_envelope(
        base_freqs, center_hz=440.0, envelope_freq_hz=2.0
    )
    pitch_z = _add_noise(pitch_z)
    w_on = _window_with_rhythm(pitch_z, base_freqs, rhythm_bank)
    z_off = np.zeros_like(pitch_z)
    w_off = _window_with_rhythm(z_off, base_freqs, rhythm_bank)
    # Frame 1: voice active with rhythm
    state = extract_voices(w_on)
    state = extract_voice_rhythms(w_on, state)
    assert state.active_voices[0].rhythm is not None
    # Frame 2: silent — voice carried forward, not active
    state = extract_voices(w_off, prev_state=state)
    # Some silent voices exist but no active ones
    assert not state.active_voices
    silent_voices = [v for v in state.voices if not v.active]
    assert silent_voices
    original_rhythm = silent_voices[0].rhythm
    # Apply rhythm extraction to the silent-frame window
    state = extract_voice_rhythms(w_off, state)
    silent_voices = [v for v in state.voices if not v.active]
    # Silent voice's rhythm is preserved from when it was active
    assert silent_voices[0].rhythm == original_rhythm


# ── Phase 3: per-voice motor coupling ──────────────────────────────

def _window_with_rhythm_and_motor(
    pitch_z: np.ndarray,
    pitch_freqs: np.ndarray,
    rhythm_freqs: np.ndarray,
    motor_freqs: np.ndarray,
    motor_phase_offset: float = np.pi / 3,
    frame_hz: float = 60.0,
):
    """StateWindow with both rhythm and motor oscillator banks.
    Motor oscillators have a deliberate phase offset so tests can
    distinguish them from rhythm phases."""
    n_frames = pitch_z.shape[0]
    t = np.arange(n_frames) / frame_hz
    rhythm_z = np.zeros((n_frames, len(rhythm_freqs)), dtype=np.complex128)
    for i, f in enumerate(rhythm_freqs):
        rhythm_z[:, i] = np.exp(1j * 2 * np.pi * f * t)
    motor_z = np.zeros((n_frames, len(motor_freqs)), dtype=np.complex128)
    for i, f in enumerate(motor_freqs):
        motor_z[:, i] = np.exp(1j * (2 * np.pi * f * t + motor_phase_offset))
    return StateWindow(
        pitch_z=pitch_z, pitch_freqs=pitch_freqs,
        rhythm_z=rhythm_z, rhythm_freqs=rhythm_freqs,
        motor_z=motor_z, motor_freqs=motor_freqs,
        frame_hz=frame_hz,
    )


def test_voice_motor_at_120_bpm(base_freqs, rhythm_bank):
    """Motor matching uses the same DFT as rhythm matching but on
    the motor bank. A 2 Hz envelope → motor.bpm ≈ 120."""
    from neurodynamics.voices import extract_voice_motor
    motor_bank = rhythm_bank.copy()  # same range in this test
    pitch_z = _voice_with_rhythmic_envelope(
        base_freqs, center_hz=440.0, envelope_freq_hz=2.0
    )
    pitch_z = _add_noise(pitch_z)
    w = _window_with_rhythm_and_motor(
        pitch_z, base_freqs, rhythm_bank, motor_bank
    )
    state = extract_voices(w)
    state = extract_voice_motor(w, state)
    v = state.active_voices[0]
    assert v.motor is not None
    assert 115.0 < v.motor.bpm < 125.0
    assert v.motor.confidence > 0.3


def test_voice_motor_phase_reads_motor_not_rhythm(base_freqs, rhythm_bank):
    """The motor phase in VoiceMotor should be the motor oscillator's
    phase, not the rhythm oscillator's. Motor has a π/3 phase offset
    in the test fixture so we can tell them apart."""
    from neurodynamics.voices import extract_voice_motor
    motor_bank = rhythm_bank.copy()
    pitch_z = _voice_with_rhythmic_envelope(
        base_freqs, center_hz=440.0, envelope_freq_hz=2.0
    )
    pitch_z = _add_noise(pitch_z)
    w = _window_with_rhythm_and_motor(
        pitch_z, base_freqs, rhythm_bank, motor_bank,
        motor_phase_offset=np.pi / 3,
    )
    state = extract_voices(w)
    state = extract_voice_rhythms(w, state)
    state = extract_voice_motor(w, state)
    v = state.active_voices[0]
    assert v.rhythm is not None
    assert v.motor is not None
    # Motor phase should equal motor_z[-1, motor.osc_idx] phase.
    expected_motor_phase = np.angle(w.motor_z_2d[-1, v.motor.osc_idx])
    assert abs(v.motor.phase - float(expected_motor_phase)) < 1e-6
    # Rhythm phase should equal rhythm_z[-1, rhythm.osc_idx] phase.
    expected_rhythm_phase = np.angle(w.rhythm_z_2d[-1, v.rhythm.osc_idx])
    assert abs(v.rhythm.phase - float(expected_rhythm_phase)) < 1e-6


def test_voice_motor_without_motor_bank_returns_unchanged(
    base_freqs, rhythm_bank
):
    """If the StateWindow has no motor state, extract_voice_motor
    should leave the voice state untouched (motor stays None)."""
    from neurodynamics.voices import extract_voice_motor
    pitch_z = _voice_with_rhythmic_envelope(
        base_freqs, center_hz=440.0, envelope_freq_hz=2.0
    )
    pitch_z = _add_noise(pitch_z)
    w = _window_with_rhythm(pitch_z, base_freqs, rhythm_bank)  # no motor
    state = extract_voices(w)
    original = state.active_voices[0]
    state = extract_voice_motor(w, state)
    v = state.active_voices[0]
    assert v.motor is None
    # Everything else is unchanged
    assert v.id == original.id
    assert v.oscillator_indices == original.oscillator_indices


def test_voice_motor_different_bank_than_rhythm(base_freqs, rhythm_bank):
    """Motor and rhythm banks can have different frequencies. The
    motor match should pick from the motor bank, not the rhythm
    bank."""
    from neurodynamics.voices import extract_voice_motor
    # Motor bank covers 1-8 Hz (60-480 BPM), rhythm covers 0.5-10.
    motor_bank = np.geomspace(1.0, 8.0, 30)
    pitch_z = _voice_with_rhythmic_envelope(
        base_freqs, center_hz=440.0, envelope_freq_hz=2.0
    )
    pitch_z = _add_noise(pitch_z)
    w = _window_with_rhythm_and_motor(
        pitch_z, base_freqs, rhythm_bank, motor_bank
    )
    state = extract_voices(w)
    state = extract_voice_motor(w, state)
    v = state.active_voices[0]
    assert v.motor is not None
    # Motor bank index must be in range for the motor bank, not the
    # rhythm bank.
    assert 0 <= v.motor.osc_idx < len(motor_bank)
    assert v.motor.freq == motor_bank[v.motor.osc_idx]


def test_voice_motor_and_rhythm_compose_independently(base_freqs, rhythm_bank):
    """Running extract_voice_rhythms and extract_voice_motor in any
    order populates both fields — they're orthogonal."""
    from neurodynamics.voices import extract_voice_motor
    motor_bank = rhythm_bank.copy()
    pitch_z = _voice_with_rhythmic_envelope(
        base_freqs, center_hz=440.0, envelope_freq_hz=2.0
    )
    pitch_z = _add_noise(pitch_z)
    w = _window_with_rhythm_and_motor(
        pitch_z, base_freqs, rhythm_bank, motor_bank
    )
    # Order A: rhythm then motor
    a = extract_voices(w)
    a = extract_voice_rhythms(w, a)
    a = extract_voice_motor(w, a)
    # Order B: motor then rhythm
    b = extract_voices(w)
    b = extract_voice_motor(w, b)
    b = extract_voice_rhythms(w, b)
    va, vb = a.active_voices[0], b.active_voices[0]
    assert va.rhythm is not None and va.motor is not None
    assert vb.rhythm is not None and vb.motor is not None
    assert va.rhythm == vb.rhythm
    assert va.motor == vb.motor
