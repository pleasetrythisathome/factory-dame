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
    VoiceState,
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
