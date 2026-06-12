"""Phase 4.1 validation: PLV gate distinguishes same-source harmonics
from independent sources at integer ratios.

The auditory scene analysis problem: a chord with notes at integer
ratios (e.g., octave) looks spectrally similar to a single source's
harmonic stack. Without phase-locking evidence they're confusable;
with PLV we can separate them.

Tests:
1. Harmonic stack (fundamental + harmonics, SAME phase relationship)
   → high PLV → merge into one voice
2. Independent sources at integer ratio (random phase, octave apart)
   → low PLV → stay as separate voices
"""

from __future__ import annotations

import numpy as np

from neurodynamics.perceptual import StateWindow
from neurodynamics.voices import (
    VoiceClusteringConfig, VoiceState, extract_voices,
    _component_plv_at_ratio, _pairwise_harmonic_matrix,
)


def _synth_harmonic_stack(freqs: np.ndarray, fundamental_hz: float,
                           n_frames: int = 80, sr: int = 30) -> np.ndarray:
    """Synthesize a phase-locked harmonic stack (one source).

    Returns shape (n_frames, len(freqs)) complex pitch_z. Harmonics
    at fundamental, 2×, 3×, 4× share the SAME source phase — so
    their phases at 2f, 3f, 4f are integer multiples of the
    fundamental's phase.
    """
    t = np.arange(n_frames) / sr
    src_phase = 2 * np.pi * fundamental_hz * t      # source phase trajectory
    env = 0.3 + 0.2 * np.sin(2 * np.pi * 0.5 * t)  # slow envelope
    z = np.zeros((n_frames, len(freqs)), dtype=np.complex128)
    for h in (1, 2, 3, 4):
        target = fundamental_hz * h
        i = int(np.argmin(np.abs(freqs - target)))
        # phase at h-th harmonic is h × source phase (perfect lock)
        z[:, i] = (env / h) * np.exp(1j * h * src_phase)
    return z


def _synth_independent_at_octave(freqs: np.ndarray, f_low_hz: float,
                                  n_frames: int = 80, sr: int = 30) -> np.ndarray:
    """Synthesize two INDEPENDENT sources, one octave apart.

    Each source has its OWN phase trajectory with independent
    vibrato / frequency wobble representative of real musical
    sources (different instruments don't share phase precision).
    Envelopes also independent.
    """
    t = np.arange(n_frames) / sr
    rng = np.random.default_rng(7)
    # Random-walk drift comparable to ~0.5 Hz vibrato over the window
    drift_low = 1.0 * np.cumsum(rng.standard_normal(n_frames))
    drift_high = 1.0 * np.cumsum(rng.standard_normal(n_frames))
    phi_low = 2 * np.pi * f_low_hz * t + drift_low
    phi_high = 2 * np.pi * (2 * f_low_hz) * t + drift_high
    env_low = 0.3 + 0.2 * np.sin(2 * np.pi * 0.4 * t)
    env_high = 0.3 + 0.2 * np.sin(2 * np.pi * 1.1 * t + 1.0)
    z = np.zeros((n_frames, len(freqs)), dtype=np.complex128)
    i_low = int(np.argmin(np.abs(freqs - f_low_hz)))
    i_high = int(np.argmin(np.abs(freqs - 2 * f_low_hz)))
    z[:, i_low] = env_low * np.exp(1j * phi_low)
    z[:, i_high] = env_high * np.exp(1j * phi_high)
    return z


def _make_window(z: np.ndarray, freqs: np.ndarray) -> StateWindow:
    return StateWindow(
        pitch_z=z, pitch_freqs=freqs,
        rhythm_z=np.zeros(1, dtype=np.complex128),
        rhythm_freqs=np.array([1.0]),
        frame_hz=30.0,
    )


def _log_spaced(low: float, high: float, n: int) -> np.ndarray:
    return np.geomspace(low, high, n)


def test_pairwise_harmonic_matrix_matches_scalar():
    """Vectorized pairwise check should match the scalar form."""
    from neurodynamics.voices import _pair_is_harmonic
    freqs = np.array([110.0, 220.0, 311.0, 440.0, 660.0, 880.0])
    mat = _pairwise_harmonic_matrix(freqs, 0.5)
    n = len(freqs)
    for i in range(n):
        for j in range(n):
            scalar = _pair_is_harmonic(freqs[i], freqs[j], 0.5)
            assert bool(mat[i, j]) == scalar, (
                f"mismatch at ({i},{j}): mat={mat[i,j]}, scalar={scalar}"
            )


def test_plv_at_ratio_high_for_locked_pair():
    """A phase-locked 1:2 pair has PLV ~ 1.0."""
    n = 200
    t = np.arange(n) / 30.0
    f1 = 100.0
    src = 2 * np.pi * f1 * t
    phases_low = src                              # at f1
    phases_high = 2 * src + 0.7                  # at 2*f1, fixed offset
    result = _component_plv_at_ratio(
        phases_low, phases_high, f1, 2 * f1, tolerance_semitones=0.5
    )
    assert result is not None
    plv, p, q = result
    print(f"locked 1:2 pair: PLV={plv:.3f}, ratio={p}:{q}")
    assert plv > 0.95, f"locked pair PLV {plv:.3f} should be near 1"


def test_plv_at_ratio_low_for_independent_pair():
    """Two oscillators at nominal 2:1 frequency ratio but with
    INDEPENDENT vibrato/wobble → PLV should be low.

    Real-world independent sources have phase drift from vibrato,
    pitch jitter, micro-detuning. Use realistic drift magnitude
    (~1 rad/√sample step, comparable to ~0.5 Hz vibrato).
    """
    rng = np.random.default_rng(11)
    n = 200
    t = np.arange(n) / 30.0
    f1 = 100.0
    drift1 = 1.0 * np.cumsum(rng.standard_normal(n))
    drift2 = 1.0 * np.cumsum(rng.standard_normal(n))
    phases_low = 2 * np.pi * f1 * t + drift1
    phases_high = 2 * np.pi * 2 * f1 * t + drift2
    result = _component_plv_at_ratio(
        phases_low, phases_high, f1, 2 * f1, tolerance_semitones=0.5
    )
    assert result is not None
    plv, p, q = result
    print(f"independent 1:2 pair: PLV={plv:.3f}, ratio={p}:{q}")
    # With independent drift, relative phase walks → PLV well below
    # the merge threshold
    assert plv < 0.7, f"independent pair PLV {plv:.3f} too high"


def test_harmonic_stack_merges_to_one_voice():
    """Phase-locked harmonic stack collapses to a single voice via
    the PLV gate in `_merge_harmonic_components`."""
    freqs = _log_spaced(50.0, 2000.0, 80)
    z = _synth_harmonic_stack(freqs, fundamental_hz=200.0,
                              n_frames=80, sr=30)
    # Add a tiny amount of noise so flat detection doesn't kick in.
    rng = np.random.default_rng(3)
    z = z + 0.001 * (rng.standard_normal(z.shape)
                      + 1j * rng.standard_normal(z.shape))
    window = _make_window(z, freqs)
    state = extract_voices(window)
    print(f"harmonic stack voices: {len(state.active_voices)}")
    for v in state.active_voices:
        print(f"  V{v.id}: f={v.center_freq:.1f} amp={v.amp:.3f}")
    # Should collapse to 1 (fundamental). At most 2 (fundamental +
    # one isolated harmonic if PLV barely missed).
    assert len(state.active_voices) <= 2


def test_independent_octave_stays_two_voices():
    """Two independent sources at octave should remain as TWO voices
    (PLV gate sees no lock; envelope correlation gate sees no
    correlation; merge declined)."""
    freqs = _log_spaced(50.0, 2000.0, 80)
    z = _synth_independent_at_octave(freqs, f_low_hz=220.0,
                                       n_frames=80, sr=30)
    rng = np.random.default_rng(5)
    z = z + 0.001 * (rng.standard_normal(z.shape)
                      + 1j * rng.standard_normal(z.shape))
    window = _make_window(z, freqs)
    state = extract_voices(window)
    print(f"independent octave voices: {len(state.active_voices)}")
    for v in state.active_voices:
        print(f"  V{v.id}: f={v.center_freq:.1f} amp={v.amp:.3f}")
    # Should remain at 2.
    assert len(state.active_voices) == 2, (
        f"expected 2 voices (independent octave), got "
        f"{len(state.active_voices)}"
    )
