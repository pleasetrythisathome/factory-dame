"""Perceptual feature extractors — key, tempo, consonance.

These extractors are thin rollups on top of oscillator state: the
dynamical system has already done the heavy lifting (phase-locking,
Hebbian learning, entrainment), and the extractors read off
interpretations.

Tests use synthetic StateWindow objects constructed directly — the
extractors are state-slice-agnostic, so we don't need to run the full
engine to test them.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurodynamics.perceptual import (
    StateWindow,
    extract_chord,
    extract_consonance,
    extract_key,
    extract_rhythm_structure,
    extract_tempo,
    hz_to_pitch_class,
)


# ── StateWindow shape handling ─────────────────────────────────────────

def test_state_window_accepts_single_frame():
    """A 1D (n_osc,) array is promoted to (1, n_osc)."""
    pz = np.zeros(10, dtype=np.complex128)
    rz = np.zeros(8, dtype=np.complex128)
    w = StateWindow(
        pitch_z=pz, pitch_freqs=np.linspace(100, 1000, 10),
        rhythm_z=rz, rhythm_freqs=np.linspace(1, 5, 8),
        frame_hz=60.0,
    )
    assert w.pitch_z_2d.shape == (1, 10)
    assert w.rhythm_z_2d.shape == (1, 8)


def test_state_window_accepts_windowed_input():
    """A 2D (frames, n_osc) array passes through unchanged."""
    pz = np.zeros((30, 10), dtype=np.complex128)
    rz = np.zeros((30, 8), dtype=np.complex128)
    w = StateWindow(
        pitch_z=pz, pitch_freqs=np.linspace(100, 1000, 10),
        rhythm_z=rz, rhythm_freqs=np.linspace(1, 5, 8),
        frame_hz=60.0,
    )
    assert w.pitch_z_2d.shape == (30, 10)
    assert w.rhythm_z_2d.shape == (30, 8)


# ── Pitch class mapping ────────────────────────────────────────────────

def test_hz_to_pitch_class_a4():
    """A4 = 440 Hz → pitch class A (index 9)."""
    assert hz_to_pitch_class(440.0) == 9


def test_hz_to_pitch_class_middle_c():
    """C4 ≈ 261.63 Hz → pitch class C (index 0)."""
    assert hz_to_pitch_class(261.63) == 0


def test_hz_to_pitch_class_octave_equivalence():
    """A3 and A5 both map to A."""
    assert hz_to_pitch_class(220.0) == 9
    assert hz_to_pitch_class(880.0) == 9


# ── Tempo extraction ───────────────────────────────────────────────────

def _synthetic_rhythm_window(peak_freq_hz: float, n_rhythm: int = 30,
                              low: float = 0.5, high: float = 10.0,
                              frames: int = 60, sharpness: float = 8.0):
    """Build a rhythm_z where an oscillator lives exactly at peak_freq_hz
    and has the largest amplitude. Gaussian amplitude profile centered
    on the peak — no grid quantization in the test."""
    base = np.geomspace(low, high, max(n_rhythm - 1, 1)).tolist()
    freqs = np.array(sorted(base + [peak_freq_hz]))
    peak_idx = int(np.argmin(np.abs(freqs - peak_freq_hz)))
    amps = np.exp(-0.5 * ((np.arange(len(freqs)) - peak_idx) / sharpness) ** 2)
    rhythm_z = np.tile(amps.astype(np.complex128), (frames, 1))
    return freqs, rhythm_z


def test_tempo_from_peak_rhythm_oscillator():
    """120 BPM = 2 Hz. A rhythm window peaked at 2 Hz should extract ≈120."""
    freqs, rz = _synthetic_rhythm_window(peak_freq_hz=2.0)
    w = StateWindow(
        pitch_z=np.zeros(1, dtype=np.complex128),
        pitch_freqs=np.array([440.0]),
        rhythm_z=rz, rhythm_freqs=freqs,
        frame_hz=60.0,
    )
    bpm, confidence = extract_tempo(w)
    assert 115.0 < bpm < 125.0
    assert confidence > 0.3  # sharp peak → decent confidence


def test_tempo_flat_rhythm_low_confidence():
    """Uniform rhythm amplitudes → confidence near zero."""
    freqs = np.geomspace(0.5, 10.0, 30)
    rz = np.ones((60, 30), dtype=np.complex128)
    w = StateWindow(
        pitch_z=np.zeros(1, dtype=np.complex128),
        pitch_freqs=np.array([440.0]),
        rhythm_z=rz, rhythm_freqs=freqs,
        frame_hz=60.0,
    )
    _, confidence = extract_tempo(w)
    assert confidence < 0.1


def test_tempo_returns_bpm_from_various_frequencies():
    """90, 120, 140 BPM all round-trip through the extractor."""
    for bpm in (90.0, 120.0, 140.0):
        f = bpm / 60.0
        freqs, rz = _synthetic_rhythm_window(peak_freq_hz=f)
        w = StateWindow(
            pitch_z=np.zeros(1, dtype=np.complex128),
            pitch_freqs=np.array([440.0]),
            rhythm_z=rz, rhythm_freqs=freqs,
            frame_hz=60.0,
        )
        extracted_bpm, _ = extract_tempo(w)
        assert abs(extracted_bpm - bpm) < 8.0, \
            f"BPM {bpm} → extracted {extracted_bpm}"


# ── Key extraction ─────────────────────────────────────────────────────

def _pitch_window_with_amplitudes(amps_per_osc: np.ndarray,
                                   low: float = 30.0, high: float = 4000.0,
                                   n_pitch: int | None = None,
                                   frames: int = 10):
    """Build a pitch window with the given per-oscillator amplitude vector."""
    n_pitch = n_pitch if n_pitch is not None else len(amps_per_osc)
    freqs = np.geomspace(low, high, n_pitch)
    pitch_z = np.tile(amps_per_osc.astype(np.complex128), (frames, 1))
    return freqs, pitch_z


def test_key_from_hebbian_diagonal():
    """A w_pitch matrix whose diagonal peaks at an A-frequency oscillator
    should report tonic="A"."""
    n_pitch = 100
    freqs = np.geomspace(30.0, 4000.0, n_pitch)
    # Find oscillator closest to A4 (440 Hz) and give it a strong diagonal.
    a_idx = int(np.argmin(np.abs(freqs - 440.0)))
    diag = np.full(n_pitch, 0.05)
    diag[a_idx] = 1.0
    # Also boost the A-related harmonics (other octaves of A).
    for octave_hz in (110.0, 220.0, 880.0, 1760.0):
        diag[int(np.argmin(np.abs(freqs - octave_hz)))] = 0.6
    w_pitch = np.diag(diag).astype(np.complex128)

    pz = np.zeros(n_pitch, dtype=np.complex128)
    w = StateWindow(
        pitch_z=pz, pitch_freqs=freqs,
        rhythm_z=np.zeros(1, dtype=np.complex128),
        rhythm_freqs=np.array([1.0]),
        frame_hz=60.0,
        w_pitch=w_pitch,
    )
    result = extract_key(w)
    assert result["tonic"] == "A"
    assert result["confidence"] > 0.3


def test_key_fallback_uses_pitch_z_when_no_w():
    """With no w_pitch, extract_key should fall back to pitch_z amplitudes."""
    n_pitch = 60
    freqs = np.geomspace(30.0, 4000.0, n_pitch)
    # Boost a C-major chord: C, E, G across octaves.
    c_major_freqs = [130.81, 164.81, 196.0, 261.63, 329.63, 392.0,
                     523.25, 659.25, 783.99]
    amps = np.full(n_pitch, 0.05)
    for f in c_major_freqs:
        amps[int(np.argmin(np.abs(freqs - f)))] = 1.0
    freqs, pz = _pitch_window_with_amplitudes(amps, n_pitch=n_pitch)

    w = StateWindow(
        pitch_z=pz, pitch_freqs=freqs,
        rhythm_z=np.zeros(1, dtype=np.complex128),
        rhythm_freqs=np.array([1.0]),
        frame_hz=60.0,
    )
    result = extract_key(w)
    assert result["tonic"] == "C"
    assert result["mode"] == "major"


def test_key_returns_minor_when_profile_matches():
    """A minor-triad weighting should return mode='minor'."""
    n_pitch = 60
    freqs = np.geomspace(30.0, 4000.0, n_pitch)
    # A minor: A, C, E
    a_minor_freqs = [110.0, 130.81, 164.81, 220.0, 261.63, 329.63,
                     440.0, 523.25, 659.25]
    amps = np.full(n_pitch, 0.05)
    for f in a_minor_freqs:
        amps[int(np.argmin(np.abs(freqs - f)))] = 1.0
    freqs, pz = _pitch_window_with_amplitudes(amps, n_pitch=n_pitch)

    w = StateWindow(
        pitch_z=pz, pitch_freqs=freqs,
        rhythm_z=np.zeros(1, dtype=np.complex128),
        rhythm_freqs=np.array([1.0]),
        frame_hz=60.0,
    )
    result = extract_key(w)
    assert result["tonic"] == "A"
    assert result["mode"] == "minor"


# ── Consonance extraction ──────────────────────────────────────────────

def _two_oscillator_window(f1: float, f2: float, a1: float = 1.0,
                            a2: float = 1.0, n_pitch: int = 100,
                            low: float = 30.0, high: float = 4000.0):
    """Pitch window with oscillators placed exactly at f1 and f2 plus a
    log-spaced filler bank. Zero quantization error — the consonance
    extractor is being tested against the algorithm, not the grid."""
    base = np.geomspace(low, high, max(n_pitch - 2, 1)).tolist()
    freqs = np.array(sorted(base + [f1, f2]))
    amps = np.zeros(len(freqs))
    amps[int(np.argmin(np.abs(freqs - f1)))] = a1
    amps[int(np.argmin(np.abs(freqs - f2)))] = a2
    pitch_z = amps.astype(np.complex128)
    return freqs, pitch_z


def test_consonance_octave_is_high():
    """A4 + A5 (2:1) → near-maximum consonance."""
    freqs, pz = _two_oscillator_window(440.0, 880.0)
    w = StateWindow(
        pitch_z=pz, pitch_freqs=freqs,
        rhythm_z=np.zeros(1, dtype=np.complex128),
        rhythm_freqs=np.array([1.0]),
        frame_hz=60.0,
    )
    c = extract_consonance(w)
    assert c > 0.8


def test_consonance_perfect_fifth_is_high():
    """A4 + E5 (3:2) → high consonance."""
    freqs, pz = _two_oscillator_window(440.0, 660.0)
    w = StateWindow(
        pitch_z=pz, pitch_freqs=freqs,
        rhythm_z=np.zeros(1, dtype=np.complex128),
        rhythm_freqs=np.array([1.0]),
        frame_hz=60.0,
    )
    c = extract_consonance(w)
    assert c > 0.6


def test_consonance_tritone_is_low():
    """A4 + D#5 (≈1:√2, not an integer ratio) → low consonance."""
    freqs, pz = _two_oscillator_window(440.0, 622.25)
    w = StateWindow(
        pitch_z=pz, pitch_freqs=freqs,
        rhythm_z=np.zeros(1, dtype=np.complex128),
        rhythm_freqs=np.array([1.0]),
        frame_hz=60.0,
    )
    c = extract_consonance(w)
    assert c < 0.4


def _synthetic_rhythm_phases(peak_freq_hz: float, companion_ratios: list,
                               n_rhythm: int = 30,
                               low: float = 0.5, high: float = 10.0,
                               frames: int = 120, frame_hz: float = 60.0):
    """Build rhythm_z with a dominant oscillator at peak_freq_hz and
    optional companions at exact rational-ratio frequencies. Insert
    peak and companion frequencies exactly into the oscillator bank
    so phase-lock is exact rather than approximate."""
    target_freqs = [peak_freq_hz]
    for ratio_p, ratio_q in companion_ratios:
        target = peak_freq_hz * (ratio_p / ratio_q)
        if low <= target <= high:
            target_freqs.append(target)
    base = np.geomspace(low, high,
                         max(n_rhythm - len(target_freqs), 1)).tolist()
    freqs = np.array(sorted(base + target_freqs))
    t = np.arange(frames) / frame_hz
    rhythm_z = np.zeros((frames, len(freqs)), dtype=np.complex128)
    peak_idx = int(np.argmin(np.abs(freqs - peak_freq_hz)))
    rhythm_z[:, peak_idx] = np.exp(1j * 2 * np.pi * peak_freq_hz * t)
    for ratio_p, ratio_q in companion_ratios:
        target = peak_freq_hz * (ratio_p / ratio_q)
        if target < low or target > high:
            continue
        comp_idx = int(np.argmin(np.abs(freqs - target)))
        if comp_idx == peak_idx:
            continue
        # Phase advance at the exact ratio-target frequency so that
        # q·phi_peak - p·phi_comp stays constant.
        rhythm_z[:, comp_idx] = 0.6 * np.exp(1j * 2 * np.pi * target * t)
    return freqs, rhythm_z


def test_rhythm_structure_peak_basic():
    """Peak oscillator at 2 Hz → bpm=120, phase reads cleanly."""
    freqs, rz = _synthetic_rhythm_phases(peak_freq_hz=2.0,
                                          companion_ratios=[])
    w = StateWindow(
        pitch_z=np.zeros(1, dtype=np.complex128),
        pitch_freqs=np.array([440.0]),
        rhythm_z=rz, rhythm_freqs=freqs,
        frame_hz=60.0,
    )
    out = extract_rhythm_structure(w)
    assert 118.0 < out["peak"]["bpm"] < 122.0
    assert out["peak"]["freq"] == pytest.approx(2.0, abs=0.05)
    assert -np.pi <= out["peak"]["phase"] <= np.pi


def test_rhythm_structure_persistence_sticks_to_prev_peak():
    """When prev_peak_idx points to a strong-enough oscillator, the
    extractor sticks to it rather than flipping to argmax.

    Also: with no prior, the initial peak pick is on *biased* amps
    (Gaussian prior centered at the musical-tempo range) rather than
    raw amps. With idx 14 ≈ 2.08 Hz sitting nearer the 2 Hz prior
    center than idx 15 ≈ 2.39 Hz, idx 14 wins even at slightly lower
    raw amplitude.
    """
    freqs = np.geomspace(0.5, 10.0, 30)
    # Two nearly-equal amplitudes, idx 14 at ≈ 2.08 Hz, idx 15 at
    # ≈ 2.39 Hz. Prior center is 2 Hz so idx 14 is favored.
    amps = np.zeros(30)
    amps[14] = 0.99
    amps[15] = 1.00
    rz = np.tile(amps.astype(np.complex128), (60, 1))
    w = StateWindow(
        pitch_z=np.zeros(1, dtype=np.complex128),
        pitch_freqs=np.array([440.0]),
        rhythm_z=rz, rhythm_freqs=freqs,
        frame_hz=60.0,
    )
    # With no prior, biased peak = idx 14 (nearest 2 Hz).
    assert extract_rhythm_structure(w)["peak"]["idx"] == 14
    # Persistence: prev_peak_idx = 14 survives (within amp threshold).
    out_stuck = extract_rhythm_structure(w, prev_peak_idx=14)
    assert out_stuck["peak"]["idx"] == 14
    # Persistence: prev_peak_idx = 15 also survives — it's still
    # above persistence threshold (amps[15] = 1.0 ≥ 0.8 * max).
    out_15 = extract_rhythm_structure(w, prev_peak_idx=15)
    assert out_15["peak"]["idx"] == 15


def test_rhythm_structure_persistence_switches_when_prev_too_weak():
    """When prev peak amplitude drops below threshold, we switch."""
    freqs = np.geomspace(0.5, 10.0, 30)
    amps = np.zeros(30)
    amps[10] = 0.3
    amps[20] = 1.0
    rz = np.tile(amps.astype(np.complex128), (60, 1))
    w = StateWindow(
        pitch_z=np.zeros(1, dtype=np.complex128),
        pitch_freqs=np.array([440.0]),
        rhythm_z=rz, rhythm_freqs=freqs,
        frame_hz=60.0,
    )
    # prev at idx 10 has amp 0.3, max is 1.0 at idx 20.
    # 0.3 < 0.8 * 1.0 → switch.
    out = extract_rhythm_structure(w, prev_peak_idx=10,
                                    persistence_threshold=0.8)
    assert out["peak"]["idx"] == 20


def test_rhythm_structure_companion_at_triplet_ratio():
    """Peak at 2 Hz with a 3:2 companion at 3 Hz should show up as a
    triplet mode-lock in the companions list."""
    freqs, rz = _synthetic_rhythm_phases(
        peak_freq_hz=2.0, companion_ratios=[(3, 2)],
    )
    w = StateWindow(
        pitch_z=np.zeros(1, dtype=np.complex128),
        pitch_freqs=np.array([440.0]),
        rhythm_z=rz, rhythm_freqs=freqs,
        frame_hz=60.0,
    )
    out = extract_rhythm_structure(w)
    triplet = [c for c in out["companions"]
               if (c["ratio_p"], c["ratio_q"]) == (3, 2)]
    assert len(triplet) == 1
    assert triplet[0]["plv"] > 0.9


def test_rhythm_structure_no_companions_on_pure_sine():
    """A single active peak with no companions should yield an empty
    companions list."""
    freqs, rz = _synthetic_rhythm_phases(peak_freq_hz=2.0,
                                          companion_ratios=[])
    w = StateWindow(
        pitch_z=np.zeros(1, dtype=np.complex128),
        pitch_freqs=np.array([440.0]),
        rhythm_z=rz, rhythm_freqs=freqs,
        frame_hz=60.0,
    )
    out = extract_rhythm_structure(w)
    # Noise floor at other oscillators is zero → no companions pass
    # the amplitude threshold.
    assert out["companions"] == []


def test_rhythm_structure_bpm_stable_under_noise():
    """Rhythm state with small amplitude noise + a clear peak should
    not have BPM jittering; persistence keeps peak stable."""
    freqs = np.geomspace(0.5, 10.0, 30)
    n_frames = 60
    rng = np.random.default_rng(0)
    # Stable peak at index 20 with tiny noise on all oscillators.
    amps = 0.02 * rng.standard_normal((n_frames, 30)).astype(np.float32)
    amps[:, 19] = 0.95 + 0.03 * rng.standard_normal(n_frames)  # near peak
    amps[:, 20] = 1.00 + 0.03 * rng.standard_normal(n_frames)  # actual peak

    bpms = []
    prev = None
    # Simulate stepping through many windows, each overlapping slightly.
    for step in range(20):
        lo = step * 2
        hi = lo + 20
        window = StateWindow(
            pitch_z=np.zeros(1, dtype=np.complex128),
            pitch_freqs=np.array([440.0]),
            rhythm_z=amps[lo:hi].astype(np.complex128),
            rhythm_freqs=freqs,
            frame_hz=60.0,
        )
        out = extract_rhythm_structure(window, prev_peak_idx=prev)
        bpms.append(out["peak"]["bpm"])
        prev = out["peak"]["idx"]

    # All should be stable at the same oscillator → identical BPMs.
    assert len(set(bpms)) == 1


def _pitch_window_with_pitchclass_amps(pitch_class_amps: dict,
                                         low: float = 30.0,
                                         high: float = 4000.0,
                                         n_pitch: int = 100,
                                         frames: int = 10):
    """Pitch window where specific pitch classes have amplitude across
    multiple octaves. Useful for chord tests."""
    pc_hz = {"C": 261.63, "C#": 277.18, "D": 293.66, "D#": 311.13,
             "E": 329.63, "F": 349.23, "F#": 369.99, "G": 392.00,
             "G#": 415.30, "A": 440.00, "A#": 466.16, "B": 493.88}
    freqs = np.geomspace(low, high, n_pitch)
    amps = np.full(n_pitch, 0.02)
    for pc, amp in pitch_class_amps.items():
        base_hz = pc_hz[pc]
        for oct_factor in (0.5, 1.0, 2.0):
            f = base_hz * oct_factor
            if low <= f <= high:
                idx = int(np.argmin(np.abs(freqs - f)))
                amps[idx] = amp
    pitch_z = np.tile(amps.astype(np.complex128), (frames, 1))
    return freqs, pitch_z


def test_chord_c_major_triad():
    """C + E + G → 'C' major chord."""
    freqs, pz = _pitch_window_with_pitchclass_amps(
        {"C": 1.0, "E": 1.0, "G": 1.0}
    )
    w = StateWindow(
        pitch_z=pz, pitch_freqs=freqs,
        rhythm_z=np.zeros(1, dtype=np.complex128),
        rhythm_freqs=np.array([1.0]),
        frame_hz=60.0,
    )
    out = extract_chord(w)
    assert out["root"] == "C"
    assert out["quality"] == "maj"
    assert out["name"] == "C"


def test_chord_a_minor_triad():
    """A + C + E → 'Am'."""
    freqs, pz = _pitch_window_with_pitchclass_amps(
        {"A": 1.0, "C": 1.0, "E": 1.0}
    )
    w = StateWindow(
        pitch_z=pz, pitch_freqs=freqs,
        rhythm_z=np.zeros(1, dtype=np.complex128),
        rhythm_freqs=np.array([1.0]),
        frame_hz=60.0,
    )
    out = extract_chord(w)
    assert out["root"] == "A"
    assert out["quality"] == "min"
    assert out["name"] == "Am"


def test_chord_g_dominant_seventh():
    """G + B + D + F → 'G7'."""
    freqs, pz = _pitch_window_with_pitchclass_amps(
        {"G": 1.0, "B": 1.0, "D": 1.0, "F": 1.0}
    )
    w = StateWindow(
        pitch_z=pz, pitch_freqs=freqs,
        rhythm_z=np.zeros(1, dtype=np.complex128),
        rhythm_freqs=np.array([1.0]),
        frame_hz=60.0,
    )
    out = extract_chord(w)
    assert out["root"] == "G"
    assert out["quality"] == "dom7"
    assert out["name"] == "G7"


def test_chord_fmaj7():
    """F + A + C + E → 'Fmaj7'."""
    freqs, pz = _pitch_window_with_pitchclass_amps(
        {"F": 1.0, "A": 1.0, "C": 1.0, "E": 1.0}
    )
    w = StateWindow(
        pitch_z=pz, pitch_freqs=freqs,
        rhythm_z=np.zeros(1, dtype=np.complex128),
        rhythm_freqs=np.array([1.0]),
        frame_hz=60.0,
    )
    out = extract_chord(w)
    assert out["root"] == "F"
    assert out["quality"] == "maj7"
    assert out["name"] == "Fmaj7"


def test_consonance_ordering_octave_gt_fifth_gt_tritone():
    """Consonance should strictly order: octave > fifth > tritone."""
    def c_for(f2):
        freqs, pz = _two_oscillator_window(440.0, f2)
        w = StateWindow(
            pitch_z=pz, pitch_freqs=freqs,
            rhythm_z=np.zeros(1, dtype=np.complex128),
            rhythm_freqs=np.array([1.0]),
            frame_hz=60.0,
        )
        return extract_consonance(w)

    octave = c_for(880.0)
    fifth = c_for(660.0)
    tritone = c_for(622.25)
    assert octave > fifth > tritone
