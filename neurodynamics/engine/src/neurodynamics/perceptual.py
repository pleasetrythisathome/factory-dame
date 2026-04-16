"""Perceptual feature extractors — key, tempo, consonance.

Thin rollups on top of oscillator state: the GrFNN dynamics have already
done the heavy lifting (phase-locking, Hebbian learning, entrainment)
and the extractors read off interpretations.

State-slice-agnostic by design: StateWindow accepts either a single
frame (n,) or a rolling window (frames, n). Offline use pipes parquet
rows; live use pipes in-memory buffers. Extractors are oblivious.

All extractors return plain Python scalars or dicts — easy to ship over
OSC/MIDI and easy to render as text overlays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "StateWindow",
    "PITCH_CLASS_NAMES",
    "hz_to_pitch_class",
    "extract_tempo",
    "extract_key",
    "extract_consonance",
    "extract_rhythm_structure",
    "extract_chord",
]

# Small-integer ratios we scan for companion phase-locks around the peak
# rhythm oscillator. Listed both "slower than peak" (1:2, 1:3) and
# "faster than peak" (2:1, 3:2, 4:3, 3:1). Multi-clock output comes for
# free — each companion is its own beat subdivision.
_RHYTHM_COMPANION_RATIOS = [
    (2, 1), (3, 1), (3, 2), (4, 3), (5, 4), (7, 4),
    (1, 2), (1, 3), (2, 3), (3, 4), (4, 5),
]


PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#",
                     "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler major and minor key profiles (1982). The 12-entry
# vectors are perceived importance of each pitch class given a tonic at
# index 0. Rotated during matching to hypothesize any of the 12 tonics.
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                     2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Chord templates — one-hot pitch-class masks for root-position chords.
# Keyed on short names; rendered as e.g. "Cmaj7" or "Am". The set is
# intentionally the common-jazz set, not exhaustive — extend as needed
# without breaking the matching loop (which rotates each template
# through 12 roots and picks argmax correlation).
#                         C  C# D  D# E  F  F# G  G# A  A# B
_CHORD_TEMPLATES: dict[str, np.ndarray] = {
    "maj":  np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]),
    "min":  np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]),
    "dom7": np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]),
    "maj7": np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]),
    "min7": np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]),
    "dim":  np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]),
    "sus4": np.array([1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]),
    "sus2": np.array([1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
}

# NRT stability hierarchy: small-integer ratios that produce phase-locked
# attractors in the GrFNN dynamics. Values taken from Large 2025 Fig. 3
# (stability ranking) and tuned so that unison/octave are the strongest.
# Tuple format: (p, q, stability_score) — p:q with score in (0, 1].
_CONSONANT_RATIOS = (
    (1, 1, 1.00),
    (2, 1, 0.92),
    (3, 2, 0.82),
    (4, 3, 0.72),
    (5, 4, 0.62),
    (5, 3, 0.55),
    (7, 4, 0.48),
    (8, 5, 0.40),
)

# Gaussian kernel width for integer-ratio matching. 0.8 semitones gives
# smooth interpolation near stable ratios and sharp falloff into
# dissonant regions (e.g., tritone at √2 is ~1 semitone from 3:2).
_CONSONANCE_SIGMA_SEMITONES = 0.8


def hz_to_pitch_class(hz: float) -> int:
    """Map a frequency to its nearest pitch class (0-11, C=0, A=9).

    Uses A4 = 440 Hz as the reference. Octave-equivalent: A3, A4, A5
    all return 9. Raises ValueError on non-positive input.
    """
    if hz <= 0:
        raise ValueError(f"hz must be positive, got {hz}")
    return int(round(12 * np.log2(hz / 440.0) + 9)) % 12


@dataclass(frozen=True)
class StateWindow:
    """A snapshot of oscillator state the extractors can chew on.

    Accepts either 1D (n_osc,) for an instantaneous slice or 2D
    (frames, n_osc) for a rolling window. The ``_2d`` properties
    normalize internally so extractors never branch on shape.

    Parameters
    ----------
    pitch_z, rhythm_z : complex state of the pitch and rhythm GrFNN.
    pitch_freqs, rhythm_freqs : natural frequencies of each oscillator (Hz).
    frame_hz : snapshot rate of the window (used by extractors that
        compute over time — tempo is derived from oscillator frequency
        directly, so frame_hz mostly matters for future extractors).
    w_pitch : optional learned Hebbian pitch weights, shape (n_p, n_p)
        complex. Diagonal is the tonal hierarchy. When None, key
        detection falls back to pitch-amplitude chroma.
    """

    pitch_z: np.ndarray
    pitch_freqs: np.ndarray
    rhythm_z: np.ndarray
    rhythm_freqs: np.ndarray
    frame_hz: float
    w_pitch: np.ndarray | None = None

    @property
    def pitch_z_2d(self) -> np.ndarray:
        return np.atleast_2d(self.pitch_z)

    @property
    def rhythm_z_2d(self) -> np.ndarray:
        return np.atleast_2d(self.rhythm_z)


def extract_tempo(window: StateWindow) -> tuple[float, float]:
    """Return (bpm, confidence).

    bpm is the windowed-mean rhythm amplitude's peak frequency × 60,
    refined with log-space quadratic peak interpolation so bpm doesn't
    jitter between adjacent log-spaced oscillators as the argmax wobbles.
    Confidence is the peak's prominence over the mean, clipped to [0, 1].
    """
    amps = np.abs(window.rhythm_z_2d).mean(axis=0)
    freqs = window.rhythm_freqs
    if amps.max() <= 0:
        return 0.0, 0.0
    peak_idx = int(np.argmax(amps))

    # Quadratic interpolation on log-amplitude around the argmax. Gives
    # sub-oscillator peak location and continuous bpm in the face of
    # small amplitude noise. Skip at array edges.
    if 0 < peak_idx < len(amps) - 1:
        a = float(amps[peak_idx - 1])
        b = float(amps[peak_idx])
        c = float(amps[peak_idx + 1])
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
        delta = max(-0.5, min(0.5, delta))
        log_freq = np.log(freqs)
        refined_log = log_freq[peak_idx] + delta * (
            log_freq[peak_idx + 1] - log_freq[peak_idx - 1]
        ) / 2
        peak_freq = float(np.exp(refined_log))
    else:
        peak_freq = float(freqs[peak_idx])

    bpm = peak_freq * 60.0
    confidence = float((amps.max() - amps.mean()) / amps.max())
    return bpm, max(0.0, min(1.0, confidence))


def _chroma_from_pitches(freqs: np.ndarray, amps: np.ndarray) -> np.ndarray:
    """Fold per-oscillator amplitudes into a 12-dim pitch-class vector."""
    chroma = np.zeros(12)
    for f, a in zip(freqs, amps):
        chroma[hz_to_pitch_class(float(f))] += float(a)
    return chroma


def extract_key(window: StateWindow) -> dict:
    """Detect tonic + mode via Krumhansl-Schmuckler template matching.

    Prefers the Hebbian pitch W diagonal (the learned tonal hierarchy)
    if available; falls back to mean |pitch_z| otherwise.

    Returns {"tonic": str, "mode": "major"|"minor", "confidence": float}.
    """
    if window.w_pitch is not None:
        amps = np.abs(np.diag(window.w_pitch))
    else:
        amps = np.abs(window.pitch_z_2d).mean(axis=0)

    chroma = _chroma_from_pitches(window.pitch_freqs, amps)
    total = chroma.sum()
    if total == 0:
        return {"tonic": "C", "mode": "major", "confidence": 0.0}

    best_score = -np.inf
    best_tonic = 0
    best_mode = "major"
    scores: list[float] = []
    for mode_name, profile in (("major", KS_MAJOR), ("minor", KS_MINOR)):
        for rot in range(12):
            rotated = np.roll(profile, rot)
            s = float(np.corrcoef(chroma, rotated)[0, 1])
            scores.append(s)
            if s > best_score:
                best_score = s
                best_tonic = rot
                best_mode = mode_name

    # Confidence: margin of winner over median correlation, mapped into
    # [0, 1]. A correlation margin of 0.5 or greater is effectively
    # full confidence; the template is unambiguous.
    margin = best_score - float(np.median(scores))
    confidence = float(max(0.0, min(1.0, margin / 0.5)))

    return {
        "tonic": PITCH_CLASS_NAMES[best_tonic],
        "mode": best_mode,
        "confidence": confidence,
    }


def _ratio_consonance(ratio: float) -> float:
    """Return the best-matching stability score for a frequency ratio.

    Iterates the NRT stability hierarchy, takes the Gaussian-weighted
    maximum over all ratios in log-frequency space.
    """
    log_ratio = np.log2(ratio)
    two_sigma_sq = 2 * _CONSONANCE_SIGMA_SEMITONES ** 2
    best = 0.0
    for p, q, stability in _CONSONANT_RATIOS:
        target_log = np.log2(p / q)
        dist_semitones = abs(log_ratio - target_log) * 12
        score = stability * np.exp(-(dist_semitones ** 2) / two_sigma_sq)
        if score > best:
            best = score
    return float(best)


def extract_chord(window: StateWindow) -> dict:
    """Detect the currently-active chord via pitch-class template matching.

    Uses the instantaneous pitch amplitudes (averaged over the window)
    — NOT the learned Hebbian W — because chord is "what's sounding
    now," whereas W captures the long-term tonal hierarchy (which is
    what ``extract_key`` is for).

    Returns ``{"root": str, "quality": str, "name": str,
    "confidence": float}``. Quality is one of the template keys (maj,
    min, dom7, …). Name is the rendered chord symbol, e.g. "Am7",
    "Fmaj7".
    """
    amps = np.abs(window.pitch_z_2d).mean(axis=0)
    chroma = _chroma_from_pitches(window.pitch_freqs, amps)
    total = chroma.sum()
    if total == 0:
        return {"root": "C", "quality": "maj", "name": "C",
                "confidence": 0.0}

    best_score = -np.inf
    best_root = 0
    best_quality = "maj"
    scores: list[float] = []
    for quality, template in _CHORD_TEMPLATES.items():
        for rot in range(12):
            rotated = np.roll(template, rot)
            s = float(np.corrcoef(chroma, rotated)[0, 1])
            scores.append(s)
            if s > best_score:
                best_score = s
                best_root = rot
                best_quality = quality

    root = PITCH_CLASS_NAMES[best_root]
    # Short chord-symbol renderer: maj → "" (just root), min → "m",
    # dom7 → "7", maj7 → "maj7", min7 → "m7", dim → "°", sus4/sus2 keep
    # their names.
    suffix = {"maj": "", "min": "m", "dom7": "7", "maj7": "maj7",
              "min7": "m7", "dim": "°", "sus4": "sus4",
              "sus2": "sus2"}[best_quality]
    name = f"{root}{suffix}"
    margin = best_score - float(np.median(scores))
    confidence = float(max(0.0, min(1.0, margin / 0.5)))
    return {
        "root": root,
        "quality": best_quality,
        "name": name,
        "confidence": confidence,
    }


def extract_rhythm_structure(
    window: StateWindow,
    *,
    prev_peak_idx: int | None = None,
    persistence_threshold: float = 0.8,
    companion_plv_threshold: float = 0.7,
    companion_amp_threshold: float = 0.3,
    tempo_prior_center_hz: float = 2.0,
    tempo_prior_sigma_log2: float = 1.0,
) -> dict:
    """Extract peak beat + phase-locked companions from rhythm state.

    This is the multi-clock primitive. The peak oscillator gives the
    main beat; companions at integer ratios give subdivisions and
    polyrhythmic structure (triplets over quarters, half-time, etc.).
    Consumers (router, viewer) turn phases into triggers.

    Returns::

        {
          "peak": {"idx", "freq", "bpm", "phase"},
          "companions": [{"ratio_p", "ratio_q", "idx", "freq",
                          "phase", "plv"}, ...],
        }

    ``phase`` is in radians, (-π, π], sampled at the latest frame in
    the window.

    **Peak persistence.** If ``prev_peak_idx`` is provided and its
    amplitude is ≥ ``persistence_threshold`` × max, we stick with it.
    This kills the argmax-wobble jitter where adjacent log-spaced
    oscillators fight for the peak. The caller maintains persistence
    state across frames — the extractor stays pure.
    """
    rz_2d = window.rhythm_z_2d
    amps = np.abs(rz_2d).mean(axis=0)
    freqs = window.rhythm_freqs
    if amps.max() <= 0:
        return {
            "peak": {"idx": 0, "freq": 0.0, "bpm": 0.0, "phase": 0.0},
            "companions": [],
        }

    # Soft tempo prior: Gaussian in log-frequency centered on the
    # musical-tempo sweet spot (~120 BPM = 2 Hz). Biases initial peak
    # selection toward the fundamental pulse without hard-gating, so
    # genuinely slow or fast tempos still win when evidence is clear.
    # Persistence overrides the prior — once locked to a peak, a
    # strong enough amplitude keeps it there.
    log_freqs = np.log2(np.clip(freqs, 1e-9, None))
    prior = np.exp(
        -0.5 * ((log_freqs - np.log2(tempo_prior_center_hz))
                / tempo_prior_sigma_log2) ** 2
    )
    biased_amps = amps * prior

    if (prev_peak_idx is not None
        and 0 <= prev_peak_idx < len(amps)
        and amps[prev_peak_idx] >= persistence_threshold * amps.max()):
        peak_idx = int(prev_peak_idx)
    else:
        peak_idx = int(np.argmax(biased_amps))

    peak_freq = float(freqs[peak_idx])
    peak_phase = float(np.angle(rz_2d[-1, peak_idx]))

    # Companion locks — for each target ratio, find the oscillator whose
    # natural frequency is closest to peak_freq × (p/q) and check PLV.
    companions: list[dict] = []
    if rz_2d.shape[0] >= 4:
        peak_phase_hist = np.angle(rz_2d[:, peak_idx])
        for ratio_p, ratio_q in _RHYTHM_COMPANION_RATIOS:
            target_freq = peak_freq * (ratio_p / ratio_q)
            if target_freq < freqs[0] or target_freq > freqs[-1]:
                continue
            comp_idx = int(np.argmin(np.abs(freqs - target_freq)))
            if comp_idx == peak_idx:
                continue
            comp_phase_hist = np.angle(rz_2d[:, comp_idx])
            # We chose comp so that f_comp = f_peak * (p/q), i.e.
            # f_peak:f_comp = q:p. Under stable lock that satisfies
            # p * phi_peak - q * phi_comp = const, giving PLV → 1.
            rel = ratio_p * peak_phase_hist - ratio_q * comp_phase_hist
            plv = float(np.abs(np.mean(np.exp(1j * rel))))
            if plv >= companion_plv_threshold \
                    and amps[comp_idx] >= companion_amp_threshold * amps.max():
                companions.append({
                    "ratio_p": ratio_p,
                    "ratio_q": ratio_q,
                    "idx": comp_idx,
                    "freq": float(freqs[comp_idx]),
                    "phase": float(np.angle(rz_2d[-1, comp_idx])),
                    "plv": plv,
                })

    return {
        "peak": {
            "idx": peak_idx,
            "freq": peak_freq,
            "bpm": peak_freq * 60.0,
            "phase": peak_phase,
        },
        "companions": companions,
    }


def extract_consonance(window: StateWindow) -> float:
    """Amplitude-weighted mean consonance of active pitch-oscillator pairs.

    For each pair of oscillators above a noise threshold, score their
    frequency ratio against the NRT stability hierarchy and weight by
    the product of their amplitudes. The result is a scalar in [0, 1]:
    near 1 when the active oscillators cluster around small-integer
    ratios (octaves, fifths), near 0 when they don't (tritones, random
    clusters).
    """
    amps = np.abs(window.pitch_z_2d).mean(axis=0)
    freqs = window.pitch_freqs
    if amps.max() <= 0:
        return 0.0

    active_idx = np.where(amps > 0.1 * amps.max())[0]
    if len(active_idx) < 2:
        return 0.0

    total_weight = 0.0
    weighted = 0.0
    for ii in range(len(active_idx)):
        for jj in range(ii + 1, len(active_idx)):
            i, j = int(active_idx[ii]), int(active_idx[jj])
            f_hi = max(freqs[i], freqs[j])
            f_lo = min(freqs[i], freqs[j])
            if f_lo <= 0:
                continue
            stability = _ratio_consonance(float(f_hi / f_lo))
            weight = float(amps[i] * amps[j])
            weighted += weight * stability
            total_weight += weight

    if total_weight == 0:
        return 0.0
    return float(weighted / total_weight)
