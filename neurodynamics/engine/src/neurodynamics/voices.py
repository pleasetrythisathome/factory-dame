"""Voice identity extraction from NRT pitch oscillator state.

A voice is a phase-coherent, amplitude-envelope-correlated cluster of
pitch oscillators that moves together as a single musical entity. A
voice may span multiple frequencies (fundamental + harmonics) and may
evolve through splits, merges, and silences over the course of a piece.

This module is the first layer of Phase 1 of the modular-bridge
product vision (``task-011``, ``Narrative_LiveNRTModularBridge``). It
decomposes the NRT pitch GrFNN state into a dynamic set of voices,
each with a stable identity tracked across frames.

Core pipeline per frame:

    1. Filter to active oscillators (above noise floor and a fraction
       of the current peak), then keep only local maxima — the cascade
       fills the bank wall-to-wall, so peak detection recovers the
       handful of meaningful candidates.
    2. Build an adjacency graph. Two oscillators share a voice if ANY
       of three signals fire:
         • proximity — within a fraction of a semitone (the same
           spectral peak);
         • learned overtone family — one is an integer multiple
           (2×…8×) of the other AND the pair is "bound", where bound
           means the Hebbian W learned a strong connection (Kim &
           Large 2021 Eq. 26) OR the per-window envelopes co-modulate.
           W is the time-integrated same-source evidence that survives
           the envelope decoupling of cascade-generated harmonics;
         • frequency-local co-modulation — co-moving bins within one
           peak's neighbourhood (a source's spectral spread), which
           chains contiguous bins but breaks at the gaps between
           distinct pitches.
    3. Connected components → clusters; a second pass merges
       cross-component overtone families bound by W.
    4. Each cluster's pitch is its LOUDEST partial (the perceptual
       fundamental), refined by sub-bin interpolation.
    5. Match clusters to previous frame's voices via Hungarian
       assignment on log-frequency distance, amplitude change, and
       oscillator-index Jaccard distance; matched voices smooth their
       frequency toward the previous value to kill jitter.
    6. Unmatched new clusters → new voice IDs. Unmatched prev voices
       → marked silent; retired after ``max_silent_frames`` frames.

Deliberate design choices:

- **Dynamic voice count.** Never fixed N. Music has 1 voice in a solo
  piano passage and 8 voices in dense jazz; the extractor follows.
- **Cluster on the learned W, not just instantaneous correlation.**
  The pitch GrFNN's Hebbian connectivity encodes which oscillators are
  harmonically bound from the whole piece's coactivity — exactly the
  signal needed to merge a cascade-enriched harmonic series back into
  one voice. An integer ratio alone is necessary but not sufficient
  (it would collapse independent sources that coincide at a ratio);
  the learned bond confirms same-source.
- **Co-modulation only confirms, never establishes.** Everything in a
  mix co-modulates with the beat, so co-modulation merges only within
  a spectral neighbourhood (one source's spread) or as confirmation of
  an overtone relationship — never across distinct pitches. This is
  what keeps a chord struck on one envelope resolved into its notes.
- **Pitch = loudest partial.** Anchoring a voice to its dominant peak
  (not the lowest member) keeps it on the true pitch; the critical-Hopf
  cascade conjures phantom subharmonics that would otherwise drag the
  reported pitch an octave low.
- **Persistence across silence.** A bassline that drops out for a bar
  and returns should keep its ID. Silent voices are held for
  ``max_silent_frames`` frames before retirement.

Extensibility note: Phase 2 (``task-011`` plan) will add per-voice
rhythm association; Phase 3 will add per-voice motor coupling. This
module exposes ``VoiceIdentity`` as the primitive those phases extend.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components

from .perceptual import StateWindow

__all__ = [
    "VoiceIdentity",
    "VoiceRhythm",
    "VoiceMotor",
    "VoiceState",
    "VoiceClusteringConfig",
    "extract_voices",
    "extract_voice_rhythms",
    "extract_voice_motor",
]


# Small-integer ratios used to boost correlation for harmonically
# related oscillator pairs. These are the same ratios the NRT network
# naturally mode-locks on, so picking them up as "same voice" is
# consistent with the dynamics.
_HARMONIC_RATIOS: tuple[tuple[int, int], ...] = (
    (1, 1),     # near-unison: bandwidth spread of one source's peak
    (1, 2), (2, 1),
    (1, 3), (3, 1),
    (2, 3), (3, 2),
    (1, 4), (4, 1),
    (3, 4), (4, 3),
    (1, 5), (5, 1),
    (4, 5), (5, 4),
    # Higher-order integer ratios for cascaded multi-layer mode:
    # the brainstem layer generates harmonics up to ~8× via the
    # canonical input + coupling kernel, so voice extraction needs
    # to recognize 6:1, 7:1, 8:1 to merge them as harmonics of the
    # same source. Per Kim & Large 2019 Eq. 14 these have narrow
    # Arnold tongues so they're physically meaningful only at
    # strong drive — exactly the cascade regime.
    (1, 6), (6, 1),
    (1, 7), (7, 1),
    (1, 8), (8, 1),
    (5, 6), (6, 5),
    (3, 5), (5, 3),
    (2, 5), (5, 2),
    (3, 7), (7, 3),
    (3, 8), (8, 3),
)


@dataclass(frozen=True)
class VoiceRhythm:
    """Rhythm association for a single voice — the rhythm oscillator
    whose natural frequency best matches the voice's amplitude
    envelope, with that oscillator's phase at the latest frame.

    This is the Phase 2 primitive: once a voice is identified, find
    the rhythm GrFNN oscillator that phase-locks to its onset
    pattern. Each voice gets its own tempo, subdivision, and beat
    phase — which is what makes the per-voice modular clock use case
    (bass at half-time while hi-hats run at double) possible.
    """

    osc_idx: int          # index into the rhythm GrFNN oscillator bank
    freq: float           # Hz — natural frequency of that oscillator
    bpm: float            # freq × 60 for convenience
    phase: float          # radians, (-π, π] — phase of the rhythm
                          # oscillator at the latest frame in the window
    confidence: float     # 0-1 — how dominant the matched frequency
                          # is in the voice's envelope spectrum


@dataclass(frozen=True)
class VoiceMotor:
    """Motor association for a single voice — the motor-GrFNN
    oscillator whose natural frequency best matches the voice's
    amplitude envelope.

    Phase 3 primitive. Structurally symmetric to ``VoiceRhythm``
    but semantically different: motor oscillators carry forward-
    predictive state through their bidirectional coupling with the
    sensory rhythm network. A voice's motor phase is its
    *anticipated* next beat — the position where the felt pulse
    will land — rather than the current sensory beat. For the
    modular use case this becomes a per-voice "beat prediction" CV
    that keeps ticking even when the voice goes silent, because
    motor oscillators sustain briefly after the sensory drive
    drops (see test_two_layer_pulse).
    """

    osc_idx: int          # index into the motor GrFNN oscillator bank
    freq: float           # Hz — natural frequency of that oscillator
    bpm: float            # freq × 60 for convenience
    phase: float          # radians, (-π, π] — anticipated-beat phase
    confidence: float     # 0-1 — DFT peak dominance in voice envelope
                          # spectrum against the motor bank


@dataclass(frozen=True)
class VoiceIdentity:
    """A single voice cluster with persistent identity across frames.

    All fields are scalars or immutable tuples so instances are safe
    to hash and to reuse between frames. Construction path is
    exclusively through ``extract_voices`` — consumers don't build
    these directly.

    ``rhythm`` is populated by a separate call to
    ``extract_voice_rhythms`` (Phase 2) and is ``None`` otherwise so
    the Phase 1 extractor stays orthogonal.
    """

    id: int
    oscillator_indices: tuple[int, ...]
    center_freq: float              # amplitude-weighted geometric mean (Hz)
    amp: float                      # mean amplitude across the cluster
    phase_centroid: float           # weighted circular mean, (-π, π]
    active: bool
    confidence: float               # 0-1, within-cluster envelope coherence
    age_frames: int                 # frames since first appearance
    silent_frames: int              # consecutive frames with active=False
    rhythm: VoiceRhythm | None = None
    motor: VoiceMotor | None = None


@dataclass
class VoiceState:
    """Rolling state threaded through successive ``extract_voices``
    calls. Carries tracked voices (active + recently silent) and the
    monotonic ID allocator so IDs never collide."""

    voices: list[VoiceIdentity] = field(default_factory=list)
    next_id: int = 0

    @property
    def active_voices(self) -> list[VoiceIdentity]:
        return [v for v in self.voices if v.active]


@dataclass(frozen=True)
class VoiceClusteringConfig:
    """Tunable parameters for voice clustering.

    Defaults assume the cleaned-up pitch GrFNN (noise.amp=0, input_gain=0.5)
    where silent baseline is ~0.003 and real signals drive the bank to
    0.01-0.15 depending on intensity. Gate at 0.005 keeps silence out while
    admitting clean synth content; the ``active_fraction`` term kicks in
    for stronger signals so only the dominant bins per frame count."""

    # Oscillator activity gating. A bin is active if its mean window
    # amplitude is above BOTH the absolute floor and a fraction of the
    # per-frame peak.
    noise_floor: float = 0.005
    active_fraction: float = 0.25   # fraction of peak amplitude to count as active

    # Correlation clustering. Two thresholds: the strict one for
    # unrelated pairs (prevents two instruments from colliding), and
    # a permissive one for harmonic pairs (fundamental + harmonics
    # belong together unless clearly anti-correlated).
    correlation_threshold: float = 0.6
    harmonic_correlation_threshold: float = 0.5
    harmonic_boost: float = 0.15    # retained for backward compatibility; unused in new logic
    harmonic_ratio_tolerance_semitones: float = 0.5
    # Frequency-proximity merging: pairs within this many semitones
    # are bandwidth-spread of a single peak — always cluster them
    # regardless of correlation or harmonic ratio. 0.5 semitones
    # admits IMMEDIATELY-ADJACENT bins (1 step at 36 bins/octave
    # spacing = 0.33 semitones) without chaining far through the
    # bank via transitive proximity, which on dense cascade-enriched
    # banks would collapse the whole bank into one component.
    proximity_merge_semitones: float = 0.5
    # Frequency-local co-modulation: two oscillators whose envelopes
    # co-modulate AND lie within this many semitones merge as the
    # spectral spread of one source. Wider than proximity (which is
    # correlation-free, for saturated sub-semitone neighbours), this
    # chains a CONTIGUOUS run of co-moving bins — one note's spectral
    # peak may be a few bins wide — into a single voice via transitive
    # closure, while the inactive gaps between distinct pitches (chord
    # notes, octaves) break the chain. Kept local on purpose:
    # everything in a mix co-modulates with the beat, so global
    # co-modulation would collapse the whole spectrum.
    comod_locality_semitones: float = 1.0

    # Peak detection within the active set: keep only local maxima
    # over a ±peak_radius_bins window. Cascade-enabled pipelines
    # produce dense activity; without peak detection, EVERY bin is
    # "active" and the cluster count balloons. At 36 bins/octave,
    # radius=3 corresponds to ±1 semitone — peaks separated by ≥1
    # semitone are distinct. Disable to revert to threshold-only
    # active filtering (legacy behavior, ok for sparse synth tests).
    peak_detection_enabled: bool = True
    # Half-width of the peak-detection window in oscillator bins.
    # At 36 bins/octave (the default pitch bank density), radius=3
    # ≈ 1 semitone — keeps adjacent-pitch resolution while collapsing
    # bandwidth tails of a mode-lock response to its center.
    peak_radius_bins: int = 3
    # Bins within `peak_shoulder_ratio` of the local maximum pass
    # peak detection (captures unison clusters + the shoulder of a
    # mode-lock response, not just the strict peak). 0.9 → admit
    # bins at ≥ 90 % of the local-window max.
    peak_shoulder_ratio: float = 0.9

    # Cluster filtering
    min_cluster_size: int = 1       # singletons allowed — a lone pitch is a voice
    min_sustain_frames: int = 1     # min window-frames of activity before counting

    # Voice tracking
    max_silent_frames: int = 40     # retire a silent voice after this many frames
    match_cost_cap: float = 5.0     # pair costs above this → no match, force new ID
    match_weight_logfreq: float = 1.0  # per-semitone penalty
    match_weight_amp: float = 0.3      # per-unit-amp penalty
    match_weight_jaccard: float = 2.0  # per-unit-dissimilarity penalty

    # ── Learned-W harmonic affinity (Phase 4.3) ──
    # The pitch GrFNN's Hebbian connectivity W (Kim & Large 2021 Eq. 26)
    # learns strong connections between oscillators that mode-lock at
    # integer ratios — i.e. the harmonics of a common source. Using W
    # as the harmonic-binding signal is strictly better than an
    # instantaneous envelope correlation: it's the time-integrated
    # evidence that the network treats two oscillators as the same
    # voice, so cascade-generated harmonics (whose per-window envelopes
    # decouple from the fundamental) still merge. A pair merges as a
    # harmonic family member only when it is BOTH near an integer ratio
    # AND learned-bound; the integer ratio alone is not sufficient
    # (that would collapse independent sources that happen to sit at a
    # small-integer ratio), and the learned bond alone is not (W also
    # encodes consonant-interval structure across distinct notes).
    use_learned_affinity: bool = True
    # "Strong" W = at/above this percentile of the global off-diagonal
    # |W| distribution, gated by the overtone mask so only harmonic
    # pairs in that top band merge. 90 was chosen against the corpus:
    # it lands dense electronic tracks at ~3-4 simultaneous voices (the
    # musically-plausible range, matching the pre-cascade baseline)
    # while keeping the cascade on. Lower over-merges harmonics into a
    # blob; higher lets cascade harmonics fragment back into separate
    # voices.
    w_affinity_percentile: float = 90.0

    # ── Fundamental-frequency stability ──
    # A tracked voice's center frequency is smoothed toward its matched
    # previous value (EMA in log-frequency) to kill the frame-to-frame
    # jitter that made voices visually "jump around". Only applied when
    # the new estimate is within freq_snap_semitones of the previous
    # one — a genuine pitch change (glissando, new note on the same ID)
    # still moves freely. 0.0 disables smoothing.
    freq_smoothing: float = 0.6        # weight on the previous frequency
    freq_snap_semitones: float = 2.0   # only smooth within this distance


def _log_freq(hz: float) -> float:
    return float(np.log(max(hz, 1e-9)))


def _circular_mean(phases: np.ndarray, weights: np.ndarray) -> float:
    """Amplitude-weighted circular mean of phases."""
    if weights.sum() <= 0:
        return 0.0
    c = float(np.sum(weights * np.cos(phases)))
    s = float(np.sum(weights * np.sin(phases)))
    return float(np.arctan2(s, c))


# Precomputed log2 of all positive small-integer ratios (p ≥ q so
# log_ratio ≥ 0). Used by both the scalar fast path and the vectorized
# pairwise routine below.
_HARMONIC_TARGET_LOGS = np.array(
    sorted({
        np.log2(p / q)
        for p, q in _HARMONIC_RATIOS
        if q > 0 and p >= q
    }),
    dtype=np.float64,
)


# Overtone ratios = integer MULTIPLES of a fundamental (2:1, 3:1, …).
# These are the unambiguous "my own harmonics" relationships: f_j is the
# n-th overtone of f_i. Distinct from the broad _HARMONIC_RATIOS set,
# which also contains chord intervals (3:2, 4:3, 5:4) that relate
# DIFFERENT notes — merging on those collapses separate voices. The
# learned-W harmonic merge keys on overtones only; the broad set is
# reserved for the weaker envelope-correlation fallback.
_OVERTONE_TARGET_LOGS = np.log2(np.arange(2, 9, dtype=np.float64))  # 2..8


def _overtone_matrix(
    freqs: np.ndarray, tolerance_semitones: float,
) -> np.ndarray:
    """(n, n) bool: entry [i, j] True iff one frequency is within
    tolerance of an integer multiple (2×…8×) of the other."""
    n = len(freqs)
    if n == 0:
        return np.zeros((0, 0), dtype=bool)
    log_f = np.log2(np.maximum(freqs.astype(np.float64), 1e-9))
    log_ratio = np.abs(log_f[:, None] - log_f[None, :])
    dists = np.abs(log_ratio[..., None] - _OVERTONE_TARGET_LOGS[None, None, :])
    return dists.min(axis=-1) * 12.0 <= tolerance_semitones


def _pair_is_overtone(
    f_i: float, f_j: float, tolerance_semitones: float,
) -> bool:
    """Scalar: True if f_i:f_j is near an integer multiple (2×…8×)."""
    if f_i <= 0 or f_j <= 0:
        return False
    ratio = max(f_i, f_j) / min(f_i, f_j)
    dists = np.abs(_OVERTONE_TARGET_LOGS - np.log2(ratio)) * 12.0
    return bool(dists.min() <= tolerance_semitones)


def _pair_is_harmonic(
    f_i: float,
    f_j: float,
    tolerance_semitones: float,
) -> bool:
    """True if f_i : f_j is near a small-integer ratio.

    Scalar fallback used by post-cluster paths (small N); the hot
    pairwise loop in extract_voices uses ``_pairwise_harmonic_matrix``
    below for a ~100× speedup.
    """
    if f_i <= 0 or f_j <= 0:
        return False
    ratio = max(f_i, f_j) / min(f_i, f_j)
    log_ratio = np.log2(ratio)
    dists = np.abs(_HARMONIC_TARGET_LOGS - log_ratio) * 12.0
    return bool(dists.min() <= tolerance_semitones)


def _pairwise_harmonic_matrix(
    freqs: np.ndarray, tolerance_semitones: float,
) -> np.ndarray:
    """Vectorized version of `_pair_is_harmonic` for all pairs.

    Returns (n, n) bool matrix where entry [i, j] is True iff the
    frequency pair freqs[i], freqs[j] is within `tolerance_semitones`
    of any small-integer ratio in _HARMONIC_RATIOS.

    Replaces a double Python loop calling `_pair_is_harmonic` per
    pair — the dominant hot-path cost in extract_voices (58 % of
    profile time at the active counts a cascaded engine produces).
    """
    n = len(freqs)
    if n == 0:
        return np.zeros((0, 0), dtype=bool)
    log_f = np.log2(np.maximum(freqs.astype(np.float64), 1e-9))
    # |log_i - log_j| → (n, n) log-ratio magnitude
    log_ratio = np.abs(log_f[:, None] - log_f[None, :])
    # For each pair, distance to nearest target (in log-octave space).
    # Broadcasting: (n, n, 1) vs (T,) → (n, n, T) → min over last axis.
    dists = np.abs(log_ratio[..., None] - _HARMONIC_TARGET_LOGS[None, None, :])
    min_dist_semitones = dists.min(axis=-1) * 12.0
    return min_dist_semitones <= tolerance_semitones


def _learned_strong_matrix(
    active_indices: np.ndarray,
    w_pitch: np.ndarray,
    percentile: float,
) -> tuple[np.ndarray, float]:
    """Return (n_active, n_active) bool matrix of learned-strong pairs.

    Entry [a, b] is True iff the magnitude of the learned Hebbian
    connection between the two active oscillators is at or above the
    given percentile of the GLOBAL off-diagonal |W| distribution — i.e.
    this pair is among the connections the network actually learned to
    bind, not background coactivity. Per Kim & Large 2021 Eq. 26 those
    learned connections concentrate on integer-ratio (harmonic) pairs,
    so AND-ing this with the integer-ratio mask isolates same-source
    harmonic families.

    Returns (matrix, threshold). The threshold is also returned for
    diagnostics. Out-of-bounds indices (W smaller than the bank) yield
    False rows/cols rather than raising.
    """
    n_active = len(active_indices)
    out = np.zeros((n_active, n_active), dtype=bool)
    if w_pitch is None or n_active == 0:
        return out, 0.0
    Wmag = np.abs(w_pitch)
    n_w = Wmag.shape[0]
    off = ~np.eye(n_w, dtype=bool)
    thresh = float(np.percentile(Wmag[off], percentile)) if n_w > 1 else 0.0
    # Strict ">" (not ">="): an unlearned W (all ~equal, e.g. early in a
    # live session before Hebbian learning kicks in) has its percentile
    # EQUAL to its values, so ">=" would mark every pair "strong" and
    # over-merge. With ">", a uniform W yields no learned bonds — the
    # extractor falls back to co-modulation until real structure forms —
    # while a partially-learned W still selects its few strong edges.
    # Map active indices into W; ignore any that fall outside W.
    in_bounds = active_indices < n_w
    valid_local = np.where(in_bounds)[0]
    valid_global = active_indices[in_bounds]
    if len(valid_global) > 1:
        sub = Wmag[np.ix_(valid_global, valid_global)] > thresh
        # scatter back into the full active-index frame
        ii, jj = np.meshgrid(valid_local, valid_local, indexing="ij")
        out[ii, jj] = sub
    return out, thresh


def _component_plv_at_ratio(
    phases_i: np.ndarray, phases_j: np.ndarray,
    f_i: float, f_j: float,
    tolerance_semitones: float,
) -> tuple[float, int, int] | None:
    """Compute phase-locking value between two components at the
    closest small-integer ratio to their frequency ratio.

    Returns (plv, p, q) where p:q is the chosen ratio, or None if
    no small-integer ratio approximates f_j/f_i within tolerance.

    The PLV is computed on the OSCILLATOR phase trajectories per
    K&L 2021 Eq. 14 — same-source harmonics have stable q*phi_a -
    p*phi_b under mode-lock, while independent sources at the same
    f-ratio have drifting relative phase.
    """
    from .modelock import phase_locking_value
    if f_i <= 0 or f_j <= 0:
        return None
    f_lo, f_hi = min(f_i, f_j), max(f_i, f_j)
    ratio = f_hi / f_lo
    log_ratio = np.log2(ratio)
    best_p, best_q = 1, 1
    best_dist = float("inf")
    for p, q in _HARMONIC_RATIOS:
        if p < q:
            continue   # only consider p >= q for the >=1 ratio direction
        target_log = np.log2(p / q)
        d = abs(log_ratio - target_log) * 12  # semitones
        if d < best_dist:
            best_dist = d
            best_p, best_q = p, q
    if best_dist > tolerance_semitones:
        return None
    # Phase A is the higher-frequency oscillator (p cycles per q of B).
    if f_i >= f_j:
        plv = phase_locking_value(phases_i, phases_j, best_p, best_q)
    else:
        plv = phase_locking_value(phases_j, phases_i, best_p, best_q)
    return plv, best_p, best_q


def _merge_harmonic_components(
    components: list[list[int]],
    active_indices: np.ndarray,
    amps_over_window: np.ndarray,
    phases_over_window: np.ndarray,
    freqs: np.ndarray,
    tolerance_semitones: float,
    min_amp_ratio: float = 0.05,
    min_envelope_correlation: float = 0.5,
    min_plv_for_merge: float = 0.7,
    w_pitch: np.ndarray | None = None,
    w_thresh: float = 0.0,
) -> list[list[int]]:
    """Merge components at integer-ratio frequencies WHEN their
    aggregated envelopes correlate.

    Without the envelope-correlation gate, two genuinely independent
    sources that happen to lie at integer ratios (e.g., a 220 Hz
    voice and an 880 Hz hi-hat at 4:1) would incorrectly collapse
    into one voice. With the gate, only harmonics of the SAME
    source — which share their amplitude envelope — merge; harmonics
    of different sources stay separate.

    Flat-envelope special case: when both components have nearly
    constant amplitude (no variation to correlate), accept the merge.
    Constant tones at integer ratios are nearly always harmonics of
    the same source — different sources at integer ratios would have
    detectable envelope differences over typical voice-extraction
    windows.

    Per Kim & Large 2019 Eq. 14, mode-locked pairs at integer ratios
    ARE physically meaningful, but only when same-source coactivity
    is established by envelope coherence.

    Union-find for transitive closure: A↔B harmonic, B↔C harmonic
    → all three end up in one merged voice.
    """
    if len(components) < 2:
        return components

    comp_envs: list[np.ndarray] = []
    rep_freqs: list[float] = []
    rep_amps: list[float] = []
    rep_phases: list[np.ndarray] = []   # peak-osc phase trajectory
    rep_global: list[int] = []          # peak-osc global index (for W lookup)
    for comp in components:
        comp_global = [int(active_indices[i]) for i in comp]
        comp_amps = amps_over_window[:, comp_global].mean(axis=0)
        peak_local = int(comp_amps.argmax())
        peak_global = comp_global[peak_local]
        rep_freqs.append(float(freqs[peak_global]))
        rep_amps.append(float(comp_amps.max()))
        comp_envs.append(amps_over_window[:, comp_global].sum(axis=1))
        rep_phases.append(phases_over_window[:, peak_global])
        rep_global.append(peak_global)

    n_w = w_pitch.shape[0] if w_pitch is not None else 0

    parent = list(range(len(components)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    max_amp = max(rep_amps) if rep_amps else 1.0
    for i in range(len(components)):
        if rep_amps[i] < min_amp_ratio * max_amp:
            continue
        for j in range(i + 1, len(components)):
            if rep_amps[j] < min_amp_ratio * max_amp:
                continue
            if not _pair_is_overtone(rep_freqs[i], rep_freqs[j],
                                     tolerance_semitones):
                continue
            # ── Learned-bond fast path ──
            # OVERTONE representatives (n:1) the Hebbian W bound strongly
            # are the same source's own harmonics — the network's
            # time-integrated evidence outweighs any single window's
            # envelope, so merge without the envelope/PLV gates below
            # (which are the fallback for the no-/weak-W case). Keyed on
            # overtones only: two distinct notes at a broad ratio (3:2)
            # must NOT collapse just because W learned their consonance.
            if w_pitch is not None and _pair_is_overtone(
                    rep_freqs[i], rep_freqs[j], tolerance_semitones):
                gi, gj = rep_global[i], rep_global[j]
                if (gi < n_w and gj < n_w
                        and abs(w_pitch[gi, gj]) > w_thresh):
                    union(i, j)
                    continue
            # ── Envelope-correlation veto (anti-merge evidence) ──
            # Two clusters with clearly ANTI-correlated envelopes
            # are different sources, regardless of how their carrier
            # phases happen to line up (synthetic pure sinusoids at
            # exact integer ratios have mathematically locked PLV
            # even though they're conceptually independent — only
            # real-world phase drift breaks PLV for those cases).
            # Check envelopes first; if they disagree strongly,
            # short-circuit out without invoking PLV.
            env_i = comp_envs[i]
            env_j = comp_envs[j]
            std_i = float(env_i.std())
            std_j = float(env_j.std())
            corr: float | None = None
            if std_i > 1e-6 and std_j > 1e-6:
                corr = float(np.corrcoef(env_i, env_j)[0, 1])
                if np.isnan(corr):
                    corr = None
            if corr is not None and corr < 0.0:
                # Anti-correlated envelopes = different sources.
                continue
            # ── PLV gate (Phase 4.1): primary same-source evidence ──
            # Per Kim & Large 2021 Eq. 14, same-source harmonics
            # have stable q*phi_a - p*phi_b (mode-locked); independent
            # sources at non-integer-ratio frequencies have drifting
            # relative phase. High PLV + non-anti envelope correlation
            # → merge as same source.
            plv_info = _component_plv_at_ratio(
                rep_phases[i], rep_phases[j],
                rep_freqs[i], rep_freqs[j],
                tolerance_semitones,
            )
            if plv_info is not None:
                plv, _, _ = plv_info
                if plv >= min_plv_for_merge:
                    # Don't merge if envelopes are uncorrelated even
                    # when PLV is high — common case for synthetic
                    # sinusoids at exact integer ratios.
                    if corr is not None and corr < min_envelope_correlation:
                        continue
                    union(i, j)
                    continue
            # Envelope-correlation fallback for cases where PLV is
            # ambiguous (mid-range PLV, off-integer-ratio pairs).
            if std_i < 1e-6 and std_j < 1e-6:
                # Both constant — without PLV evidence, stay conservative.
                continue
            if std_i < 1e-6 or std_j < 1e-6:
                continue
            if corr is not None and corr >= min_envelope_correlation:
                union(i, j)

    # Pass 2 — harmonic-child collapse. After the same-source merge
    # above, scan remaining root components for "child" clusters: a
    # cluster at an integer ratio of a louder cluster is likely its
    # harmonic (per the typical 1/n amplitude falloff of natural
    # harmonic series). Two clusters at integer ratios with similar
    # amplitudes are more likely musical chord intervals (each note
    # carries its own fundamental energy) and should stay distinct.
    #
    # Discriminator: amplitude ratio. If A is louder than B AND they
    # are at integer ratio AND f_A < f_B (A is the lower fundamental
    # candidate), merge B into A. Use a strict amp-ratio cutoff to
    # avoid merging equal-loudness chord intervals.
    min_amp_dominance = 1.3
    # Require POSITIVE envelope correlation: real-music harmonics
    # share their fundamental's envelope (rising-falling together),
    # while independent sources at integer ratios have ~zero or
    # negative correlation over typical windows. 0.2 is loose enough
    # to admit harmonics with natural vibrato/attack-time variation,
    # but blocks orthogonal-envelope pairs (different tempos →
    # corr ≈ 0).
    min_env_corr_for_child_merge = 0.2
    root_indices = sorted(set(find(i) for i in range(len(components))))
    root_rep_freq: dict[int, float] = {}
    root_rep_amp: dict[int, float] = {}
    root_rep_env: dict[int, np.ndarray] = {}
    for r in root_indices:
        members = [i for i in range(len(components)) if find(i) == r]
        idx_loudest = max(members, key=lambda x: rep_amps[x])
        root_rep_freq[r] = rep_freqs[idx_loudest]
        root_rep_amp[r] = rep_amps[idx_loudest]
        # Aggregate envelope across all union members — captures
        # the full source's amplitude profile, not just one bin.
        env_acc = np.zeros_like(comp_envs[0])
        for m in members:
            env_acc = env_acc + comp_envs[m]
        root_rep_env[r] = env_acc
    for ri, r_low in enumerate(root_indices):
        for r_high in root_indices[ri + 1:]:
            f_low, amp_low = root_rep_freq[r_low], root_rep_amp[r_low]
            f_high, amp_high = root_rep_freq[r_high], root_rep_amp[r_high]
            if f_low > f_high:
                f_low, f_high = f_high, f_low
                amp_low, amp_high = amp_high, amp_low
                low_root, high_root = r_high, r_low
            else:
                low_root, high_root = r_low, r_high
            if not _pair_is_overtone(
                f_low, f_high, tolerance_semitones
            ):
                continue
            if amp_low < min_amp_dominance * amp_high:
                continue
            # Envelope correlation gate: a harmonic CHILD of A must
            # share A's envelope (the source's amplitude profile).
            # Skip the check for very dominant parents (≥ 4×) — at
            # that ratio the child is almost certainly a harmonic
            # tail, not an independent voice (independent voices
            # rarely have one ≥ 4× louder than another at integer
            # ratios in normal music).
            env_lo = root_rep_env[low_root]
            env_hi = root_rep_env[high_root]
            very_dominant = amp_low >= 4.0 * amp_high
            if not very_dominant and env_lo.std() > 1e-6 and env_hi.std() > 1e-6:
                corr = float(np.corrcoef(env_lo, env_hi)[0, 1])
                if np.isnan(corr) or corr < min_env_corr_for_child_merge:
                    continue
            union(low_root, high_root)

    merged: dict[int, list[int]] = {}
    for i, comp in enumerate(components):
        root = find(i)
        merged.setdefault(root, []).extend(comp)
    return list(merged.values())


def _connected_components_from_adj(adj: np.ndarray) -> list[list[int]]:
    """Return a list of index lists, one per connected component."""
    n = adj.shape[0]
    if n == 0:
        return []
    n_components, labels = connected_components(
        csgraph=adj.astype(np.int32), directed=False, return_labels=True,
    )
    groups: list[list[int]] = [[] for _ in range(n_components)]
    for idx, lab in enumerate(labels):
        groups[int(lab)].append(int(idx))
    return groups


def _cluster_stats(
    cluster_osc_indices: Sequence[int],
    amps_over_window: np.ndarray,       # (frames, n_osc)
    phases_last_frame: np.ndarray,      # (n_osc,)
    freqs: np.ndarray,                  # (n_osc,)
    corr_matrix: np.ndarray | None,     # (n_active, n_active) or None
    active_to_cluster_local: dict[int, int] | None,
) -> tuple[float, float, float, float]:
    """Compute (center_freq, amp, phase_centroid, confidence) for a cluster.

    ``center_freq`` is the frequency of the cluster's **loudest
    partial**, refined by quadratic interpolation in log-frequency for
    sub-bin accuracy. The loudest partial is the perceptually salient
    pitch of a voice: for a pure tone it is the tone itself; for a
    natural harmonic series the fundamental carries the most energy, so
    it wins. This is deliberately NOT a descend-to-lowest-member
    heuristic — the critical-Hopf cascade conjures phantom subharmonics
    beneath even a pure tone, and anchoring to the lowest member put
    voices an octave (or two) below their true pitch. Anchoring to the
    peak is correct for the common case and stable frame to frame
    (the peak bin doesn't flicker the way a "best candidate fundamental"
    search does).

    ``confidence`` is the mean of within-cluster pairwise correlations.
    If the cluster has a single oscillator, confidence is 1.0
    (trivially coherent with itself).
    """
    idx_list = list(cluster_osc_indices)
    cluster_mean_amps = amps_over_window[:, idx_list].mean(axis=0)
    cluster_freqs = freqs[idx_list]
    total_amp = float(cluster_mean_amps.sum())
    if total_amp <= 0:
        center_freq = float(cluster_freqs[0]) if len(cluster_freqs) else 0.0
        return center_freq, 0.0, 0.0, 0.0
    # Voice pitch = the loudest partial's frequency.
    peak_local = int(cluster_mean_amps.argmax())
    center_freq = float(cluster_freqs[peak_local])
    # Sub-bin refinement: quadratic interpolation on the log-amplitude
    # of the peak and its two frequency-neighbours. cluster_freqs is
    # ascending (the bank is tonotopic and idx_list is sorted), so the
    # neighbours bracket the peak; for a contiguous spectral peak they
    # are bin-adjacent and the parabola gives the true sub-bin centre.
    if 0 < peak_local < len(cluster_freqs) - 1:
        a = float(cluster_mean_amps[peak_local - 1])
        b = float(cluster_mean_amps[peak_local])
        c = float(cluster_mean_amps[peak_local + 1])
        denom = a - 2 * b + c
        if denom != 0:
            delta = max(-0.5, min(0.5, 0.5 * (a - c) / denom))
            lf = np.log(cluster_freqs)
            center_freq = float(np.exp(
                lf[peak_local]
                + delta * (lf[peak_local + 1] - lf[peak_local - 1]) / 2.0))
    amp = float(cluster_mean_amps.mean())
    phase = _circular_mean(phases_last_frame[idx_list], cluster_mean_amps)
    if len(idx_list) < 2 or corr_matrix is None or active_to_cluster_local is None:
        confidence = 1.0
    else:
        local = [active_to_cluster_local[i] for i in idx_list]
        pairs: list[float] = []
        for a in range(len(local)):
            for b in range(a + 1, len(local)):
                pairs.append(float(corr_matrix[local[a], local[b]]))
        confidence = float(np.mean(pairs)) if pairs else 1.0
    return center_freq, amp, phase, max(0.0, min(1.0, confidence))


def _match_cost(
    candidate: dict,
    prev: VoiceIdentity,
    config: VoiceClusteringConfig,
) -> float:
    """Cost of matching a candidate cluster to a previous voice."""
    # Log-frequency distance (in octaves)
    log_dist = abs(_log_freq(candidate["center_freq"]) - _log_freq(prev.center_freq))
    # In semitones so the weight is interpretable.
    log_dist_semi = log_dist * 12 / np.log(2)
    amp_dist = abs(candidate["amp"] - prev.amp)
    # Jaccard distance on oscillator index sets
    a = set(candidate["oscillator_indices"])
    b = set(prev.oscillator_indices)
    union = a | b
    jaccard = 1.0 - (len(a & b) / len(union)) if union else 1.0
    cost = (
        config.match_weight_logfreq * log_dist_semi
        + config.match_weight_amp * amp_dist
        + config.match_weight_jaccard * jaccard
    )
    return float(cost)


def _hungarian_assign(
    candidates: list[dict],
    prev_voices: list[VoiceIdentity],
    config: VoiceClusteringConfig,
) -> tuple[dict[int, int], set[int]]:
    """Assign candidates to prev voices minimizing total cost.

    Returns (candidate_index → prev_voice_index mapping, set of
    matched prev_voice_indices). Candidates whose best-match cost
    exceeds ``match_cost_cap`` are left unassigned (→ new IDs).
    """
    n_cand = len(candidates)
    n_prev = len(prev_voices)
    if n_cand == 0 or n_prev == 0:
        return {}, set()

    # Build cost matrix. Use a large sentinel for gated pairs so
    # Hungarian prefers real matches.
    BIG = 1e6
    cost = np.full((n_cand, n_prev), BIG, dtype=np.float64)
    for i, cand in enumerate(candidates):
        for j, prev in enumerate(prev_voices):
            c = _match_cost(cand, prev, config)
            if c <= config.match_cost_cap:
                cost[i, j] = c
            # else leave as BIG

    row_ind, col_ind = linear_sum_assignment(cost)
    assignments: dict[int, int] = {}
    matched_prev: set[int] = set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < BIG:
            assignments[int(r)] = int(c)
            matched_prev.add(int(c))
    return assignments, matched_prev


def _decayed_prev_voices(
    prev_voices: list[VoiceIdentity],
    matched_prev: set[int],
    config: VoiceClusteringConfig,
) -> list[VoiceIdentity]:
    """Return the carry-forward list of unmatched prev voices, with
    silent_frames incremented and stale entries retired."""
    carried: list[VoiceIdentity] = []
    for j, v in enumerate(prev_voices):
        if j in matched_prev:
            continue
        new_silent = v.silent_frames + 1
        if new_silent > config.max_silent_frames:
            continue  # retire
        carried.append(replace(
            v,
            active=False,
            silent_frames=new_silent,
            age_frames=v.age_frames + 1,
        ))
    return carried


def extract_voices(
    window: StateWindow,
    *,
    prev_state: VoiceState | None = None,
    config: VoiceClusteringConfig | None = None,
) -> VoiceState:
    """Identify voices in the current pitch state and track them
    against the previous frame's state.

    This is the core of the voice-identity primitive. Pure-ish (reads
    no module-level state; writes no side effects) — but takes
    ``prev_state`` so identity can persist across frames. Callers
    thread ``prev_state`` through time and should pass ``None`` only
    on the first frame.

    Parameters
    ----------
    window : StateWindow
        Must carry ``pitch_z`` of shape ``(frames, n_pitch)`` — a
        single-frame snapshot won't yield meaningful amplitude
        envelope correlations. A ~2 s window is a reasonable default.
    prev_state : VoiceState | None
        The return value of the previous ``extract_voices`` call. If
        ``None``, all detected clusters become new voices with fresh
        IDs.
    config : VoiceClusteringConfig | None
        Override clustering parameters. Defaults are tuned for
        mastered audio.

    Returns
    -------
    VoiceState
        The updated state. ``voices`` contains every tracked voice
        (active + silent-but-not-retired). ``active_voices`` is the
        convenience filter.
    """
    cfg = config or VoiceClusteringConfig()
    prev = prev_state or VoiceState()

    pitch_z_2d = window.pitch_z_2d
    if pitch_z_2d.ndim != 2 or pitch_z_2d.shape[0] < 2:
        # Not enough frames for correlation. Decay prev voices and
        # return.
        new_voices = _decayed_prev_voices(prev.voices, set(), cfg)
        return VoiceState(voices=new_voices, next_id=prev.next_id)

    amps = np.abs(pitch_z_2d)                # (frames, n_osc)
    phases_last = np.angle(pitch_z_2d[-1])   # (n_osc,)
    mean_amps = amps.mean(axis=0)
    peak = float(mean_amps.max())

    # Active oscillator filter
    active_threshold = max(cfg.noise_floor, cfg.active_fraction * peak)
    active_mask = mean_amps > active_threshold
    # Peak detection on the active set: keep ONLY oscillators that
    # are local maxima within a ±peak_radius_bins window. The
    # cascade-enabled pipeline produces wall-to-wall activity in the
    # pitch bank for dense music (every harmonic + combination tone
    # of every note registers above threshold), so a simple threshold
    # admits hundreds of "active" bins that are really bandwidth
    # tails of a few peaks. Peak detection collapses each tail to
    # its center, recovering a sparse set of meaningful candidates.
    if cfg.peak_detection_enabled and active_mask.any():
        radius = cfg.peak_radius_bins
        shoulder = cfg.peak_shoulder_ratio
        is_peak = np.zeros_like(active_mask)
        for i in range(len(mean_amps)):
            if not active_mask[i]:
                continue
            lo = max(0, i - radius)
            hi = min(len(mean_amps), i + radius + 1)
            local_max = mean_amps[lo:hi].max()
            # Accept the bin if it's within `shoulder` of the local
            # maximum — captures unison clusters (multiple bins at
            # near-identical amplitude) as well as the strict peak.
            if mean_amps[i] >= shoulder * local_max:
                is_peak[i] = True
        active_mask = is_peak
    active_indices = np.where(active_mask)[0]

    if len(active_indices) == 0:
        # Silence: carry prev voices forward as silent, retire stale.
        carried = _decayed_prev_voices(prev.voices, set(), cfg)
        return VoiceState(voices=carried, next_id=prev.next_id)

    # Pairwise amplitude-envelope correlation over active oscillators.
    #
    # Two regimes: modulated oscillators (amp varies over the window)
    # carry envelope information we can correlate on; flat oscillators
    # (truly sustained tone, near-zero amp std) don't. Earlier the
    # code added a deterministic ramp to all flat rows so corrcoef
    # wouldn't NaN — but that made every flat row perfectly correlated
    # with every other flat row, so any sustained signal collapsed its
    # fundamental, harmonics, *and* the entire noise-floor bank into
    # one giant cluster. Instead we compute correlation only where it
    # has meaning, and cluster flat pairs purely by harmonic ratio.
    active_amps = amps[:, active_indices]    # (frames, n_active)
    if len(active_indices) == 1:
        corr = np.array([[1.0]])
        flat_mask = np.zeros(1, dtype=bool)
    else:
        stds = active_amps.std(axis=0)
        means = active_amps.mean(axis=0)
        # "Flat" = coefficient of variation below a small threshold.
        # A row is flat if its envelope varies by less than ~2% over
        # the window, relative to its mean amplitude. This catches
        # sustained synth tones (truly constant) AND near-flat signals
        # with small measurement noise, but stays False for musical
        # envelopes that genuinely vary (>10% typically).
        with np.errstate(invalid="ignore", divide="ignore"):
            cv = np.where(means > 1e-6, stds / means, 0.0)
        flat_mask = cv < 0.02
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(active_amps.T)
        corr = np.nan_to_num(corr, nan=0.0)
        # Zero out correlations involving flat rows — correlation was
        # ill-defined for them; harmonic-ratio adjacency (below) is
        # the real clustering signal for sustained content.
        if flat_mask.any():
            flat_idx = np.where(flat_mask)[0]
            corr[flat_idx, :] = 0.0
            corr[:, flat_idx] = 0.0
            corr[flat_idx, flat_idx] = 1.0

    # Build adjacency: per-pair decision based on whether each
    # oscillator's envelope carries information, AND whether the
    # pair is harmonically related.
    #
    # Non-harmonic pairs: cluster only if correlation is strong
    #   (>= correlation_threshold, default 0.6). Two unrelated
    #   instruments playing together won't collide.
    # Harmonic pairs: cluster UNLESS clearly anti-correlated. Two
    #   oscillators at a 2:1 ratio driven by the same source will
    #   phase-lock (and show up with correlated or weakly-correlated
    #   envelopes); two oscillators at 2:1 driven by different sources
    #   with anti-phase envelopes will have strong negative
    #   correlation and won't cluster. This is the closest proxy for
    #   the NRT phase-lock primitive with the correlation signal
    #   we currently compute — see Task #32 for potential replacement
    #   with direct phase-locking value (PLV).
    # Flat rows (std < 1e-9): correlation meaningless, defer entirely
    #   to harmonic relationship.
    n_active = len(active_indices)
    adj = np.zeros((n_active, n_active), dtype=bool)
    if n_active > 1:
        freqs = window.pitch_freqs
        active_freqs = freqs[active_indices]
        flat_mat = flat_mask[:, None] | flat_mask[None, :]

        # Three independent signals decide whether two oscillators
        # belong to the same voice. A pair connects if ANY fires.
        #
        # (1) Proximity — same spectral peak. Pairs within
        #     proximity_merge_semitones are bandwidth spread of one
        #     mode-lock response, not distinct pitches.
        log_f = np.log2(np.maximum(active_freqs.astype(np.float64), 1e-9))
        log_diff_semi = np.abs(log_f[:, None] - log_f[None, :]) * 12.0
        proximity_mat = log_diff_semi <= cfg.proximity_merge_semitones

        # (2) Overtone family — one oscillator is an integer multiple
        #     (2×…8×) of the other AND the pair is "bound". The overtone
        #     ratio is the structural prerequisite for "harmonic of the
        #     same note"; broad ratios like 3:2 / 5:4 are chord
        #     intervals between DISTINCT notes and are deliberately
        #     excluded. "Bound" means either the Hebbian W learned a
        #     strong connection (Kim & Large 2021 Eq. 26 — the
        #     time-integrated same-source signal that survives a single
        #     window's envelope decoupling, which is exactly how cascade
        #     harmonics behave) OR the per-window envelopes co-modulate.
        #     Co-modulation only ever CONFIRMS an overtone relationship;
        #     it never merges on its own, because everything in a mix
        #     co-moves with the beat — a chord struck on one envelope
        #     must still resolve into separate notes.
        # Overtone matching must absorb the bank's frequency
        # quantization: a true n:1 ratio between two QUANTIZED bins can
        # be off by up to ~one bin spacing (half a bin at each end). In
        # the dense engine bank (~0.33 st/bin) that's below the 0.5 st
        # default; in a coarse bank it dominates, so widen to the bin
        # spacing when larger. Overtone targets (2×…8×) sit many
        # semitones from any chord interval, so this never admits a
        # chord relationship as an "overtone".
        bank_logf = np.log2(np.maximum(freqs.astype(np.float64), 1e-9))
        bin_semi = (float(np.median(np.abs(np.diff(bank_logf))) * 12.0)
                    if len(freqs) > 1 else 0.0)
        overtone_tol = max(cfg.harmonic_ratio_tolerance_semitones, bin_semi)
        overtone_mat = _overtone_matrix(active_freqs, overtone_tol)
        comod = corr >= cfg.harmonic_correlation_threshold
        if cfg.use_learned_affinity and window.w_pitch is not None:
            learned_strong, _ = _learned_strong_matrix(
                active_indices, window.w_pitch, cfg.w_affinity_percentile
            )
            bound = learned_strong | comod
        else:
            # No learned W (early live / pre-convergence synthetic):
            # co-modulation is the only same-source evidence, and flat
            # overtone pairs (no envelope to correlate) merge on ratio.
            bound = comod | flat_mat

        # (3) Frequency-local co-modulation — contiguous co-moving bins
        #     are the spectral spread of one source's peak. Gated to a
        #     narrow neighbourhood so it chains one peak's bins (and
        #     near-unison) without bridging the gaps between distinct
        #     pitches; the gaps are where a chord resolves into notes.
        comod_local = (comod & ~flat_mat
                       & (log_diff_semi <= cfg.comod_locality_semitones))

        connect_mat = proximity_mat | (overtone_mat & bound) | comod_local
        np.fill_diagonal(connect_mat, False)
        adj |= connect_mat
        adj |= connect_mat.T

    components = _connected_components_from_adj(adj)
    # Phase 4: merge components at integer-ratio relationships. Catches
    # harmonics generated by canonical input + coupling kernel + Hebbian
    # learning that don't envelope-correlate strongly with their
    # fundamental but track its frequency with constant phase offset.
    # Tolerance is wider than the pre-cluster harmonic-edge tolerance:
    # post-cluster representative frequencies have bandwidth spread
    # from the cluster's member oscillators (e.g., a "110 Hz harmonic"
    # cluster might be centered at 104 or 117 Hz depending on which
    # bin happens to be loudest). 1.5 semitones tolerance catches
    # this spread without admitting clearly non-harmonic pairs.
    phases_full = np.angle(pitch_z_2d)
    w_for_merge = (window.w_pitch
                   if cfg.use_learned_affinity else None)
    w_thresh_merge = 0.0
    if w_for_merge is not None:
        Wmag = np.abs(w_for_merge)
        n_w = Wmag.shape[0]
        if n_w > 1:
            w_thresh_merge = float(np.percentile(
                Wmag[~np.eye(n_w, dtype=bool)], cfg.w_affinity_percentile))
    components = _merge_harmonic_components(
        components,
        np.asarray(active_indices),
        amps,
        phases_full,
        window.pitch_freqs,
        tolerance_semitones=1.5,
        w_pitch=w_for_merge,
        w_thresh=w_thresh_merge,
    )

    # Build candidate cluster descriptors
    active_to_local = {int(global_idx): local
                       for local, global_idx in enumerate(active_indices)}
    candidates: list[dict] = []
    for comp in components:
        osc_global = sorted(int(active_indices[i]) for i in comp)
        if len(osc_global) < cfg.min_cluster_size:
            continue
        center_freq, amp, phase, conf = _cluster_stats(
            osc_global, amps, phases_last, window.pitch_freqs,
            corr, active_to_local,
        )
        candidates.append({
            "oscillator_indices": tuple(osc_global),
            "center_freq": center_freq,
            "amp": amp,
            "phase_centroid": phase,
            "confidence": conf,
        })

    # Sort candidates by amplitude descending — so the dominant voice
    # tends to inherit the dominant prev ID when costs are close.
    candidates.sort(key=lambda c: -c["amp"])

    # Hungarian matching against prev voices
    assignments, matched_prev = _hungarian_assign(candidates, prev.voices, cfg)

    # Build new voices
    new_voices: list[VoiceIdentity] = []
    next_id = prev.next_id
    for i, cand in enumerate(candidates):
        if i in assignments:
            prev_v = prev.voices[assignments[i]]
            # Frequency smoothing: a tracked voice's pitch is an EMA in
            # log-frequency toward its previous value, but only while the
            # new estimate stays within freq_snap_semitones. This kills
            # the per-frame jitter (the fundamental estimate flickering
            # between adjacent bins) that made voices "jump around",
            # while still letting a real pitch move (glissando, a new
            # note on the same ID) track freely past the snap distance.
            center_freq = cand["center_freq"]
            if (cfg.freq_smoothing > 0.0 and prev_v.center_freq > 0
                    and center_freq > 0):
                step_semi = abs(12.0 * np.log2(center_freq / prev_v.center_freq))
                if step_semi <= cfg.freq_snap_semitones:
                    a = cfg.freq_smoothing
                    center_freq = float(np.exp(
                        a * np.log(prev_v.center_freq)
                        + (1.0 - a) * np.log(center_freq)))
            new_voices.append(VoiceIdentity(
                id=prev_v.id,
                oscillator_indices=cand["oscillator_indices"],
                center_freq=center_freq,
                amp=cand["amp"],
                phase_centroid=cand["phase_centroid"],
                active=True,
                confidence=cand["confidence"],
                age_frames=prev_v.age_frames + 1,
                silent_frames=0,
            ))
        else:
            new_voices.append(VoiceIdentity(
                id=next_id,
                oscillator_indices=cand["oscillator_indices"],
                center_freq=cand["center_freq"],
                amp=cand["amp"],
                phase_centroid=cand["phase_centroid"],
                active=True,
                confidence=cand["confidence"],
                age_frames=0,
                silent_frames=0,
            ))
            next_id += 1

    # Carry unmatched prev voices forward as silent
    carried = _decayed_prev_voices(prev.voices, matched_prev, cfg)
    new_voices.extend(carried)

    return VoiceState(voices=new_voices, next_id=next_id)


def _associate_voice_with_bank(
    voice_envelopes: dict[int, np.ndarray],
    bank_freqs: np.ndarray,
    bank_phase_last: np.ndarray,
    frame_hz: float,
    min_bpm: float,
    max_bpm: float,
) -> dict[int, tuple[int, float, float, float]]:
    """Match each voice's envelope against an oscillator bank via DFT.

    ``voice_envelopes`` maps voice.id → envelope (1D amp-over-time).
    Returns {voice_id: (osc_idx, freq_hz, phase_rad, confidence)} for
    voices where a rhythm match was found. Voices with flat envelopes
    or below-SNR peaks are omitted.

    Shared by ``extract_voice_rhythms`` (rhythm bank) and
    ``extract_voice_motor`` (motor bank). The two extractors are
    structurally identical; they differ only in which bank they
    associate against and in the semantic label on the output.
    """
    if not voice_envelopes:
        return {}
    n_frames = next(iter(voice_envelopes.values())).shape[0]
    if n_frames < 8:
        return {}
    min_freq = min_bpm / 60.0
    max_freq = max_bpm / 60.0
    eligible_idx = np.where(
        (bank_freqs >= min_freq) & (bank_freqs <= max_freq)
    )[0]
    if len(eligible_idx) == 0:
        return {}
    t = np.arange(n_frames) / frame_hz
    freqs_eligible = bank_freqs[eligible_idx]
    basis = np.exp(-1j * 2 * np.pi * np.outer(freqs_eligible, t)) / n_frames

    out: dict[int, tuple[int, float, float, float]] = {}
    for voice_id, env in voice_envelopes.items():
        env_mean = float(env.mean())
        env_std = float(env.std())
        if env_std < 1e-6 or env_mean <= 0:
            continue
        env_centered = env - env_mean
        magnitudes = np.abs(basis @ env_centered)
        if magnitudes.max() <= 0:
            continue
        best_local = int(np.argmax(magnitudes))
        best_global = int(eligible_idx[best_local])
        peak_mag = float(magnitudes[best_local])
        mean_mag = float(magnitudes.mean())
        conf = (peak_mag - mean_mag) / peak_mag if peak_mag > 0 else 0.0
        out[voice_id] = (
            best_global,
            float(bank_freqs[best_global]),
            float(bank_phase_last[best_global]),
            float(max(0.0, min(1.0, conf))),
        )
    return out


def _voice_envelopes(
    voice_state: VoiceState,
    pitch_amps: np.ndarray,
) -> dict[int, np.ndarray]:
    """Compute the per-voice amplitude envelope for each active
    voice — mean |z| across the voice's oscillator indices over the
    window."""
    envs: dict[int, np.ndarray] = {}
    for v in voice_state.voices:
        if not v.active or not v.oscillator_indices:
            continue
        envs[v.id] = pitch_amps[:, list(v.oscillator_indices)].mean(axis=1)
    return envs


def extract_voice_rhythms(
    window: StateWindow,
    voice_state: VoiceState,
    *,
    min_bpm: float = 60.0,
    max_bpm: float = 240.0,
) -> VoiceState:
    """Associate each active voice with its own rhythm.

    For each active voice, compute its amplitude envelope across the
    window (mean over the voice's pitch oscillators), then find the
    rhythm oscillator whose natural frequency best matches the
    envelope's dominant periodicity via DFT.

    Why this is novel for the modular use case: conventional beat
    trackers output a single tempo for the whole track. Here, each
    voice gets its own tempo — a bass part at 60 BPM coexists with
    a hi-hat pattern at 240 BPM as two distinct per-voice clocks,
    both read off the same oscillator network.

    Implementation note: this reads from the existing rhythm GrFNN
    state rather than introducing per-voice rhythm networks. The
    rhythm GrFNN is already entrained to the overall music; we're
    picking the oscillator from its bank whose frequency matches
    each voice's envelope. Per-voice rhythm GrFNNs (Phase 3
    territory) would be a strict upgrade if this proves insufficient.

    Returns a new ``VoiceState`` with each active voice's
    ``rhythm`` field populated. Silent voices and voices with flat
    envelopes are left with their existing ``rhythm`` (possibly
    ``None``).
    """
    pitch_z_2d = window.pitch_z_2d
    rhythm_z_2d = window.rhythm_z_2d
    rhythm_freqs = window.rhythm_freqs
    n_frames = pitch_z_2d.shape[0]
    if n_frames < 8 or rhythm_z_2d.shape[0] == 0:
        return voice_state

    pitch_amps = np.abs(pitch_z_2d)
    rhythm_phase_last = np.angle(rhythm_z_2d[-1])
    envelopes = _voice_envelopes(voice_state, pitch_amps)
    matches = _associate_voice_with_bank(
        envelopes, rhythm_freqs, rhythm_phase_last,
        window.frame_hz, min_bpm, max_bpm,
    )

    new_voices: list[VoiceIdentity] = []
    for v in voice_state.voices:
        if v.id in matches:
            osc_idx, freq, phase, conf = matches[v.id]
            new_voices.append(replace(
                v,
                rhythm=VoiceRhythm(
                    osc_idx=osc_idx,
                    freq=freq,
                    bpm=freq * 60.0,
                    phase=phase,
                    confidence=conf,
                ),
            ))
        else:
            new_voices.append(v)
    return VoiceState(voices=new_voices, next_id=voice_state.next_id)


def extract_voice_motor(
    window: StateWindow,
    voice_state: VoiceState,
    *,
    min_bpm: float = 60.0,
    max_bpm: float = 240.0,
) -> VoiceState:
    """Associate each active voice with a motor-GrFNN oscillator.

    Structurally identical to ``extract_voice_rhythms`` — DFT the
    voice envelope against an oscillator bank, pick the best-matching
    frequency, report that oscillator's phase. The difference is
    semantic: motor oscillators carry *forward-predictive* state via
    the bidirectional coupling with sensory rhythm. A voice's motor
    phase is the anticipated next-beat position, not the current
    beat. For the modular-bridge use case this becomes a per-voice
    "beat prediction" CV that keeps ticking through brief silences
    (motor sustains when sensory drive drops — see
    test_two_layer_pulse).

    If the window lacks motor state (``motor_z is None``), returns
    ``voice_state`` unchanged — caller's state remains valid and the
    motor field stays at whatever its previous value was (likely
    None in that case).
    """
    motor_z_2d = window.motor_z_2d
    motor_freqs = window.motor_freqs
    if motor_z_2d is None or motor_freqs is None:
        return voice_state
    pitch_z_2d = window.pitch_z_2d
    n_frames = pitch_z_2d.shape[0]
    if n_frames < 8 or motor_z_2d.shape[0] == 0:
        return voice_state

    pitch_amps = np.abs(pitch_z_2d)
    motor_phase_last = np.angle(motor_z_2d[-1])
    envelopes = _voice_envelopes(voice_state, pitch_amps)
    matches = _associate_voice_with_bank(
        envelopes, motor_freqs, motor_phase_last,
        window.frame_hz, min_bpm, max_bpm,
    )

    new_voices: list[VoiceIdentity] = []
    for v in voice_state.voices:
        if v.id in matches:
            osc_idx, freq, phase, conf = matches[v.id]
            new_voices.append(replace(
                v,
                motor=VoiceMotor(
                    osc_idx=osc_idx,
                    freq=freq,
                    bpm=freq * 60.0,
                    phase=phase,
                    confidence=conf,
                ),
            ))
        else:
            new_voices.append(v)
    return VoiceState(voices=new_voices, next_id=voice_state.next_id)
