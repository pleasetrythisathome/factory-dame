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

    1. Filter to active oscillators (amplitude above noise floor and
       above a fraction of the current peak).
    2. Compute pairwise Pearson correlation between their amplitude
       envelopes over the window — "do these oscillators move
       together?"
    3. Boost correlation for pairs whose frequency ratio is a small
       integer (harmonically related — likely parts of the same
       voice).
    4. Threshold → adjacency graph → connected components → clusters.
    5. Match clusters to previous frame's voices via Hungarian
       assignment on a cost function combining log-frequency distance,
       amplitude change, and oscillator-index Jaccard distance.
    6. Unmatched new clusters → new voice IDs. Unmatched prev voices
       → marked silent; retired after ``max_silent_frames`` frames.

Deliberate design choices:

- **Dynamic voice count.** Never fixed N. Music has 1 voice in a solo
  piano passage and 8 voices in dense jazz; the extractor follows.
- **Amplitude envelope correlation**, not 1:1 PLV. A voice spans
  multiple frequencies. What makes oscillators share a voice is that
  they rise and fall together, not that they're at the same pitch.
- **Harmonic ratio boost.** A fundamental and its harmonics will have
  correlated envelopes already; the boost is belt-and-suspenders and
  helps noisy or weakly-articulated voices.
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
    (1, 2), (2, 1),
    (1, 3), (3, 1),
    (2, 3), (3, 2),
    (1, 4), (4, 1),
    (3, 4), (4, 3),
    (1, 5), (5, 1),
    (4, 5), (5, 4),
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
    """Tunable parameters for voice clustering. Defaults are chosen to
    work on mastered audio (pitch amplitudes peaking ~0.3-0.6 on the
    NRT Hopf limit cycle); tweak for other signal regimes."""

    # Oscillator activity gating
    noise_floor: float = 0.02
    active_fraction: float = 0.08   # fraction of peak amplitude to count as active

    # Correlation clustering
    correlation_threshold: float = 0.6
    harmonic_boost: float = 0.15    # added to correlation when pair has harmonic ratio
    harmonic_ratio_tolerance_semitones: float = 0.5

    # Cluster filtering
    min_cluster_size: int = 1       # singletons allowed — a lone pitch is a voice
    min_sustain_frames: int = 1     # min window-frames of activity before counting

    # Voice tracking
    max_silent_frames: int = 40     # retire a silent voice after this many frames
    match_cost_cap: float = 5.0     # pair costs above this → no match, force new ID
    match_weight_logfreq: float = 1.0  # per-semitone penalty
    match_weight_amp: float = 0.3      # per-unit-amp penalty
    match_weight_jaccard: float = 2.0  # per-unit-dissimilarity penalty


def _log_freq(hz: float) -> float:
    return float(np.log(max(hz, 1e-9)))


def _circular_mean(phases: np.ndarray, weights: np.ndarray) -> float:
    """Amplitude-weighted circular mean of phases."""
    if weights.sum() <= 0:
        return 0.0
    c = float(np.sum(weights * np.cos(phases)))
    s = float(np.sum(weights * np.sin(phases)))
    return float(np.arctan2(s, c))


def _pair_is_harmonic(
    f_i: float,
    f_j: float,
    tolerance_semitones: float,
) -> bool:
    """True if f_i : f_j is near a small-integer ratio."""
    if f_i <= 0 or f_j <= 0:
        return False
    ratio = max(f_i, f_j) / min(f_i, f_j)
    log_ratio = np.log2(ratio)
    for p, q in _HARMONIC_RATIOS:
        if q == 0:
            continue
        target_log = np.log2(p / q)
        if target_log < 0:
            continue  # symmetric pair already covered
        if abs(log_ratio - target_log) * 12 <= tolerance_semitones:
            return True
    return False


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
    log_f_weighted = (cluster_mean_amps * np.log(cluster_freqs)).sum() / total_amp
    center_freq = float(np.exp(log_f_weighted))
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
    active_indices = np.where(active_mask)[0]

    if len(active_indices) == 0:
        # Silence: carry prev voices forward as silent, retire stale.
        carried = _decayed_prev_voices(prev.voices, set(), cfg)
        return VoiceState(voices=carried, next_id=prev.next_id)

    # Pairwise amplitude-envelope correlation over active oscillators
    active_amps = amps[:, active_indices]    # (frames, n_active)
    # np.corrcoef wants features as rows
    if len(active_indices) == 1:
        corr = np.array([[1.0]])
    else:
        # Guard against zero-variance rows (degenerate amps). corrcoef
        # would otherwise emit NaN.
        stds = active_amps.std(axis=0)
        if (stds == 0).any():
            # Replace flat rows with tiny noise to get well-defined
            # correlations (self-correlation = 1, cross = 0).
            active_amps = active_amps + (stds == 0) * 1e-12 * np.arange(
                1, active_amps.shape[0] + 1
            )[:, None]
        corr = np.corrcoef(active_amps.T)
        corr = np.nan_to_num(corr, nan=0.0)

    # Harmonic ratio boost
    if cfg.harmonic_boost > 0 and len(active_indices) > 1:
        freqs = window.pitch_freqs
        for a in range(len(active_indices)):
            for b in range(a + 1, len(active_indices)):
                f_a = float(freqs[active_indices[a]])
                f_b = float(freqs[active_indices[b]])
                if _pair_is_harmonic(
                    f_a, f_b, cfg.harmonic_ratio_tolerance_semitones
                ):
                    corr[a, b] = min(1.0, corr[a, b] + cfg.harmonic_boost)
                    corr[b, a] = corr[a, b]

    # Threshold → adjacency → connected components
    adj = corr > cfg.correlation_threshold
    np.fill_diagonal(adj, False)
    components = _connected_components_from_adj(adj)

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
            new_voices.append(VoiceIdentity(
                id=prev_v.id,
                oscillator_indices=cand["oscillator_indices"],
                center_freq=cand["center_freq"],
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
