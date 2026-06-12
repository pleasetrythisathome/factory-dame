"""Ground-truth reference voices + scoring metrics for voice extraction.

The hard problem with validating voice extraction on real music is that
there is no ground truth: nobody labelled "this frame contains a 110 Hz
bass and a 440 Hz lead". This module defines what a ground-truth voice
IS and how to SCORE an extraction against it, so correctness becomes a
number instead of an eyeballed bound.

A reference is built one of three ways (see the sibling modules):

  - arrangements.py  — parametric multi-instrument synthesis. We choose
    the notes, so the reference is EXACT.
  - midi_ground_truth.py — render a MIDI score; the notes ARE the
    reference, exact.
  - corpus_reference.py — derive an APPROXIMATE reference from real audio
    (source separation + pitch tracking). Lower trust, but real audio.

The scoring is deliberately assignment-based: at each analysis frame we
optimally match extracted voices to the reference notes sounding at that
instant (Hungarian on log-frequency distance, capped by a tolerance),
then tally precision / recall / pitch error. This rewards "found the
right pitches" and penalises both missed notes and spurious voices,
without demanding that voice IDs line up with reference labels (IDs are
arbitrary; pitch-over-time is what's real).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


# ── Ground-truth data model ───────────────────────────────────────


@dataclass(frozen=True)
class Note:
    """A single sounding note: a fundamental over a time span."""

    onset_s: float
    offset_s: float
    freq_hz: float

    def active_at(self, t: float) -> bool:
        return self.onset_s <= t < self.offset_s


@dataclass
class ReferenceVoice:
    """One ground-truth voice — a monophonic line of notes (a bass part,
    a lead, one note of a sustained chord). ``role`` distinguishes tonal
    voices (carry pitch, scored against extracted voices) from percussive
    ones (broadband transients; counted separately, not pitch-scored)."""

    label: str
    notes: list[Note] = field(default_factory=list)
    role: str = "tonal"          # "tonal" | "percussive"

    def freq_at(self, t: float) -> float | None:
        for n in self.notes:
            if n.active_at(t):
                return n.freq_hz
        return None


@dataclass
class ReferenceAnnotation:
    """Complete ground truth for one audio clip."""

    voices: list[ReferenceVoice]
    duration_s: float
    sample_rate: int = 16000

    # — queries —
    def tonal_voices(self) -> list[ReferenceVoice]:
        return [v for v in self.voices if v.role == "tonal"]

    def active_freqs_at(self, t: float) -> list[float]:
        """Fundamentals of all tonal voices sounding at time t."""
        out = []
        for v in self.tonal_voices():
            f = v.freq_at(t)
            if f is not None and f > 0:
                out.append(f)
        return out

    def n_tonal_active_at(self, t: float) -> int:
        return len(self.active_freqs_at(t))

    # — serialisation (so generated references persist next to parquets) —
    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ReferenceAnnotation":
        voices = [
            ReferenceVoice(
                label=v["label"],
                role=v.get("role", "tonal"),
                notes=[Note(**n) for n in v["notes"]],
            )
            for v in d["voices"]
        ]
        return ReferenceAnnotation(
            voices=voices,
            duration_s=d["duration_s"],
            sample_rate=d.get("sample_rate", 16000),
        )


# ── Scoring ───────────────────────────────────────────────────────


def _match_frame(
    extracted_freqs: Sequence[float],
    reference_freqs: Sequence[float],
    tol_semitones: float,
) -> tuple[int, list[float]]:
    """Optimally match extracted voices to reference notes at one frame.

    Returns (n_matched, pitch_errors_semitones). A pair may match only if
    within ``tol_semitones``; the assignment maximises the number of
    matched pairs (minimising summed log-frequency distance) under that
    cap, so neither side is double-counted.
    """
    e = np.asarray([f for f in extracted_freqs if f and f > 0], dtype=np.float64)
    r = np.asarray([f for f in reference_freqs if f and f > 0], dtype=np.float64)
    if len(e) == 0 or len(r) == 0:
        return 0, []
    # cost = absolute semitone distance; gate at tolerance with a big sentinel
    dist = np.abs(12.0 * np.log2(e[:, None] / r[None, :]))   # (n_e, n_r)
    BIG = 1e6
    cost = np.where(dist <= tol_semitones, dist, BIG)
    ri, ci = linear_sum_assignment(cost)
    errors = []
    for a, b in zip(ri, ci):
        if cost[a, b] < BIG:
            errors.append(float(dist[a, b]))
    return len(errors), errors


def score_extraction(
    reference: ReferenceAnnotation,
    extracted_frames: Sequence[tuple[float, Sequence[dict]]],
    *,
    pitch_tol_semitones: float = 1.0,
) -> dict:
    """Score an extraction against ground truth.

    Parameters
    ----------
    reference : the ground-truth annotation.
    extracted_frames : sequence of (time_s, voices) where each ``voices``
        is the list of active extracted voices at that time — dicts with
        at least a ``center_freq`` key (and optionally ``id`` for the
        fragmentation metric).
    pitch_tol_semitones : a matched extracted voice must be within this of
        a reference fundamental (default 1 semitone).

    Returns a dict of metrics:
      count_mae        — mean |n_extracted - n_reference_tonal| per frame
      count_within_1   — fraction of frames with |count diff| <= 1
      pitch_precision  — matched extracted / total extracted
      pitch_recall     — matched reference / total reference
      pitch_f1         — harmonic mean of the two
      pitch_mae_semis  — mean pitch error over matched pairs
      ref_coverage     — fraction of reference note-frames covered by a
                         correctly-pitched extracted voice (== recall here,
                         reported separately for readability)
      id_fragmentation — mean number of distinct extracted IDs that match a
                         given reference voice over its life (1.0 = perfect;
                         higher = the voice keeps getting re-IDed)
    """
    n_frames = 0
    count_abs_err = 0.0
    count_within = 0
    tot_extracted = 0
    tot_reference = 0
    tot_matched = 0
    pitch_errs: list[float] = []
    # id_fragmentation bookkeeping: reference-voice-index -> set of extracted ids
    ref_ids: dict[int, set] = {}

    tonal = reference.tonal_voices()

    for t, voices in extracted_frames:
        ref_freqs = reference.active_freqs_at(t)
        ext_freqs = [float(v["center_freq"]) for v in voices
                     if v.get("center_freq", 0) and v["center_freq"] > 0]
        n_ref = len(ref_freqs)
        n_ext = len(ext_freqs)
        n_frames += 1
        count_abs_err += abs(n_ext - n_ref)
        if abs(n_ext - n_ref) <= 1:
            count_within += 1
        tot_extracted += n_ext
        tot_reference += n_ref
        n_matched, errs = _match_frame(ext_freqs, ref_freqs, pitch_tol_semitones)
        tot_matched += n_matched
        pitch_errs.extend(errs)

        # fragmentation: which extracted IDs sit on which reference voice
        for v in voices:
            f = float(v.get("center_freq", 0) or 0)
            vid = v.get("id")
            if f <= 0 or vid is None:
                continue
            # nearest reference tonal voice within tolerance
            best, best_d = None, pitch_tol_semitones
            for vi, rv in enumerate(tonal):
                rf = rv.freq_at(t)
                if rf and rf > 0:
                    d = abs(12.0 * np.log2(f / rf))
                    if d <= best_d:
                        best, best_d = vi, d
            if best is not None:
                ref_ids.setdefault(best, set()).add(vid)

    precision = tot_matched / tot_extracted if tot_extracted else 0.0
    recall = tot_matched / tot_reference if tot_reference else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    frag = (float(np.mean([len(s) for s in ref_ids.values()]))
            if ref_ids else 0.0)

    return {
        "n_frames": n_frames,
        "count_mae": count_abs_err / n_frames if n_frames else 0.0,
        "count_within_1": count_within / n_frames if n_frames else 0.0,
        "pitch_precision": precision,
        "pitch_recall": recall,
        "pitch_f1": f1,
        "pitch_mae_semis": float(np.mean(pitch_errs)) if pitch_errs else 0.0,
        "ref_coverage": recall,
        "id_fragmentation": frag,
    }
