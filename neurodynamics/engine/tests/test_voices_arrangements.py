"""Score voice extraction against EXACT ground truth on synthetic
multi-instrument arrangements.

Where test_voices_ground_truth asserts simple known-answer counts, these
arrangements have a full time-resolved reference (every note's onset,
offset, fundamental) so we score extraction the way a transcription task
is scored: per-frame optimal matching of extracted voices to the notes
actually sounding, yielding pitch precision / recall / F1, count error,
and ID fragmentation (see tests/reference.py).

Each arrangement is rebuilt from tests/arrangements.py (deterministic, so
the reference is exact) and scored against its parquet
(output/arr_<slug>.parquet, produced by `uv run python -m
tests.arrangements`). Skips cleanly if the parquet is absent.

Run `uv run python -m tests.test_voices_arrangements` to print the full
score table (calibration) without pytest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from neurodynamics.perceptual import StateWindow
from neurodynamics.voices import VoiceState, extract_voices

from tests.arrangements import ALL_ARRANGEMENTS
from tests.reference import score_extraction

ENGINE_DIR = Path(__file__).resolve().parent.parent
STATE_DIR = ENGINE_DIR / "output"


def _load_w(slug: str) -> np.ndarray | None:
    p = STATE_DIR / f"arr_{slug}.weights.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return z["pitch_W"] if "pitch_W" in z.files else None


def _timeline(parquet: Path, w_pitch, *, feature_hz=5.0, window_s=2.5):
    """Run extraction over the parquet, return [(time_s, [voice dicts])]."""
    t = pq.read_table(parquet)
    layer = t.column("layer").to_pylist()
    pitch = t.filter([x == "pitch" for x in layer])
    times = np.array(pitch.column("t").to_pylist(), dtype=np.float64)
    amps = np.array(pitch.column("amp").to_pylist(), dtype=np.float32)
    phases = np.array(pitch.column("phase").to_pylist(), dtype=np.float32)
    meta = t.schema.metadata or {}
    freqs = np.array([float(x) for x in
                      meta.get(b"layer.pitch.f", b"").decode().split(",") if x])
    if len(times) < 2:
        return []
    snap_hz = 1.0 / (times[1] - times[0])
    stride = max(1, int(round(snap_hz / feature_hz)))
    half = max(1, int(snap_hz * window_s / 2))
    state = VoiceState()
    out = []
    for i in range(0, len(times), stride):
        lo, hi = max(0, i - half), min(len(times), i + half + 1)
        z = (amps[lo:hi].astype(np.complex128)
             * np.exp(1j * phases[lo:hi].astype(np.complex128)))
        sw = StateWindow(pitch_z=z, pitch_freqs=freqs,
                         rhythm_z=np.zeros(1, dtype=np.complex128),
                         rhythm_freqs=np.array([1.0]),
                         frame_hz=float(snap_hz), w_pitch=w_pitch)
        state = extract_voices(sw, prev_state=state)
        out.append((float(times[i]),
                    [{"id": v.id, "center_freq": float(v.center_freq),
                      "amp": float(v.amp)} for v in state.active_voices]))
    return out


def _score(slug: str) -> dict | None:
    parquet = STATE_DIR / f"arr_{slug}.parquet"
    if not parquet.exists():
        return None
    _audio, ref = ALL_ARRANGEMENTS[slug]()
    frames = _timeline(parquet, _load_w(slug))
    if not frames:
        return None
    return score_extraction(ref, frames, pitch_tol_semitones=1.0)


# Per-arrangement REGRESSION FLOORS (not aspirational targets).
#
# KNOWN GAP (2026-05-13): the W-community extractor was tuned for clean
# voice COUNTS (musical plausibility), and on polyphonic content that
# tuning actively hurts pitch ACCURACY. Recall on these arrangements is
# only ~0.15-0.35: most simultaneous notes are NOT recovered, even though
# the pitch bank contains clean, strong peaks at every reference note
# (verified: triad_held's top-3 bins are exactly 262/330/392 Hz). Root
# cause: in a harmonic-rich chord an upper note's lower harmonic coincides
# with a lower note's higher harmonic (785 Hz ~ 3x262 ~ 2x392), and
# union-find chains the chord notes together through that shared bin,
# collapsing the chord to one voice. PRECISION stays high (the voices it
# DOES report are at the right pitch, <1 st error) — it under-reports
# rather than hallucinates.
#
# These floors catch REGRESSION below the current baseline; the real
# output is the calibration table (`python -m tests.test_voices_arrangements`).
# The aspirational target is recall >= 0.7 via fundamental-assignment
# clustering (harmonic stripping) instead of union-find — tracked
# separately. Do NOT loosen these to hide a regression.
_THRESHOLDS = {
    #             recall>= precision>= count_mae<= pitch_mae<= frag<=
    "duet":              (0.22, 0.28, 1.0, 1.0, 3.0),
    "bass_and_lead":     (0.08, 0.10, 1.2, 1.5, 3.0),
    "triad_held":        (0.25, 0.65, 2.5, 1.0, 2.0),
    "bass_pad_lead":     (0.12, 0.28, 2.5, 1.0, 2.0),
    "unison_then_split": (0.22, 0.28, 1.2, 1.0, 2.0),
}


@pytest.mark.parametrize("slug", list(ALL_ARRANGEMENTS))
def test_arrangement_extraction_quality(slug):
    parquet = STATE_DIR / f"arr_{slug}.parquet"
    if not parquet.exists():
        pytest.skip(f"{parquet} not present — run `python -m tests.arrangements`")
    s = _score(slug)
    if s is None:
        pytest.skip(f"{slug}: no extraction frames")
    rec, prec, cmae, pmae, frag = _THRESHOLDS[slug]
    assert s["pitch_recall"] >= rec, f"{slug}: recall {s['pitch_recall']:.2f} < {rec}; {s}"
    assert s["pitch_precision"] >= prec, f"{slug}: precision {s['pitch_precision']:.2f} < {prec}; {s}"
    assert s["count_mae"] <= cmae, f"{slug}: count_mae {s['count_mae']:.2f} > {cmae}; {s}"
    assert s["pitch_mae_semis"] <= pmae, f"{slug}: pitch_mae {s['pitch_mae_semis']:.2f} > {pmae}; {s}"
    assert s["id_fragmentation"] <= frag, f"{slug}: fragmentation {s['id_fragmentation']:.2f} > {frag}; {s}"


def main() -> None:
    cols = ["recall", "precis", "f1", "cMAE", "pMAE", "frag", "cWin1"]
    print(f"{'arrangement':20s} " + " ".join(f"{c:>6}" for c in cols))
    for slug in ALL_ARRANGEMENTS:
        s = _score(slug)
        if s is None:
            print(f"{slug:20s}  (no parquet)")
            continue
        print(f"{slug:20s} "
              f"{s['pitch_recall']:6.2f} {s['pitch_precision']:6.2f} "
              f"{s['pitch_f1']:6.2f} {s['count_mae']:6.2f} "
              f"{s['pitch_mae_semis']:6.2f} {s['id_fragmentation']:6.2f} "
              f"{s['count_within_1']:6.2f}")


if __name__ == "__main__":
    main()
