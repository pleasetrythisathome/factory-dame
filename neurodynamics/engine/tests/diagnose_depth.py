"""Deep diagnostic for the cascade-vs-voice-count problem.

Answers, against existing corpus parquets + weights (no engine re-run):

  D1. Does the learned pitch_W encode HARMONIC structure? I.e. is
      |W[i,j]| systematically larger when f_i:f_j is a small-integer
      ratio than when it isn't? This is the make-or-break test for
      W-community voice clustering (Phase 4.3). If yes, W is a
      paper-accurate (Kim & Large 2021 Eq. 26) affinity signal we can
      cluster on. If no, W is unusable and we need another route.

  D2. Failure modes of the current extractor on a real track:
        - voice-count distribution over the track
        - ID churn / mean lifetime
        - per-ID center-frequency jitter ("jumping around")
        - are the "extra" voices harmonics of a louder voice
          (mergeable) or genuinely independent?

Usage:
    cd neurodynamics/engine
    uv run python -m tests.diagnose_depth [slug]
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ENGINE_DIR = Path(__file__).resolve().parent.parent
STATE = ENGINE_DIR / "test_audio" / "state"

sys.path.insert(0, str(ENGINE_DIR / "tests"))
from test_voices_real_audio import _extract_voices_over_track  # noqa: E402

# Small-integer ratios that count as "harmonically related" for D1.
HARM = [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1),
        (3, 2), (4, 3), (5, 3), (5, 4), (5, 2), (7, 2), (8, 3)]
HARM_LOGS = np.array(sorted({np.log2(p / q) for p, q in HARM}), float)


def d1_w_is_harmonic(slug: str) -> None:
    npz = STATE / f"{slug}.weights.npz"
    if not npz.exists():
        print(f"[D1] no weights for {slug}")
        return
    d = np.load(npz)
    W = np.abs(d["pitch_W"])
    f = d["pitch_f"]
    n = W.shape[0]
    f = f[:n]
    # symmetry
    asym = np.abs(W - W.T).max() / (W.max() + 1e-12)
    # log-ratio of every pair, distance to nearest small-integer ratio
    logf = np.log2(np.maximum(f, 1e-9))
    lr = np.abs(logf[:, None] - logf[None, :])
    dist_semi = np.min(np.abs(lr[..., None] - HARM_LOGS[None, None, :]),
                       axis=-1) * 12.0
    off = ~np.eye(n, dtype=bool)
    harm = (dist_semi <= 0.5) & off
    nonharm = (dist_semi > 0.5) & off
    wh = W[harm]
    wn = W[nonharm]
    print(f"[D1] {slug}: pitch_W {n}x{n}, asymmetry={asym:.3f}")
    print(f"     |W| on HARMONIC pairs:    mean={wh.mean():.4g} "
          f"median={np.median(wh):.4g} p90={np.percentile(wh,90):.4g}")
    print(f"     |W| on NON-harmonic pairs: mean={wn.mean():.4g} "
          f"median={np.median(wn):.4g} p90={np.percentile(wn,90):.4g}")
    ratio = wh.mean() / (wn.mean() + 1e-12)
    print(f"     harmonic/non-harmonic mean-|W| ratio = {ratio:.2f}x "
          f"({'USABLE' if ratio > 1.5 else 'WEAK — W not clearly harmonic'})")
    # Of the strongest connections, what fraction are harmonic?
    thr = np.percentile(W[off], 99)
    strong = (W > thr) & off
    frac_h = harm[strong].mean()
    print(f"     of top-1% strongest |W| edges, {frac_h:.1%} are "
          f"integer-ratio pairs (chance ~{harm.mean():.1%})")


def d2_failure_modes(slug: str) -> None:
    p = STATE / f"{slug}.parquet"
    if not p.exists():
        print(f"[D2] no parquet for {slug}")
        return
    frames = _extract_voices_over_track(p)
    counts = [len(fr) for fr in frames]
    print(f"\n[D2] {slug}: {len(frames)} feature frames")
    print(f"     voice count: mean={np.mean(counts):.1f} "
          f"median={np.median(counts):.0f} max={max(counts)} min={min(counts)}")
    # ID churn / lifetime
    id_frames: dict[int, int] = defaultdict(int)
    id_freqs: dict[int, list[float]] = defaultdict(list)
    id_amps: dict[int, list[float]] = defaultdict(list)
    for fr in frames:
        for v in fr:
            id_frames[v["id"]] += 1
            id_freqs[v["id"]].append(v["center_freq"])
            id_amps[v["id"]].append(v["amp"])
    n_ids = len(id_frames)
    mean_life = np.mean(list(id_frames.values()))
    print(f"     distinct IDs={n_ids}  mean lifetime={mean_life:.1f} frames "
          f"({mean_life/5:.1f}s at 5Hz)")
    # frequency jitter for persistent IDs (>= 5 frames)
    jit = []
    for vid, fs in id_freqs.items():
        if len(fs) >= 5:
            fs = np.array(fs)
            # semitone std around median
            semi = 12 * np.log2(fs / np.median(fs))
            jit.append(semi.std())
    if jit:
        print(f"     center-freq jitter (persistent IDs): "
              f"median={np.median(jit):.2f} semitones, "
              f"p90={np.percentile(jit,90):.2f} st  "
              f"({(np.array(jit)>1).mean():.0%} jump >1 st)")
    # Are extra voices harmonics? Inspect the densest frame.
    densest = max(range(len(frames)), key=lambda i: len(frames[i]))
    fr = sorted(frames[densest], key=lambda v: -v["amp"])
    print(f"     densest frame #{densest}: {len(fr)} voices")
    fund = [v for v in fr[:3]]
    n_harm_of_louder = 0
    for v in fr:
        for u in fr:
            if u is v or u["amp"] <= v["amp"]:
                continue
            r = max(v["center_freq"], u["center_freq"]) / max(
                1e-9, min(v["center_freq"], u["center_freq"]))
            d = np.min(np.abs(np.log2(r) - HARM_LOGS)) * 12
            if d <= 0.7:
                n_harm_of_louder += 1
                break
    print(f"     of {len(fr)} voices in densest frame, {n_harm_of_louder} "
          f"are within 0.7st of an integer ratio to a LOUDER voice "
          f"(candidate harmonic children)")
    print("     top voices (freq Hz / amp):  " + "  ".join(
        f"{v['center_freq']:.0f}/{v['amp']:.3f}" for v in fr[:8]))


if __name__ == "__main__":
    slug = sys.argv[1] if len(sys.argv) > 1 else "four_tet_angel_echoes"
    d1_w_is_harmonic(slug)
    d2_failure_modes(slug)
