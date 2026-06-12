"""Parametric multi-instrument arrangements with EXACT ground truth.

Each builder returns ``(audio, ReferenceAnnotation)``: we choose every
note (onset, offset, fundamental) and the instrument timbre, so the
reference voices are exact by construction. Tonal parts are rendered as
harmonic stacks with ADSR envelopes — rich enough that the Lerud 2014
cascade generates real harmonic enrichment, so the W-community merge is
actually exercised — and percussion as shaped noise bursts (role
"percussive": present in the audio to tempt the extractor, but NOT
counted as a tonal voice, so spurious voices on the kick cost precision).

Pipeline mirrors synth_ground_truth: ``generate`` writes a wav and runs
``nd-run`` to produce a parquet; the test rebuilds the reference from the
builder (deterministic) and scores the extraction against it. Audio and
parquets are gitignored/regenerable; the ground truth lives in code.

Usage:
    cd neurodynamics/engine
    uv run python -m tests.arrangements            # generate all
    uv run python -m tests.arrangements --only duet
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from tests.reference import Note, ReferenceVoice, ReferenceAnnotation

ENGINE_DIR = Path(__file__).resolve().parent.parent
WAV_DIR = ENGINE_DIR / "test_audio" / "arrangements"
STATE_DIR = ENGINE_DIR / "output"
FS = 16000
PEAK = 0.85


# ── pitch + synth helpers ─────────────────────────────────────────

def midi_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


def _adsr(n: int, fs: int, a=0.02, d=0.08, s=0.7, r=0.12) -> np.ndarray:
    env = np.zeros(n, dtype=np.float64)
    na, nd, nr = int(a * fs), int(d * fs), int(r * fs)
    ns = max(n - na - nd - nr, 0)
    i = 0
    if na:
        env[i:i + na] = np.linspace(0, 1, na); i += na
    if nd:
        env[i:i + nd] = np.linspace(1, s, nd); i += nd
    if ns:
        env[i:i + ns] = s; i += ns
    if nr and i < n:
        env[i:i + nr] = np.linspace(s, 0, min(nr, n - i))
    return env


def _render_note(freq: float, dur_s: float, *, fs: int = FS,
                 n_harmonics: int = 5, falloff: float = 0.7,
                 vibrato_hz: float = 0.0, vibrato_cents: float = 0.0,
                 adsr=(0.02, 0.08, 0.7, 0.12)) -> np.ndarray:
    """One note as a harmonic stack with ADSR (and optional vibrato)."""
    n = int(dur_s * fs)
    t = np.arange(n) / fs
    # Per-sample instantaneous frequency. MUST be a length-n array even
    # without vibrato — a scalar here makes np.cumsum return a single
    # element, collapsing every harmonic to DC (silent-but-for-a-step).
    if vibrato_hz > 0 and vibrato_cents > 0:
        fmod = freq * 2.0 ** ((vibrato_cents / 1200.0)
                              * np.sin(2 * np.pi * vibrato_hz * t))
    else:
        fmod = np.full(n, float(freq))
    sig = np.zeros(n, dtype=np.float64)
    for h in range(1, n_harmonics + 1):
        # cumulative-phase integration so vibrato tracks correctly
        phase = 2 * np.pi * np.cumsum(fmod * h) / fs
        sig += (falloff ** (h - 1)) * np.sin(phase)
    return sig * _adsr(n, fs, *adsr)


def _render_part(notes: list[tuple[float, float, float]], total_s: float, *,
                 fs: int = FS, gain: float = 1.0, **note_kw) -> np.ndarray:
    """Render a monophonic part. notes = [(onset_s, dur_s, freq_hz)]."""
    out = np.zeros(int(total_s * fs), dtype=np.float64)
    for onset, dur, freq in notes:
        seg = _render_note(freq, dur, fs=fs, **note_kw) * gain
        i0 = int(onset * fs)
        i1 = min(i0 + len(seg), len(out))
        out[i0:i1] += seg[:i1 - i0]
    return out


def _kick(times: list[float], total_s: float, *, fs: int = FS,
          gain: float = 0.9, seed: int = 0) -> np.ndarray:
    """Percussive kick: a fast pitch-dropping sine + click. Role is
    percussive, so it is NOT a tonal reference voice — but it IS in the
    audio, so the extractor is tempted to make a voice of it."""
    rng = np.random.default_rng(seed)
    out = np.zeros(int(total_s * fs), dtype=np.float64)
    for tk in times:
        nlen = int(0.12 * fs)
        tt = np.arange(nlen) / fs
        f = 120.0 * np.exp(-tt * 35.0) + 45.0
        env = np.exp(-tt * 22.0)
        body = np.sin(2 * np.pi * np.cumsum(f) / fs) * env
        click = rng.standard_normal(nlen) * np.exp(-tt * 120.0) * 0.3
        seg = (body + click) * gain
        i0 = int(tk * fs)
        i1 = min(i0 + nlen, len(out))
        out[i0:i1] += seg[:i1 - i0]
    return out


def _voice(label: str, notes: list[tuple[float, float, float]],
           role: str = "tonal") -> ReferenceVoice:
    return ReferenceVoice(
        label=label, role=role,
        notes=[Note(onset_s=o, offset_s=o + d, freq_hz=f) for o, d, f in notes],
    )


def _finalize(tracks: list[np.ndarray], voices: list[ReferenceVoice],
              total_s: float) -> tuple[np.ndarray, ReferenceAnnotation]:
    mix = np.zeros(int(total_s * FS), dtype=np.float64)
    for tr in tracks:
        m = min(len(mix), len(tr))
        mix[:m] += tr[:m]
    peak = np.max(np.abs(mix)) + 1e-9
    audio = (mix / peak * PEAK).astype(np.float32)
    return audio, ReferenceAnnotation(voices=voices, duration_s=total_s,
                                      sample_rate=FS)


# ── arrangements ──────────────────────────────────────────────────
# Each returns (audio float32, ReferenceAnnotation). Note frequencies via
# midi_hz; MIDI 60 = C4 = 261.6 Hz.

def arr_duet() -> tuple[np.ndarray, ReferenceAnnotation]:
    """Two independent melodic lines (a 2-voice duet). The lines move in
    contrary motion so no instant is a unison; always exactly 2 voices."""
    total = 8.0
    upper = [(0, 1.9, midi_hz(72)), (2, 1.9, midi_hz(74)),
             (4, 1.9, midi_hz(76)), (6, 1.9, midi_hz(77))]   # C5 D5 E5 F5
    lower = [(0, 1.9, midi_hz(60)), (2, 1.9, midi_hz(59)),
             (4, 1.9, midi_hz(57)), (6, 1.9, midi_hz(55))]   # C4 B3 A3 G3
    t_up = _render_part(upper, total, gain=0.9, n_harmonics=4,
                        vibrato_hz=5, vibrato_cents=12)
    t_lo = _render_part(lower, total, gain=1.0, n_harmonics=6)
    return _finalize([t_up, t_lo],
                     [_voice("upper", upper), _voice("lower", lower)], total)


def arr_bass_and_lead() -> tuple[np.ndarray, ReferenceAnnotation]:
    """A walking bass + a lead melody = 2 tonal voices, far apart in
    register (octaves between them — the case where the old extractor
    would fold the lead into the bass's harmonics)."""
    total = 8.0
    bass = [(0, 0.95, midi_hz(36)), (1, 0.95, midi_hz(43)),
            (2, 0.95, midi_hz(41)), (3, 0.95, midi_hz(36)),
            (4, 0.95, midi_hz(36)), (5, 0.95, midi_hz(43)),
            (6, 0.95, midi_hz(38)), (7, 0.95, midi_hz(36))]   # C2..
    lead = [(0.5, 1.4, midi_hz(67)), (2, 1.9, midi_hz(72)),
            (4.5, 1.4, midi_hz(69)), (6, 1.9, midi_hz(64))]   # G4 C5 A4 E4
    t_b = _render_part(bass, total, gain=1.0, n_harmonics=7, falloff=0.75)
    t_l = _render_part(lead, total, gain=0.7, n_harmonics=4,
                       vibrato_hz=5.5, vibrato_cents=15)
    return _finalize([t_b, t_l],
                     [_voice("bass", bass), _voice("lead", lead)], total)


def arr_triad_held() -> tuple[np.ndarray, ReferenceAnnotation]:
    """A single sustained major triad held for the clip = 3 tonal voices
    at a 4:5:6 ratio. The chord must resolve into THREE notes, not collapse
    to one (its implied fundamental) nor inflate beyond three."""
    total = 6.0
    c = [(0.2, 5.6, midi_hz(60))]   # C4
    e = [(0.2, 5.6, midi_hz(64))]   # E4
    g = [(0.2, 5.6, midi_hz(67))]   # G4
    tracks = [_render_part(p, total, gain=0.9, n_harmonics=5,
                           adsr=(0.05, 0.2, 0.8, 0.3)) for p in (c, e, g)]
    return _finalize(tracks,
                     [_voice("C", c), _voice("E", e), _voice("G", g)], total)


def arr_bass_pad_lead() -> tuple[np.ndarray, ReferenceAnnotation]:
    """Bass (1) + a sustained two-note pad (2) + a lead (1) = 4 tonal
    voices, plus a 4-on-the-floor kick (percussive, not counted). The
    medium-density realistic case."""
    total = 8.0
    bass = [(b, 1.95, midi_hz(40 if (b // 2) % 2 == 0 else 38))
            for b in (0, 2, 4, 6)]                  # E2 / D2 alternating
    pad_a = [(0, 7.8, midi_hz(64))]                 # E4
    pad_b = [(0, 7.8, midi_hz(71))]                 # B4
    lead = [(1, 1.4, midi_hz(76)), (3, 1.4, midi_hz(74)),
            (5, 1.4, midi_hz(72)), (7, 0.9, midi_hz(69))]
    kick_t = [i * 0.5 for i in range(16)]           # 120 BPM
    t_bass = _render_part(bass, total, gain=1.0, n_harmonics=7, falloff=0.78)
    t_pa = _render_part(pad_a, total, gain=0.55, n_harmonics=4,
                        adsr=(0.3, 0.3, 0.85, 0.5))
    t_pb = _render_part(pad_b, total, gain=0.5, n_harmonics=4,
                        adsr=(0.3, 0.3, 0.85, 0.5))
    t_lead = _render_part(lead, total, gain=0.6, n_harmonics=4,
                          vibrato_hz=5.5, vibrato_cents=14)
    t_kick = _kick(kick_t, total, gain=0.8)
    voices = [_voice("bass", bass), _voice("pad_low", pad_a),
              _voice("pad_high", pad_b), _voice("lead", lead),
              _voice("kick", kick_t and [], role="percussive")]
    # percussive voice carries no notes (counted as percussive presence only)
    return _finalize([t_bass, t_pa, t_pb, t_lead, t_kick], voices, total)


def arr_unison_then_split() -> tuple[np.ndarray, ReferenceAnnotation]:
    """Two voices in unison for the first half, then splitting to an
    octave apart. Tests that unison reads as ~1 and the split as 2 —
    the dynamic-voice-count behaviour."""
    total = 6.0
    a = [(0, 2.9, midi_hz(60)), (3, 2.9, midi_hz(60))]     # C4, C4
    b = [(0, 2.9, midi_hz(60)), (3, 2.9, midi_hz(72))]     # C4, C5
    t_a = _render_part(a, total, gain=0.9, n_harmonics=5)
    t_b = _render_part(b, total, gain=0.9, n_harmonics=5)
    return _finalize([t_a, t_b], [_voice("a", a), _voice("b", b)], total)


ALL_ARRANGEMENTS = {
    "duet": arr_duet,
    "bass_and_lead": arr_bass_and_lead,
    "triad_held": arr_triad_held,
    "bass_pad_lead": arr_bass_pad_lead,
    "unison_then_split": arr_unison_then_split,
}


# ── generation (wav + nd-run parquet) ─────────────────────────────

def generate(slug: str, *, force: bool = False) -> Path:
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    audio, _ref = ALL_ARRANGEMENTS[slug]()
    wav = WAV_DIR / f"arr_{slug}.wav"
    parquet = STATE_DIR / f"arr_{slug}.parquet"
    if force or not wav.exists():
        sf.write(wav, audio, FS, subtype="FLOAT")
    if force or not parquet.exists():
        print(f"[{slug}] nd-run …")
        subprocess.run([
            "uv", "run", "nd-run", "--audio", str(wav),
            "--output", str(parquet), "--no-osc",
        ], cwd=ENGINE_DIR, check=True,
           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return parquet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    slugs = ([args.only] if args.only else list(ALL_ARRANGEMENTS))
    for slug in slugs:
        p = generate(slug, force=args.force)
        print(f"[{slug}] -> {p}")


if __name__ == "__main__":
    main()
