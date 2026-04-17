"""Ground-truth test harness: generate synthetic audio with known
musical content, run the engine, report what the voice extractor
sees vs. what should be there.

Each fixture synthesizes a specific scenario (single tone, hi-hat
pulses, harmonic stack, two independent voices, glissando, chord
progression), writes a WAV + expected-voices JSON, runs the engine
via nd-run, and prints a diff between what came out and what should
have.

Intended for interactive validation, not a pytest gate — small
differences in engine behavior across revisions are easier to read
visually than to encode as strict assertions. But the JSON
annotations give a concrete "this is what I meant" anchor so the
diffs are interpretable.

Usage:
    uv run python -m tests.synth_ground_truth            # all fixtures
    uv run python -m tests.synth_ground_truth --only pulsed_tone
    uv run python -m tests.synth_ground_truth --screenshot
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from neurodynamics.perceptual import StateWindow
from neurodynamics.voices import VoiceState, extract_voices


ENGINE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = ENGINE_DIR / "test_audio" / "synthetic"
STATE_DIR = ENGINE_DIR / "output"

FS = 16000  # matches engine target rate

# Engine's pitch-bank input_gain (0.1 in the default config) + its
# small noise amp (0.01) means low-amplitude synthetic audio gets
# dominated by deterministic noise. Mastered music sits near full
# scale; synthesize at roughly the same peak so the drive actually
# excites oscillators.
PEAK_AMP = 0.85


def _normalize_to_peak(audio: np.ndarray, peak: float = PEAK_AMP) -> np.ndarray:
    m = float(np.max(np.abs(audio)))
    if m <= 0:
        return audio
    return (audio / m * peak).astype(np.float32)


# ── Fixture data model ────────────────────────────────────────────

@dataclass
class ExpectedVoice:
    """What we expect to see for one voice over the track."""

    freq_range_hz: tuple[float, float]
    when_active_s: tuple[float, float]
    amp_profile: str  # 'constant' | 'rising' | 'pulsed' | 'decaying'
    notes: str = ""


@dataclass
class Fixture:
    slug: str
    description: str
    duration_s: float
    expected_voices: list[ExpectedVoice] = field(default_factory=list)
    expected_simultaneous: tuple[int, int] = (0, 10)  # (min, max) at any instant


# ── Synth helpers ─────────────────────────────────────────────────

def _sine(freq_hz: float, duration_s: float,
           envelope: np.ndarray | None = None,
           fs: int = FS) -> np.ndarray:
    t = np.arange(int(duration_s * fs)) / fs
    s = np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
    if envelope is not None:
        if len(envelope) != len(s):
            envelope = np.interp(np.linspace(0, 1, len(s)),
                                  np.linspace(0, 1, len(envelope)),
                                  envelope)
        s = s * envelope.astype(np.float32)
    return s


def _adsr(duration_s: float, attack_s: float = 0.01,
          decay_s: float = 0.1, sustain: float = 0.7,
          release_s: float = 0.1, fs: int = FS) -> np.ndarray:
    n = int(duration_s * fs)
    a = int(attack_s * fs)
    d = int(decay_s * fs)
    r = int(release_s * fs)
    s = max(n - a - d - r, 0)
    env = np.zeros(n, dtype=np.float32)
    if a > 0:
        env[:a] = np.linspace(0, 1, a)
    if d > 0:
        env[a:a + d] = np.linspace(1, sustain, d)
    if s > 0:
        env[a + d:a + d + s] = sustain
    if r > 0:
        env[-r:] = np.linspace(sustain, 0, r)
    return env


def _pulsed(duration_s: float, period_s: float,
            attack_s: float = 0.005, decay_s: float = 0.05,
            fs: int = FS) -> np.ndarray:
    n = int(duration_s * fs)
    env = np.zeros(n, dtype=np.float32)
    pulse_n = int((attack_s + decay_s) * fs)
    pulse = np.concatenate([
        np.linspace(0, 1, max(1, int(attack_s * fs))),
        np.linspace(1, 0, max(1, int(decay_s * fs))),
    ]).astype(np.float32)
    period_samples = int(period_s * fs)
    for i in range(0, n, period_samples):
        end = min(i + len(pulse), n)
        env[i:end] += pulse[:end - i]
    return env


# ── Fixture definitions ───────────────────────────────────────────

def fx_single_tone() -> tuple[Fixture, np.ndarray]:
    fx = Fixture(
        slug="single_tone",
        description="5 s 440 Hz tone with gentle 2 Hz tremolo (±10% amp). "
                    "NRT likely produces a driven voice near 440 Hz AND "
                    "phantom subharmonics at 220, 110, ~55 Hz due to the "
                    "Hopf integer-division nonlinearity — the Large et al. "
                    "phantom-fundamental prediction.",
        duration_s=5.0,
        expected_voices=[ExpectedVoice(
            freq_range_hz=(420, 460),
            when_active_s=(0.5, 5.0),
            amp_profile="constant",
            notes="Primary driven voice. Expect additional phantom "
                  "voices at subharmonics — that's signature NRT, "
                  "not an extractor bug.",
        )],
        expected_simultaneous=(1, 5),  # primary + phantom subharmonics
    )
    t = np.arange(int(5.0 * FS)) / FS
    tremolo = 0.3 * (1.0 + 0.1 * np.sin(2 * np.pi * 2.0 * t))
    audio = tremolo.astype(np.float32) * _sine(440.0, 5.0)
    return fx, _normalize_to_peak(audio)


def fx_ramp_tone() -> tuple[Fixture, np.ndarray]:
    fx = Fixture(
        slug="ramp_tone",
        description="6 s 440 Hz tone ramping linearly 0 → 0.4 amp — "
                    "voice line should go thin → thick",
        duration_s=6.0,
        expected_voices=[ExpectedVoice(
            freq_range_hz=(420, 460),
            when_active_s=(2.0, 6.0),
            amp_profile="rising",
        )],
        expected_simultaneous=(0, 1),
    )
    env = np.linspace(0, 0.4, int(6.0 * FS), dtype=np.float32)
    audio = _sine(440.0, 6.0, envelope=env)
    return fx, _normalize_to_peak(audio)


def fx_pulsed_hihat() -> tuple[Fixture, np.ndarray]:
    fx = Fixture(
        slug="pulsed_hihat",
        description="6 s of brief 2 kHz pulses every 250 ms "
                    "(~120 BPM hi-hat) — voice line should pulse "
                    "thin → thick → thin with each hit",
        duration_s=6.0,
        expected_voices=[ExpectedVoice(
            freq_range_hz=(1900, 2100),
            when_active_s=(0.5, 6.0),
            amp_profile="pulsed",
            notes="Peak amp during each burst; near-zero between",
        )],
        expected_simultaneous=(0, 1),
    )
    env = _pulsed(6.0, period_s=0.25, attack_s=0.005, decay_s=0.06)
    audio = 0.5 * _sine(2000.0, 6.0, envelope=env)
    return fx, _normalize_to_peak(audio)


def fx_harmonic_stack() -> tuple[Fixture, np.ndarray]:
    fx = Fixture(
        slug="harmonic_stack",
        description="5 s fundamental + 3 harmonics (110, 220, 440, 880) "
                    "all on same envelope. Envelope-correlation clustering "
                    "should merge these into a single voice spanning multiple "
                    "pitch oscillators.",
        duration_s=5.0,
        expected_voices=[ExpectedVoice(
            freq_range_hz=(100, 900),
            when_active_s=(0.5, 5.0),
            amp_profile="rising",
            notes="center_freq amplitude-weighted across all 4 partials; "
                  "voice's oscillator_indices should span the fundamental "
                  "and upper harmonics.",
        )],
        expected_simultaneous=(1, 1),
    )
    env = np.linspace(0.05, 0.35, int(5.0 * FS), dtype=np.float32)
    audio = np.zeros(int(5.0 * FS), dtype=np.float32)
    for harmonic, weight in zip((110.0, 220.0, 440.0, 880.0),
                                 (1.0, 0.6, 0.4, 0.25)):
        audio = audio + weight * _sine(harmonic, 5.0, envelope=env)
    audio = audio / np.max(np.abs(audio)) * 0.5
    return fx, _normalize_to_peak(audio)


def fx_two_independent() -> tuple[Fixture, np.ndarray]:
    fx = Fixture(
        slug="two_independent",
        description="6 s: two independent voices at 330 Hz and 880 Hz "
                    "with uncorrelated envelopes (slow sine × 0.6 Hz and "
                    "fast sine × 1.3 Hz offset by π/2). Should cluster as "
                    "TWO distinct voices.",
        duration_s=6.0,
        expected_voices=[
            ExpectedVoice(
                freq_range_hz=(310, 350),
                when_active_s=(0.5, 6.0),
                amp_profile="rising",
            ),
            ExpectedVoice(
                freq_range_hz=(850, 910),
                when_active_s=(0.5, 6.0),
                amp_profile="rising",
            ),
        ],
        expected_simultaneous=(1, 2),
    )
    t = np.arange(int(6.0 * FS)) / FS
    env_a = (0.2 + 0.15 * np.sin(2 * np.pi * 0.6 * t)).astype(np.float32)
    env_b = (0.2 + 0.15 * np.sin(2 * np.pi * 1.3 * t + np.pi / 2)).astype(np.float32)
    audio = _sine(330.0, 6.0, envelope=env_a) + _sine(880.0, 6.0, envelope=env_b)
    audio = audio * 0.5
    return fx, _normalize_to_peak(audio)


def fx_glissando() -> tuple[Fixture, np.ndarray]:
    fx = Fixture(
        slug="glissando",
        description="5 s pitch sweep from 220 Hz → 880 Hz (2 octaves) "
                    "with gentle 2 Hz tremolo. Primary voice tracks the "
                    "sweep; phantom subharmonics trail it.",
        duration_s=5.0,
        expected_voices=[ExpectedVoice(
            freq_range_hz=(220, 880),
            when_active_s=(0.5, 5.0),
            amp_profile="constant",
            notes="Driven voice whose oscillator_indices should drift "
                  "from low to high. Voice ID ideally persists across "
                  "the sweep but may re-register if drift outruns the "
                  "match window.",
        )],
        expected_simultaneous=(1, 5),
    )
    n = int(5.0 * FS)
    t = np.arange(n) / FS
    freq = 220.0 * (880.0 / 220.0) ** (t / 5.0)
    phase = 2 * np.pi * np.cumsum(freq) / FS
    tremolo = 0.3 * (1.0 + 0.1 * np.sin(2 * np.pi * 2.0 * t))
    audio = (tremolo * np.sin(phase)).astype(np.float32)
    return fx, _normalize_to_peak(audio)


def fx_chord_progression() -> tuple[Fixture, np.ndarray]:
    fx = Fixture(
        slug="chord_progression",
        description="8 s, four 2-s chords: C major (C4 E4 G4), F major "
                    "(F4 A4 C5), G major (G4 B4 D5), C major again. "
                    "Each chord has 3 notes in unison envelope → should "
                    "collapse to 3 voices (or fewer if clustering merges "
                    "closely-spaced notes).",
        duration_s=8.0,
        expected_voices=[],  # too many across chord changes to annotate flatly
        expected_simultaneous=(2, 4),
    )
    chords = [
        (261.63, 329.63, 392.00),   # C4 E4 G4
        (349.23, 440.00, 523.25),   # F4 A4 C5
        (392.00, 493.88, 587.33),   # G4 B4 D5
        (261.63, 329.63, 392.00),   # C again
    ]
    audio = np.zeros(int(8.0 * FS), dtype=np.float32)
    for i, chord in enumerate(chords):
        env = _adsr(2.0, attack_s=0.05, decay_s=0.1,
                    sustain=0.3, release_s=0.2)
        start = i * 2 * FS
        chunk = np.zeros(int(2.0 * FS), dtype=np.float32)
        for freq in chord:
            chunk = chunk + _sine(freq, 2.0, envelope=env)
        chunk = chunk / max(np.max(np.abs(chunk)), 1e-9) * 0.5
        audio[start:start + len(chunk)] += chunk
    return fx, _normalize_to_peak(audio)


ALL_FIXTURES = [
    fx_single_tone, fx_ramp_tone, fx_pulsed_hihat, fx_harmonic_stack,
    fx_two_independent, fx_glissando, fx_chord_progression,
]


# ── Extraction + diff ─────────────────────────────────────────────

def run_engine(wav_path: Path, parquet_path: Path,
                config_path: Path, force: bool = False) -> None:
    if parquet_path.exists() and not force:
        return
    print(f"  [engine] processing {wav_path.name}")
    subprocess.run([
        "uv", "run", "nd-run",
        "--config", str(config_path),
        "--audio", str(wav_path),
        "--output", str(parquet_path),
    ], cwd=ENGINE_DIR, check=True,
       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def extract_voice_timeline(parquet_path: Path) -> list[dict]:
    t = pq.read_table(parquet_path)
    mask = [x == "pitch" for x in t.column("layer").to_pylist()]
    pitch = t.filter(mask)
    times = np.array(pitch.column("t").to_pylist(), dtype=np.float64)
    amps = np.array(pitch.column("amp").to_pylist(), dtype=np.float32)
    phases = np.array(pitch.column("phase").to_pylist(), dtype=np.float32)
    meta = t.schema.metadata or {}
    freq_str = meta.get(b"layer.pitch.f", b"").decode()
    freqs = np.array([float(x) for x in freq_str.split(",") if x])
    if len(times) < 2:
        return []
    snap_hz = 1.0 / (times[1] - times[0])
    feat_stride = max(1, int(round(snap_hz / 5.0)))
    feat_half = max(1, int(snap_hz * 1.25))
    state = VoiceState()
    timeline: list[dict] = []
    for i in range(0, len(times), feat_stride):
        lo = max(0, i - feat_half)
        hi = min(len(times), i + feat_half + 1)
        pitch_z = (amps[lo:hi].astype(np.complex128)
                   * np.exp(1j * phases[lo:hi].astype(np.complex128)))
        sw = StateWindow(
            pitch_z=pitch_z, pitch_freqs=freqs,
            rhythm_z=np.zeros(1, dtype=np.complex128),
            rhythm_freqs=np.array([1.0]),
            frame_hz=float(snap_hz),
        )
        state = extract_voices(sw, prev_state=state)
        timeline.append({
            "t": float(times[i]),
            "voices": [
                {
                    "id": v.id,
                    "center_freq": float(v.center_freq),
                    "amp": float(v.amp),
                    "n_osc": len(v.oscillator_indices),
                }
                for v in state.active_voices
            ],
        })
    return timeline


def summarize(fx: Fixture, timeline: list[dict]) -> str:
    counts = [len(t["voices"]) for t in timeline]
    if not counts:
        return "  [no snapshots]"
    mean_count = float(np.mean(counts))
    max_count = max(counts)
    # Distinct ids seen + per-id freq range
    id_freqs: dict[int, list[float]] = {}
    for t in timeline:
        for v in t["voices"]:
            id_freqs.setdefault(v["id"], []).append(v["center_freq"])
    lines = [
        f"  simultaneous voices:  mean={mean_count:.1f}  max={max_count}"
        f"  (expected {fx.expected_simultaneous[0]}-"
        f"{fx.expected_simultaneous[1]})",
        f"  distinct voice IDs:   {len(id_freqs)}",
    ]
    for vid in sorted(id_freqs)[:8]:
        freqs = id_freqs[vid]
        lines.append(
            f"    V{vid}: {len(freqs)} frames, "
            f"freq {min(freqs):.0f}-{max(freqs):.0f} Hz "
            f"(median {float(np.median(freqs)):.0f})"
        )
    if fx.expected_voices:
        lines.append("  expected:")
        for ev in fx.expected_voices:
            lines.append(
                f"    {ev.freq_range_hz[0]:.0f}-{ev.freq_range_hz[1]:.0f} Hz "
                f"active {ev.when_active_s[0]:.1f}-"
                f"{ev.when_active_s[1]:.1f}s  ({ev.amp_profile})"
            )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None,
                    help="Run only the fixture with this slug")
    ap.add_argument("--force", action="store_true",
                    help="Regenerate WAVs and parquets even if they exist")
    ap.add_argument("--screenshot", action="store_true",
                    help="Also render nd-view --screenshot-at for each fixture")
    ap.add_argument("--config", type=Path, default=ENGINE_DIR / "config.toml")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fixtures = ALL_FIXTURES
    if args.only:
        fixtures = [f for f in fixtures if f()[0].slug == args.only]
        if not fixtures:
            print(f"no fixture with slug {args.only!r}")
            return

    for factory in fixtures:
        fx, audio = factory()
        wav_path = OUT_DIR / f"{fx.slug}.wav"
        json_path = OUT_DIR / f"{fx.slug}.expected.json"
        parquet_path = STATE_DIR / f"{fx.slug}.parquet"

        print(f"\n=== {fx.slug} ===")
        print(f"  {fx.description}")

        if args.force or not wav_path.exists():
            sf.write(wav_path, audio, FS, subtype="FLOAT")
            json_path.write_text(json.dumps(asdict(fx), indent=2))

        run_engine(wav_path, parquet_path, args.config, force=args.force)

        timeline = extract_voice_timeline(parquet_path)
        print(summarize(fx, timeline))

        if args.screenshot:
            shot = OUT_DIR / f"{fx.slug}.png"
            subprocess.run([
                "uv", "run", "nd-view",
                "--config", str(args.config),
                "--state", str(parquet_path),
                "--audio", str(wav_path),
                "--screenshot-at", str(fx.duration_s * 0.75),
                "--screenshot-path", str(shot),
            ], cwd=ENGINE_DIR, check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  screenshot → {shot}")


if __name__ == "__main__":
    main()
