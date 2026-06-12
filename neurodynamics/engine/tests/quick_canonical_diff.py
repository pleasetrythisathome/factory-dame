"""Fast A/B smoke test for canonical nonlinear input form.

Runs one short audio fixture through the engine twice — linear vs
canonical input — and prints how much the pitch-bank state differs.

This validates that:
1. Enabling nonlinear_input=True doesn't crash on real audio
2. The two paths produce meaningfully different engine state
3. Canonical produces additional energy at harmonic positions

Use before the full Tier A audit. Runs in ~5-10 s vs hours for full.

Usage:
    .venv/bin/python tests/quick_canonical_diff.py
    .venv/bin/python tests/quick_canonical_diff.py --audio test_audio/triggers/single.overtone.wav
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import soundfile as sf
import tomllib

from neurodynamics.live import LiveEngine

ENGINE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_AUDIO = ENGINE_DIR / "test_audio" / "triggers" / "phantom_fundamental.overtone.wav"


def run_with_canonical(cfg: dict, audio: np.ndarray, sr: int,
                       nonlinear_input: bool) -> dict:
    cfg2 = deepcopy(cfg)
    cfg2.setdefault("pitch_grfnn", {})["nonlinear_input"] = nonlinear_input
    engine = LiveEngine(cfg2)
    engine.process(audio[:4096])  # JIT warm-up
    chunk_size = 1024
    z_history = []
    voice_counts = []
    for i in range(0, len(audio) - chunk_size, chunk_size):
        engine.process(audio[i:i + chunk_size])
        z_history.append(engine.pitch.z.copy())
        voice_counts.append(len(engine._voice_state.active_voices))
    z_arr = np.array(z_history)
    return {
        "amps_2d": np.abs(z_arr),
        "voice_counts": voice_counts,
        "freqs": engine.pitch.f.copy(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    args = ap.parse_args()

    config = ENGINE_DIR / "config.toml"
    with open(config, "rb") as f:
        cfg = tomllib.load(f)
    a, sr = sf.read(str(args.audio))
    if a.ndim > 1:
        a = a.mean(axis=1)
    a = a.astype(np.float32)
    print(f"audio: {args.audio.name}  {len(a)/sr:.2f}s @ {sr}Hz")

    print("running linear (nonlinear_input=False)...")
    off = run_with_canonical(cfg, a, sr, nonlinear_input=False)
    print("running canonical (nonlinear_input=True)...")
    on = run_with_canonical(cfg, a, sr, nonlinear_input=True)

    print()
    print(f"voice count avg: linear {np.mean(off['voice_counts']):.2f}  "
          f"canonical {np.mean(on['voice_counts']):.2f}")
    amp_off = off["amps_2d"]
    amp_on = on["amps_2d"]
    print(f"mean |z| over time/osc: linear {amp_off.mean():.4f}  "
          f"canonical {amp_on.mean():.4f}")
    print(f"max |z|: linear {amp_off.max():.4f}  canonical {amp_on.max():.4f}")
    print(f"active osc fraction (|z|>0.01): "
          f"linear {(amp_off > 0.01).mean():.3f}  "
          f"canonical {(amp_on > 0.01).mean():.3f}")
    # Relative L2 difference
    rel_diff = np.linalg.norm(amp_on - amp_off) / np.linalg.norm(amp_off)
    print(f"relative L2 diff in |z| trajectory: {rel_diff:.3f}")
    if rel_diff < 0.01:
        print("\n⚠ canonical and linear produce essentially identical state")
    else:
        print(f"\n✓ canonical produces meaningfully different state "
              f"(rel diff {rel_diff*100:.1f}%)")


if __name__ == "__main__":
    main()
