"""Fast feedback loop for tuning the internal coupling kernel.

Runs ONE short audio fixture through the engine twice — once with
coupling disabled, once with it enabled at the configured gain — and
prints how much the pitch-bank state actually differs.

Use this BEFORE the full multi-track baseline capture to confirm a
parameter change actually moved the engine. Each invocation runs in
~5-10 seconds vs the full baseline's 10+ minutes.

Usage:
    uv run python -m tests.quick_kernel_diff
    uv run python -m tests.quick_kernel_diff --gain 0.5
    uv run python -m tests.quick_kernel_diff --audio test_audio/triggers/single.overtone.wav
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


def run_with_gain(cfg: dict, audio: np.ndarray, sr: int,
                   coupling_gain: float) -> dict:
    """Run LiveEngine with a specific coupling gain and return state stats."""
    cfg2 = deepcopy(cfg)
    cfg2.setdefault("pitch_grfnn", {}).setdefault("coupling", {})
    cfg2["pitch_grfnn"]["coupling"]["enabled"] = (coupling_gain != 0)
    cfg2["pitch_grfnn"]["coupling"]["gain"] = coupling_gain
    engine = LiveEngine(cfg2)
    engine.process(audio[:4096])  # JIT warm-up
    chunk_size = 1024
    z_history = []
    voice_counts = []
    for i in range(0, len(audio) - chunk_size, chunk_size):
        engine.process(audio[i:i + chunk_size])
        z_history.append(engine.pitch.z.copy())
        voice_counts.append(len(engine._voice_state.active_voices))
    z_arr = np.array(z_history)  # (T, n_pitch)
    return {
        "z_history": z_arr,
        "amps_2d": np.abs(z_arr),
        "voice_counts": voice_counts,
        "freqs": engine.pitch.f.copy(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    ap.add_argument("--gain", type=float, default=None,
                    help="Override coupling gain. If unset, uses config.")
    args = ap.parse_args()

    config = ENGINE_DIR / "config.toml"
    with open(config, "rb") as f:
        cfg = tomllib.load(f)
    a, sr = sf.read(str(args.audio))
    if a.ndim > 1:
        a = a.mean(axis=1)
    a = a.astype(np.float32)
    print(f"audio: {args.audio.name}  {len(a)/sr:.2f}s @ {sr}Hz")

    target_gain = args.gain if args.gain is not None else float(
        cfg.get("pitch_grfnn", {}).get("coupling", {}).get("gain", 0.0)
    )
    print(f"comparing: coupling OFF (gain=0) vs ON (gain={target_gain})")

    off = run_with_gain(cfg, a, sr, coupling_gain=0.0)
    on = run_with_gain(cfg, a, sr, coupling_gain=target_gain)

    # Per-frame max amp comparison
    max_off = off["amps_2d"].max(axis=1)
    max_on = on["amps_2d"].max(axis=1)
    diff_per_frame = np.abs(off["z_history"] - on["z_history"]).sum(axis=1)
    print(f"\n{'metric':<30}  {'off':>8}  {'on':>8}  {'delta':>10}")
    print("-" * 64)
    print(f"{'mean_peak_amp':<30}  {max_off.mean():>8.4f}  {max_on.mean():>8.4f}  "
          f"{(max_on.mean()-max_off.mean()):>+10.4f}")
    print(f"{'max_peak_amp':<30}  {max_off.max():>8.4f}  {max_on.max():>8.4f}  "
          f"{(max_on.max()-max_off.max()):>+10.4f}")
    print(f"{'mean_voice_count':<30}  "
          f"{np.mean(off['voice_counts']):>8.2f}  "
          f"{np.mean(on['voice_counts']):>8.2f}  "
          f"{(np.mean(on['voice_counts'])-np.mean(off['voice_counts'])):>+10.2f}")
    print(f"{'mean_state_distance':<30}  {'-':>8}  {'-':>8}  "
          f"{diff_per_frame.mean():>+10.4f}")

    # If audio is phantom_fundamental, look at the f=220 bin specifically
    if "phantom" in args.audio.name:
        idx_f = int(np.argmin(np.abs(off["freqs"] - 220.0)))
        idx_2f = int(np.argmin(np.abs(off["freqs"] - 440.0)))
        idx_3f = int(np.argmin(np.abs(off["freqs"] - 660.0)))
        print(f"\nPhantom fundamental targets:")
        print(f"  bin {idx_f} @ {off['freqs'][idx_f]:.1f}Hz (target f=220):")
        print(f"    off: max amp = {off['amps_2d'][:, idx_f].max():.4f}")
        print(f"    on:  max amp = {on['amps_2d'][:, idx_f].max():.4f}")
        print(f"  bin {idx_2f} @ {off['freqs'][idx_2f]:.1f}Hz (driven 2f=440):")
        print(f"    off: max amp = {off['amps_2d'][:, idx_2f].max():.4f}")
        print(f"    on:  max amp = {on['amps_2d'][:, idx_2f].max():.4f}")
        print(f"  bin {idx_3f} @ {off['freqs'][idx_3f]:.1f}Hz (driven 3f=660):")
        print(f"    off: max amp = {off['amps_2d'][:, idx_3f].max():.4f}")
        print(f"    on:  max amp = {on['amps_2d'][:, idx_3f].max():.4f}")


if __name__ == "__main__":
    main()
