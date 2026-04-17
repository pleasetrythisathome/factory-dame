"""Per-stage benchmark of the engine hot paths.

Runs a synthetic 60 s audio fixture through a stripped-down version
of the pipeline, timing each stage (cochlea, pitch RK4, rhythm RK4,
Hebbian update). Emits a breakdown so optimization effort targets
the actual bottleneck, not a guessed one.

Usage:
    uv run python -m tests.benchmark_engine

Not a pytest target — run as a script.
"""

from __future__ import annotations

import time

import numpy as np

from neurodynamics.cochlea import GammatoneFilterbank
from neurodynamics.grfnn import GrFNN, GrFNNParams, channel_to_oscillator_weights


def _pitch_params():
    return GrFNNParams(
        alpha=-0.05, beta1=-1.0, beta2=-1.0,
        delta1=0.0, delta2=0.0, epsilon=1.0, input_gain=2.0,
    )


def _rhythm_params():
    return GrFNNParams(
        alpha=-0.02, beta1=-1.0, beta2=-1.0,
        delta1=0.0, delta2=0.0, epsilon=1.0, input_gain=80.0,
    )


def main() -> None:
    # Synthetic audio: 60 s at 16 kHz — tone + noise burst, doesn't
    # matter musically, just representative load.
    fs = 16000
    duration = 60.0
    n = int(fs * duration)
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    audio = (0.3 * np.sin(2 * np.pi * 440.0 * t)
             + 0.05 * rng.standard_normal(n)).astype(np.float32)

    print(f"Benchmarking {duration:.0f} s of audio at {fs} Hz "
          f"({n} samples)")

    # Cochlea
    t0 = time.perf_counter()
    fb = GammatoneFilterbank(n_channels=64, low_hz=30.0, high_hz=4000.0, fs=fs)
    env = fb.envelope(audio)
    pitch_bands = fb.filter(audio)
    cochlea_elapsed = time.perf_counter() - t0
    print(f"  cochlea (64 ch, env + filter):     "
          f"{cochlea_elapsed:6.2f} s  ({duration/cochlea_elapsed:5.2f}x realtime)")

    # Pitch GrFNN — the audio-rate hot loop
    pitch = GrFNN(
        n_oscillators=100, low_hz=30.0, high_hz=4000.0,
        dt=1.0 / fs, params=_pitch_params(),
        hebbian=True, learn_rate=0.2, weight_decay=0.02,
    )
    W_pitch = channel_to_oscillator_weights(fb.fc, pitch.f)
    # Precompute the per-sample input vectors once to isolate the
    # GrFNN cost from input preparation.
    drives = (W_pitch @ pitch_bands.astype(np.float64)).T.astype(np.complex128)
    t0 = time.perf_counter()
    for i in range(n):
        pitch.step(drives[i])
    pitch_elapsed = time.perf_counter() - t0
    print(f"  pitch GrFNN step (100 osc, 16 kHz): "
          f"{pitch_elapsed:6.2f} s  ({duration/pitch_elapsed:5.2f}x realtime)")

    # Same load via the batched API — one JIT call for the whole run
    pitch2 = GrFNN(
        n_oscillators=100, low_hz=30.0, high_hz=4000.0,
        dt=1.0 / fs, params=_pitch_params(),
        hebbian=True, learn_rate=0.2, weight_decay=0.02,
    )
    t0 = time.perf_counter()
    pitch2.step_many(drives.copy())
    pitch_many_elapsed = time.perf_counter() - t0
    print(f"  pitch GrFNN step_many (batch full): "
          f"{pitch_many_elapsed:6.2f} s  "
          f"({duration/pitch_many_elapsed:5.2f}x realtime)")

    # Rhythm GrFNN
    rhythm_dt = 0.002  # 500 Hz
    rhythm_fs = 1.0 / rhythm_dt
    n_rhythm_steps = int(duration * rhythm_fs)
    idx = (np.arange(n_rhythm_steps) * (fs * rhythm_dt)).astype(np.int64)
    idx = np.clip(idx, 0, len(env.sum(axis=0)) - 1)
    rhythm_drive = env.sum(axis=0)
    rhythm_drive = np.diff(rhythm_drive, prepend=rhythm_drive[0])
    rhythm_drive = np.maximum(rhythm_drive, 0.0)
    rhythm_drive /= rhythm_drive.max() + 1e-9
    rhythm_drive_stepped = rhythm_drive[idx]
    rhythm = GrFNN(
        n_oscillators=50, low_hz=0.5, high_hz=10.0,
        dt=rhythm_dt, params=_rhythm_params(),
        hebbian=True, learn_rate=0.5, weight_decay=0.05,
    )
    t0 = time.perf_counter()
    for i in range(n_rhythm_steps):
        drive = np.full(rhythm.n, rhythm_drive_stepped[i], dtype=np.complex128)
        rhythm.step(drive)
    rhythm_elapsed = time.perf_counter() - t0
    print(f"  rhythm GrFNN (50 osc, 500 Hz):     "
          f"{rhythm_elapsed:6.2f} s  ({duration/rhythm_elapsed:5.2f}x realtime)")

    total = cochlea_elapsed + pitch_elapsed + rhythm_elapsed
    print(f"  total:                             "
          f"{total:6.2f} s  ({duration/total:5.2f}x realtime)")
    print()
    print("Proportion of time:")
    print(f"  cochlea:  {cochlea_elapsed/total*100:5.1f}%")
    print(f"  pitch:    {pitch_elapsed/total*100:5.1f}%")
    print(f"  rhythm:   {rhythm_elapsed/total*100:5.1f}%")
    print()
    if duration / total < 1.0:
        print("Engine is slower than realtime — live mode blocked "
              f"({duration/total:.2f}x).")
    else:
        print(f"Engine runs at {duration/total:.2f}x realtime.")


if __name__ == "__main__":
    main()
