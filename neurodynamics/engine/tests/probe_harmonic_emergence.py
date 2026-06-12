"""Probe: does enabling canonical input + integer-ratio coupling on
the existing pitch bank produce harmonic emergence?

Drives an oscillator bank with a 220 Hz pure tone projected to a
single oscillator. Checks whether neighboring oscillators at 440,
660, 880 Hz get excited (harmonic emergence) vs only the 220 Hz
oscillator (no emergence).

Three configurations tested:
  A. Linear input, no coupling (baseline)
  B. Canonical input only
  C. Canonical input + coupling kernel at moderate gain

If C produces measurable amplitude at the harmonic positions while
A does not, the existing pitch bank can do harmonic emergence with
the right config — no need for a separate cochlear layer.
"""

from __future__ import annotations

import numpy as np

from neurodynamics.grfnn import (
    GrFNN, GrFNNParams, build_integer_ratio_coupling,
)


def _build_bank(*, alpha: float, beta1: float, beta2: float,
                epsilon: float, input_gain: float,
                nonlinear_input: bool, coupling_gain: float,
                freqs: np.ndarray, dt: float,
                noise_amp: float = 1e-4) -> GrFNN:
    params = GrFNNParams(
        alpha=alpha, beta1=beta1, beta2=beta2,
        delta1=0.0, delta2=0.0, epsilon=epsilon,
        input_gain=input_gain,
    )
    coupling_kernel = None
    if coupling_gain > 0:
        coupling_kernel = build_integer_ratio_coupling(
            freqs, max_octave_distance=2.5, tolerance_semitones=0.4,
        )
    return GrFNN(
        n_oscillators=len(freqs), low_hz=freqs[0], high_hz=freqs[-1],
        dt=dt, params=params, freqs=freqs,
        nonlinear_input=nonlinear_input,
        coupling_kernel=coupling_kernel,
        coupling_gain=coupling_gain,
        noise_amp=noise_amp, noise_seed=42,
    )


def _drive_220hz(amp: float, duration: float, dt: float,
                  freqs: np.ndarray) -> np.ndarray:
    """Drive osc 0 (220 Hz target) with a pure 220 Hz sinusoid.

    Returns shape (n_samples, n_osc) complex. Only osc 0 gets drive
    (full projection); others get nothing.
    """
    n_samples = int(duration / dt)
    t = np.arange(n_samples) * dt
    f0 = freqs[0]  # 220 Hz
    sinusoid = amp * np.exp(1j * 2 * np.pi * f0 * t)
    xs = np.zeros((n_samples, len(freqs)), dtype=np.complex128)
    xs[:, 0] = sinusoid
    return xs


def _run(config: dict, freqs: np.ndarray) -> dict:
    dt = 1.0 / 16000.0
    bank = _build_bank(freqs=freqs, dt=dt, **config)
    # Seed each oscillator at small amplitude — gives canonical
    # input form material to bootstrap from.
    bank.z[:] = 0.02 * np.exp(1j * 2 * np.pi * np.arange(len(freqs)) / len(freqs))
    # Audio-realistic drive amplitude (cochlear output of normalized
    # audio at typical levels)
    xs = _drive_220hz(amp=0.3, duration=4.0, dt=dt, freqs=freqs)
    # Run in chunks and capture amp history
    chunk = 1024
    amp_history = []
    for start in range(0, xs.shape[0], chunk):
        bank.step_many(xs[start:start + chunk].copy())
        amp_history.append(np.abs(bank.z.copy()))
    return {
        "amps_final": np.abs(bank.z.copy()),
        "amp_history": np.array(amp_history),
        "freqs": freqs,
    }


def main():
    # 220 Hz fundamental + harmonics at 440 (2:1), 660 (3:1), 880 (4:1)
    freqs = np.array([220.0, 440.0, 660.0, 880.0])

    # Use a realistic audio-level drive (amp=0.3 — corresponds to a
    # moderately loud tone in cochlear output) and existing pitch
    # config defaults.
    configs = {
        "A: linear, no coupling (current default)": dict(
            alpha=-0.05, beta1=-1.0, beta2=-1.0, epsilon=1.0,
            input_gain=0.5,
            nonlinear_input=False, coupling_gain=0.0,
        ),
        "B: canonical only": dict(
            alpha=-0.05, beta1=-1.0, beta2=-1.0, epsilon=1.0,
            input_gain=0.5,
            nonlinear_input=True, coupling_gain=0.0,
        ),
        "C: canonical + coupling gain=0.1": dict(
            alpha=-0.05, beta1=-1.0, beta2=-1.0, epsilon=1.0,
            input_gain=0.5,
            nonlinear_input=True, coupling_gain=0.1,
        ),
        "D: canonical + coupling gain=0.5": dict(
            alpha=-0.05, beta1=-1.0, beta2=-1.0, epsilon=1.0,
            input_gain=0.5,
            nonlinear_input=True, coupling_gain=0.5,
        ),
        "E: canonical + coupling gain=1.0": dict(
            alpha=-0.05, beta1=-1.0, beta2=-1.0, epsilon=1.0,
            input_gain=0.5,
            nonlinear_input=True, coupling_gain=1.0,
        ),
        "F: canonical + coupling gain=2.0": dict(
            alpha=-0.05, beta1=-1.0, beta2=-1.0, epsilon=1.0,
            input_gain=0.5,
            nonlinear_input=True, coupling_gain=2.0,
        ),
    }

    print(f"{'Config':<48} {'220 Hz':>9} {'440 Hz':>9} "
          f"{'660 Hz':>9} {'880 Hz':>9}")
    print("-" * 92)
    for name, cfg in configs.items():
        res = _run(cfg, freqs)
        amps = res["amps_final"]
        print(f"{name:<48} {amps[0]:>9.4f} {amps[1]:>9.4f} "
              f"{amps[2]:>9.4f} {amps[3]:>9.4f}")

    print()
    print("Interpretation:")
    print("  - 220 Hz amplitude: how well the bank tracks the drive")
    print("  - 440/660/880 amplitude: how much harmonic energy emerges")
    print("  - If only 220 has amplitude: NO harmonic emergence")
    print("  - If 440/660/880 also have amplitude: harmonic emergence works")


if __name__ == "__main__":
    main()
