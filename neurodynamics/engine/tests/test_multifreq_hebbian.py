"""Phase 5 validation: multi-frequency Hebbian learns integer-ratio
community structure in the W matrix.

Per Kim & Large 2021 Eq. 26 + Large 2016 Eq. A2, the multi-frequency
Hebbian rule

    dW_ij/dt = -λ W_ij + κ · P(z_i) · conj(P(z_j))

with P(z) = z/(1-√ε z) develops strong connections between
oscillator pairs whose frequencies are in small-integer ratios.

The test trains a bank with oscillators at 220, 440, 660, 880 Hz
(integer multiples of a 220 Hz fundamental) plus a "distractor" at
311 Hz (irrational ratio with 220). Drives osc 0 (220 Hz) with a
strong tone. After training, W[0, 1] (220→440, 1:2 ratio) and
W[0, 2] (220→660, 1:3 ratio) should be substantially stronger than
W[0, 4] (220→311, irrational).

This is the substrate for W-based voice extraction: harmonic
stacks form communities, voices = communities.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurodynamics.grfnn import GrFNN, GrFNNParams


def _train_with_drives(freqs: np.ndarray, drive_freqs: list[float],
                        drive_amp: float, duration_s: float,
                        learn_rate: float, weight_decay: float,
                        alpha: float = 0.0,
                        nonlinear_input: bool = True,
                        input_gain: float = 0.5,
                        dt: float = 1.0 / 16000.0,
                        noise_amp: float = 1e-4) -> GrFNN:
    """Train a bank with sinusoidal drives at multiple frequencies.

    Each drive_freqs entry is mapped to the oscillator with the
    closest natural frequency. Same-amplitude excitation per drive.
    """
    params = GrFNNParams(
        alpha=alpha, beta1=-1.0, beta2=-1.0,
        delta1=0.0, delta2=0.0, epsilon=1.0,
        input_gain=input_gain,
    )
    net = GrFNN(
        n_oscillators=len(freqs), low_hz=freqs[0], high_hz=freqs[-1],
        dt=dt, params=params, freqs=freqs,
        hebbian=True, learn_rate=learn_rate, weight_decay=weight_decay,
        nonlinear_input=nonlinear_input,
        noise_amp=noise_amp, noise_seed=7,
    )
    n_samples = int(duration_s / dt)
    t = np.arange(n_samples) * dt
    xs = np.zeros((n_samples, len(freqs)), dtype=np.complex128)
    for f_drive in drive_freqs:
        sinusoid = drive_amp * np.exp(1j * 2 * np.pi * f_drive * t)
        idx = int(np.argmin(np.abs(freqs - f_drive)))
        xs[:, idx] += sinusoid
    chunk = 256
    for start in range(0, n_samples, chunk):
        end = min(start + chunk, n_samples)
        net.step_many(xs[start:end].copy())
    return net


class TestMultiFreqHebbian:
    def test_w_learns_harmonic_stack_community(self):
        """Train on a 220 Hz fundamental + harmonics at 440, 660 Hz
        AND a distractor at 311 Hz. All four drives at same
        amplitude.

        After training, W[fundamental, harmonic] connections should
        be MUCH stronger than W[fundamental, distractor] because
        the 220↔440 (1:2) and 220↔660 (1:3) outer-product monomials
        have stationary DC components, while 220↔311 (irrational
        ratio) oscillates and time-averages to zero.
        """
        # idx:   0      1      2      3
        # freq: 220    440    660    311 (distractor)
        freqs = np.array([220.0, 440.0, 660.0, 311.0])

        net = _train_with_drives(
            freqs, drive_freqs=[220.0, 440.0, 660.0, 311.0],
            drive_amp=0.3, duration_s=3.0,
            learn_rate=2.0, weight_decay=0.3,
            alpha=0.0, nonlinear_input=True, input_gain=0.5,
        )
        W = np.abs(net.W)
        w_1to2 = W[0, 1]
        w_1to3 = W[0, 2]
        w_distractor = W[0, 3]

        print(f"W[220→440] (1:2): {w_1to2:.4f}")
        print(f"W[220→660] (1:3): {w_1to3:.4f}")
        print(f"W[220→311] (distractor): {w_distractor:.4f}")

        # Harmonic connections at integer ratios should dominate.
        # Per K&L 2021, the lowest-order resonant monomial is
        # stationary for k:m integer ratios and oscillating for
        # irrational ratios — the latter time-averages to zero.
        assert w_1to2 > 2 * w_distractor, (
            f"1:2 ({w_1to2:.4f}) not above distractor "
            f"({w_distractor:.4f}) by enough margin"
        )
        assert w_1to3 > w_distractor, (
            f"1:3 ({w_1to3:.4f}) not above distractor "
            f"({w_distractor:.4f})"
        )
        # 1:2 is stronger than 1:3 per stability hierarchy
        assert w_1to2 >= 0.8 * w_1to3, (
            f"1:2 expected ≥ 1:3 in stability "
            f"(w_1to2={w_1to2:.4f}, w_1to3={w_1to3:.4f})"
        )

    def test_distractor_alone_doesnt_learn(self):
        """Drive ONLY the distractor (311 Hz). Verify no connection
        develops between 311 and 220 (or 311 and 440). Validates
        that learning requires CO-ACTIVATION at integer ratios.
        """
        freqs = np.array([220.0, 440.0, 311.0])
        net = _train_with_drives(
            freqs, drive_freqs=[311.0],
            drive_amp=0.3, duration_s=2.0,
            learn_rate=2.0, weight_decay=0.3,
            alpha=0.0, nonlinear_input=True, input_gain=0.5,
        )
        W = np.abs(net.W)
        # Distractor isn't at integer ratio with any other osc,
        # AND nothing else is driven, so all distractor connections
        # should be near zero.
        assert W[2, 0] < 0.01 and W[2, 1] < 0.01, (
            f"distractor connections grew unexpectedly: "
            f"W[311→220]={W[2,0]:.4f}, W[311→440]={W[2,1]:.4f}"
        )
