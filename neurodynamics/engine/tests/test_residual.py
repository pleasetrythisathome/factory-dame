"""Tests for per-oscillator prediction residual (surprise signal).

An oscillator "expects" the input magnitude that would exactly sustain its
current amplitude against damping. For a resonantly driven oscillator at
steady state, |input| ≈ |alpha| * |z|. The residual is how much the actual
input deviates from this expectation:

    residual = |input| - |alpha| * |z|

A positive residual means "more energy than needed to sustain" — typical of
a fresh onset, syncopated beat, or surprise event. A negative residual means
"less energy than the oscillator's current state would predict" — typical
of silence mid-ringing (the inner voice / phantom regime).

This is the single-scalar surprise/novelty signal that downstream consumers
(particle turbulence, tension-release synth) can read directly.
"""

from __future__ import annotations

import numpy as np

from neurodynamics.grfnn import GrFNN, GrFNNParams


class TestPredictionResidual:
    def test_exists_and_zero_initially(self):
        net = GrFNN(n_oscillators=3, low_hz=1.0, high_hz=5.0, dt=0.001,
                    params=GrFNNParams())
        assert hasattr(net, "last_residual")
        assert net.last_residual.shape == (3,)
        assert np.allclose(net.last_residual, 0.0)

    def test_negative_when_ringing_without_drive(self):
        """An oscillator with residual amplitude and zero input has
        |input| - |alpha|*|z| = 0 - |alpha|*|z| < 0. That's the 'expected
        sustenance was not delivered' signal — the phantom/inner-voice regime."""
        net = GrFNN(n_oscillators=1, low_hz=5.0, high_hz=5.0, dt=0.001,
                    params=GrFNNParams(alpha=-0.1, input_gain=1.0))
        net.z[0] = 0.5 + 0j  # non-trivial amplitude
        net.step(np.zeros(1, dtype=np.complex128))
        # |alpha| * |z| = 0.1 * 0.5 = 0.05. residual = 0 - 0.05 = -0.05.
        assert net.last_residual[0] < -0.01

    def test_positive_at_transient_onset(self):
        """A sudden input kick to an otherwise-quiet oscillator has a large
        positive residual — the surprise spike."""
        net = GrFNN(n_oscillators=1, low_hz=5.0, high_hz=5.0, dt=0.001,
                    params=GrFNNParams(alpha=-0.1, input_gain=1.0))
        # First: quiet. Initial noise in z produces a tiny residual of
        # order |alpha| * 1e-3 ≈ 1e-4.
        net.step(np.zeros(1, dtype=np.complex128))
        assert abs(net.last_residual[0]) < 1e-3
        # Now a big kick (far bigger than |alpha| * |z| ≈ 0.0001).
        net.step(np.array([0.8 + 0j]))
        assert net.last_residual[0] > 0.5

    def test_small_at_steady_state_resonance(self):
        """A sustained sinusoidal drive at natural frequency should reach
        steady state where |input| ≈ |alpha| * |z| and residual is small."""
        dt = 0.0005
        fs = 1.0 / dt
        f = 5.0
        net = GrFNN(n_oscillators=1, low_hz=f, high_hz=f, dt=dt,
                    params=GrFNNParams(alpha=-0.3, input_gain=1.0,
                                       beta1=-1.0, beta2=-1.0))
        # Run well past steady state (~ 5 time constants of 1/0.3 = 3.3 s).
        for i in range(int(fs * 15.0)):
            val = 0.3 * np.cos(2 * np.pi * f * i * dt)
            net.step(np.array([val + 0j]))
        # At steady state the oscillator's amplitude tracks the drive, and
        # the residual oscillates but its magnitude is small relative to
        # the drive amplitude.
        # Look at final residuals over one period.
        final_residuals = []
        for i in range(int(fs / f)):
            val = 0.3 * np.cos(2 * np.pi * f * (int(fs * 15.0) + i) * dt)
            net.step(np.array([val + 0j]))
            final_residuals.append(net.last_residual[0])
        peak_residual = max(abs(r) for r in final_residuals)
        # Drive peak is 0.3. Residual should be small compared to that.
        assert peak_residual < 0.25, (
            f"expected residual small at resonance, got {peak_residual}"
        )
