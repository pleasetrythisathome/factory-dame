"""Unit tests for the cochlear front-end (gammatone filterbank + envelope)."""

from __future__ import annotations

import numpy as np
import pytest

from neurodynamics.cochlea import GammatoneFilterbank, erb_space


class TestErbSpacing:
    def test_endpoints(self):
        fc = erb_space(100.0, 8000.0, 32)
        assert fc[0] == pytest.approx(100.0, rel=1e-6)
        assert fc[-1] == pytest.approx(8000.0, rel=1e-6)
        # erb_space returns frequencies high → low (matches Slaney's convention).
        assert fc[0] < fc[-1]

    def test_monotonic(self):
        fc = erb_space(50.0, 4000.0, 20)
        assert np.all(np.diff(fc) > 0)

    def test_log_like_density(self):
        """ERB spacing puts more channels at low frequencies. Verify that the
        ratio between adjacent channels shrinks as frequency increases."""
        fc = erb_space(50.0, 4000.0, 30)
        ratios = fc[1:] / fc[:-1]
        # High-end ratios should be smaller than low-end ratios.
        assert ratios[-1] < ratios[0]


class TestGammatoneFilterbank:
    def test_filter_output_shape(self):
        fs = 16000
        fb = GammatoneFilterbank(n_channels=8, low_hz=100.0, high_hz=4000.0, fs=fs)
        audio = np.random.default_rng(0).standard_normal(fs).astype(np.float32)
        out = fb.filter(audio)
        assert out.shape == (8, fs)

    def test_peak_response_at_center_freq(self):
        """Drive each channel with a pure tone at its center frequency;
        that channel should respond more strongly than off-center channels."""
        fs = 16000
        fb = GammatoneFilterbank(n_channels=16, low_hz=100.0, high_hz=4000.0, fs=fs)
        # Pick a center frequency not too close to the endpoints.
        target_idx = 8
        fc = fb.fc[target_idx]
        t = np.arange(fs) / fs
        tone = np.sin(2 * np.pi * fc * t).astype(np.float32)
        out = fb.filter(tone)
        # Skip the filter warm-up transient.
        rms = np.sqrt(np.mean(out[:, fs // 2:] ** 2, axis=1))
        assert rms.argmax() == target_idx

    def test_envelope_nonnegative(self):
        fs = 16000
        fb = GammatoneFilterbank(n_channels=4, low_hz=100.0, high_hz=2000.0, fs=fs)
        audio = np.random.default_rng(0).standard_normal(fs).astype(np.float32)
        env = fb.envelope(audio)
        assert (env >= 0).all()

    def test_envelope_smoother_than_signal(self):
        """Hilbert envelope has lower high-frequency energy than the raw band."""
        fs = 16000
        fb = GammatoneFilterbank(n_channels=4, low_hz=200.0, high_hz=2000.0, fs=fs)
        t = np.arange(fs) / fs
        # 500 Hz tone modulated at 10 Hz — envelope should follow the 10 Hz mod.
        audio = (np.sin(2 * np.pi * 500 * t)
                 * (0.5 + 0.5 * np.sin(2 * np.pi * 10 * t))).astype(np.float32)
        bands = fb.filter(audio)
        env = fb.envelope(audio)
        # Signal oscillates at 500 Hz (many zero crossings). Envelope oscillates
        # at 10 Hz (few crossings).
        def zero_crossings(x):
            return int(np.sum(np.diff(np.signbit(x - x.mean())) != 0))
        # Pick the most active band (around 500 Hz).
        band_idx = np.argmax(np.abs(bands).mean(axis=1))
        assert zero_crossings(bands[band_idx, fs // 2:]) \
               > 5 * zero_crossings(env[band_idx, fs // 2:])
