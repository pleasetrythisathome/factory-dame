"""Stochastic noise in the GrFNN.

Adding a small Gaussian perturbation per step models intrinsic neural noise.
It lets oscillators spontaneously explore their attractor basins, enhances
phantom behavior in silence (the network "imagines" quietly), and breaks
pathological symmetric fixed points.
"""

from __future__ import annotations

import numpy as np

from neurodynamics.grfnn import GrFNN, GrFNNParams


class TestNoise:
    def test_off_by_default(self):
        net = GrFNN(n_oscillators=2, low_hz=5.0, high_hz=10.0, dt=0.001,
                    params=GrFNNParams())
        assert net.noise_amp == 0.0

    def test_zero_noise_is_deterministic(self):
        """Two nets with noise=0 produce identical trajectories."""
        params = GrFNNParams(alpha=-0.1, input_gain=1.0)
        kwargs = dict(n_oscillators=3, low_hz=5.0, high_hz=5.0, dt=0.001,
                      params=params)
        a = GrFNN(**kwargs, noise_amp=0.0)
        b = GrFNN(**kwargs, noise_amp=0.0)
        for _ in range(200):
            drive = np.full(3, 0.1, dtype=np.complex128)
            a.step(drive)
            b.step(drive)
        np.testing.assert_array_equal(a.z, b.z)

    def test_noise_breaks_determinism(self):
        """With noise > 0, two fresh nets diverge from the same input."""
        params = GrFNNParams(alpha=-0.1, input_gain=1.0)
        kwargs = dict(n_oscillators=3, low_hz=5.0, high_hz=5.0, dt=0.001,
                      params=params)
        a = GrFNN(**kwargs, noise_amp=0.05, noise_seed=1)
        b = GrFNN(**kwargs, noise_amp=0.05, noise_seed=2)
        for _ in range(200):
            net_input = np.full(3, 0.0, dtype=np.complex128)
            a.step(net_input)
            b.step(net_input)
        # The two trajectories must have diverged non-trivially.
        assert np.max(np.abs(a.z - b.z)) > 1e-4

    def test_same_seed_reproducible(self):
        """Same noise_seed on two fresh nets → identical trajectories."""
        params = GrFNNParams(alpha=-0.1, input_gain=1.0)
        kwargs = dict(n_oscillators=3, low_hz=5.0, high_hz=5.0, dt=0.001,
                      params=params)
        a = GrFNN(**kwargs, noise_amp=0.05, noise_seed=42)
        b = GrFNN(**kwargs, noise_amp=0.05, noise_seed=42)
        for _ in range(200):
            net_input = np.full(3, 0.0, dtype=np.complex128)
            a.step(net_input)
            b.step(net_input)
        np.testing.assert_array_equal(a.z, b.z)

    def test_noise_produces_nonzero_activity_in_silence(self):
        """With no drive and noise on, some oscillator should develop
        measurable amplitude above its initial noise floor."""
        net = GrFNN(
            n_oscillators=5, low_hz=1.0, high_hz=5.0, dt=0.001,
            params=GrFNNParams(alpha=-0.05, input_gain=0.0),
            noise_amp=0.02, noise_seed=0,
        )
        initial = np.abs(net.z).max()
        for _ in range(2000):
            net.step(np.zeros(5, dtype=np.complex128))
        final = np.abs(net.z).max()
        # Noise should keep oscillators above the seeded 1e-3 noise floor.
        assert final > max(10 * initial, 1e-3)
