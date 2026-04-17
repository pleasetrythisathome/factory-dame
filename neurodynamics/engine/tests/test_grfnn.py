"""Unit tests for the GrFNN oscillator bank (core NRT math).

These tests check the oscillator's dynamical behavior against properties
derivable from the equation, not against an external reference implementation.
Failure here means the canonical form is mis-coded.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurodynamics.grfnn import (
    GrFNN,
    GrFNNParams,
    channel_to_oscillator_weights,
)


def _run_steps(net: GrFNN, drive: np.ndarray) -> np.ndarray:
    """Run drive (T, n_osc) through net, returning per-step |z| (T, n_osc)."""
    out = np.zeros_like(drive, dtype=np.float64).real
    out = np.abs(np.zeros((drive.shape[0], net.n), dtype=np.complex128))
    for i in range(drive.shape[0]):
        net.step(drive[i].astype(np.complex128))
        out[i] = np.abs(net.z)
    return out


def _make_net(alpha: float = -0.05, beta: float = -1.0, input_gain: float = 1.0,
              low: float = 1.0, high: float = 10.0, n: int = 20,
              dt: float = 0.001) -> GrFNN:
    params = GrFNNParams(
        alpha=alpha, beta1=beta, beta2=beta,
        delta1=0.0, delta2=0.0, epsilon=1.0, input_gain=input_gain,
    )
    return GrFNN(n_oscillators=n, low_hz=low, high_hz=high, dt=dt, params=params)


class TestCanonicalDynamics:
    def test_undriven_decay_when_subcritical(self):
        """alpha < 0 with no input → |z| decays from initial noise toward 0."""
        net = _make_net(alpha=-0.5, n=10, dt=0.001)
        steps = 2000
        no_drive = np.zeros((steps, net.n), dtype=np.complex128)
        amps = _run_steps(net, no_drive)
        # Initial noise is ~1e-3; after decay should be far smaller still.
        assert amps[-1].max() < amps[0].max()
        assert amps[-1].max() < 1e-3

    def test_undriven_limit_cycle_when_supercritical(self):
        """alpha > 0 with no input → spontaneous limit cycle at predictable
        amplitude.

        For the canonical Hopf form with both cubic and quintic terms
        (beta1 = beta2 = -1, eps = 1), the fixed-point equation
            alpha + beta1*|z|^2 + eps*beta2*|z|^4 / (1 - eps*|z|^2) = 0
        simplifies with x=|z|^2 to x = alpha/(alpha+1).
        """
        alpha, beta = 0.5, -1.0
        net = _make_net(alpha=alpha, beta=beta, n=5, dt=0.0005,
                        low=5.0, high=5.0)
        net.z[:] = 0.05  # kick the limit cycle
        steps = 30000   # 15 s @ dt=0.0005; >> time constant 1/alpha = 2 s
        no_drive = np.zeros((steps, net.n), dtype=np.complex128)
        amps = _run_steps(net, no_drive)
        expected_amp = float(np.sqrt(alpha / (alpha + 1.0)))
        final = amps[-500:].mean(axis=0)
        assert np.all(final > 0.8 * expected_amp), f"final={final}, exp={expected_amp}"
        assert np.all(final < 1.2 * expected_amp), f"final={final}, exp={expected_amp}"

    def test_driven_resonance_at_natural_freq(self):
        """A sinusoidal drive at the oscillator's natural frequency produces
        larger |z| than the same-amplitude drive at a far-off frequency."""
        fs = 1000.0
        dt = 1.0 / fs
        natural_hz = 5.0
        net_on = _make_net(alpha=-0.05, beta=-1.0, input_gain=3.0,
                           low=natural_hz, high=natural_hz, n=1, dt=dt)
        net_off = _make_net(alpha=-0.05, beta=-1.0, input_gain=3.0,
                            low=natural_hz, high=natural_hz, n=1, dt=dt)
        steps = int(fs * 4.0)
        t = np.arange(steps) / fs
        on_drive = np.cos(2 * np.pi * natural_hz * t)[:, None]
        off_drive = np.cos(2 * np.pi * (natural_hz * 3) * t)[:, None]
        amps_on = _run_steps(net_on, on_drive.astype(np.complex128))
        amps_off = _run_steps(net_off, off_drive.astype(np.complex128))
        assert amps_on[-500:].mean() > 3 * amps_off[-500:].mean()

    def test_amplitude_stays_below_pole(self):
        """The integrator's clamp must keep |z| strictly under 1/sqrt(eps),
        even under heavy drive that would otherwise blow through the quintic
        pole to NaN/inf."""
        net = _make_net(alpha=-0.05, beta=-1.0, input_gain=1000.0,
                        n=5, dt=0.001)
        steps = 5000
        strong = np.ones((steps, net.n), dtype=np.complex128) * 100.0
        amps = _run_steps(net, strong)
        assert np.isfinite(amps).all()
        # epsilon = 1 → clamp at 0.98
        assert amps.max() < 0.9801


class TestPhantomMask:
    def test_phantom_when_amp_high_and_drive_low(self):
        net = _make_net(n=3, dt=0.001)
        net.z[:] = 0.5 + 0j  # force amplitude above threshold
        net.last_input_mag[:] = 0.0  # no drive
        mask = net.phantom_mask(amp_thresh=0.1, drive_thresh=0.02)
        assert mask.all()

    def test_no_phantom_when_drive_high(self):
        net = _make_net(n=3, dt=0.001)
        net.z[:] = 0.5 + 0j
        net.last_input_mag[:] = 1.0  # strong drive — not phantom
        mask = net.phantom_mask(amp_thresh=0.1, drive_thresh=0.02)
        assert not mask.any()

    def test_no_phantom_when_amp_low(self):
        net = _make_net(n=3, dt=0.001)
        net.z[:] = 0.01 + 0j  # below amp threshold
        net.last_input_mag[:] = 0.0
        mask = net.phantom_mask(amp_thresh=0.1, drive_thresh=0.02)
        assert not mask.any()


class TestFrequencySpacing:
    def test_log_spacing(self):
        net = _make_net(low=1.0, high=100.0, n=3, dt=0.001)
        # geomspace: ratio between consecutive is constant.
        ratios = net.f[1:] / net.f[:-1]
        assert np.allclose(ratios, ratios[0])

    def test_endpoints_match_requested(self):
        net = _make_net(low=2.0, high=50.0, n=10, dt=0.001)
        assert net.f[0] == pytest.approx(2.0)
        assert net.f[-1] == pytest.approx(50.0)


class TestChannelWeights:
    def test_row_sums_to_one(self):
        channels = np.array([100.0, 200.0, 400.0, 800.0])
        osc = np.array([100.0, 300.0, 1000.0])
        W = channel_to_oscillator_weights(channels, osc)
        assert np.allclose(W.sum(axis=1), 1.0)

    def test_closest_channel_dominates(self):
        channels = np.array([100.0, 200.0, 400.0, 800.0])
        osc = np.array([210.0])  # right next to channel 1 (200)
        W = channel_to_oscillator_weights(channels, osc, sharpness=6.0)
        assert W[0].argmax() == 1  # the 200 Hz channel should dominate


class TestStepMany:
    """Verify step_many (batched JIT) produces the same end state as
    calling step() sample-by-sample — up to float tolerance. This
    guards against regressions when the JIT loop is modified."""

    def _params(self, noise_amp=0.0):
        return GrFNNParams(
            alpha=-0.05, beta1=-1.0, beta2=-1.0,
            delta1=0.0, delta2=0.0, epsilon=1.0,
            input_gain=1.0,
        ), noise_amp

    def _make_pair(self, *, hebbian, delay_tau=0.0, noise_amp=0.0):
        p, noise = self._params(noise_amp=noise_amp)
        a = GrFNN(
            n_oscillators=20, low_hz=1.0, high_hz=100.0, dt=0.001,
            params=p, hebbian=hebbian,
            learn_rate=0.1 if hebbian else 0.0,
            weight_decay=0.01 if hebbian else 0.0,
            delay_tau=delay_tau, delay_gain=0.1 if delay_tau else 0.0,
            noise_amp=noise, noise_seed=42,
        )
        b = GrFNN(
            n_oscillators=20, low_hz=1.0, high_hz=100.0, dt=0.001,
            params=p, hebbian=hebbian,
            learn_rate=0.1 if hebbian else 0.0,
            weight_decay=0.01 if hebbian else 0.0,
            delay_tau=delay_tau, delay_gain=0.1 if delay_tau else 0.0,
            noise_amp=noise, noise_seed=42,
        )
        return a, b

    def test_step_vs_step_many_no_hebbian(self):
        """Pure nonlinear dynamics — step_many matches step() exactly
        (no Hebbian timing difference)."""
        a, b = self._make_pair(hebbian=False)
        t = np.arange(200) / 1000.0
        xs = (0.3 * np.sin(2 * np.pi * 10.0 * t)[:, None]
              * np.ones(20)).astype(np.complex128)
        for i in range(len(xs)):
            a.step(xs[i].copy())
        b.step_many(xs.copy())
        np.testing.assert_allclose(a.z, b.z, atol=1e-9)

    def test_step_vs_step_many_with_hebbian(self):
        """With Hebbian, step_many updates W once per batch rather
        than once per sample — so the final state drifts slightly
        from per-step over long runs. Short batches (here 50 samples)
        should stay within a loose tolerance."""
        a, b = self._make_pair(hebbian=True)
        t = np.arange(50) / 1000.0
        xs = (0.3 * np.sin(2 * np.pi * 10.0 * t)[:, None]
              * np.ones(20)).astype(np.complex128)
        for i in range(len(xs)):
            a.step(xs[i].copy())
        b.step_many(xs.copy())
        # End state should be close, not bit-identical, due to
        # batched Hebbian.
        np.testing.assert_allclose(a.z, b.z, atol=1e-3)

    def test_step_many_accepts_single_sample(self):
        """A 1-row batch is a valid input."""
        p, _ = self._params()
        g = GrFNN(n_oscillators=5, low_hz=1.0, high_hz=10.0, dt=0.01,
                   params=p)
        xs = np.ones((1, 5), dtype=np.complex128) * 0.1
        g.step_many(xs)
        assert g.z.shape == (5,)
