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
        """alpha < 0 with no input + perturbation → |z| decays toward 0.

        With zero-init (not random-noise init), we need to kick the
        oscillators off the origin to test decay behavior — otherwise
        they just sit at 0.
        """
        net = _make_net(alpha=-0.5, n=10, dt=0.001)
        # Seed a small perturbation that should decay below the
        # strict assertion threshold within the test's 2 s horizon.
        # For alpha=-0.5 time constant is 2 s, so 2 s = 1 e-fold
        # (|z| × 0.37). Start at 1e-3 so the final is < 4e-4.
        net.z = np.full(net.n, 1e-3 + 0.0j, dtype=np.complex128)
        steps = 2000
        no_drive = np.zeros((steps, net.n), dtype=np.complex128)
        amps = _run_steps(net, no_drive)
        assert amps[-1].max() < amps[0].max()
        assert amps[-1].max() < 1e-3


# ── Internal coupling kernel ──────────────────────────────────────

class TestCouplingKernel:
    """Tests for the integer-ratio internal coupling kernel D.

    Public domain (US 7,376,562, expired 2024-11-17). The kernel
    pre-wires oscillator pairs at small-integer frequency ratios so
    the network exhibits cross-resonances (phantom fundamentals,
    harmonic reinforcement) without requiring Hebbian learning."""

    def test_kernel_diagonal_is_zero(self):
        """An oscillator does not couple to itself via the kernel."""
        from neurodynamics.grfnn import build_integer_ratio_coupling
        freqs = np.geomspace(100.0, 1600.0, 50)
        C = build_integer_ratio_coupling(freqs)
        assert np.allclose(np.diag(C), 0)

    def test_kernel_connects_octaves(self):
        """Pairs at exact 2:1 and 1:2 ratios get non-zero weight."""
        from neurodynamics.grfnn import build_integer_ratio_coupling
        # Hand-pick freqs so 200:400 and 400:800 are exact octaves.
        freqs = np.array([100.0, 200.0, 400.0, 800.0])
        C = build_integer_ratio_coupling(freqs, max_octave_distance=4.0)
        # 200 ↔ 400 is 1 octave apart
        assert abs(C[1, 2]) > 0
        assert abs(C[2, 1]) > 0
        # 400 ↔ 800 too
        assert abs(C[2, 3]) > 0
        # 100 ↔ 200 too
        assert abs(C[0, 1]) > 0

    def test_kernel_skips_non_harmonic_pairs(self):
        """Pairs at unrelated ratios (e.g. ~tritone) get zero or
        tiny weight."""
        from neurodynamics.grfnn import build_integer_ratio_coupling
        # 440 (A4) and 622 (D#5) — tritone, ratio sqrt(2) ≈ 1.414, not
        # in our integer-ratio set
        freqs = np.array([440.0, 622.25])
        C = build_integer_ratio_coupling(
            freqs, tolerance_semitones=0.4, max_octave_distance=2.0
        )
        assert abs(C[0, 1]) == 0
        assert abs(C[1, 0]) == 0

    def test_kernel_respects_octave_distance(self):
        """Pairs more than max_octave_distance apart are not connected
        even if their ratio is harmonic."""
        from neurodynamics.grfnn import build_integer_ratio_coupling
        # 100 and 800 are 3 octaves apart — exact 8:1 ratio, but
        # outside max_octave_distance=2.0
        freqs = np.array([100.0, 800.0])
        C = build_integer_ratio_coupling(freqs, max_octave_distance=2.0)
        assert abs(C[0, 1]) == 0
        assert abs(C[1, 0]) == 0

    def test_engine_with_coupling_runs_stable(self):
        """Engine with coupling enabled doesn't blow up on a sustained
        sine input. Smoke test for the JIT integration path."""
        from neurodynamics.grfnn import (
            GrFNN, GrFNNParams, build_integer_ratio_coupling,
        )
        freqs = np.geomspace(100.0, 1600.0, 30)
        C = build_integer_ratio_coupling(freqs)
        net = GrFNN(
            n_oscillators=30, low_hz=100.0, high_hz=1600.0, dt=1e-4,
            params=GrFNNParams(alpha=-0.05, beta1=-1.0, beta2=-1.0,
                                epsilon=1.0, input_gain=0.5),
            freqs=freqs,
            coupling_kernel=C, coupling_gain=0.05,
        )
        # 1 s of constant drive at the 5th oscillator
        n_samples = 10000
        x = np.zeros((n_samples, 30), dtype=np.complex128)
        x[:, 5] = 0.1
        net.step_many(x)
        # |z| stays bounded
        assert np.all(np.abs(net.z) < 1.0)
        # The driven oscillator has reasonable amp
        assert np.abs(net.z[5]) > 0
        assert np.all(np.isfinite(net.z))

    def test_per_oscillator_tau_changes_dynamics(self):
        """With per_oscillator_tau=True, the (α, β, drive) terms scale
        by f_n. Effect: low-freq oscillators decay faster in real time
        than the uniform-τ default (where decay is the same wall-clock
        time for all freqs)."""
        from neurodynamics.grfnn import GrFNN, GrFNNParams
        freqs = np.array([100.0, 1000.0])
        params = GrFNNParams(alpha=-0.1, beta1=-1.0, beta2=-1.0,
                              epsilon=1.0, input_gain=0.5)
        # Same perturbation applied; decay observed.
        net_uniform = GrFNN(
            n_oscillators=2, low_hz=100.0, high_hz=1000.0, dt=1e-4,
            params=params, freqs=freqs,
            per_oscillator_tau=False,
        )
        net_uniform.z = np.array([0.1 + 0.0j, 0.1 + 0.0j],
                                  dtype=np.complex128)
        net_per = GrFNN(
            n_oscillators=2, low_hz=100.0, high_hz=1000.0, dt=1e-4,
            params=params, freqs=freqs,
            per_oscillator_tau=True, tau_reference_hz=1.0,
        )
        net_per.z = np.array([0.1 + 0.0j, 0.1 + 0.0j],
                              dtype=np.complex128)
        # 0.1 s of zero drive
        n = 1000
        x = np.zeros((n, 2), dtype=np.complex128)
        net_uniform.step_many(x.copy())
        net_per.step_many(x.copy())
        # Uniform τ: both oscillators decay at the same alpha rate
        # over real time. Per-osc τ: 1000 Hz osc decays 10× faster
        # than 100 Hz osc.
        amps_uniform = np.abs(net_uniform.z)
        amps_per = np.abs(net_per.z)
        # 1000 Hz amp under per-osc τ should be much smaller than
        # under uniform τ (faster decay).
        assert amps_per[1] < 0.5 * amps_uniform[1], (
            f"per-τ 1000Hz amp {amps_per[1]:.4f} not smaller than "
            f"uniform-τ {amps_uniform[1]:.4f}"
        )

    def test_engine_phantom_emerges_with_coupling(self):
        """A bank driven only at 2f and 3f develops a response at f
        via the 2:1 and 3:1 coupling — the phantom fundamental.

        Without coupling, the f bin should stay at zero for clean
        inputs at 2f and 3f. With coupling, it should pick up
        cross-resonance amplitude."""
        from neurodynamics.grfnn import (
            GrFNN, GrFNNParams, build_integer_ratio_coupling,
        )
        # Bank with bins exactly at 220 (A3), 440 (A4), 660 (E5)
        freqs = np.array([220.0, 440.0, 660.0])
        C = build_integer_ratio_coupling(freqs, max_octave_distance=4.0,
                                          tolerance_semitones=2.0)
        params = GrFNNParams(alpha=-0.05, beta1=-1.0, beta2=-1.0,
                              epsilon=1.0, input_gain=0.5)
        # Without coupling
        net_off = GrFNN(
            n_oscillators=3, low_hz=220.0, high_hz=660.0, dt=1e-4,
            params=params, freqs=freqs,
        )
        # With coupling
        net_on = GrFNN(
            n_oscillators=3, low_hz=220.0, high_hz=660.0, dt=1e-4,
            params=params, freqs=freqs,
            coupling_kernel=C, coupling_gain=0.2,
        )
        # 0.5s of drive at 440 and 660 — sinusoidal at the bank's
        # natural frequencies (since the input is the per-oscillator
        # complex input; we simulate strong drive on bins 1 and 2).
        n_samples = 5000
        t_arr = np.arange(n_samples) * 1e-4
        x = np.zeros((n_samples, 3), dtype=np.complex128)
        x[:, 1] = 0.2 * np.exp(1j * 2 * np.pi * 440.0 * t_arr)
        x[:, 2] = 0.2 * np.exp(1j * 2 * np.pi * 660.0 * t_arr)
        net_off.step_many(x.copy())
        net_on.step_many(x.copy())
        # Without coupling, bin 0 (220 Hz) gets no drive → ~0 amp
        assert np.abs(net_off.z[0]) < 1e-3
        # With coupling, bin 0 should pick up phantom response
        assert np.abs(net_on.z[0]) > 5 * np.abs(net_off.z[0]), (
            f"phantom not stronger: off={np.abs(net_off.z[0]):.4f} "
            f"on={np.abs(net_on.z[0]):.4f}"
        )

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
