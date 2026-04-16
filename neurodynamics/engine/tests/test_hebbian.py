"""Tests for Hebbian plasticity in the GrFNN.

Reference: Large, "A Dynamical Systems Approach to Musical Tonality" (2011)
and the Harding/Kim/Large 2025 Musical Neurodynamics Perspective, Fig. 2c.

Hebbian plasticity in an oscillator network: complex connection weights
W_ij between oscillators grow when osc_i and osc_j are both active with
a stable phase relationship, and decay when they are not. Learned weights
form the attractor landscape that produces tonal hierarchies, implied
harmony, and learned rhythmic expectations.

Our Hebbian rule (simplified form from Large 2011):

    dW_ij/dt = -lambda_w * W_ij + kappa_w * z_i * conj(z_j)

where lambda_w is the decay rate and kappa_w is the learning rate.
|W_ij| grows with sustained co-activation; arg(W_ij) settles to the stable
phase offset between the two oscillators.
"""

from __future__ import annotations

import numpy as np

from neurodynamics.grfnn import GrFNN, GrFNNParams


def _drive_at(net: GrFNN, freq: float, amp: float, duration_s: float,
              phase: float = 0.0) -> None:
    """Run the network for duration_s driven by a sinusoid at freq
    (broadcast to every oscillator)."""
    fs = 1.0 / net.dt
    t = np.arange(int(fs * duration_s)) / fs
    sig = amp * np.cos(2 * np.pi * freq * t + phase)
    for v in sig:
        net.step(np.full(net.n, v, dtype=np.complex128))


class TestHebbianAllocation:
    def test_off_by_default(self):
        """Existing networks stay backward-compatible — no W unless enabled."""
        net = GrFNN(n_oscillators=3, low_hz=1.0, high_hz=5.0, dt=0.001,
                    params=GrFNNParams())
        assert net.W is None
        assert not net.hebbian_enabled

    def test_enabled_allocates_weight_matrix(self):
        net = GrFNN(
            n_oscillators=4, low_hz=1.0, high_hz=5.0, dt=0.001,
            params=GrFNNParams(),
            hebbian=True, learn_rate=0.1, weight_decay=0.01,
        )
        assert net.W is not None
        assert net.W.shape == (4, 4)
        assert net.W.dtype == np.complex128
        # Weights start at zero.
        assert np.allclose(net.W, 0.0)

    def test_runs_without_regressing_non_hebbian_behavior(self):
        """A GrFNN with hebbian=True but zero-coupling (learn_rate=0) should
        produce the same trajectory as an unenabled one under identical drive."""
        params = GrFNNParams(alpha=-0.1, input_gain=1.0)
        kwargs = dict(n_oscillators=3, low_hz=5.0, high_hz=5.0, dt=0.001,
                      params=params)
        a = GrFNN(**kwargs)
        b = GrFNN(**kwargs, hebbian=True, learn_rate=0.0, weight_decay=0.0)
        for _ in range(500):
            drive = np.full(3, 0.1, dtype=np.complex128)
            a.step(drive)
            b.step(drive)
        np.testing.assert_allclose(a.z, b.z, rtol=1e-10, atol=1e-12)


class TestHebbianLearning:
    def test_weights_grow_under_coactivation(self):
        """Two oscillators driven simultaneously at their natural frequency
        should develop a non-zero mutual connection."""
        net = GrFNN(
            n_oscillators=2, low_hz=5.0, high_hz=5.0, dt=0.001,
            params=GrFNNParams(alpha=-0.1, input_gain=2.0),
            hebbian=True, learn_rate=2.0, weight_decay=0.01,
        )
        _drive_at(net, freq=5.0, amp=0.3, duration_s=4.0)
        # Off-diagonal weight magnitudes should be clearly above initial zero.
        off_diag = np.abs(net.W[0, 1]) + np.abs(net.W[1, 0])
        assert off_diag > 0.01, (
            f"expected weights to grow under co-activation, got {off_diag}"
        )

    def test_weights_decay_when_oscillators_quiet(self):
        """With no oscillator activity, W should decay at rate weight_decay.
        We isolate this by preloading W and then stepping with z forced to
        zero each step — so the only active term is the -lambda * W decay."""
        net = GrFNN(
            n_oscillators=2, low_hz=5.0, high_hz=5.0, dt=0.001,
            params=GrFNNParams(alpha=-0.1),
            hebbian=True, learn_rate=0.0, weight_decay=5.0,
        )
        net.W[0, 1] = 1.0 + 0j
        net.W[1, 0] = 1.0 + 0j
        initial = np.abs(net.W[0, 1])
        # Force z to zero each step so no learning term is active.
        for _ in range(int(2.0 / net.dt)):
            net.z[:] = 0.0
            net.step(np.zeros(2, dtype=np.complex128))
        # weight_decay=5 => time constant 0.2s; 2s is 10 time constants.
        after = np.abs(net.W[0, 1])
        assert after < 0.05 * initial, (
            f"expected decay; initial={initial}, after={after}"
        )

    def test_weights_are_bounded(self):
        """Even under extreme drive, Hebbian weights must not diverge."""
        net = GrFNN(
            n_oscillators=3, low_hz=5.0, high_hz=5.0, dt=0.001,
            params=GrFNNParams(alpha=-0.05, input_gain=10.0),
            hebbian=True, learn_rate=10.0, weight_decay=0.5,
        )
        _drive_at(net, freq=5.0, amp=1.0, duration_s=6.0)
        assert np.isfinite(net.W).all()
        # With |z| saturated near 1 and decay=0.5, learn=10, the fixed point
        # for |W| is |z|^2 * kappa / lambda = 1 * 10 / 0.5 = 20. Accept a
        # little overshoot — the integrator isn't at strict fixed-point yet.
        assert np.abs(net.W).max() < 25.0


class TestHebbianCoupling:
    def test_pipeline_persists_weights_when_enabled(
        self, pulse_train_120bpm, temp_audio_file, minimal_config, tmp_path,
    ):
        """End-to-end: a config with Hebbian enabled on the rhythm layer
        produces a <stem>.weights.npz file with learned non-zero W."""
        from neurodynamics.run import run
        audio, sr = pulse_train_120bpm
        audio_file = temp_audio_file(audio, sr, "pulse.wav")
        cfg = minimal_config(
            str(audio_file.name),
            overrides={"rhythm_grfnn": {"hebbian": {
                "enabled": True, "learn_rate": 1.0, "weight_decay": 0.1,
            }}},
        )
        out = tmp_path / "state.parquet"
        run(cfg, audio_override=audio_file, output_override=out)

        weights = tmp_path / "state.weights.npz"
        assert weights.exists(), "Hebbian-enabled run must emit .weights.npz"
        npz = np.load(weights)
        assert "rhythm_W" in npz.files
        W = npz["rhythm_W"]
        assert W.shape[0] == W.shape[1] == 30
        # Something must have been learned from the 2 Hz pulse train.
        assert np.abs(W).max() > 1e-4

    def test_weights_influence_dynamics(self):
        """A preloaded Hebbian weight matrix should perturb the unforced
        trajectory. With no weights, oscillators are independent; with
        weights, activity in one affects the other."""
        alpha = -0.1
        params = GrFNNParams(alpha=alpha, input_gain=1.0)
        kwargs = dict(n_oscillators=2, low_hz=5.0, high_hz=5.0, dt=0.001,
                      params=params)
        a = GrFNN(**kwargs, hebbian=True, learn_rate=0.0, weight_decay=0.0)
        b = GrFNN(**kwargs, hebbian=True, learn_rate=0.0, weight_decay=0.0)
        # Inject initial state: osc 0 active, osc 1 at rest.
        a.z[:] = np.array([0.5 + 0j, 0.0 + 0j])
        b.z[:] = np.array([0.5 + 0j, 0.0 + 0j])
        # b has a non-zero coupling: osc 0 drives osc 1.
        b.W[1, 0] = 0.3 + 0j
        for _ in range(300):
            a.step(np.zeros(2, dtype=np.complex128))
            b.step(np.zeros(2, dtype=np.complex128))
        # In a, osc 1 stays near 0 (independent oscillators, initial condition 0).
        # In b, osc 1 should have picked up amplitude from coupling to osc 0.
        assert np.abs(b.z[1]) > 0.01
        assert np.abs(b.z[1]) > 5 * np.abs(a.z[1])
