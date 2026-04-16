"""Tests for delay-coupled oscillator dynamics.

Transmission delays in the intra-layer coupling let a GrFNN produce "strong
anticipation" — the network's response can *lead* a periodic driver rather
than lag it (Stepp & Turvey 2010; Harding/Kim/Large 2025 Fig. 2d).

In the simplest form we implement: each oscillator receives a self-coupling
from its own past state z(t - tau):

    dz/dt = z*(drift) + x + delay_gain * z(t - tau)

For appropriate tau this produces phase lead, modeling the anticipation
in finger-tapping tasks that evolutionary predictive coders cannot easily
reproduce.
"""

from __future__ import annotations

import numpy as np

from neurodynamics.grfnn import GrFNN, GrFNNParams


class TestDelayAllocation:
    def test_off_by_default(self):
        net = GrFNN(n_oscillators=2, low_hz=5.0, high_hz=10.0, dt=0.001,
                    params=GrFNNParams())
        assert not net.delay_enabled
        assert net.delay_buffer is None

    def test_enabled_allocates_ring_buffer(self):
        net = GrFNN(
            n_oscillators=3, low_hz=5.0, high_hz=10.0, dt=0.001,
            params=GrFNNParams(),
            delay_tau=0.05, delay_gain=0.3,
        )
        assert net.delay_enabled
        assert net.delay_buffer is not None
        # tau=0.05, dt=0.001 → 50-sample ring.
        assert net.delay_buffer.shape == (50, 3)
        assert np.allclose(net.delay_buffer, 0.0)

    def test_zero_gain_matches_no_delay(self):
        """delay_gain=0 should produce identical trajectory to no-delay."""
        params = GrFNNParams(alpha=-0.1, input_gain=1.0)
        kwargs = dict(n_oscillators=2, low_hz=5.0, high_hz=5.0, dt=0.001,
                      params=params)
        a = GrFNN(**kwargs)
        b = GrFNN(**kwargs, delay_tau=0.05, delay_gain=0.0)
        for i in range(500):
            drive = np.full(2, 0.1 * np.cos(2 * np.pi * 5 * i * 0.001),
                            dtype=np.complex128)
            a.step(drive)
            b.step(drive)
        np.testing.assert_allclose(a.z, b.z, rtol=1e-10, atol=1e-12)


class TestDelayDynamics:
    def test_delay_changes_trajectory(self):
        """Non-zero delay coupling must produce a different trajectory from
        the baseline (no coupling). The sign and magnitude of the phase shift
        depend on tau and delay_gain; here we only assert they differ."""
        params = GrFNNParams(alpha=-0.1, input_gain=1.0,
                             beta1=-1.0, beta2=-1.0)
        kwargs = dict(n_oscillators=1, low_hz=5.0, high_hz=5.0, dt=0.001,
                      params=params)
        a = GrFNN(**kwargs)
        b = GrFNN(**kwargs, delay_tau=0.05, delay_gain=0.5)
        # Run long enough for the delay buffer to fill and the system to
        # settle into a driven steady state.
        for i in range(5000):
            drive = np.array([0.3 * np.cos(2 * np.pi * 5 * i * 0.001)],
                             dtype=np.complex128)
            a.step(drive)
            b.step(drive)
        # At least one of amplitude or phase should differ noticeably.
        amp_diff = abs(abs(a.z[0]) - abs(b.z[0]))
        phase_diff = abs(np.angle(a.z[0]) - np.angle(b.z[0]))
        assert amp_diff > 1e-3 or phase_diff > 0.05, (
            f"delay coupling must change dynamics. "
            f"amp_diff={amp_diff}, phase_diff={phase_diff}"
        )

    def test_delay_buffer_reads_past(self):
        """The buffer read at step t should reflect z from ~tau seconds ago,
        not the current z."""
        # Construct a network with explicit preloaded past state.
        net = GrFNN(
            n_oscillators=1, low_hz=5.0, high_hz=5.0, dt=0.001,
            params=GrFNNParams(alpha=-0.0, input_gain=0.0),  # neutral dynamics
            delay_tau=0.005, delay_gain=1.0,
        )
        # Preload the ring buffer with a known non-zero value in the slot
        # that will be read on the next step.
        net.delay_buffer[:] = 0.42 + 0j
        net.z[:] = 0.0 + 0j
        net.step(np.zeros(1, dtype=np.complex128))
        # After one step with delay_gain=1 and tau-sample-old value 0.42,
        # z should have gained ~ 0.42 * dt (Euler magnitude).
        # It won't be exactly 0.42*dt due to RK4 using intermediate stages,
        # but it will be close and clearly nonzero.
        assert abs(net.z[0]) > 1e-5, (
            f"delay buffer failed to feed back into dynamics: z={net.z[0]}"
        )
