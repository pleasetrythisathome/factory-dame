"""Tests for mode-lock detection.

Mode-locking is the signature NRT prediction: when two oscillators couple
strongly enough, their phases lock at an integer ratio (1:1, 2:1, 3:2, 7:4…).
This is what produces perceived polyrhythms, consonant intervals, and the
stability hierarchy in Fig. 3 of Harding/Kim/Large 2025.

Detecting a mode-lock between two oscillators with natural frequencies f_i, f_j
and phases φ_i, φ_j at a ratio p:q means:

    n * φ_i - m * φ_j  stays roughly constant over time
    (for the integer ratio p:q where n=p, m=q and f_i/f_j ≈ m/n)

We compute the "phase-locking value" (PLV) over a rolling window:

    PLV_{i,j,p:q} = | mean_t exp(i * (p * φ_j - q * φ_i)) |

PLV is 1.0 for perfect lock, 0.0 for uncorrelated. Threshold ≈ 0.85 picks out
robust locks.
"""

from __future__ import annotations

import numpy as np

from neurodynamics.grfnn import GrFNN, GrFNNParams
from neurodynamics.modelock import detect_mode_locks, phase_locking_value


def _drive_coupled_pair(f_driver: float, ratio_p: int, ratio_q: int,
                        duration_s: float = 4.0, dt: float = 0.0005,
                        alpha: float = -0.1, input_gain: float = 2.0):
    """Create two oscillators at natural frequencies f_1 = f_driver,
    f_2 = f_driver * q/p. Drive both at f_driver (the fundamental).

    When p=1, q=2 → osc2 is at 2×f_driver, should 2:1 mode-lock.
    When p=2, q=3 → osc2 is at 1.5×f_driver, should 3:2 mode-lock (harder).
    """
    f1 = f_driver
    f2 = f_driver * ratio_q / ratio_p
    net = GrFNN(
        n_oscillators=2, low_hz=f1, high_hz=f2, dt=dt,
        params=GrFNNParams(alpha=alpha, input_gain=input_gain,
                           beta1=-1.0, beta2=-1.0, epsilon=1.0),
    )
    # geomspace(f1, f2, 2) gives exactly [f1, f2].
    assert np.isclose(net.f[0], f1)
    assert np.isclose(net.f[1], f2)

    fs = 1.0 / dt
    t = np.arange(int(fs * duration_s)) / fs
    drive_sig = 0.3 * np.cos(2 * np.pi * f_driver * t)
    phases = np.zeros((len(t), 2))
    for i, v in enumerate(drive_sig):
        net.step(np.full(2, v, dtype=np.complex128))
        phases[i] = np.angle(net.z)
    return phases, net


class TestPhaseLockingValue:
    def test_plv_one_for_identical_phases(self):
        """Two identical phase signals → PLV = 1 at ratio 1:1."""
        phi = np.linspace(0, 10 * 2 * np.pi, 1000)
        plv = phase_locking_value(phi, phi, ratio_p=1, ratio_q=1)
        assert plv == 1.0

    def test_plv_zero_for_independent_noise(self):
        rng = np.random.default_rng(0)
        phi_a = rng.uniform(-np.pi, np.pi, 10000)
        phi_b = rng.uniform(-np.pi, np.pi, 10000)
        plv = phase_locking_value(phi_a, phi_b, ratio_p=1, ratio_q=1)
        # Independent uniform phases: PLV ~ 1/sqrt(N). For N=10000, ~0.01.
        assert plv < 0.05

    def test_plv_detects_freq_ratio(self):
        """phi_b = 2 * phi_a + offset means f_a : f_b = 1 : 2 (B is faster).
        Convention: phase_locking_value(phi_a, phi_b, p, q) expects f_a:f_b=p:q,
        so this should PLV=1 at (p=1, q=2)."""
        phi_a = np.linspace(0, 50, 1000)
        phi_b = 2 * phi_a + 0.7
        assert phase_locking_value(phi_a, phi_b, ratio_p=1, ratio_q=2) > 0.999
        # The 1:1 ratio on the same pair should fail.
        assert phase_locking_value(phi_a, phi_b, ratio_p=1, ratio_q=1) < 0.5


class TestModeLockDetection:
    def test_detects_1_1_at_driven_frequency(self):
        """Two oscillators at the same natural freq, co-driven, should
        1:1 mode-lock."""
        phases, _net = _drive_coupled_pair(f_driver=5.0, ratio_p=1, ratio_q=1,
                                           duration_s=4.0)
        # Analyze the second half (after transient).
        tail = phases[len(phases) // 2:]
        plv = phase_locking_value(tail[:, 0], tail[:, 1], 1, 1)
        assert plv > 0.9, f"1:1 PLV too low: {plv}"

    def test_detect_mode_locks_returns_expected_schema(self):
        """detect_mode_locks returns a list of (i, j, p, q, plv) tuples."""
        phases, _net = _drive_coupled_pair(f_driver=5.0, ratio_p=1, ratio_q=1,
                                           duration_s=4.0)
        tail = phases[len(phases) // 2:]
        locks = detect_mode_locks(
            tail, threshold=0.85,
            ratios=[(1, 1), (2, 1), (3, 2)],
        )
        assert isinstance(locks, list)
        assert len(locks) >= 1
        for lock in locks:
            assert set(lock.keys()) == {"i", "j", "p", "q", "plv"}
            assert 0.0 <= lock["plv"] <= 1.0
