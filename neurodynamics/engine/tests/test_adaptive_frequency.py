"""Phase 2 validation: adaptive natural frequencies (Roman 2023 ASHLE).

Drives an oscillator at a frequency slightly OFF its natural rate
and verifies that the oscillator's natural frequency f_i drifts
toward the drive frequency. Then verifies that when the drive
stops, f_i elastically relaxes back to f_0.

Per Roman et al. 2023 Eqs.:
    ḟ = f · (λ₁ |x| sin(φ_x - φ_z) - λ_e (exp((f - f_0)/f_0) - 1))

The first term pulls f toward the drive phase rate; the second
term (elasticity) pulls f back to f_0 in absence of drive.
"""

from __future__ import annotations

import numpy as np

from neurodynamics.grfnn import GrFNN, GrFNNParams


def _drive_complex(f_hz: float, amp: float, n_samples: int,
                   dt: float) -> np.ndarray:
    t = np.arange(n_samples) * dt
    return (amp * np.exp(1j * 2 * np.pi * f_hz * t))[:, np.newaxis]


def test_freq_tracks_offset_drive():
    """Drive at 2.1 Hz on an oscillator with natural freq 2.0 Hz.

    Per Roman 2023, the adaptive frequency mechanism should pull
    the oscillator's f toward the drive rate. 5% offset is well
    inside the Arnold tongue at amp=0.7, so the oscillator first
    mode-locks (via z dynamics), then f adapts (via the ASHLE
    frequency rule).
    """
    f_natural = 2.0  # Hz
    f_drive = 2.1    # Hz, offset by 5%
    dt = 0.002
    duration = 12.0  # ASHLE adaptation is slow (multi-second)

    params = GrFNNParams(
        alpha=0.0, beta1=-1.0, beta2=-1.0,
        delta1=0.0, delta2=0.0, epsilon=1.0,
        input_gain=1.0,
    )
    freqs = np.array([f_natural], dtype=np.float64)
    net = GrFNN(
        n_oscillators=1, low_hz=f_natural, high_hz=f_natural,
        dt=dt, params=params, freqs=freqs,
        adaptive_frequency=True,
        adaptive_lambda_freq=4.0,
        adaptive_lambda_elastic=0.2,  # weak elasticity for adaptation test
        noise_amp=1e-4, noise_seed=11,
    )
    net.z[:] = 0.1 + 0.0j
    n_samples = int(duration / dt)
    drive = _drive_complex(f_drive, amp=0.7, n_samples=n_samples, dt=dt)
    chunk = 256
    for start in range(0, n_samples, chunk):
        end = min(start + chunk, n_samples)
        net.step_many(drive[start:end].copy())

    final_f = float(net.f[0])
    f_0 = float(net.f_natural[0])
    drift = (final_f - f_0) / (f_drive - f_0)
    print(f"f_0={f_0:.3f}  drive={f_drive:.3f}  "
          f"final f={final_f:.3f}  drift fraction={drift:.3f}")
    assert drift > 0.2, (
        f"adaptive frequency didn't track drive: f={final_f:.3f}, "
        f"f_0={f_0:.3f}, drive={f_drive:.3f}, drift={drift:.3f}"
    )


def test_freq_relaxes_after_drive_stops():
    """Drive at 2.5 Hz for 6 s, then stop. After another 6 s of
    silence, frequency should have relaxed back toward f_0 = 2.0.
    """
    f_natural = 2.0
    f_drive = 2.5
    dt = 0.002

    params = GrFNNParams(
        alpha=0.0, beta1=-1.0, beta2=-1.0,
        delta1=0.0, delta2=0.0, epsilon=1.0,
        input_gain=1.0,
    )
    freqs = np.array([f_natural], dtype=np.float64)
    net = GrFNN(
        n_oscillators=1, low_hz=f_natural, high_hz=f_natural,
        dt=dt, params=params, freqs=freqs,
        adaptive_frequency=True,
        adaptive_lambda_freq=4.0,
        adaptive_lambda_elastic=2.0,  # full elasticity for relaxation test
        noise_amp=1e-4, noise_seed=13,
    )
    net.z[:] = 0.1 + 0.0j

    # Phase 1: drive
    n_drive = int(6.0 / dt)
    drive = _drive_complex(f_drive, amp=0.5, n_samples=n_drive, dt=dt)
    chunk = 256
    for start in range(0, n_drive, chunk):
        end = min(start + chunk, n_drive)
        net.step_many(drive[start:end].copy())
    f_after_drive = float(net.f[0])

    # Phase 2: silence
    n_silence = int(6.0 / dt)
    silence = np.zeros((n_silence, 1), dtype=np.complex128)
    for start in range(0, n_silence, chunk):
        end = min(start + chunk, n_silence)
        net.step_many(silence[start:end].copy())
    f_after_silence = float(net.f[0])
    f_0 = float(net.f_natural[0])

    print(f"f_0={f_0:.3f}  after drive={f_after_drive:.3f}  "
          f"after silence={f_after_silence:.3f}")
    # Should relax noticeably back toward f_0
    drift_before = abs(f_after_drive - f_0)
    drift_after = abs(f_after_silence - f_0)
    assert drift_after < drift_before * 0.8, (
        f"elasticity not relaxing: drift before silence={drift_before:.3f}, "
        f"after silence={drift_after:.3f}"
    )
