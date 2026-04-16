"""Gradient Frequency Neural Network: bank of canonical Hopf oscillators.

Implements the equation from Large et al. 2025 "Musical neurodynamics" Fig. 2:

    tau * dz/dt = z * (alpha + i*omega
                       + (beta1 + i*delta1) * |z|^2
                       + epsilon * (beta2 + i*delta2) * |z|^4 / (1 - epsilon * |z|^2))
                  + input(t)

Each oscillator is a complex-valued state z ∈ ℂ with a natural frequency ω.
The parameters alpha/beta control bifurcation (damped vs. limit-cycle),
and the higher-order |z|^4/(1-eps|z|^2) term is what produces mode-locking
at complex integer ratios (3:2, 7:4 etc.) — the distinctive NRT prediction.

Integration: RK4 for stability at reasonable step sizes.

Input coupling: each oscillator receives a linear combination of input
channels weighted by proximity of input-channel center frequency to the
oscillator's natural frequency. This is a simplification of the full NRT
afferent connectivity — good enough for v0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GrFNNParams:
    alpha: float = -0.05
    beta1: float = -0.05
    beta2: float = -0.05
    delta1: float = 0.0
    delta2: float = 0.0
    epsilon: float = 1.0
    input_gain: float = 0.5


class GrFNN:
    """A bank of nonlinear oscillators log-spaced across a frequency range.

    Optional Hebbian plasticity: when ``hebbian=True``, the network carries
    a complex weight matrix W (n×n). Weights grow when oscillator pairs
    co-activate with stable phase, and decay otherwise. They feed back into
    the dynamics as an intra-layer coupling term (W @ z). This gives the
    network the attunement mechanism from NRT — learned tonal hierarchies,
    rhythmic attractors, implied harmony.

    The update rule (simplified from Large 2011, eq. 7):

        dW_ij/dt = -lambda_w * W_ij + kappa_w * z_i * conj(z_j)

    where ``kappa_w = learn_rate`` and ``lambda_w = weight_decay``.
    """

    def __init__(
        self,
        n_oscillators: int,
        low_hz: float,
        high_hz: float,
        dt: float,
        params: GrFNNParams,
        *,
        hebbian: bool = False,
        learn_rate: float = 0.0,
        weight_decay: float = 0.0,
        delay_tau: float = 0.0,
        delay_gain: float = 0.0,
        noise_amp: float = 0.0,
        noise_seed: int = 0,
    ):
        self.n = n_oscillators
        self.dt = dt
        self.p = params
        self.f = np.geomspace(low_hz, high_hz, n_oscillators).astype(np.float64)
        self.omega = 2 * np.pi * self.f
        rng = np.random.default_rng(0)
        self.z = (rng.standard_normal(n_oscillators)
                  + 1j * rng.standard_normal(n_oscillators)) * 1e-3
        self.z = self.z.astype(np.complex128)
        self.last_input_mag = np.zeros(n_oscillators, dtype=np.float64)
        # Per-oscillator prediction residual (surprise). Positive = more
        # input than needed to sustain current amplitude (onset / syncopation).
        # Negative = sustained amplitude with less input than required
        # (phantom / inner-voice regime).
        self.last_residual = np.zeros(n_oscillators, dtype=np.float64)

        self.hebbian_enabled = bool(hebbian)
        self.learn_rate = float(learn_rate)
        self.weight_decay = float(weight_decay)
        if self.hebbian_enabled:
            self.W = np.zeros((n_oscillators, n_oscillators), dtype=np.complex128)
        else:
            self.W = None

        # Strong anticipation via delayed self-coupling. tau > 0 allocates
        # a ring buffer of past z values; each step injects
        # delay_gain * z(t - tau) into the dynamics.
        self.delay_tau = float(delay_tau)
        self.delay_gain = float(delay_gain)
        self.delay_enabled = self.delay_tau > 0.0
        if self.delay_enabled:
            n_delay = max(1, int(round(self.delay_tau / self.dt)))
            self.delay_buffer = np.zeros((n_delay, n_oscillators),
                                         dtype=np.complex128)
            self._delay_write = 0  # next write index
        else:
            self.delay_buffer = None
            self._delay_write = 0

        # Stochastic noise: small Gaussian kicks per step model intrinsic
        # neural fluctuation. Doubles as a symmetry-breaker and keeps
        # phantom dynamics alive when the drive stops.
        self.noise_amp = float(noise_amp)
        self._noise_rng = np.random.default_rng(int(noise_seed))

    def _deriv(self, z: np.ndarray, x: np.ndarray,
               delayed: np.ndarray | None = None) -> np.ndarray:
        p = self.p
        abs2 = (z * z.conj()).real
        abs2_sat = np.minimum(abs2, 0.95 / max(p.epsilon, 1e-9))
        denom = 1.0 - p.epsilon * abs2_sat
        cubic = (p.beta1 + 1j * p.delta1) * abs2
        quintic = p.epsilon * (p.beta2 + 1j * p.delta2) * abs2_sat * abs2_sat / denom
        linear = p.alpha + 1j * self.omega
        rhs = z * (linear + cubic + quintic) + x
        if self.W is not None:
            # Mean-field normalization: divide by n so the intra-layer
            # coupling term is the AVERAGE partner contribution, not the
            # sum. Without this, network sizes affect dynamics scale and
            # large layers easily cascade into the integrator clamp.
            rhs = rhs + (self.W @ z) / self.n
        if delayed is not None:
            rhs = rhs + self.delay_gain * delayed
        return rhs

    def step(self, x: np.ndarray) -> np.ndarray:
        """Advance one dt using RK4 + Hebbian weight update.

        x: complex input vector, length = n_oscillators.
        """
        x = x * self.p.input_gain
        self.last_input_mag = np.abs(x)
        # Read delayed state (from ~tau ago) before advancing z. We treat
        # `delayed` as constant across the 4 RK4 stages — acceptable since
        # delay dynamics are slow relative to dt.
        delayed = None
        if self.delay_enabled:
            read_idx = self._delay_write  # oldest entry (ring buffer)
            delayed = self.delay_buffer[read_idx].copy()
        dt = self.dt
        k1 = self._deriv(self.z, x, delayed)
        k2 = self._deriv(self.z + 0.5 * dt * k1, x, delayed)
        k3 = self._deriv(self.z + 0.5 * dt * k2, x, delayed)
        k4 = self._deriv(self.z + dt * k3, x, delayed)
        self.z = self.z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if self.noise_amp > 0.0:
            # Complex Gaussian with amplitude proportional to sqrt(dt) for
            # correct Wiener scaling.
            re = self._noise_rng.standard_normal(self.n)
            im = self._noise_rng.standard_normal(self.n)
            self.z = self.z + self.noise_amp * np.sqrt(dt) * (re + 1j * im)
        limit = 0.98 / np.sqrt(max(self.p.epsilon, 1e-9))
        mag = np.abs(self.z)
        over = mag > limit
        if over.any():
            self.z[over] *= limit / mag[over]
        # Prediction residual: how much the actual input differs from what
        # the oscillator "expects" to sustain its current amplitude against
        # damping. At steady-state resonance |x| ≈ |alpha| * |z|; deviation
        # from this is the surprise signal.
        self.last_residual = (self.last_input_mag
                              - abs(self.p.alpha) * np.abs(self.z))
        if self.W is not None:
            self._hebbian_update()
        if self.delay_enabled:
            # Ring buffer: write new z over the slot just read (now the oldest).
            self.delay_buffer[self._delay_write] = self.z
            self._delay_write = (self._delay_write + 1) % self.delay_buffer.shape[0]
        return self.z

    def _hebbian_update(self) -> None:
        """Euler step on the weight matrix.

        dW_ij/dt = -lambda * W_ij + kappa * z_i * conj(z_j)

        Diagonal is kept at zero — an oscillator does not connect to itself.
        """
        if self.learn_rate == 0.0 and self.weight_decay == 0.0:
            return
        outer = np.outer(self.z, self.z.conj())
        self.W = self.W + self.dt * (
            self.learn_rate * outer - self.weight_decay * self.W
        )
        np.fill_diagonal(self.W, 0.0)

    def phantom_mask(self, amp_thresh: float, drive_thresh: float) -> np.ndarray:
        """Boolean mask: oscillators resonating without current drive.

        True = oscillator is "imagining" — high amplitude, low input. The
        missing-pulse / inner-voice signal.
        """
        return (np.abs(self.z) > amp_thresh) & (self.last_input_mag < drive_thresh)


def channel_to_oscillator_weights(
    channel_fc: np.ndarray, osc_f: np.ndarray, sharpness: float = 15.0,
) -> np.ndarray:
    """Build (n_osc, n_channels) coupling matrix weighting by log-frequency proximity.

    Used to project cochlear channel signals onto oscillators at their natural
    frequencies.

    IMPORTANT: band signals are oscillating in real time (not envelopes), so
    averaging adjacent channels leads to destructive phase interference at
    the driving frequency. We default to high sharpness (~one-hot) so each
    oscillator reads essentially one channel. For envelope-based drive (as
    used in the rhythm layer) a smaller sharpness averages multiple bands
    cleanly since envelopes are slowly varying and in-phase.
    """
    log_osc = np.log(osc_f)[:, None]
    log_ch = np.log(channel_fc)[None, :]
    d = (log_osc - log_ch) * sharpness
    w = np.exp(-d * d)
    # For oscillators whose nearest channel is too distant for the Gaussian
    # weights to be numerically significant, fall back to a one-hot on the
    # nearest channel. This preserves the row-sum-to-1 invariant.
    row_sums = w.sum(axis=1)
    degenerate = row_sums < 1e-9
    if degenerate.any():
        nearest = np.argmin(np.abs(log_osc - log_ch), axis=1)
        w[degenerate] = 0.0
        rows = np.where(degenerate)[0]
        w[rows, nearest[rows]] = 1.0
    w /= w.sum(axis=1, keepdims=True)
    return w.astype(np.float64)
