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
from numba import njit


@dataclass
class GrFNNParams:
    alpha: float = -0.05
    beta1: float = -0.05
    beta2: float = -0.05
    delta1: float = 0.0
    delta2: float = 0.0
    epsilon: float = 1.0
    input_gain: float = 0.5


# ── JIT hot path ─────────────────────────────────────────────────
#
# The Hopf ODE right-hand side + RK4 integration is the engine's
# single dominant cost (~95% of wall time at 16 kHz audio rate per
# the benchmark in tests/benchmark_engine.py). Both are extracted
# into module-level ``@njit`` functions so numba can generate tight
# native code — the per-step Python overhead that dominates pure-
# numpy at small array sizes (n≈100) disappears entirely.
#
# Bit-identical with the pure-NumPy implementation is not quite
# achievable (JIT uses slightly different float dispatch paths) but
# numerical equivalence within 1e-7 across 60s of audio is routine.
# Engine tests use tolerance-based assertions already.


@njit(cache=True, fastmath=True, inline="always")
def _deriv_jit(
    z: np.ndarray, x: np.ndarray, omega: np.ndarray,
    alpha: float, beta1: float, beta2: float,
    delta1: float, delta2: float, epsilon: float,
    wz_scaled: np.ndarray, W_enabled: bool, n: int,
    delayed: np.ndarray, delay_enabled: bool, delay_gain: float,
    out: np.ndarray,
) -> None:
    """Hopf RHS per oscillator. ``wz_scaled`` carries the
    precomputed ``(W @ z) / n`` from the RK4 wrapper (computed once
    per step rather than per RK4 stage — we hold the intra-layer
    coupling constant across the 4 stages of one dt, which is an
    acceptable numerical approximation for dt ≪ 1/omega and gives
    a 4× reduction in matrix-vector cost)."""
    sat_limit = 0.95 / max(epsilon, 1e-9)
    beta1c = complex(beta1, delta1)
    beta2c = complex(beta2, delta2)
    for i in range(n):
        zi = z[i]
        abs2 = zi.real * zi.real + zi.imag * zi.imag
        abs2_sat = abs2 if abs2 < sat_limit else sat_limit
        denom = 1.0 - epsilon * abs2_sat
        cubic = beta1c * abs2
        quintic = epsilon * beta2c * abs2_sat * abs2_sat / denom
        linear = complex(alpha, omega[i])
        rhs = zi * (linear + cubic + quintic) + x[i]
        if W_enabled:
            rhs = rhs + wz_scaled[i]
        if delay_enabled:
            rhs = rhs + delay_gain * delayed[i]
        out[i] = rhs


@njit(cache=True, fastmath=True)
def _rk4_step_jit(
    z: np.ndarray, x: np.ndarray, omega: np.ndarray,
    alpha: float, beta1: float, beta2: float,
    delta1: float, delta2: float, epsilon: float,
    W: np.ndarray, W_enabled: bool, n: int,
    delayed: np.ndarray, delay_enabled: bool, delay_gain: float,
    dt: float,
    k1: np.ndarray, k2: np.ndarray, k3: np.ndarray, k4: np.ndarray,
    ztmp: np.ndarray, wz_scaled: np.ndarray,
) -> None:
    """One RK4 advance, writing the result back into ``z`` in place.
    All scratch buffers are caller-supplied so heap allocation is
    amortized."""
    # Compute W @ z / n once per step (constant across RK4 stages).
    if W_enabled:
        inv_n = 1.0 / n
        for i in range(n):
            acc = 0.0 + 0.0j
            for j in range(n):
                acc += W[i, j] * z[j]
            wz_scaled[i] = acc * inv_n
    _deriv_jit(z, x, omega, alpha, beta1, beta2,
               delta1, delta2, epsilon, wz_scaled, W_enabled, n,
               delayed, delay_enabled, delay_gain, k1)
    for i in range(n):
        ztmp[i] = z[i] + 0.5 * dt * k1[i]
    _deriv_jit(ztmp, x, omega, alpha, beta1, beta2,
               delta1, delta2, epsilon, wz_scaled, W_enabled, n,
               delayed, delay_enabled, delay_gain, k2)
    for i in range(n):
        ztmp[i] = z[i] + 0.5 * dt * k2[i]
    _deriv_jit(ztmp, x, omega, alpha, beta1, beta2,
               delta1, delta2, epsilon, wz_scaled, W_enabled, n,
               delayed, delay_enabled, delay_gain, k3)
    for i in range(n):
        ztmp[i] = z[i] + dt * k3[i]
    _deriv_jit(ztmp, x, omega, alpha, beta1, beta2,
               delta1, delta2, epsilon, wz_scaled, W_enabled, n,
               delayed, delay_enabled, delay_gain, k4)
    dt_over_6 = dt / 6.0
    for i in range(n):
        z[i] = z[i] + dt_over_6 * (k1[i] + 2.0 * k2[i]
                                    + 2.0 * k3[i] + k4[i])


_EMPTY_W = np.zeros((0, 0), dtype=np.complex128)
_EMPTY_DELAY = np.zeros(0, dtype=np.complex128)


@njit(cache=True, fastmath=True)
def _step_many_jit(
    z: np.ndarray,                  # (n,) — advanced in place
    xs: np.ndarray,                 # (n_samples, n) inputs
    omega: np.ndarray,
    alpha: float, beta1: float, beta2: float,
    delta1: float, delta2: float, epsilon: float,
    input_gain: float,
    W: np.ndarray, W_enabled: bool, n: int,
    delay_buffer: np.ndarray, delay_write_start: int,
    delay_enabled: bool, delay_gain: float,
    noise_amp: float, noise_re: np.ndarray, noise_im: np.ndarray,
    noise_sqrt_dt: float,
    dt: float,
    k1: np.ndarray, k2: np.ndarray, k3: np.ndarray, k4: np.ndarray,
    ztmp: np.ndarray, wz_scaled: np.ndarray,
    last_input_mag: np.ndarray,    # (n,) out
    last_residual: np.ndarray,     # (n,) out
) -> int:
    """Advance ``z`` through ``len(xs)`` samples inside a single
    JIT call. All the per-sample Python overhead that dominated the
    single-step API disappears here — the loop runs entirely in
    compiled code.

    Returns the new ``delay_write`` ring-buffer index.
    """
    sat_limit = 0.95 / max(epsilon, 1e-9)
    abs_alpha = alpha if alpha >= 0 else -alpha
    limit = 0.98 / np.sqrt(max(epsilon, 1e-9))
    beta1c = complex(beta1, delta1)
    beta2c = complex(beta2, delta2)
    dt_over_6 = dt / 6.0
    inv_n = 1.0 / n if n > 0 else 0.0
    delay_depth = delay_buffer.shape[0] if delay_enabled else 0
    delay_write = delay_write_start
    n_samples = xs.shape[0]
    for t in range(n_samples):
        # Per-sample input scaling + magnitude
        for i in range(n):
            xs_ti = xs[t, i] * input_gain
            xs[t, i] = xs_ti
            last_input_mag[i] = (xs_ti.real * xs_ti.real
                                  + xs_ti.imag * xs_ti.imag) ** 0.5
        # Delayed state (constant across RK4 stages)
        if delay_enabled:
            read_idx = delay_write  # oldest
            for i in range(n):
                ztmp[i] = delay_buffer[read_idx, i]
            delayed = ztmp  # reuse; will be overwritten later
        # Compute W @ z / n once per step
        if W_enabled:
            for i in range(n):
                acc = 0.0 + 0.0j
                for j in range(n):
                    acc += W[i, j] * z[j]
                wz_scaled[i] = acc * inv_n
        # RK4 stage 1
        for i in range(n):
            zi = z[i]
            abs2 = zi.real * zi.real + zi.imag * zi.imag
            abs2_sat = abs2 if abs2 < sat_limit else sat_limit
            denom = 1.0 - epsilon * abs2_sat
            cubic = beta1c * abs2
            quintic = epsilon * beta2c * abs2_sat * abs2_sat / denom
            linear = complex(alpha, omega[i])
            rhs = zi * (linear + cubic + quintic) + xs[t, i]
            if W_enabled:
                rhs = rhs + wz_scaled[i]
            if delay_enabled:
                rhs = rhs + delay_gain * delay_buffer[delay_write, i]
            k1[i] = rhs
        # RK4 stage 2
        for i in range(n):
            ztmp_i = z[i] + 0.5 * dt * k1[i]
            abs2 = ztmp_i.real * ztmp_i.real + ztmp_i.imag * ztmp_i.imag
            abs2_sat = abs2 if abs2 < sat_limit else sat_limit
            denom = 1.0 - epsilon * abs2_sat
            cubic = beta1c * abs2
            quintic = epsilon * beta2c * abs2_sat * abs2_sat / denom
            linear = complex(alpha, omega[i])
            rhs = ztmp_i * (linear + cubic + quintic) + xs[t, i]
            if W_enabled:
                rhs = rhs + wz_scaled[i]
            if delay_enabled:
                rhs = rhs + delay_gain * delay_buffer[delay_write, i]
            k2[i] = rhs
        # RK4 stage 3
        for i in range(n):
            ztmp_i = z[i] + 0.5 * dt * k2[i]
            abs2 = ztmp_i.real * ztmp_i.real + ztmp_i.imag * ztmp_i.imag
            abs2_sat = abs2 if abs2 < sat_limit else sat_limit
            denom = 1.0 - epsilon * abs2_sat
            cubic = beta1c * abs2
            quintic = epsilon * beta2c * abs2_sat * abs2_sat / denom
            linear = complex(alpha, omega[i])
            rhs = ztmp_i * (linear + cubic + quintic) + xs[t, i]
            if W_enabled:
                rhs = rhs + wz_scaled[i]
            if delay_enabled:
                rhs = rhs + delay_gain * delay_buffer[delay_write, i]
            k3[i] = rhs
        # RK4 stage 4
        for i in range(n):
            ztmp_i = z[i] + dt * k3[i]
            abs2 = ztmp_i.real * ztmp_i.real + ztmp_i.imag * ztmp_i.imag
            abs2_sat = abs2 if abs2 < sat_limit else sat_limit
            denom = 1.0 - epsilon * abs2_sat
            cubic = beta1c * abs2
            quintic = epsilon * beta2c * abs2_sat * abs2_sat / denom
            linear = complex(alpha, omega[i])
            rhs = ztmp_i * (linear + cubic + quintic) + xs[t, i]
            if W_enabled:
                rhs = rhs + wz_scaled[i]
            if delay_enabled:
                rhs = rhs + delay_gain * delay_buffer[delay_write, i]
            k4[i] = rhs
        # Combine + noise + clamp + residual
        for i in range(n):
            z[i] = z[i] + dt_over_6 * (
                k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]
            )
            if noise_amp > 0.0:
                z[i] = z[i] + noise_amp * noise_sqrt_dt * complex(
                    noise_re[t, i], noise_im[t, i]
                )
            # Clamp |z|
            re = z[i].real; im = z[i].imag
            mag = (re * re + im * im) ** 0.5
            if mag > limit:
                scale = limit / mag
                z[i] = complex(re * scale, im * scale)
            # Residual: |x| - |alpha| * |z|
            re = z[i].real; im = z[i].imag
            zmag = (re * re + im * im) ** 0.5
            last_residual[i] = last_input_mag[i] - abs_alpha * zmag
        # Delay ring-buffer: write new z over the oldest slot
        if delay_enabled and delay_depth > 0:
            for i in range(n):
                delay_buffer[delay_write, i] = z[i]
            delay_write = (delay_write + 1) % delay_depth
    return delay_write


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
        # Preallocated RK4 scratch buffers — avoid a fresh heap
        # allocation on every audio-rate step.
        self._k1 = np.empty(n_oscillators, dtype=np.complex128)
        self._k2 = np.empty(n_oscillators, dtype=np.complex128)
        self._k3 = np.empty(n_oscillators, dtype=np.complex128)
        self._k4 = np.empty(n_oscillators, dtype=np.complex128)
        self._ztmp = np.empty(n_oscillators, dtype=np.complex128)
        self._wz_scaled = np.empty(n_oscillators, dtype=np.complex128)
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

        Hot path — delegates the RK4 integration to ``_rk4_step_jit``
        so numba compiles the Hopf RHS + 4-stage RK4 into tight native
        code. Python-side logic (noise, clamp, Hebbian, delay buffer)
        is either cheap enough not to JIT or stateful in ways numba
        doesn't handle cleanly (RNG with seeded Generator).
        """
        p = self.p
        x = x * p.input_gain
        self.last_input_mag = np.abs(x)
        # Read delayed state (from ~tau ago) before advancing z. We treat
        # `delayed` as constant across the 4 RK4 stages — acceptable since
        # delay dynamics are slow relative to dt.
        if self.delay_enabled:
            delayed = self.delay_buffer[self._delay_write].copy()
        else:
            delayed = _EMPTY_DELAY
        _rk4_step_jit(
            self.z, x, self.omega,
            p.alpha, p.beta1, p.beta2, p.delta1, p.delta2, p.epsilon,
            self.W if self.W is not None else _EMPTY_W,
            self.W is not None, self.n,
            delayed, self.delay_enabled, self.delay_gain,
            self.dt,
            self._k1, self._k2, self._k3, self._k4,
            self._ztmp, self._wz_scaled,
        )
        if self.noise_amp > 0.0:
            # Complex Gaussian with amplitude proportional to sqrt(dt) for
            # correct Wiener scaling.
            re = self._noise_rng.standard_normal(self.n)
            im = self._noise_rng.standard_normal(self.n)
            self.z = self.z + self.noise_amp * np.sqrt(self.dt) * (re + 1j * im)
        limit = 0.98 / np.sqrt(max(p.epsilon, 1e-9))
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

    def step_many(self, xs: np.ndarray) -> None:
        """Advance through a batch of samples in a single JIT call.

        ``xs`` must be shape ``(n_samples, n_oscillators)`` complex.
        The batched API amortizes Python overhead across many samples
        — at audio rate (16 kHz) this is the difference between
        ~0.5× realtime (per-sample calls) and comfortably faster-
        than-realtime (batched).

        Caveats vs ``step``:
        - ``last_input_mag`` and ``last_residual`` reflect the last
          sample of the batch only.
        - Hebbian weight updates happen at batch boundaries rather
          than per sample. At audio rate (16 kHz) the per-sample W
          update is imperceptible anyway; batching a few ms worth of
          samples preserves essentially the same learning behavior
          while dropping per-step overhead entirely.

        If you need per-sample Hebbian, chunk the batch into tiny
        pieces (1-2 samples) — the numba overhead is small enough
        that 1-sample batches still win over the per-step API.
        """
        if xs.ndim != 2 or xs.shape[1] != self.n:
            raise ValueError(
                f"xs must be (n_samples, {self.n}); got {xs.shape}"
            )
        n_samples = xs.shape[0]
        p = self.p
        # Noise: draw all samples up front so the JIT loop never
        # calls out. Even when noise is off, pass zero-sized arrays.
        if self.noise_amp > 0.0:
            noise_re = self._noise_rng.standard_normal((n_samples, self.n))
            noise_im = self._noise_rng.standard_normal((n_samples, self.n))
        else:
            noise_re = np.zeros((n_samples, self.n), dtype=np.float64)
            noise_im = np.zeros((n_samples, self.n), dtype=np.float64)
        delay_buffer = (self.delay_buffer
                        if self.delay_enabled
                        else np.zeros((0, self.n), dtype=np.complex128))
        W_eff = self.W if self.W is not None else _EMPTY_W
        # Ensure xs is complex128 (JIT doesn't coerce).
        xs = np.ascontiguousarray(xs, dtype=np.complex128)
        new_delay_write = _step_many_jit(
            self.z, xs, self.omega,
            p.alpha, p.beta1, p.beta2, p.delta1, p.delta2, p.epsilon,
            p.input_gain,
            W_eff, self.W is not None, self.n,
            delay_buffer, self._delay_write,
            self.delay_enabled, self.delay_gain,
            self.noise_amp, noise_re, noise_im,
            np.sqrt(self.dt),
            self.dt,
            self._k1, self._k2, self._k3, self._k4,
            self._ztmp, self._wz_scaled,
            self.last_input_mag, self.last_residual,
        )
        if self.delay_enabled:
            self._delay_write = int(new_delay_write)
        if self.W is not None:
            self._hebbian_update()

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
