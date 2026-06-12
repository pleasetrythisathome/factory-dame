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


# Default integer-ratio set for the internal coupling kernel D
# (canonical NRT cross-resonances). The most-stable ratios in NRT's
# stability hierarchy: 1:1 > 2:1 ≈ 1:2 > 3:2 ≈ 2:3 > 3:1 ≈ 1:3.
DEFAULT_RESONANCE_RATIOS: tuple[tuple[int, int], ...] = (
    (1, 1),  # unison / same freq
    (2, 1),  # octave up
    (1, 2),  # octave down (subharmonic)
    (3, 2),  # perfect fifth above
    (2, 3),  # perfect fourth below
    (3, 1),  # octave + perfect fifth above
    (1, 3),  # octave + perfect fifth below
)


def build_integer_ratio_coupling(
    freqs: np.ndarray,
    *,
    ratios: tuple[tuple[int, int], ...] = DEFAULT_RESONANCE_RATIOS,
    tolerance_semitones: float = 0.4,
    max_octave_distance: float = 2.5,
    decay_with_distance: bool = True,
) -> np.ndarray:
    """Construct the internal coupling matrix C for a pitch GrFNN.

    Returns a complex (n, n) matrix where C[i, j] is the coupling
    weight from oscillator j into oscillator i. Non-zero entries
    exist where (f_i / f_j) approximates one of the integer ratios in
    ``ratios``. Local kernel: pairs more than ``max_octave_distance``
    octaves apart are not connected.

    The weights decay with distance from the perfect ratio (Gaussian
    in log-frequency space) and with octave separation when
    ``decay_with_distance`` is True. The diagonal is left at zero —
    an oscillator does not couple to itself via this kernel (it
    self-resonates through the canonical Hopf dynamics).

    Public domain: this construction is disclosed in US 7,376,562
    (expired Nov 17, 2024).
    """
    n = len(freqs)
    log_freqs = np.log2(np.asarray(freqs, dtype=np.float64))
    C = np.zeros((n, n), dtype=np.complex128)
    target_logs = np.array(
        [np.log2(p / q) for p, q in ratios], dtype=np.float64
    )
    base_weights = np.array(
        [1.0 / max(p, q) for p, q in ratios], dtype=np.float64
    )
    sigma = max(tolerance_semitones / 12.0, 1e-6)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            log_ratio = log_freqs[i] - log_freqs[j]
            if abs(log_ratio) > max_octave_distance:
                continue
            # Match against the catalogue of ratios; pick the closest.
            best_idx = -1
            best_dist_oct = float("inf")
            for k, target in enumerate(target_logs):
                d = abs(log_ratio - target)
                if d < best_dist_oct:
                    best_dist_oct = d
                    best_idx = k
            # Hard cutoff outside 2σ — ensures clearly non-harmonic
            # pairs get zero weight rather than a tiny exponential
            # tail. Inside the window, Gaussian falloff smooths the
            # edge so slightly mistuned pairs still couple.
            if best_dist_oct * 12 > 2 * tolerance_semitones:
                continue
            falloff = np.exp(-(best_dist_oct / sigma) ** 2 * 0.5)
            w = base_weights[best_idx] * falloff
            if decay_with_distance:
                # Mild penalty for octave separation — connections across
                # multiple octaves are real (octaves are real harmonic
                # relationships) but should be slightly weaker than
                # local resonances.
                w *= np.exp(-abs(log_ratio) / 4.0)
            C[i, j] = w
    return C


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
    internal_coupling: np.ndarray, C_enabled: bool, coupling_gain: float,
    tau_scale: np.ndarray, tau_enabled: bool,
    canonical_drive: np.ndarray, nonlinear_input: bool,
) -> None:
    """Hopf RHS per oscillator.

    ``wz_scaled`` carries the precomputed ``(W @ z) / n`` from the RK4
    wrapper (computed once per step rather than per RK4 stage — we
    hold the intra-layer coupling constant across the 4 stages of
    one dt, which is an acceptable numerical approximation for
    dt ≪ 1/omega and gives a 4× reduction in matrix-vector cost).

    ``tau_scale`` (when enabled) is the per-oscillator timescale
    factor f_n that scales the dimensionless dynamical coefficients
    (α, β1, β2, drive) into per-second units. Public domain: the
    τₙ = 1/fₙ scaling is disclosed in US 7,376,562 (expired Nov 17,
    2024). Effect: all oscillators integrate over the same number of
    cycles (∼20) instead of the same number of seconds — fundamentally
    correct for audio analysis where time resolution should scale
    with frequency.

    ``canonical_drive`` (when ``nonlinear_input`` is True) is the
    precomputed per-oscillator canonical input transformation
    P(ε,x)·A(ε,z̄) = x/(1-√ε x) · 1/(1-√ε z̄). It replaces the linear
    ``x`` term entirely. The expansion of P·A generates resonant drive
    at all integer ratios of input to oscillator frequency — the
    mechanism behind harmonic enrichment, subharmonic mode-locking,
    and missing-fundamental perception. Per Large 2010 Eq. 15, Lerud
    2014 Eq. 2, Kim & Large 2019 Eq. 15.
    """
    sat_limit = 0.95 / max(epsilon, 1e-9)
    beta1c = complex(beta1, delta1)
    beta2c = complex(beta2, delta2)
    for i in range(n):
        s = tau_scale[i] if tau_enabled else 1.0
        zi = z[i]
        abs2 = zi.real * zi.real + zi.imag * zi.imag
        abs2_sat = abs2 if abs2 < sat_limit else sat_limit
        denom = 1.0 - epsilon * abs2_sat
        cubic = beta1c * abs2
        quintic = epsilon * beta2c * abs2_sat * abs2_sat / denom
        # Per-oscillator τ scaling: (α + cubic + quintic) and drive
        # scale by s = f_n; ω stays unscaled (already in rad/s).
        linear = complex(s * alpha, omega[i])
        if nonlinear_input:
            drive = canonical_drive[i]
        else:
            drive = x[i]
        rhs = zi * (linear + s * cubic + s * quintic) + s * drive
        if W_enabled:
            rhs = rhs + wz_scaled[i]
        if C_enabled:
            rhs = rhs + coupling_gain * internal_coupling[i]
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
    C: np.ndarray, C_enabled: bool, coupling_gain: float,
    P_scratch: np.ndarray, A_scratch: np.ndarray,
    internal_scratch: np.ndarray,
    tau_scale: np.ndarray, tau_enabled: bool,
    nonlinear_input: bool, P_input_scratch: np.ndarray,
    canonical_drive: np.ndarray,
    C_csr_row_start: np.ndarray,
    C_csr_col: np.ndarray,
    C_csr_val: np.ndarray,
) -> None:
    """One RK4 advance, writing the result back into ``z`` in place.
    All scratch buffers are caller-supplied so heap allocation is
    amortized.

    Both W coupling (Hebbian-learned) and C coupling (integer-ratio
    kernel) are computed once at the start of the step and held
    constant across the 4 RK4 stages — same approximation, valid
    when dt ≪ 1/ω_max."""
    # Compute W @ z / n once per step (constant across RK4 stages).
    if W_enabled:
        inv_n = 1.0 / n
        for i in range(n):
            acc = 0.0 + 0.0j
            for j in range(n):
                acc += W[i, j] * z[j]
            wz_scaled[i] = acc * inv_n
    # Precompute A_i = 1/(1-√ε z̄_i) when needed by either the
    # integer-ratio coupling kernel OR the canonical nonlinear input
    # form. The two consumers share the same A factor (per Large 2015
    # Eq. 2 / Lerud 2014 Eq. 2).
    need_A = C_enabled or nonlinear_input
    if need_A:
        sqrt_eps = np.sqrt(max(epsilon, 1e-9))
        eps_guard = 1e-12
        for i in range(n):
            zc = complex(z[i].real, -z[i].imag)
            d2 = 1.0 - sqrt_eps * zc
            d2_mag2 = d2.real * d2.real + d2.imag * d2.imag
            if d2_mag2 < eps_guard:
                A_scratch[i] = 0.0 + 0.0j
            else:
                A_scratch[i] = 1.0 / d2
    # Integer-ratio internal coupling: P_z = z_j/(1-√ε z_j), then
    # internal_i = A_i · Σ_j C[i,j] · P_z[j]. CSR sparse matvec.
    if C_enabled:
        sqrt_eps = np.sqrt(max(epsilon, 1e-9))
        eps_guard = 1e-12
        for i in range(n):
            d = 1.0 - sqrt_eps * z[i]
            d_mag2 = d.real * d.real + d.imag * d.imag
            if d_mag2 < eps_guard:
                P_scratch[i] = 0.0 + 0.0j
            else:
                P_scratch[i] = z[i] / d
        for i in range(n):
            acc = 0.0 + 0.0j
            row_lo = C_csr_row_start[i]
            row_hi = C_csr_row_start[i + 1]
            for k in range(row_lo, row_hi):
                acc += C_csr_val[k] * P_scratch[C_csr_col[k]]
            internal_scratch[i] = A_scratch[i] * acc
    # Canonical nonlinear input: drive_i = P(ε, x_i)·A(ε, z̄_i)
    # where P(ε, x) = x/(1-√ε x). Held constant across RK4 stages
    # (same approximation as W and C — valid when dt ≪ 1/ω).
    if nonlinear_input:
        sqrt_eps = np.sqrt(max(epsilon, 1e-9))
        eps_guard = 1e-12
        for i in range(n):
            d = 1.0 - sqrt_eps * x[i]
            d_mag2 = d.real * d.real + d.imag * d.imag
            if d_mag2 < eps_guard:
                P_input_scratch[i] = 0.0 + 0.0j
            else:
                P_input_scratch[i] = x[i] / d
            canonical_drive[i] = P_input_scratch[i] * A_scratch[i]
    _deriv_jit(z, x, omega, alpha, beta1, beta2,
               delta1, delta2, epsilon, wz_scaled, W_enabled, n,
               delayed, delay_enabled, delay_gain, k1,
               internal_scratch, C_enabled, coupling_gain,
               tau_scale, tau_enabled,
               canonical_drive, nonlinear_input)
    for i in range(n):
        ztmp[i] = z[i] + 0.5 * dt * k1[i]
    _deriv_jit(ztmp, x, omega, alpha, beta1, beta2,
               delta1, delta2, epsilon, wz_scaled, W_enabled, n,
               delayed, delay_enabled, delay_gain, k2,
               internal_scratch, C_enabled, coupling_gain,
               tau_scale, tau_enabled,
               canonical_drive, nonlinear_input)
    for i in range(n):
        ztmp[i] = z[i] + 0.5 * dt * k2[i]
    _deriv_jit(ztmp, x, omega, alpha, beta1, beta2,
               delta1, delta2, epsilon, wz_scaled, W_enabled, n,
               delayed, delay_enabled, delay_gain, k3,
               internal_scratch, C_enabled, coupling_gain,
               tau_scale, tau_enabled,
               canonical_drive, nonlinear_input)
    for i in range(n):
        ztmp[i] = z[i] + dt * k3[i]
    _deriv_jit(ztmp, x, omega, alpha, beta1, beta2,
               delta1, delta2, epsilon, wz_scaled, W_enabled, n,
               delayed, delay_enabled, delay_gain, k4,
               internal_scratch, C_enabled, coupling_gain,
               tau_scale, tau_enabled,
               canonical_drive, nonlinear_input)
    dt_over_6 = dt / 6.0
    for i in range(n):
        z[i] = z[i] + dt_over_6 * (k1[i] + 2.0 * k2[i]
                                    + 2.0 * k3[i] + k4[i])


_EMPTY_W = np.zeros((0, 0), dtype=np.complex128)
_EMPTY_DELAY = np.zeros(0, dtype=np.complex128)
_EMPTY_C = np.zeros((0, 0), dtype=np.complex128)


@njit(cache=True, fastmath=True)
def _compute_internal_coupling_jit(
    z: np.ndarray,                 # (n,) current state
    C: np.ndarray,                 # (n, n) coupling kernel
    sqrt_eps: float,
    n: int,
    P: np.ndarray,                 # (n,) scratch
    A: np.ndarray,                 # (n,) scratch
    out: np.ndarray,               # (n,) result
) -> None:
    """Compute the canonical NRT internal coupling input per oscillator.

    For oscillator i:
        internal_i = (1 / (1 - √ε z̄_i)) · Σ_j C[i, j] · (z_j / (1 - √ε z_j))

    The series expansions of P_j = z_j/(1 - √ε z_j) and
    A_i = 1/(1 - √ε z̄_i) carry all integer-ratio resonance terms
    automatically — the user's choice of which entries to make
    non-zero in C selects which resonances participate in the
    network's dynamics.

    Public domain: this canonical coupling form is disclosed in
    US 7,376,562 (expired 2024-11-17).
    """
    # P_j = z_j / (1 - √ε z_j)
    # Guard against denominator → 0 when |z| ≈ 1/√ε; clamp introduced
    # in the integrator already keeps |z| < 0.98/√ε so denom magnitude
    # ≥ 0.02; we add a small epsilon to be safe in case the helper is
    # called pre-clamp.
    eps_guard = 1e-12
    for i in range(n):
        d = 1.0 - sqrt_eps * z[i]
        # Inline complex reciprocal with guard
        d_mag2 = d.real * d.real + d.imag * d.imag
        if d_mag2 < eps_guard:
            P[i] = 0.0 + 0.0j
        else:
            P[i] = z[i] / d
        zc = complex(z[i].real, -z[i].imag)
        d2 = 1.0 - sqrt_eps * zc
        d2_mag2 = d2.real * d2.real + d2.imag * d2.imag
        if d2_mag2 < eps_guard:
            A[i] = 0.0 + 0.0j
        else:
            A[i] = 1.0 / d2
    # Matvec: out_i = A_i · Σ_j C[i, j] · P_j
    for i in range(n):
        acc = 0.0 + 0.0j
        for j in range(n):
            acc += C[i, j] * P[j]
        out[i] = A[i] * acc


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
    C: np.ndarray,                 # (n, n) integer-ratio coupling kernel
    C_enabled: bool, coupling_gain: float,
    P_scratch: np.ndarray,         # (n,) scratch for P_j = z_j/(1-√ε z_j)
    A_scratch: np.ndarray,         # (n,) scratch for A_i = 1/(1-√ε z̄_i)
    internal_scratch: np.ndarray,  # (n,) scratch for kernel result
    tau_scale: np.ndarray,         # (n,) per-osc τ scale, 1.0 if disabled
    tau_enabled: bool,             # whether to apply tau_scale
    nonlinear_input: bool,         # canonical input form P(ε,x)·A(ε,z̄)
    P_input_scratch: np.ndarray,   # (n,) scratch for P(ε, x_i)
    canonical_drive: np.ndarray,   # (n,) scratch for P_input_i · A_i
    record_z: bool,                # write per-sample z to history
    z_history: np.ndarray,         # (n_samples, n) out or (0, 0)
    C_csr_row_start: np.ndarray,   # (n+1,) CSR row pointers
    C_csr_col: np.ndarray,         # (nnz,) CSR column indices
    C_csr_val: np.ndarray,         # (nnz,) CSR values
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
    sqrt_eps = np.sqrt(max(epsilon, 1e-9))
    delay_depth = delay_buffer.shape[0] if delay_enabled else 0
    delay_write = delay_write_start
    n_samples = xs.shape[0]
    eps_guard = 1e-12
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
        # Precompute A_i = 1/(1-√ε z̄_i) when needed by either
        # integer-ratio coupling OR canonical nonlinear input.
        need_A = C_enabled or nonlinear_input
        if need_A:
            for i in range(n):
                zc = complex(z[i].real, -z[i].imag)
                d2 = 1.0 - sqrt_eps * zc
                d2_mag2 = d2.real * d2.real + d2.imag * d2.imag
                if d2_mag2 < eps_guard:
                    A_scratch[i] = 0.0 + 0.0j
                else:
                    A_scratch[i] = 1.0 / d2
        # Integer-ratio internal coupling: P_z = z_j/(1-√ε z_j), then
        # internal_i = A_i · Σ_j C[i,j] · P_z[j]. Series expansion of
        # P*A captures all integer-ratio resonances naturally.
        if C_enabled:
            for i in range(n):
                d = 1.0 - sqrt_eps * z[i]
                d_mag2 = d.real * d.real + d.imag * d.imag
                if d_mag2 < eps_guard:
                    P_scratch[i] = 0.0 + 0.0j
                else:
                    P_scratch[i] = z[i] / d
            # CSR sparse matvec: ~9× faster than dense at our typical
            # 10% non-zero coupling density.
            for i in range(n):
                acc = 0.0 + 0.0j
                row_lo = C_csr_row_start[i]
                row_hi = C_csr_row_start[i + 1]
                for k in range(row_lo, row_hi):
                    acc += C_csr_val[k] * P_scratch[C_csr_col[k]]
                internal_scratch[i] = A_scratch[i] * acc
        # Canonical nonlinear input: drive_i = P(ε, x_i)·A(ε, z̄_i)
        # where P(ε, x) = x/(1-√ε x). Per Large 2010 Eq. 15, Lerud
        # 2014 Eq. 2, Kim & Large 2019 Eq. 15. Held constant across
        # RK4 stages (same approximation as W and C).
        if nonlinear_input:
            for i in range(n):
                d = 1.0 - sqrt_eps * xs[t, i]
                d_mag2 = d.real * d.real + d.imag * d.imag
                if d_mag2 < eps_guard:
                    P_input_scratch[i] = 0.0 + 0.0j
                else:
                    P_input_scratch[i] = xs[t, i] / d
                canonical_drive[i] = P_input_scratch[i] * A_scratch[i]
        # RK4 stage 1
        for i in range(n):
            s = tau_scale[i] if tau_enabled else 1.0
            zi = z[i]
            abs2 = zi.real * zi.real + zi.imag * zi.imag
            abs2_sat = abs2 if abs2 < sat_limit else sat_limit
            denom = 1.0 - epsilon * abs2_sat
            cubic = beta1c * abs2
            quintic = epsilon * beta2c * abs2_sat * abs2_sat / denom
            linear = complex(s * alpha, omega[i])
            if nonlinear_input:
                drive = canonical_drive[i]
            else:
                drive = xs[t, i]
            rhs = zi * (linear + s * cubic + s * quintic) + s * drive
            if W_enabled:
                rhs = rhs + wz_scaled[i]
            if C_enabled:
                rhs = rhs + coupling_gain * internal_scratch[i]
            if delay_enabled:
                rhs = rhs + delay_gain * delay_buffer[delay_write, i]
            k1[i] = rhs
        # RK4 stage 2
        for i in range(n):
            s = tau_scale[i] if tau_enabled else 1.0
            ztmp_i = z[i] + 0.5 * dt * k1[i]
            abs2 = ztmp_i.real * ztmp_i.real + ztmp_i.imag * ztmp_i.imag
            abs2_sat = abs2 if abs2 < sat_limit else sat_limit
            denom = 1.0 - epsilon * abs2_sat
            cubic = beta1c * abs2
            quintic = epsilon * beta2c * abs2_sat * abs2_sat / denom
            linear = complex(s * alpha, omega[i])
            if nonlinear_input:
                drive = canonical_drive[i]
            else:
                drive = xs[t, i]
            rhs = ztmp_i * (linear + s * cubic + s * quintic) + s * drive
            if W_enabled:
                rhs = rhs + wz_scaled[i]
            if C_enabled:
                rhs = rhs + coupling_gain * internal_scratch[i]
            if delay_enabled:
                rhs = rhs + delay_gain * delay_buffer[delay_write, i]
            k2[i] = rhs
        # RK4 stage 3
        for i in range(n):
            s = tau_scale[i] if tau_enabled else 1.0
            ztmp_i = z[i] + 0.5 * dt * k2[i]
            abs2 = ztmp_i.real * ztmp_i.real + ztmp_i.imag * ztmp_i.imag
            abs2_sat = abs2 if abs2 < sat_limit else sat_limit
            denom = 1.0 - epsilon * abs2_sat
            cubic = beta1c * abs2
            quintic = epsilon * beta2c * abs2_sat * abs2_sat / denom
            linear = complex(s * alpha, omega[i])
            if nonlinear_input:
                drive = canonical_drive[i]
            else:
                drive = xs[t, i]
            rhs = ztmp_i * (linear + s * cubic + s * quintic) + s * drive
            if W_enabled:
                rhs = rhs + wz_scaled[i]
            if C_enabled:
                rhs = rhs + coupling_gain * internal_scratch[i]
            if delay_enabled:
                rhs = rhs + delay_gain * delay_buffer[delay_write, i]
            k3[i] = rhs
        # RK4 stage 4
        for i in range(n):
            s = tau_scale[i] if tau_enabled else 1.0
            ztmp_i = z[i] + dt * k3[i]
            abs2 = ztmp_i.real * ztmp_i.real + ztmp_i.imag * ztmp_i.imag
            abs2_sat = abs2 if abs2 < sat_limit else sat_limit
            denom = 1.0 - epsilon * abs2_sat
            cubic = beta1c * abs2
            quintic = epsilon * beta2c * abs2_sat * abs2_sat / denom
            linear = complex(s * alpha, omega[i])
            if nonlinear_input:
                drive = canonical_drive[i]
            else:
                drive = xs[t, i]
            rhs = ztmp_i * (linear + s * cubic + s * quintic) + s * drive
            if W_enabled:
                rhs = rhs + wz_scaled[i]
            if C_enabled:
                rhs = rhs + coupling_gain * internal_scratch[i]
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
        # Record per-sample z to history (for cascaded multi-layer)
        if record_z:
            for i in range(n):
                z_history[t, i] = z[i]
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
        freqs: np.ndarray | None = None,
        coupling_kernel: np.ndarray | None = None,
        coupling_gain: float = 0.0,
        per_oscillator_tau: bool = False,
        tau_reference_hz: float = 1.0,
        nonlinear_input: bool = False,
        adaptive_frequency: bool = False,
        adaptive_lambda_freq: float = 4.0,
        adaptive_lambda_elastic: float = 2.0,
        adaptive_freq_min_ratio: float = 0.5,
        adaptive_freq_max_ratio: float = 2.0,
    ):
        # When ``freqs`` is provided, it fully determines the oscillator
        # layout — the pitch bank passes a 12-TET-aligned grid here so
        # named notes land on bins. ``n_oscillators``/``low_hz``/
        # ``high_hz`` are ignored in that case. Other banks (rhythm,
        # motor) still use the log-uniform default via geomspace.
        if freqs is not None:
            self.f = np.asarray(freqs, dtype=np.float64).copy()
            n_oscillators = int(self.f.size)
        else:
            self.f = np.geomspace(low_hz, high_hz, n_oscillators).astype(np.float64)
        self.n = n_oscillators
        self.dt = dt
        self.p = params
        self.omega = 2 * np.pi * self.f
        # Preallocated RK4 scratch buffers — avoid a fresh heap
        # allocation on every audio-rate step.
        self._k1 = np.empty(n_oscillators, dtype=np.complex128)
        self._k2 = np.empty(n_oscillators, dtype=np.complex128)
        self._k3 = np.empty(n_oscillators, dtype=np.complex128)
        self._k4 = np.empty(n_oscillators, dtype=np.complex128)
        self._ztmp = np.empty(n_oscillators, dtype=np.complex128)
        self._wz_scaled = np.empty(n_oscillators, dtype=np.complex128)
        # Initial state: zero. With alpha<0 (damped) and noise.amp=0,
        # the oscillators need to be driven from silence by real input.
        # The previous random-seed initialization (~1e-3 per osc) took
        # ~20 seconds to decay with alpha=-0.05 and produced a
        # structured baseline (bins near 31 Hz at amp ~0.003) that
        # drowned quiet signals. If stochastic initialization is needed
        # in a specific scenario, add a config option.
        self.z = np.zeros(n_oscillators, dtype=np.complex128)
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

        # Internal coupling kernel D — pre-wired connections at integer
        # frequency ratios. Generates phantom fundamentals and reinforces
        # harmonic coherence directly through the dynamics. Held as a
        # complex (n, n) matrix; entries non-zero only where two
        # oscillators are at small-integer frequency ratios. Public domain
        # (US 7,376,562, expired Nov 2024).
        self.coupling_gain = float(coupling_gain)
        self.coupling_enabled = (coupling_kernel is not None
                                  and self.coupling_gain != 0.0)
        if self.coupling_enabled:
            self.C = np.asarray(coupling_kernel, dtype=np.complex128).copy()
            if self.C.shape != (n_oscillators, n_oscillators):
                raise ValueError(
                    f"coupling_kernel shape {self.C.shape} does not match "
                    f"({n_oscillators}, {n_oscillators})"
                )
            self._P_scratch = np.empty(n_oscillators, dtype=np.complex128)
            self._internal_scratch = np.empty(n_oscillators, dtype=np.complex128)
            # Sparse (CSR) representation of C — coupling kernels are
            # ~10% non-zero (local kernel limits + integer-ratio
            # selectivity). Dense per-row inner products dominated
            # the hot path; sparse iteration is ~9× faster at typical
            # density.
            nonzero_per_row = (np.abs(self.C) > 0).sum(axis=1)
            self._C_sparse_row_start = np.zeros(
                n_oscillators + 1, dtype=np.int64
            )
            self._C_sparse_row_start[1:] = np.cumsum(nonzero_per_row)
            total_nnz = int(self._C_sparse_row_start[-1])
            self._C_sparse_col = np.zeros(total_nnz, dtype=np.int64)
            self._C_sparse_val = np.zeros(total_nnz, dtype=np.complex128)
            idx = 0
            for i in range(n_oscillators):
                for j in range(n_oscillators):
                    c_ij = self.C[i, j]
                    if c_ij != 0:
                        self._C_sparse_col[idx] = j
                        self._C_sparse_val[idx] = c_ij
                        idx += 1
        else:
            self.C = None
            self._P_scratch = None
            self._internal_scratch = None
            self._C_sparse_row_start = None
            self._C_sparse_col = None
            self._C_sparse_val = None

        # Canonical nonlinear input transformation P(ε,x)·A(ε,z̄).
        # When enabled, external stimulus enters via the canonical NRT
        # form rather than linearly. Generates harmonics, subharmonics,
        # combination frequencies, and 1:n mode-locking natively. Per
        # Large 2010 Eq. 15, Lerud 2014 Eq. 2, Kim & Large 2019 Eq. 15.
        self.nonlinear_input = bool(nonlinear_input)
        if self.nonlinear_input:
            self._P_input_scratch = np.empty(n_oscillators, dtype=np.complex128)
            self._canonical_drive = np.empty(n_oscillators, dtype=np.complex128)
        else:
            self._P_input_scratch = None
            self._canonical_drive = None
        # A_i = 1/(1-√ε z̄_i) is shared between integer-ratio coupling
        # and canonical input. Allocate when either is enabled.
        if self.coupling_enabled or self.nonlinear_input:
            self._A_scratch = np.empty(n_oscillators, dtype=np.complex128)
        else:
            self._A_scratch = None

        # Per-oscillator time scale τ_n = 1/f_n. When enabled, the
        # dimensionless dynamical coefficients (α, β1, β2, drive) are
        # scaled by f_n / tau_reference_hz per oscillator. Effect: all
        # oscillators integrate over the same number of cycles
        # (∼1/|α| × tau_reference_hz cycles) instead of the same
        # number of seconds, which is fundamentally correct for audio
        # analysis. Public domain (US 7,376,562, expired Nov 2024).
        self.tau_enabled = bool(per_oscillator_tau)
        if self.tau_enabled:
            self.tau_scale = (self.f / max(float(tau_reference_hz), 1e-9)
                              ).astype(np.float64)
        else:
            self.tau_scale = np.ones(n_oscillators, dtype=np.float64)

        # Adaptive natural frequencies per Roman et al. 2023 (ASHLE).
        # Each oscillator's natural frequency f_i tracks the
        # instantaneous frequency of its drive when driven, then
        # elastically returns to its original value f_0_i in absence
        # of drive. Implements the simplest single-osc-per-bank ASHLE
        # form on the bank as a whole — each oscillator behaves like
        # an independent ASHLE sensory tracker with f_0 = its initial
        # log-spaced frequency.
        #
        # Equation per oscillator (Roman 2023):
        #   ḟ = f · (λ₁ |x| sin(φ_x − φ_z) − λ_e (exp((f − f_0)/f_0) − 1))
        # First term: Hebbian frequency learning toward drive phase
        # rate (scaled by drive amplitude so silent osc doesn't drift)
        # Second term: elasticity — exponential pull back to f_0.
        self.adaptive_freq_enabled = bool(adaptive_frequency)
        self.f_natural = self.f.copy()
        self.adaptive_lambda_freq = float(adaptive_lambda_freq)
        self.adaptive_lambda_elastic = float(adaptive_lambda_elastic)
        self.adaptive_freq_min_ratio = float(adaptive_freq_min_ratio)
        self.adaptive_freq_max_ratio = float(adaptive_freq_max_ratio)

    def _deriv(self, z: np.ndarray, x: np.ndarray,
               delayed: np.ndarray | None = None) -> np.ndarray:
        p = self.p
        abs2 = (z * z.conj()).real
        abs2_sat = np.minimum(abs2, 0.95 / max(p.epsilon, 1e-9))
        denom = 1.0 - p.epsilon * abs2_sat
        cubic = (p.beta1 + 1j * p.delta1) * abs2
        quintic = p.epsilon * (p.beta2 + 1j * p.delta2) * abs2_sat * abs2_sat / denom
        linear = p.alpha + 1j * self.omega
        if self.nonlinear_input:
            sqrt_eps = np.sqrt(max(p.epsilon, 1e-9))
            P_in = x / (1.0 - sqrt_eps * x + 1e-12)
            A_z = 1.0 / (1.0 - sqrt_eps * z.conj() + 1e-12)
            drive = P_in * A_z
        else:
            drive = x
        rhs = z * (linear + cubic + quintic) + drive
        if self.W is not None:
            # Mean-field normalization: divide by n so the intra-layer
            # coupling term is the AVERAGE partner contribution, not the
            # sum. Without this, network sizes affect dynamics scale and
            # large layers easily cascade into the integrator clamp.
            rhs = rhs + (self.W @ z) / self.n
        if self.coupling_enabled:
            sqrt_eps = np.sqrt(max(p.epsilon, 1e-9))
            P = z / (1.0 - sqrt_eps * z + 1e-12)
            A = 1.0 / (1.0 - sqrt_eps * z.conj() + 1e-12)
            rhs = rhs + self.coupling_gain * A * (self.C @ P)
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
        if self.coupling_enabled:
            C_eff = self.C
            P_eff = self._P_scratch
            internal_eff = self._internal_scratch
            csr_row = self._C_sparse_row_start
            csr_col = self._C_sparse_col
            csr_val = self._C_sparse_val
        else:
            C_eff = _EMPTY_C
            P_eff = np.empty(0, dtype=np.complex128)
            internal_eff = np.empty(0, dtype=np.complex128)
            csr_row = np.zeros(1, dtype=np.int64)
            csr_col = np.zeros(0, dtype=np.int64)
            csr_val = np.zeros(0, dtype=np.complex128)
        if self.coupling_enabled or self.nonlinear_input:
            A_eff = self._A_scratch
        else:
            A_eff = np.empty(0, dtype=np.complex128)
        if self.nonlinear_input:
            P_input_eff = self._P_input_scratch
            canonical_eff = self._canonical_drive
        else:
            P_input_eff = np.empty(0, dtype=np.complex128)
            canonical_eff = np.empty(0, dtype=np.complex128)
        _rk4_step_jit(
            self.z, x, self.omega,
            p.alpha, p.beta1, p.beta2, p.delta1, p.delta2, p.epsilon,
            self.W if self.W is not None else _EMPTY_W,
            self.W is not None, self.n,
            delayed, self.delay_enabled, self.delay_gain,
            self.dt,
            self._k1, self._k2, self._k3, self._k4,
            self._ztmp, self._wz_scaled,
            C_eff, self.coupling_enabled, self.coupling_gain,
            P_eff, A_eff, internal_eff,
            self.tau_scale, self.tau_enabled,
            self.nonlinear_input, P_input_eff, canonical_eff,
            csr_row, csr_col, csr_val,
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
            self._hebbian_update(n_steps=1)
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
        # Coupling kernel: pass real matrix + scratch buffers when
        # enabled; pass empty otherwise (JIT signature still resolves).
        if self.coupling_enabled:
            C_eff = self.C
            P_eff = self._P_scratch
            internal_eff = self._internal_scratch
            csr_row = self._C_sparse_row_start
            csr_col = self._C_sparse_col
            csr_val = self._C_sparse_val
        else:
            C_eff = _EMPTY_C
            P_eff = np.empty(0, dtype=np.complex128)
            internal_eff = np.empty(0, dtype=np.complex128)
            csr_row = np.zeros(1, dtype=np.int64)
            csr_col = np.zeros(0, dtype=np.int64)
            csr_val = np.zeros(0, dtype=np.complex128)
        if self.coupling_enabled or self.nonlinear_input:
            A_eff = self._A_scratch
        else:
            A_eff = np.empty(0, dtype=np.complex128)
        if self.nonlinear_input:
            P_input_eff = self._P_input_scratch
            canonical_eff = self._canonical_drive
        else:
            P_input_eff = np.empty(0, dtype=np.complex128)
            canonical_eff = np.empty(0, dtype=np.complex128)
        # Ensure xs is complex128 (JIT doesn't coerce).
        xs = np.ascontiguousarray(xs, dtype=np.complex128)
        empty_history = np.zeros((0, 0), dtype=np.complex128)
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
            C_eff, self.coupling_enabled, self.coupling_gain,
            P_eff, A_eff, internal_eff,
            self.tau_scale, self.tau_enabled,
            self.nonlinear_input, P_input_eff, canonical_eff,
            False, empty_history,
            csr_row, csr_col, csr_val,
        )
        if self.delay_enabled:
            self._delay_write = int(new_delay_write)
        if self.W is not None:
            self._hebbian_update(n_steps=xs.shape[0])
        if self.adaptive_freq_enabled:
            self._update_adaptive_freq(xs, n_steps=xs.shape[0])

    def step_many_record(self, xs: np.ndarray) -> np.ndarray:
        """Like step_many, but records z trajectory per sample.

        Returns shape ``(n_samples, n_oscillators)`` complex —
        the oscillator state after each sample step. Used for
        cascaded multi-layer architectures (Lerud 2014):
        downstream layers receive the per-sample trajectory of
        upstream layers rather than just the final-state snapshot.
        """
        if xs.ndim != 2 or xs.shape[1] != self.n:
            raise ValueError(
                f"xs must be (n_samples, {self.n}); got {xs.shape}"
            )
        n_samples = xs.shape[0]
        p = self.p
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
        if self.coupling_enabled:
            C_eff = self.C
            P_eff = self._P_scratch
            internal_eff = self._internal_scratch
            csr_row = self._C_sparse_row_start
            csr_col = self._C_sparse_col
            csr_val = self._C_sparse_val
        else:
            C_eff = _EMPTY_C
            P_eff = np.empty(0, dtype=np.complex128)
            internal_eff = np.empty(0, dtype=np.complex128)
            csr_row = np.zeros(1, dtype=np.int64)
            csr_col = np.zeros(0, dtype=np.int64)
            csr_val = np.zeros(0, dtype=np.complex128)
        if self.coupling_enabled or self.nonlinear_input:
            A_eff = self._A_scratch
        else:
            A_eff = np.empty(0, dtype=np.complex128)
        if self.nonlinear_input:
            P_input_eff = self._P_input_scratch
            canonical_eff = self._canonical_drive
        else:
            P_input_eff = np.empty(0, dtype=np.complex128)
            canonical_eff = np.empty(0, dtype=np.complex128)
        xs = np.ascontiguousarray(xs, dtype=np.complex128)
        z_history = np.empty((n_samples, self.n), dtype=np.complex128)
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
            C_eff, self.coupling_enabled, self.coupling_gain,
            P_eff, A_eff, internal_eff,
            self.tau_scale, self.tau_enabled,
            self.nonlinear_input, P_input_eff, canonical_eff,
            True, z_history,
            csr_row, csr_col, csr_val,
        )
        if self.delay_enabled:
            self._delay_write = int(new_delay_write)
        if self.W is not None:
            self._hebbian_update(n_steps=xs.shape[0])
        if self.adaptive_freq_enabled:
            self._update_adaptive_freq(xs, n_steps=xs.shape[0])
        return z_history

    def _update_adaptive_freq(self, xs_chunk: np.ndarray,
                              n_steps: int) -> None:
        """Adaptive natural frequency update per Roman 2023 (ASHLE).

        Updates each oscillator's natural frequency toward the drive
        phase rate when driven, with elasticity pulling back toward
        the original f_0. Called once per step_many chunk; the time
        scale of frequency dynamics (~seconds) is much slower than
        chunk size (~ms), so chunk-level Euler integration is
        sufficient — no per-sample JIT overhead needed.
        """
        if not self.adaptive_freq_enabled or n_steps < 1:
            return
        # End-of-chunk drive: instantaneous phase at the same moment
        # self.z is measured (after step_many advances). Using the
        # mean over a sinusoidal chunk would average to zero and lose
        # the phase entirely (~|mean_x| ≪ |x_inst| for chunks longer
        # than a fraction of an oscillation period).
        last_x = xs_chunk[-1]
        phi_x = np.angle(last_x)
        phi_z = np.angle(self.z)
        # Use the chunk-averaged magnitude as the activation
        # weight — that is robust to phase rotation and reflects
        # how strongly this oscillator is being driven over the
        # window.
        amp_x = np.abs(xs_chunk).mean(axis=0)
        # Frequency-learning term: gradient toward sin(φ_x − φ_z) = 0
        # (i.e., toward phase-aligned, which means same instantaneous
        # frequency). Scaled by |x| so silent oscillators don't drift.
        sin_dphi = np.sin(phi_x - phi_z)
        # Elasticity: pull back toward original f_0_i
        rel_f = (self.f - self.f_natural) / np.maximum(
            self.f_natural, 1e-9
        )
        elastic = np.exp(rel_f) - 1.0
        df_dt = self.f * (
            self.adaptive_lambda_freq * amp_x * sin_dphi
            - self.adaptive_lambda_elastic * elastic
        )
        effective_dt = self.dt * float(n_steps)
        self.f = self.f + effective_dt * df_dt
        # Clamp to a physiologically reasonable band around f_0 to
        # prevent runaway under pathological drive (e.g., DC drives
        # could push f → 0).
        f_min = self.adaptive_freq_min_ratio * self.f_natural
        f_max = self.adaptive_freq_max_ratio * self.f_natural
        np.clip(self.f, f_min, f_max, out=self.f)
        # ω in rad/s; keep tau_scale in sync if enabled.
        self.omega = 2 * np.pi * self.f

    def _hebbian_update(self, n_steps: int = 1) -> None:
        """Euler step on the weight matrix.

        ``n_steps`` is the number of audio samples represented by
        this update. ``step()`` (per-sample API) passes 1;
        ``step_many()`` passes the chunk size. The effective dt of
        the Hebbian Euler step is ``self.dt * n_steps`` — this
        keeps the learning rate per unit time consistent regardless
        of chunk size. Previously the update used self.dt alone,
        which made step_many learn ``chunk_size``× slower than
        step for the same effective audio duration.

        Multi-frequency Hebbian rule per Kim & Large 2021 Eq. 26 /
        Large 2016 Eq. A2:

            dW_ij/dt = -lambda * W_ij + kappa * P(z_i) * conj(P(z_j))

        where P(z) = z / (1 - √ε z). The P-transform expands z into
        a series with content at all integer multiples of z's
        frequency. Therefore the outer product P(z_i)·conj(P(z_j))
        becomes STATIONARY (time-constant) when oscillators i and j
        are mode-locked at any small-integer k:m ratio, growing
        W[i,j] until decay balances it. Non-locked pairs oscillate
        through the outer product and time-average to zero.

        This is the mechanism that lets W community structure
        encode harmonic relationships: oscillators at integer
        ratios of one voice's fundamental develop strong mutual
        connections, while oscillators at unrelated ratios stay
        decoupled.

        Reduces to the classical 1:1 Hebbian rule
        ``ċ_ij = -λ c_ij + κ z_i z̄_j`` (Hoppensteadt-Izhikevich
        1996b Eq. 14) at low |z|, so existing single-frequency
        learning behavior is preserved at typical audio amplitudes.

        Diagonal kept at zero — an oscillator does not connect
        to itself.

        Patent IP: covered by US 8,930,292 (active to 2032) Claims
        1, 10. For research/personal use: fine. For commercial use
        a license or a patent-safe variant (cubic damping on c_ij
        per K&L 2021 Eq. 9 Strategy 2) is needed.
        """
        if self.learn_rate == 0.0 and self.weight_decay == 0.0:
            return
        p = self.p
        sqrt_eps = np.sqrt(max(p.epsilon, 1e-9))
        # P-transforms with denominator guard. The 1e-12 keeps the
        # division stable even if |z| approaches 1/√ε (the canonical
        # model's hard amplitude limit).
        z_P = self.z / (1.0 - sqrt_eps * self.z + 1e-12)
        z_conj_P = self.z.conj() / (1.0 - sqrt_eps * self.z.conj()
                                     + 1e-12)
        # Outer product: pairs P(z_i) with conj(P(z_j)). Per K&L
        # 2021, this is dominated locally by the lowest-order
        # resonant monomial for each k:m ratio.
        outer = np.outer(z_P, z_conj_P)
        effective_dt = self.dt * float(n_steps)
        self.W = self.W + effective_dt * (
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
