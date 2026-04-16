"""Mode-lock detection for GrFNN oscillator banks.

Mode-locking is the signature NRT prediction: oscillators coupled via
driven resonance (or direct intra-layer connections) phase-lock at
small-integer frequency ratios. The stability hierarchy is 1:1 > 2:1 > 3:2
> 7:4 (see Fig. 3 of Harding/Kim/Large 2025).

This module exposes two primitives:

- ``phase_locking_value(phi_a, phi_b, p, q)`` — scalar PLV for a phase pair
  at ratio p:q. PLV ∈ [0, 1]: 1 = perfect lock, 0 = uncorrelated.
- ``detect_mode_locks(phases, threshold, ratios)`` — scan all oscillator
  pairs for mode-locks at any of the given integer ratios.

Typical use: accumulate a rolling buffer of phase snapshots from a GrFNN,
then run ``detect_mode_locks`` every few hundred ms to emit a stream of
active locks. This is the input to "polyrhythm detected" style consumer
signals.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def phase_locking_value(
    phi_a: np.ndarray, phi_b: np.ndarray,
    ratio_p: int, ratio_q: int,
) -> float:
    """Compute PLV between two phase trajectories at frequency ratio p:q.

    Convention: ratio p:q means f_a : f_b = p : q, i.e. oscillator A
    completes p cycles for every q cycles of B. Under stable lock,
    ``q * phi_a - p * phi_b`` stays constant over time, driving PLV → 1.

    PLV = | (1/T) Σ_t exp(i * (q * phi_a - p * phi_b)) |

    Parameters
    ----------
    phi_a, phi_b : 1D arrays of same length. Phase of each oscillator
        at each time step (can be wrapped or unwrapped — the mod-2π of
        the difference is what matters).
    ratio_p, ratio_q : positive integers. Frequency ratio f_a : f_b = p : q.

    Returns
    -------
    float in [0.0, 1.0]. 1 = perfect lock, 0 = uncorrelated.
    """
    phi_a = np.asarray(phi_a, dtype=np.float64)
    phi_b = np.asarray(phi_b, dtype=np.float64)
    if phi_a.shape != phi_b.shape:
        raise ValueError("phi_a and phi_b must have the same shape")
    rel = ratio_q * phi_a - ratio_p * phi_b
    return float(np.abs(np.mean(np.exp(1j * rel))))


def detect_mode_locks(
    phases: np.ndarray,
    threshold: float = 0.85,
    ratios: Iterable[tuple[int, int]] = (
        (1, 1), (2, 1), (3, 2), (4, 3), (5, 4), (7, 4),
    ),
) -> list[dict]:
    """Scan all oscillator pairs for mode-locks at any of the given ratios.

    Parameters
    ----------
    phases : ndarray shape (T, n_osc). Phase of each oscillator at each step.
    threshold : minimum PLV to count as a lock.
    ratios : sequence of (p, q) integer ratios to test. The default covers
             the most stable ratios in NRT's hierarchy.

    Returns
    -------
    List of dicts with keys i, j, p, q, plv — one entry per detected lock.
    We test each unordered oscillator pair (i < j) at each ratio; a given
    pair can lock at multiple ratios (rare but possible under noise).
    """
    phases = np.asarray(phases)
    if phases.ndim != 2:
        raise ValueError(f"phases must be 2D (T, n_osc), got shape {phases.shape}")
    _T, n = phases.shape
    out: list[dict] = []
    for i in range(n):
        for j in range(i + 1, n):
            for p, q in ratios:
                plv = phase_locking_value(phases[:, i], phases[:, j], p, q)
                if plv >= threshold:
                    out.append({"i": i, "j": j, "p": p, "q": q, "plv": plv})
    return out
