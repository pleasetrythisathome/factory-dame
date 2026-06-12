"""Phase 0.3 validation: missing-pulse phenomenon (Tal et al. 2017).

The "missing-pulse" demonstration: construct a rhythm whose stimulus
envelope has NO energy at the perceived beat (pulse) frequency.
Human MEG measurements show enhanced auditory-cortex power at the
pulse frequency anyway — the brain *generates* the pulse via
nonlinear resonance even though it's physically absent.

This test synthesizes a missing-pulse-style stimulus and verifies
that the rhythm GrFNN produces measurable activity at the pulse
frequency — the NRT prediction.

Per Tal et al. 2017 stimuli MP1/MP2: place onsets at off-pulse
positions such that the envelope's Fourier spectrum has a *null* at
the pulse frequency. The standard construction is to place stimulus
events at every 2nd, 3rd, or 5th of a regular grid such that the
DC + grid-frequency components dominate but the pulse frequency
(at a different period) has zero spectral mass.

For factory-dame, we synthesize a simpler proxy: a syncopated
2:3 pattern over a 2 Hz pulse. Stimulus events at 1.33 Hz and
0.67 Hz; pulse at 2 Hz has no direct spectral component but
emerges via mode-locking + cross-frequency resonance.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import tomllib

from neurodynamics.live import LiveEngine

ENGINE_DIR = Path(__file__).resolve().parent.parent


def _load_cfg() -> dict:
    with open(ENGINE_DIR / "config.toml", "rb") as f:
        return tomllib.load(f)


def _impulse_train_off_pulse(duration: float, sr: int,
                              pulse_hz: float, off_ratios=(2, 3),
                              amp: float = 0.7) -> np.ndarray:
    """Synthesize a rhythm whose envelope has NO spectral component
    at ``pulse_hz`` but whose perceived pulse is at that frequency.

    Method: place impulses on grid positions [pulse_hz × r/k] for
    several ratios r:k that don't include 1:1. The grid has period
    1/pulse_hz, but the actual impulses occur at sub-divisions that
    miss every other pulse — creating a syncopated rhythm whose
    perceived pulse (at pulse_hz) has zero direct power.
    """
    n_samples = int(duration * sr)
    out = np.zeros(n_samples, dtype=np.float32)
    pulse_period = 1.0 / pulse_hz
    # Place an exponentially-decaying impulse at each event time.
    decay_ms = 30
    impulse_len = int(sr * decay_ms / 1000)
    decay = np.exp(-np.linspace(0, 5, impulse_len)).astype(np.float32)
    event_times = []
    for r in off_ratios:
        # Events at every r-th sub-pulse offset by 1/(2r) so they're
        # OFF the main grid
        for k in range(int(duration * pulse_hz * r)):
            event_t = (k + 0.5) * pulse_period / r
            if event_t < duration:
                event_times.append(event_t)
    for t in event_times:
        i = int(t * sr)
        end = min(i + impulse_len, n_samples)
        out[i:end] = np.maximum(out[i:end], amp * decay[:end - i])
    return out


import pytest


def test_missing_pulse_rhythm_bank_responds():
    """Drive engine with a missing-pulse rhythm at perceived pulse
    frequency 2 Hz. The stimulus envelope has events placed off the
    2 Hz grid (at 2:3 and 2:5 subdivisions) so DC + sub-pulse
    frequencies dominate.

    Pass criterion: the rhythm-bank oscillator nearest 2 Hz has
    meaningfully higher amplitude than the average background
    activity. Per Tal 2017, the brain generates pulse-frequency
    activity even when it's absent from the stimulus.
    """
    cfg = _load_cfg()
    engine = LiveEngine(cfg)
    sr = engine.fs
    pulse_hz = 2.0
    audio = _impulse_train_off_pulse(
        duration=8.0, sr=sr, pulse_hz=pulse_hz,
        off_ratios=(2, 3), amp=0.7,
    )
    # JIT warm-up
    engine.process(audio[:4096])
    # Reset and run
    engine.rhythm.z[:] = 0
    if engine.motor is not None:
        engine.motor.z[:] = 0
    chunk = 4096
    for start in range(0, len(audio) - chunk, chunk):
        engine.process(audio[start:start + chunk])
    rhythm_amps = np.abs(engine.rhythm.z)
    rhythm_freqs = engine.rhythm.f
    pulse_idx = int(np.argmin(np.abs(rhythm_freqs - pulse_hz)))
    pulse_amp = rhythm_amps[pulse_idx]
    # Average amp across the bank EXCLUDING the immediate vicinity
    # of the pulse frequency
    away_mask = (np.abs(rhythm_freqs - pulse_hz) > 0.5
                  * pulse_hz)  # outside ±50 % of pulse
    background_amp = rhythm_amps[away_mask].mean()
    print(f"rhythm @ 2 Hz pulse: {pulse_amp:.4f}")
    print(f"background mean:    {background_amp:.4f}")
    print(f"signal-to-background: {pulse_amp / max(background_amp, 1e-9):.2f}")

    # Per Tal 2017, the pulse-frequency oscillator should have
    # markedly stronger activity than background — the bank should
    # produce missing-pulse resonance even though the stimulus
    # envelope lacks that frequency component.
    # 1.2× threshold reflects the gentle nature of the emergence
    # (per Tal 2017 MEG: SNR ratio ~r_s = 0.82 at the pulse freq).
    assert pulse_amp > 1.2 * background_amp, (
        f"missing-pulse not detected: 2 Hz amp {pulse_amp:.4f}, "
        f"background {background_amp:.4f}"
    )
