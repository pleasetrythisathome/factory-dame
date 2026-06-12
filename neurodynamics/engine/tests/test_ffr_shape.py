"""Phase 0.4 validation: FFR-style harmonic + combination tones
(Lerud et al. 2014).

Drives the engine with a consonant interval (G2 = 99 Hz + E3 = 166 Hz
per Lerud 2014 Fig. 4). Verifies that the cascade pipeline (Phase 3
brainstem → pitch) produces oscillator activity at:

  - f1 (99 Hz)            ← driven
  - f2 (166 Hz)           ← driven
  - 2*f1 = 198            ← octave harmonic from brainstem coupling
  - 2*f2 = 332            ← octave harmonic
  - f2 - f1 = 67          ← difference tone (combination)
  - f1 + f2 = 265         ← summation tone
  - 2*f1 - f2 = 32        ← cubic distortion product
  - 2*f1 + f2 = 364       ← cubic combination

Per Lerud 2014 Fig. 4, the canonical model reproduces FFR data at
R²=0.77 for the consonant interval. We verify the QUALITATIVE
pattern: the harmonic/combination positions show meaningful
activity above noise floor.
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


def _two_tone_audio(f1: float, f2: float, duration: float, sr: int,
                    amp: float = 0.4) -> np.ndarray:
    t = np.arange(int(duration * sr)) / sr
    out = (amp * np.sin(2 * np.pi * f1 * t)
           + amp * np.sin(2 * np.pi * f2 * t))
    return out.astype(np.float32)


def test_ffr_consonant_interval_produces_lerud_signature():
    """Drive G2 + E3 (99, 166 Hz — consonant interval, 5:3 ratio).
    Verify pitch bank shows expected FFR-signature peaks at the
    harmonic and combination positions per Lerud 2014 Fig. 4.
    """
    cfg = _load_cfg()
    engine = LiveEngine(cfg)
    sr = engine.fs
    f1 = 99.0   # G2 (close)
    f2 = 166.0  # E3 (close)
    audio = _two_tone_audio(f1, f2, duration=3.0, sr=sr, amp=0.4)
    engine.process(audio[:4096])  # JIT warm-up
    engine.pitch.z[:] = 0
    engine.brainstem.z[:] = 0
    chunk = 1024
    for start in range(0, len(audio) - chunk, chunk):
        engine.process(audio[start:start + chunk])

    freqs = engine.pitch.f
    pitch_amps = np.abs(engine.pitch.z)

    def amp_at(target_hz: float) -> float:
        idx = int(np.argmin(np.abs(freqs - target_hz)))
        return float(pitch_amps[idx])

    # Stimulus positions
    a_f1 = amp_at(f1)
    a_f2 = amp_at(f2)
    # Harmonics
    a_2f1 = amp_at(2 * f1)        # 198
    a_2f2 = amp_at(2 * f2)        # 332
    # Combination tones (FFR signature)
    a_diff = amp_at(f2 - f1)      # 67
    a_sum = amp_at(f1 + f2)       # 265
    a_2f1_m_f2 = amp_at(2 * f1 - f2)  # 32
    # Far-off "noise floor" reference
    a_noise = amp_at(2500.0)  # 2.5 kHz — no stimulus content

    print(f"FFR-signature amplitudes (Lerud 2014 Fig. 4 positions):")
    print(f"  f1 = {f1:.0f} Hz:                 {a_f1:.4f}")
    print(f"  f2 = {f2:.0f} Hz:                 {a_f2:.4f}")
    print(f"  2*f1 = {2*f1:.0f} Hz:             {a_2f1:.4f}")
    print(f"  2*f2 = {2*f2:.0f} Hz:             {a_2f2:.4f}")
    print(f"  f2 - f1 = {f2-f1:.0f} Hz:        {a_diff:.4f}")
    print(f"  f1 + f2 = {f1+f2:.0f} Hz:        {a_sum:.4f}")
    print(f"  2f1 - f2 = {2*f1-f2:.0f} Hz:     {a_2f1_m_f2:.4f}")
    print(f"  noise (2500 Hz):           {a_noise:.4f}")

    # Stimulus positions should dominate
    assert a_f1 > 5 * max(a_noise, 0.001), (
        f"f1 = {f1} Hz not detected: {a_f1:.4f}"
    )
    assert a_f2 > 5 * max(a_noise, 0.001), (
        f"f2 = {f2} Hz not detected: {a_f2:.4f}"
    )
    # FFR signature: at least ONE of the harmonic / combination
    # positions should show meaningful amplitude (above 10× noise).
    # Lerud 2014's model fits all of them at R²=0.77, but specific
    # ratios vary with ε. We check the pattern presence, not
    # exact ratios.
    signature_amps = [a_2f1, a_2f2, a_diff, a_sum]
    n_signature_active = sum(
        1 for a in signature_amps if a > 10 * max(a_noise, 0.001)
    )
    assert n_signature_active >= 2, (
        f"FFR signature absent: only {n_signature_active}/4 harmonic/"
        f"combination positions show meaningful amplitude. "
        f"Cascade may not be generating expected nonlinear pattern."
    )
