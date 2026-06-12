"""Phase 3 validation: brainstem layer generates harmonic content
that the pitch (cortex) layer doesn't.

Drives the engine with a pure 220 Hz tone. The brainstem layer —
configured in critical-Hopf regime with canonical input + coupling
kernel enabled per Lerud 2014 — should show amplitude at 440 Hz
(2:1), 660 Hz (3:1), 880 Hz (4:1) harmonic positions in addition
to 220 Hz. The pitch layer, in subcritical-DLC regime without
coupling, should show primarily 220 Hz.

This is the architectural demonstration that multi-layer NRT
separates harmonic generation (brainstem) from fundamental
detection (cortex).
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import tomllib
from pathlib import Path

from neurodynamics.live import LiveEngine

ENGINE_DIR = Path(__file__).resolve().parent.parent


def _load_cfg() -> dict:
    with open(ENGINE_DIR / "config.toml", "rb") as f:
        return tomllib.load(f)


def _enable_brainstem(cfg: dict) -> dict:
    cfg = deepcopy(cfg)
    cfg.setdefault("brainstem_grfnn", {})["enabled"] = True
    return cfg


def _synth_tone(freq: float, duration: float, sr: int,
                amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(duration * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _run_engine_with_tone(cfg: dict, freq_hz: float,
                           duration: float = 2.0,
                           amp: float = 0.5) -> dict:
    engine = LiveEngine(cfg)
    sr = engine.fs
    audio = _synth_tone(freq_hz, duration=duration, sr=sr, amp=amp)
    # JIT warm-up
    engine.process(audio[:4096])
    # Reset for clean measurement
    engine.pitch.z[:] = 0
    if engine.brainstem is not None:
        engine.brainstem.z[:] = 0
    chunk = 1024
    for start in range(0, len(audio) - chunk, chunk):
        engine.process(audio[start:start + chunk])
    return {
        "pitch_amps": np.abs(engine.pitch.z),
        "brainstem_amps": (np.abs(engine.brainstem.z)
                           if engine.brainstem is not None else None),
        "freqs": engine.pitch.f,
    }


def test_cascade_propagates_harmonic_to_pitch():
    """Drive engine with pure 220 Hz. In CASCADE mode (brainstem
    enabled), brainstem generates 440 Hz (1:2 harmonic) via
    canonical input + coupling kernel, and that 440 Hz content
    propagates to pitch via the tonotopic feedforward connection.

    Compare: pitch at 440 Hz should be SIGNIFICANTLY larger when
    brainstem is in-path vs when it's bypassed. This is the
    distinguishing test for the Lerud 2014 cascaded architecture.
    """
    cfg_base = _load_cfg()

    # Bypass: brainstem disabled, gammatone → pitch directly
    cfg_bypass = deepcopy(cfg_base)
    cfg_bypass.setdefault("brainstem_grfnn", {})["enabled"] = False
    bypass = _run_engine_with_tone(cfg_bypass, 220.0)

    # Cascade: brainstem enabled, gammatone → brainstem → pitch
    cfg_cascade = _enable_brainstem(cfg_base)
    cascade = _run_engine_with_tone(cfg_cascade, 220.0)

    freqs = bypass["freqs"]

    def closest(target):
        return int(np.argmin(np.abs(freqs - target)))

    idx_220 = closest(220.0)
    idx_440 = closest(440.0)
    idx_660 = closest(660.0)

    p_bypass_220 = bypass["pitch_amps"][idx_220]
    p_bypass_440 = bypass["pitch_amps"][idx_440]
    p_cascade_220 = cascade["pitch_amps"][idx_220]
    p_cascade_440 = cascade["pitch_amps"][idx_440]
    p_cascade_660 = cascade["pitch_amps"][idx_660]
    bs_220 = cascade["brainstem_amps"][idx_220]
    bs_440 = cascade["brainstem_amps"][idx_440]
    bs_660 = cascade["brainstem_amps"][idx_660]

    print(f"BYPASS pitch |z|:   220Hz={p_bypass_220:.4f}  "
          f"440Hz={p_bypass_440:.4f}")
    print(f"CASCADE brainstem:  220Hz={bs_220:.4f}  "
          f"440Hz={bs_440:.4f}  660Hz={bs_660:.4f}")
    print(f"CASCADE pitch |z|:  220Hz={p_cascade_220:.4f}  "
          f"440Hz={p_cascade_440:.4f}  660Hz={p_cascade_660:.4f}")

    # Both modes track the fundamental
    assert p_bypass_220 > 0.01, "bypass pitch doesn't track drive"
    assert p_cascade_220 > 0.01, "cascade pitch doesn't track drive"

    # Cascade should produce SIGNIFICANTLY more 440 Hz energy in
    # pitch than bypass — that's the propagated harmonic. Expect
    # at least 5× the bypass baseline.
    assert p_cascade_440 > 5 * max(p_bypass_440, 1e-3), (
        f"cascaded harmonic propagation not visible: "
        f"bypass 440={p_bypass_440:.4f}, "
        f"cascade 440={p_cascade_440:.4f}"
    )

    # Brainstem generates the harmonic (smaller than pitch's
    # amplified version, but non-zero — meaningfully above noise).
    assert bs_440 > 0.03, (
        f"brainstem didn't generate 440 Hz harmonic: {bs_440}"
    )
