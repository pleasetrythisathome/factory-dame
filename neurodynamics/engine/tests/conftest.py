"""Shared pytest fixtures for the neurodynamics engine test suite.

Philosophy: deterministic synthetic audio with known properties is the
backbone of every integration test. We never mock the engine — we feed it
signals whose correct response we can derive analytically, then assert the
oscillator bank produces that response.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def rng():
    """Deterministic RNG for any stochastic test fixtures."""
    return np.random.default_rng(12345)


@pytest.fixture
def pulse_train_120bpm():
    """3 s of a clean 2 Hz pulse train at 16 kHz (120 BPM).

    Each pulse is a short noise burst with an exponential decay — sharp
    attack so rhythm GrFNN has strong Fourier energy at 2 Hz.
    """
    sr = 16000
    duration = 3.0
    n = int(sr * duration)
    audio = np.zeros(n, dtype=np.float32)
    beat_period = 0.5
    burst_len = int(0.02 * sr)
    env = np.exp(-np.linspace(0, 8, burst_len)).astype(np.float32)
    rng = np.random.default_rng(7)
    for k in range(int(duration / beat_period)):
        onset = int(k * beat_period * sr)
        burst = rng.standard_normal(burst_len).astype(np.float32) * env * 0.6
        audio[onset:onset + burst_len] += burst
    peak = float(np.abs(audio).max()) or 1.0
    return audio / peak * 0.8, sr


@pytest.fixture
def sine_440hz():
    """3 s of a pure 440 Hz sine at 16 kHz. A4. Pitch GrFNN should peak
    near the 440 Hz oscillator."""
    sr = 16000
    duration = 3.0
    t = np.arange(int(sr * duration)) / sr
    audio = (np.sin(2 * np.pi * 440.0 * t) * 0.4).astype(np.float32)
    return audio, sr


@pytest.fixture
def temp_audio_file(tmp_path):
    """Return a helper that writes arbitrary (audio, sr) to a temp wav and
    returns the path."""
    def _write(audio: np.ndarray, sr: int, name: str = "test.wav") -> Path:
        p = tmp_path / name
        sf.write(str(p), audio, sr)
        return p
    return _write


@pytest.fixture
def minimal_config(tmp_path):
    """Return a factory that writes a minimal runnable config.toml for
    integration tests. Default parameters match the repo config but with
    reduced oscillator counts and duration for fast tests."""
    def _build(audio_file: str, overrides: dict | None = None) -> Path:
        cfg = {
            "audio": {"input_file": audio_file, "sample_rate": 16000},
            "cochlea": {"n_channels": 32, "low_hz": 30.0, "high_hz": 4000.0},
            "rhythm_grfnn": {
                "n_oscillators": 30, "low_hz": 0.5, "high_hz": 10.0,
                "dt": 0.002,
                "alpha": -0.02, "beta1": -1.0, "beta2": -1.0,
                "delta1": 0.0, "delta2": 0.0, "epsilon": 1.0,
                "input_gain": 80.0,
            },
            "pitch_grfnn": {
                "n_oscillators": 60, "low_hz": 30.0, "high_hz": 4000.0,
                "dt": 0.0000625,
                # Test-only aggressive params: alpha=-0.3 ≈ 3s time constant
                # (vs 20s in production) so short test audio still reaches
                # clearly-resonant amplitudes. input_gain high so drive
                # dominates initial noise within the test window.
                "alpha": -0.3, "beta1": -1.0, "beta2": -1.0,
                "delta1": 0.0, "delta2": 0.0, "epsilon": 1.0,
                "input_gain": 2.0,
            },
            "phantom": {"amp_thresh": 0.1, "drive_thresh": 0.02},
            "state_log": {
                "snapshot_hz": 60,
                "output_dir": "output",
            },
            "osc": {"enabled": False, "host": "127.0.0.1", "port": 57120},
        }
        if overrides:
            for section, updates in overrides.items():
                cfg.setdefault(section, {}).update(updates)

        def fmt(v):
            if isinstance(v, bool):
                return str(v).lower()
            if isinstance(v, str):
                return f'"{v}"'
            return str(v)

        # Two-pass TOML emit: primitives into [section], nested dicts into
        # [section.subkey] blocks after. Supports one level of nesting
        # (enough for rhythm_grfnn.hebbian etc.).
        lines = []
        for section, body in cfg.items():
            primitives = {k: v for k, v in body.items() if not isinstance(v, dict)}
            tables = {k: v for k, v in body.items() if isinstance(v, dict)}
            lines.append(f"[{section}]")
            for k, v in primitives.items():
                lines.append(f"{k} = {fmt(v)}")
            lines.append("")
            for sub, subbody in tables.items():
                lines.append(f"[{section}.{sub}]")
                for k, v in subbody.items():
                    lines.append(f"{k} = {fmt(v)}")
                lines.append("")
        p = tmp_path / "config.toml"
        p.write_text("\n".join(lines))
        return p
    return _build
