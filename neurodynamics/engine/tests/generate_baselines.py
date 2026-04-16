"""Generate / regenerate regression baselines.

Run this when the engine's behavior intentionally changes — new features,
tuned defaults, etc. The baselines record the engine's output so future runs
can be asserted identical.

Usage:
    uv run python -m tests.generate_baselines

This will overwrite any existing baselines under tests/baselines/.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from tests.conftest import pulse_train_120bpm  # type: ignore  # noqa: F401
# ^ pytest fixtures are not directly callable outside pytest. We rebuild the
# synthetic audio inline below to keep this script independent of pytest.

from neurodynamics.run import run
from tests.test_regression import _summarize, BASELINE_DIR


def _build_pulse_audio():
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
    audio /= max(float(np.abs(audio).max()), 1e-6)
    audio *= 0.8
    return audio, sr


def _build_sine_audio():
    sr = 16000
    t = np.arange(int(sr * 3.0)) / sr
    return (np.sin(2 * np.pi * 440.0 * t) * 0.4).astype(np.float32), sr


def _write_test_config(tmp: Path, audio_rel: str) -> Path:
    """Mirror of tests/conftest.py minimal_config — kept in sync manually."""
    cfg = {
        "audio": {"input_file": audio_rel, "sample_rate": 16000},
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
            "alpha": -0.3, "beta1": -1.0, "beta2": -1.0,
            "delta1": 0.0, "delta2": 0.0, "epsilon": 1.0,
            "input_gain": 2.0,
        },
        "phantom": {"amp_thresh": 0.1, "drive_thresh": 0.02},
        "state_log": {"snapshot_hz": 60, "output_dir": "output"},
        "osc": {"enabled": False, "host": "127.0.0.1", "port": 57120},
    }
    lines = []
    for section, body in cfg.items():
        lines.append(f"[{section}]")
        for k, v in body.items():
            if isinstance(v, bool):
                lines.append(f"{k} = {str(v).lower()}")
            elif isinstance(v, str):
                lines.append(f'{k} = "{v}"')
            else:
                lines.append(f"{k} = {v}")
        lines.append("")
    p = tmp / "config.toml"
    p.write_text("\n".join(lines))
    return p


def main() -> None:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    cases = [
        ("pulse_train_120bpm", _build_pulse_audio, "pulse.wav"),
        ("sine_440hz", _build_sine_audio, "a4.wav"),
    ]
    for name, audio_fn, audio_file in cases:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio, sr = audio_fn()
            audio_path = tmp_path / audio_file
            sf.write(str(audio_path), audio, sr)
            cfg_path = _write_test_config(tmp_path, audio_file)
            state_path = tmp_path / "state.parquet"
            print(f"\n=== {name} ===")
            run(cfg_path, audio_override=audio_path, output_override=state_path)
            summary = _summarize(state_path)
        dest = BASELINE_DIR / f"{name}.json"
        dest.write_text(json.dumps(summary, indent=2))
        print(f"wrote {dest}")


if __name__ == "__main__":
    main()
