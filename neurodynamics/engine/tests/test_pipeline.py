"""Integration tests — run the full pipeline on synthetic audio and assert
the engine responds how NRT predicts."""

from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
import pytest

from neurodynamics.run import run


def _load_state(path):
    """Return (pitch, rhythm) dicts with amp array and frequency vector."""
    t = pq.read_table(str(path))
    layer = np.array(t.column("layer").to_pylist())
    amp_col = t.column("amp").to_pylist()
    out = {}
    for which in ("pitch", "rhythm"):
        mask = layer == which
        amp = np.stack([np.asarray(a) for a, m in zip(amp_col, mask) if m])
        f_key = f"layer.{which}.f".encode()
        freqs = np.array(
            [float(x) for x in t.schema.metadata[f_key].decode().split(",")]
        )
        out[which] = {"amp": amp, "f": freqs}
    return out


@pytest.mark.slow
class TestPipelineSyntheticPulse:
    def test_rhythm_locks_to_2hz_pulse(
        self, pulse_train_120bpm, temp_audio_file, minimal_config, tmp_path,
    ):
        """A 2 Hz pulse train drives the rhythm GrFNN. The most active
        oscillator should be at (or adjacent to) the 2 Hz natural frequency."""
        audio, sr = pulse_train_120bpm
        audio_file = temp_audio_file(audio, sr, "pulse.wav")
        cfg = minimal_config(str(audio_file.name))
        run(cfg, audio_override=audio_file,
            output_override=tmp_path / "state.parquet")

        state = _load_state(tmp_path / "state.parquet")
        r = state["rhythm"]
        # Per-oscillator peak amplitude over the whole run.
        per_osc_max = r["amp"].max(axis=0)
        top = np.argsort(per_osc_max)[-5:][::-1]
        top_freqs = r["f"][top]
        # Expect a 2 Hz oscillator (or its 2:1 harmonic at 4 Hz) among the top-5.
        assert any(abs(f - 2.0) < 0.3 for f in top_freqs) \
            or any(abs(f - 4.0) < 0.6 for f in top_freqs), \
            f"no 2 Hz or 4 Hz locking among top-5: {top_freqs}"


@pytest.mark.slow
class TestPipelineSyntheticTone:
    def test_pitch_locks_to_440hz_tone(
        self, sine_440hz, temp_audio_file, minimal_config, tmp_path,
    ):
        """A 440 Hz sine should drive the pitch oscillator nearest 440 Hz
        into the top-5 most active."""
        audio, sr = sine_440hz
        audio_file = temp_audio_file(audio, sr, "a4.wav")
        cfg = minimal_config(str(audio_file.name))
        run(cfg, audio_override=audio_file,
            output_override=tmp_path / "state.parquet")

        state = _load_state(tmp_path / "state.parquet")
        p = state["pitch"]
        per_osc_max = p["amp"].max(axis=0)
        top_idx = np.argsort(per_osc_max)[-10:][::-1]
        top_freqs = p["f"][top_idx]
        # With only 60 oscillators over 30–4000 Hz (log-spaced) adjacent steps
        # are ~14%. So we expect an oscillator within 20% of 440 Hz.
        assert any(abs(f - 440.0) / 440.0 < 0.2 for f in top_freqs), \
            f"no oscillator near 440 Hz among top-10: {top_freqs}"


@pytest.mark.slow
class TestPipelineContract:
    """Structural tests on what run() produces — shape, schema, contents."""

    def test_state_file_schema(
        self, pulse_train_120bpm, temp_audio_file, minimal_config, tmp_path,
    ):
        audio, sr = pulse_train_120bpm
        audio_file = temp_audio_file(audio, sr, "pulse.wav")
        cfg = minimal_config(str(audio_file.name))
        run(cfg, audio_override=audio_file,
            output_override=tmp_path / "state.parquet")

        table = pq.read_table(tmp_path / "state.parquet")
        required = {"t", "layer", "z_real", "z_imag", "amp", "phase",
                    "phantom", "drive", "residual"}
        assert required <= set(table.column_names)
        layers = set(table.column("layer").to_pylist())
        assert layers == {"rhythm", "pitch"}
        # Metadata carries per-layer frequency vectors.
        md = table.schema.metadata
        assert b"layer.rhythm.f" in md
        assert b"layer.pitch.f" in md

    def test_runs_with_osc_disabled(
        self, pulse_train_120bpm, temp_audio_file, minimal_config, tmp_path,
    ):
        """Disabling OSC must not break the run. Minimal config already sets
        enabled=false — this just confirms no tooling regresses that."""
        audio, sr = pulse_train_120bpm
        audio_file = temp_audio_file(audio, sr, "pulse.wav")
        cfg = minimal_config(str(audio_file.name), overrides={"osc": {"enabled": False}})
        run(cfg, audio_override=audio_file,
            output_override=tmp_path / "state.parquet")
        assert (tmp_path / "state.parquet").exists()

    def test_deterministic_output(
        self, pulse_train_120bpm, temp_audio_file, minimal_config, tmp_path,
    ):
        """Same input + config must produce bit-identical state amplitudes.
        This is the prerequisite for the regression test suite."""
        audio, sr = pulse_train_120bpm
        audio_file = temp_audio_file(audio, sr, "pulse.wav")
        cfg = minimal_config(str(audio_file.name))

        run(cfg, audio_override=audio_file,
            output_override=tmp_path / "a.parquet")
        run(cfg, audio_override=audio_file,
            output_override=tmp_path / "b.parquet")

        a = _load_state(tmp_path / "a.parquet")
        b = _load_state(tmp_path / "b.parquet")
        np.testing.assert_array_equal(a["rhythm"]["amp"], b["rhythm"]["amp"])
        np.testing.assert_array_equal(a["pitch"]["amp"], b["pitch"]["amp"])
