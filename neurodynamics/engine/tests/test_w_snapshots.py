"""Tests for logging Hebbian W snapshots over time.

When Hebbian plasticity is on and snapshot logging is enabled, the engine
records the weight matrix at a configured cadence so downstream consumers
(primarily the viewer's W matrix panel) can animate the learned structure
as it evolves during a run.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurodynamics.run import run


@pytest.mark.slow
class TestWSnapshotPersistence:
    def test_history_written_when_enabled(
        self, pulse_train_120bpm, temp_audio_file, minimal_config, tmp_path,
    ):
        audio, sr = pulse_train_120bpm
        audio_file = temp_audio_file(audio, sr, "pulse.wav")
        cfg = minimal_config(
            str(audio_file.name),
            overrides={
                "rhythm_grfnn": {"hebbian": {
                    "enabled": True, "learn_rate": 1.0, "weight_decay": 0.1,
                }},
                "state_log": {"w_snapshot_hz": 4},
            },
        )
        out = tmp_path / "state.parquet"
        run(cfg, audio_override=audio_file, output_override=out)

        weights = tmp_path / "state.weights.npz"
        assert weights.exists()
        npz = np.load(weights)
        # History arrays exist alongside the legacy rhythm_W/pitch_W finals.
        assert "rhythm_W_history" in npz.files
        assert "rhythm_W_times" in npz.files
        # 3 s audio × 4 Hz = ~12 snapshots.
        times = npz["rhythm_W_times"]
        assert 10 <= len(times) <= 15
        # Monotonic.
        assert np.all(np.diff(times) > 0)
        # History shape (T, n, n).
        hist = npz["rhythm_W_history"]
        assert hist.shape == (len(times), 30, 30)
        # Final history entry should match the final W (within numerical
        # equality — both come from the same in-memory state).
        np.testing.assert_allclose(hist[-1], npz["rhythm_W"])

    def test_no_history_when_disabled(
        self, pulse_train_120bpm, temp_audio_file, minimal_config, tmp_path,
    ):
        """Back-compat: old configs without w_snapshot_hz still produce
        weights.npz with only the final matrix."""
        audio, sr = pulse_train_120bpm
        audio_file = temp_audio_file(audio, sr, "pulse.wav")
        cfg = minimal_config(
            str(audio_file.name),
            overrides={"rhythm_grfnn": {"hebbian": {
                "enabled": True, "learn_rate": 1.0, "weight_decay": 0.1,
            }}},
        )
        out = tmp_path / "state.parquet"
        run(cfg, audio_override=audio_file, output_override=out)

        npz = np.load(tmp_path / "state.weights.npz")
        assert "rhythm_W" in npz.files
        assert "rhythm_W_history" not in npz.files

    def test_no_weights_file_when_no_hebbian(
        self, pulse_train_120bpm, temp_audio_file, minimal_config, tmp_path,
    ):
        """No plasticity anywhere → no weights.npz at all."""
        audio, sr = pulse_train_120bpm
        audio_file = temp_audio_file(audio, sr, "pulse.wav")
        cfg = minimal_config(str(audio_file.name))
        out = tmp_path / "state.parquet"
        run(cfg, audio_override=audio_file, output_override=out)
        assert not (tmp_path / "state.weights.npz").exists()

    def test_history_shows_learning_progression(
        self, pulse_train_120bpm, temp_audio_file, minimal_config, tmp_path,
    ):
        """Early W snapshots should have smaller off-diagonal magnitude than
        late ones — the network is actively learning, not static."""
        audio, sr = pulse_train_120bpm
        audio_file = temp_audio_file(audio, sr, "pulse.wav")
        cfg = minimal_config(
            str(audio_file.name),
            overrides={
                "rhythm_grfnn": {"hebbian": {
                    "enabled": True, "learn_rate": 2.0, "weight_decay": 0.05,
                }},
                "state_log": {"w_snapshot_hz": 4},
            },
        )
        out = tmp_path / "state.parquet"
        run(cfg, audio_override=audio_file, output_override=out)
        npz = np.load(tmp_path / "state.weights.npz")
        hist = npz["rhythm_W_history"]
        # Skip the first snapshot (essentially zero) and compare early vs late.
        early = np.abs(hist[1]).max()
        late = np.abs(hist[-1]).max()
        assert late > early, (
            f"W should grow during the run: early={early}, late={late}"
        )
