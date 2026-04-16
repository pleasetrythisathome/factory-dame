"""Tests for the two-layer pulse network (sensory + motor rhythm GrFNNs).

Harding/Kim/Large 2025 Fig. 4 describes the two-layer architecture needed
for the full missing-pulse effect: a sensory rhythm layer driven by audio
onsets, plus a motor rhythm layer bidirectionally coupled to it. The motor
layer embodies the "felt pulse" — it can sustain pulse activity across
silences and reinforce the sensory network's response at the expected beat.

These tests exercise the structural contract (motor net allocated when
enabled, state logged with layer="motor") and the critical dynamical
claim (motor persists when sensory drive drops).
"""

from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
import pytest

from neurodynamics.run import run


@pytest.mark.slow
class TestTwoLayerPulse:
    def test_motor_layer_absent_by_default(
        self, pulse_train_120bpm, temp_audio_file, minimal_config, tmp_path,
    ):
        """Without a [motor_grfnn] section in config, only rhythm + pitch
        layers appear in the state log."""
        audio, sr = pulse_train_120bpm
        audio_file = temp_audio_file(audio, sr, "pulse.wav")
        cfg = minimal_config(str(audio_file.name))
        out = tmp_path / "state.parquet"
        run(cfg, audio_override=audio_file, output_override=out)
        table = pq.read_table(out)
        layers = set(table.column("layer").to_pylist())
        assert layers == {"rhythm", "pitch"}

    def test_motor_layer_logged_when_enabled(
        self, pulse_train_120bpm, temp_audio_file, minimal_config, tmp_path,
    ):
        """With motor coupling enabled, the state log contains a 'motor'
        layer with the same frequency vector as rhythm."""
        audio, sr = pulse_train_120bpm
        audio_file = temp_audio_file(audio, sr, "pulse.wav")
        cfg = minimal_config(
            str(audio_file.name),
            overrides={"motor_grfnn": {
                "enabled": True,
                "n_oscillators": 30, "low_hz": 0.5, "high_hz": 10.0,
                "dt": 0.002,
                "alpha": -0.005, "beta1": -1.0, "beta2": -1.0,
                "delta1": 0.0, "delta2": 0.0, "epsilon": 1.0,
                "input_gain": 1.0,
                "forward_gain": 2.0,
                "backward_gain": 0.3,
            }},
        )
        out = tmp_path / "state.parquet"
        run(cfg, audio_override=audio_file, output_override=out)
        table = pq.read_table(out)
        layers = set(table.column("layer").to_pylist())
        assert "motor" in layers
        md = table.schema.metadata
        assert b"layer.motor.f" in md

    def test_motor_sustains_during_sensory_silence(
        self, temp_audio_file, minimal_config, tmp_path,
    ):
        """The headline NRT claim: motor layer keeps pulsing when sensory
        drive stops. We drive the network with a 2 Hz pulse train for 2 s,
        then silence for 2 s. Motor amplitude at 2 Hz should remain non-
        trivially above its initial (pre-training) level during the silence,
        while sensory amplitude decays much faster."""
        sr = 16000
        duration = 4.0
        n = int(sr * duration)
        audio = np.zeros(n, dtype=np.float32)
        beat_period = 0.5
        burst_len = int(0.02 * sr)
        env = np.exp(-np.linspace(0, 8, burst_len)).astype(np.float32)
        rng = np.random.default_rng(7)
        # Beats for the first half only — second half is silence.
        for k in range(int((duration / 2) / beat_period)):
            onset = int(k * beat_period * sr)
            burst = (rng.standard_normal(burst_len).astype(np.float32)
                     * env * 0.6)
            audio[onset:onset + burst_len] += burst
        audio /= max(float(np.abs(audio).max()), 1e-6)
        audio *= 0.8
        audio_file = temp_audio_file(audio, sr, "halfpulse.wav")

        cfg = minimal_config(
            str(audio_file.name),
            overrides={
                # Moderate rhythm input_gain so sensory reaches observable
                # amplitudes without immediately clamping at the quintic pole
                # (which would mask the motor-sustain dynamic).
                "rhythm_grfnn": {"input_gain": 15.0, "alpha": -0.1},
                "motor_grfnn": {
                    "enabled": True,
                    "n_oscillators": 30, "low_hz": 0.5, "high_hz": 10.0,
                    "dt": 0.002,
                    # Motor is very lightly damped so it sustains once
                    # energized. Lower than sensory's alpha.
                    "alpha": -0.005, "beta1": -1.0, "beta2": -1.0,
                    "delta1": 0.0, "delta2": 0.0, "epsilon": 1.0,
                    "input_gain": 1.0,
                    # Forward strong enough to energize motor without
                    # saturating it; backward small so the feedback doesn't
                    # create a positive-feedback runaway with sensory.
                    "forward_gain": 1.0,
                    "backward_gain": 0.05,
                },
            },
        )
        out = tmp_path / "state.parquet"
        run(cfg, audio_override=audio_file, output_override=out)

        t = pq.read_table(out)
        times = np.array(t.column("t").to_pylist())
        layer = np.array(t.column("layer").to_pylist())
        amp_col = t.column("amp").to_pylist()

        def amp_at_2hz(layer_name):
            mask = layer == layer_name
            a = np.stack([np.asarray(x) for x, m in zip(amp_col, mask) if m])
            tt = times[mask]
            f_key = f"layer.{layer_name}.f".encode()
            freqs = np.array([float(x)
                              for x in t.schema.metadata[f_key].decode().split(",")])
            idx_2hz = int(np.argmin(np.abs(freqs - 2.0)))
            return tt, a[:, idx_2hz]

        _, motor_2 = amp_at_2hz("motor")
        _, sens_2 = amp_at_2hz("rhythm")

        # Split between pulse phase (first half) and silence phase (second).
        mid = len(motor_2) // 2
        motor_silent = motor_2[mid + 30:]  # skip transient at the cut
        sens_silent = sens_2[mid + 30:]
        motor_silent_peak = motor_silent.max()
        sens_silent_peak = sens_silent.max()

        # Motor should still be ringing after sensory drive stops.
        assert motor_silent_peak > 0.05, (
            f"motor should persist during silence; got peak {motor_silent_peak}"
        )
        # And it should be clearly more sustained than the sensory layer.
        assert motor_silent.mean() > sens_silent.mean(), (
            f"motor should outlast sensory in silence "
            f"(motor mean {motor_silent.mean():.4f} vs "
            f"sensory mean {sens_silent.mean():.4f})"
        )
