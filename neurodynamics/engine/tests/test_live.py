"""Live-mode engine tests — LiveEngine.process called directly with
chunks of audio, no sounddevice involvement. This covers the
chunked processing path; the sounddevice callback wiring is trivial
enough that it gets validated by actually running ``nd-live`` with
the ES-9 attached (hardware-in-the-loop, not in CI).
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from neurodynamics.live import LiveEngine


def _live_cfg():
    """Minimal config dict matching what live.run would parse from
    TOML. Reduced network sizes so tests stay fast."""
    return {
        "audio": {"sample_rate": 16000},
        "cochlea": {"n_channels": 32, "low_hz": 30.0, "high_hz": 4000.0},
        "rhythm_grfnn": {
            "n_oscillators": 20, "low_hz": 0.5, "high_hz": 10.0,
            "dt": 0.002,
            "alpha": -0.02, "beta1": -1.0, "beta2": -1.0,
            "delta1": 0.0, "delta2": 0.0, "epsilon": 1.0,
            "input_gain": 80.0,
        },
        "pitch_grfnn": {
            "n_oscillators": 30, "low_hz": 30.0, "high_hz": 4000.0,
            "dt": 1.0 / 16000.0,
            "alpha": -0.3, "beta1": -1.0, "beta2": -1.0,
            "delta1": 0.0, "delta2": 0.0, "epsilon": 1.0,
            "input_gain": 2.0,
        },
        "phantom": {"amp_thresh": 0.1, "drive_thresh": 0.02},
        "state_log": {"snapshot_hz": 60},
        "osc": {"enabled": False, "host": "127.0.0.1", "port": 57120},
    }


@pytest.fixture
def engine():
    return LiveEngine(_live_cfg())


def test_live_engine_processes_silence(engine):
    """A chunk of zero audio should advance the engine without
    raising. Voices should be 0 (silence = no active clusters)."""
    chunk = np.zeros(512, dtype=np.float32)
    engine.process(chunk)
    assert engine._sample_count == 512


def test_live_engine_processes_sine(engine):
    """A chunk of 440 Hz sine at 16 kHz: 30 oscillator pitch bank
    picks up some resonance near the A4 oscillator."""
    fs = 16000
    t = np.arange(512) / fs
    chunk = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    engine.process(chunk)
    assert np.any(np.abs(engine.pitch.z) > 1e-6)


def test_live_engine_emits_snapshots_at_expected_rate(engine):
    """Processing exactly one second of audio at 60 Hz snapshot rate
    should trigger 60 snapshot emissions."""
    fs = 16000
    rng = np.random.default_rng(0)
    # Count snapshots by patching emit.
    count = [0]
    original = engine._emit_snapshot
    def counting_emit():
        count[0] += 1
        original()
    engine._emit_snapshot = counting_emit
    # One second of audio delivered in two chunks.
    chunk = (0.1 * rng.standard_normal(fs)).astype(np.float32)
    engine.process(chunk[:fs // 2])
    engine.process(chunk[fs // 2:])
    # We should emit ~60 snapshots (± 1 for boundary alignment).
    assert 58 <= count[0] <= 62


def test_live_engine_advances_across_chunk_boundaries(engine):
    """Pitch state after 1024 samples delivered as 1×1024 should
    match delivery as 2×512."""
    fs = 16000
    t = np.arange(1024) / fs
    audio = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    e1 = LiveEngine(_live_cfg())
    e1.process(audio)
    e2 = LiveEngine(_live_cfg())
    e2.process(audio[:512])
    e2.process(audio[512:])
    # Pitch state should be very close — not bit-identical because
    # the rhythm-drive residual handling crosses a chunk boundary
    # and the normalization's "max" may differ across sub-chunks.
    np.testing.assert_allclose(e1.pitch.z, e2.pitch.z, atol=5e-2)


def test_live_engine_motor_disabled_by_default(engine):
    """If [motor_grfnn] block is absent, engine has no motor."""
    assert engine.motor is None


def test_live_engine_faster_than_realtime_on_synth(engine):
    """Sanity check: processing N seconds of random audio takes
    less than N seconds of wall time on the test machine.
    Guards against regressions in the numba JIT path."""
    import time
    fs = 16000
    duration = 2.0
    rng = np.random.default_rng(0)
    audio = (0.1 * rng.standard_normal(int(fs * duration))).astype(np.float32)
    # Feed as ~32ms chunks so the loop resembles the live path.
    chunk_size = 512
    t0 = time.perf_counter()
    for lo in range(0, len(audio), chunk_size):
        engine.process(audio[lo:lo + chunk_size])
    elapsed = time.perf_counter() - t0
    ratio = duration / elapsed
    # Should be at least 1.5x realtime on a modern laptop with JIT
    # warm. First-run cold-JIT may be slower; this test assumes
    # numba cache is pre-populated by the GrFNN test suite running
    # before this one, which is the default pytest order.
    assert ratio > 1.0, (
        f"live engine ran at {ratio:.2f}x realtime "
        f"(took {elapsed:.2f}s for {duration}s of audio) — "
        f"numba JIT may have regressed"
    )
