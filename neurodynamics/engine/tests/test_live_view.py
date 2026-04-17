"""Tests for the live viewer's state buffer and OSC plumbing.

Matplotlib / Tk UI path isn't tested (GUI automation is out of
scope); validation there is hands-on. What's covered:

- Ring buffer field accumulation + advance-on-residual behavior
- OSC message dispatch into the buffer
- Window assembly across ring boundary (wraparound)
"""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest

from neurodynamics.live_view import LiveStateBuffer, _build_server
from pythonosc.udp_client import SimpleUDPClient


def _push_snapshot(buf: LiveStateBuffer, layer: str, amp, phase,
                    phantom=None, drive=None, residual=None) -> None:
    """Push a full layer's worth of fields, triggering advance when
    ``layer == "pitch"`` (the engine emits pitch last per snapshot)."""
    n = {"pitch": buf.n_pitch, "rhythm": buf.n_rhythm,
         "motor": buf.n_motor}[layer]
    phantom = phantom if phantom is not None else np.zeros(n)
    drive = drive if drive is not None else np.zeros(n)
    residual = residual if residual is not None else np.zeros(n)
    buf.push_field(layer, "amp", list(amp))
    buf.push_field(layer, "phase", list(phase))
    buf.push_field(layer, "phantom", list(phantom))
    buf.push_field(layer, "drive", list(drive))
    buf.push_field(layer, "residual", list(residual))


def test_buffer_initial_latest_is_empty():
    """A brand-new buffer has no samples; latest returns empty dict."""
    buf = LiveStateBuffer(n_pitch=10, n_rhythm=5, n_motor=0, snap_hz=60)
    result = buf.latest(60)
    assert result == {}


def test_buffer_push_and_read():
    """Push N snapshots, read window — gets back N samples.
    Pitch push is the advance trigger, so rhythm must be pushed
    before pitch for a well-formed frame."""
    buf = LiveStateBuffer(n_pitch=10, n_rhythm=5, n_motor=0, snap_hz=60)
    for i in range(5):
        _push_snapshot(buf, "rhythm", np.zeros(5), np.zeros(5))
        _push_snapshot(buf, "pitch", np.ones(10) * i, np.zeros(10))
    snap = buf.latest(10)
    assert snap["pamp"].shape == (5, 10)
    assert snap["pamp"][0, 0] == 0.0
    assert snap["pamp"][-1, 0] == 4.0


def test_buffer_ring_wraparound():
    """Write more samples than the ring depth; read-back returns
    the last buf_depth samples in correct order."""
    buf = LiveStateBuffer(n_pitch=4, n_rhythm=4, n_motor=0, snap_hz=60,
                          window_s=0.5)
    total = 100
    for i in range(total):
        _push_snapshot(buf, "rhythm", np.zeros(4), np.zeros(4))
        _push_snapshot(buf, "pitch", np.ones(4) * i, np.zeros(4))
    snap = buf.latest(buf.buf_depth)
    pa = snap["pamp"]
    expected_first = total - buf.buf_depth
    assert pa[0, 0] == expected_first
    assert pa[-1, 0] == total - 1


def test_buffer_motor_layer_populated():
    """When n_motor>0, motor data flows through."""
    buf = LiveStateBuffer(n_pitch=4, n_rhythm=4, n_motor=3, snap_hz=60)
    for i in range(3):
        _push_snapshot(buf, "rhythm", np.zeros(4), np.zeros(4))
        _push_snapshot(buf, "motor", np.ones(3) * i, np.zeros(3))
        _push_snapshot(buf, "pitch", np.zeros(4), np.zeros(4))
    snap = buf.latest(10)
    ma = snap["mamp"]
    assert ma.shape == (3, 3)
    assert ma[-1, 0] == 2.0


def test_buffer_motor_absent_when_n_motor_zero():
    """When n_motor=0, motor buffers stay None."""
    buf = LiveStateBuffer(n_pitch=4, n_rhythm=4, n_motor=0, snap_hz=60)
    for _ in range(2):
        _push_snapshot(buf, "rhythm", np.zeros(4), np.zeros(4))
        _push_snapshot(buf, "pitch", np.zeros(4), np.zeros(4))
    snap = buf.latest(10)
    assert snap.get("mamp") is None


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_osc_server_populates_buffer():
    """End-to-end: send full layer OSC bursts, verify the buffer
    advances after pitch residual lands."""
    buf = LiveStateBuffer(n_pitch=4, n_rhythm=4, n_motor=0, snap_hz=60)
    port = _free_udp_port()
    server = _build_server(buf, "127.0.0.1", port)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        client = SimpleUDPClient("127.0.0.1", port)
        # Rhythm layer
        client.send_message("/rhythm/amp", [0.5, 0.5, 0.5, 0.5])
        client.send_message("/rhythm/phase", [0.0, 0.0, 0.0, 0.0])
        client.send_message("/rhythm/phantom", [0, 0, 0, 0])
        client.send_message("/rhythm/drive", [0.0, 0.0, 0.0, 0.0])
        client.send_message("/rhythm/residual", [0.0, 0.0, 0.0, 0.0])
        # Pitch layer (advance trigger)
        client.send_message("/pitch/amp", [0.1, 0.2, 0.3, 0.4])
        client.send_message("/pitch/phase", [0.0, 0.0, 0.0, 0.0])
        client.send_message("/pitch/phantom", [0, 0, 0, 0])
        client.send_message("/pitch/drive", [0.0, 0.0, 0.0, 0.0])
        client.send_message("/pitch/residual", [0.0, 0.0, 0.0, 0.0])
        time.sleep(0.1)
        snap = buf.latest(10)
        assert snap["pamp"].shape == (1, 4)
        np.testing.assert_allclose(snap["pamp"][0], [0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(snap["ramp"][0], [0.5, 0.5, 0.5, 0.5])
    finally:
        server.shutdown()
        server.server_close()


def test_osc_features_messages_populate_dict():
    """Features arrive as individual OSC messages and land in
    buffer.features."""
    buf = LiveStateBuffer(n_pitch=4, n_rhythm=4, n_motor=0, snap_hz=60)
    port = _free_udp_port()
    server = _build_server(buf, "127.0.0.1", port)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        client = SimpleUDPClient("127.0.0.1", port)
        client.send_message("/features/tempo", 125.0)
        client.send_message("/features/key", "F")
        client.send_message("/features/mode", "major")
        client.send_message("/features/chord", "Am7")
        client.send_message("/features/consonance", 0.42)
        client.send_message("/voice/active_count", 5)
        time.sleep(0.1)
        assert buf.features.get("tempo") == pytest.approx(125.0)
        assert buf.features.get("tonic") == "F"
        assert buf.features.get("mode") == "major"
        assert buf.features.get("chord") == "Am7"
        assert buf.features.get("consonance") == pytest.approx(0.42)
        assert buf.features.get("voice_count") == 5
    finally:
        server.shutdown()
        server.server_close()
