"""Tests for the live viewer's state buffer and OSC plumbing.

Matplotlib / Tk UI path isn't tested (GUI automation is out of
scope); validation there is hands-on. What's covered:

- Ring buffer push/advance/window-read behavior
- OSC message dispatch into the buffer
- Window assembly across ring boundary (wraparound)
"""

from __future__ import annotations

import threading
import time
import socket

import numpy as np
import pytest

from neurodynamics.live_view import LiveStateBuffer, _build_server
from pythonosc.udp_client import SimpleUDPClient


def test_buffer_initial_latest_is_empty():
    """A brand-new buffer has no samples; window read returns
    (None, ...)."""
    buf = LiveStateBuffer(n_pitch=10, n_rhythm=5, n_motor=0, snap_hz=60)
    result = buf.latest_window(60)
    assert result[0] is None


def test_buffer_push_and_read():
    """Push N snapshots, read the window — gets back N samples."""
    buf = LiveStateBuffer(n_pitch=10, n_rhythm=5, n_motor=0, snap_hz=60)
    for i in range(5):
        buf.push_pitch(np.ones(10) * i, np.zeros(10))
        buf.push_rhythm(np.zeros(5), np.zeros(5))
        buf.advance()
    pa, _, _, _, _, _ = buf.latest_window(10)
    assert pa.shape == (5, 10)
    # Oldest first, newest last
    assert pa[0, 0] == 0.0
    assert pa[-1, 0] == 4.0


def test_buffer_ring_wraparound():
    """Write more samples than the ring depth; read-back returns
    the last buf_depth samples in correct order."""
    buf = LiveStateBuffer(n_pitch=4, n_rhythm=4, n_motor=0, snap_hz=60,
                          window_s=0.5)  # buf_depth ~= 45
    total = 100
    for i in range(total):
        buf.push_pitch(np.ones(4) * i, np.zeros(4))
        buf.push_rhythm(np.zeros(4), np.zeros(4))
        buf.advance()
    pa, _, _, _, _, _ = buf.latest_window(buf.buf_depth)
    # Last buf_depth samples should be contiguous: total-buf_depth .. total-1
    expected_first = total - buf.buf_depth
    assert pa[0, 0] == expected_first
    assert pa[-1, 0] == total - 1


def test_buffer_motor_layer_optional():
    """When n_motor=0, motor buffers are None and push_motor is a
    no-op."""
    buf = LiveStateBuffer(n_pitch=4, n_rhythm=4, n_motor=0, snap_hz=60)
    buf.push_motor(np.ones(4), np.zeros(4))  # no-op, no raise
    assert buf.mamp is None


def test_buffer_motor_layer_populated():
    """When n_motor>0, motor data flows through."""
    buf = LiveStateBuffer(n_pitch=4, n_rhythm=4, n_motor=3, snap_hz=60)
    for i in range(3):
        buf.push_pitch(np.zeros(4), np.zeros(4))
        buf.push_rhythm(np.zeros(4), np.zeros(4))
        buf.push_motor(np.ones(3) * i, np.zeros(3))
        buf.advance()
    _, _, _, _, ma, _ = buf.latest_window(10)
    assert ma.shape == (3, 3)
    assert ma[-1, 0] == 2.0


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_osc_server_populates_buffer():
    """End-to-end: send OSC over UDP loopback, verify the buffer
    receives pitch + rhythm data and advances on residual trigger."""
    buf = LiveStateBuffer(n_pitch=4, n_rhythm=4, n_motor=0, snap_hz=60)
    port = _free_udp_port()
    server = _build_server(buf, "127.0.0.1", port)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        client = SimpleUDPClient("127.0.0.1", port)
        # Simulate one full snapshot: pitch amp + phase, rhythm amp
        # + phase, pitch residual (advance trigger).
        client.send_message("/pitch/amp", [0.1, 0.2, 0.3, 0.4])
        client.send_message("/pitch/phase", [0.0, 0.0, 0.0, 0.0])
        client.send_message("/rhythm/amp", [0.5, 0.5, 0.5, 0.5])
        client.send_message("/rhythm/phase", [0.0, 0.0, 0.0, 0.0])
        client.send_message("/pitch/residual", [0.0, 0.0, 0.0, 0.0])
        # Brief wait for the server thread
        time.sleep(0.1)
        pa, _, ra, _, _, _ = buf.latest_window(10)
        assert pa is not None
        assert pa.shape == (1, 4)
        np.testing.assert_allclose(pa[0], [0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(ra[0], [0.5, 0.5, 0.5, 0.5])
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
