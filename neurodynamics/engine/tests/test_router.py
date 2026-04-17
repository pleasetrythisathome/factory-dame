"""Router tests — Phase 4 of task-011.

Tests scale functions in isolation, OSC address matching, and
end-to-end dispatch from an OSC message through the routing table
to mock backends. Hardware-in-the-loop (ES-9, real MIDI device)
validation is left to the operator since CI can't plug in cables.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from neurodynamics.router import (
    CVBackend,
    MIDIBackend,
    Mapping,
    OSCForwardBackend,
    Router,
    RouterConfig,
    _osc_address_matches,
    phase_wrapped,
    scale_phase_to_sawtooth,
    scale_tonic_to_1v_per_oct,
    scale_unit_to_5v,
)


# ── Scale functions ──────────────────────────────────────────────

def test_tonic_to_1v_per_oct_c_octave4():
    assert scale_tonic_to_1v_per_oct("C", octave=4) == 4.0


def test_tonic_to_1v_per_oct_a_octave4():
    # A at octave 4 = 4 + 9/12 semitones.
    assert scale_tonic_to_1v_per_oct("A", octave=4) == pytest.approx(4.75)


def test_tonic_to_1v_per_oct_fsharp():
    assert scale_tonic_to_1v_per_oct("F#", octave=3) == pytest.approx(3.5)


def test_tonic_to_1v_per_oct_unknown_raises():
    with pytest.raises(ValueError):
        scale_tonic_to_1v_per_oct("Q")


def test_phase_to_sawtooth_wraps():
    """Phase at -π → 0V, phase at 0 → half range, phase at +π → top."""
    assert scale_phase_to_sawtooth(-np.pi) == pytest.approx(0.0, abs=1e-6)
    assert scale_phase_to_sawtooth(0.0) == pytest.approx(2.5, abs=1e-6)
    # +π maps to the top but wraps to 0 when strictly at the boundary;
    # any nudge past π wraps to 0.
    v = scale_phase_to_sawtooth(np.pi - 1e-9)
    assert v > 4.9


def test_phase_to_sawtooth_custom_range():
    v = scale_phase_to_sawtooth(0.0, v_min=-1.0, v_max=1.0)
    assert v == pytest.approx(0.0, abs=1e-6)


def test_unit_to_5v_clips():
    assert scale_unit_to_5v(0.5) == 2.5
    assert scale_unit_to_5v(1.5) == 5.0
    assert scale_unit_to_5v(-0.3) == 0.0


def test_phase_wrapped_detects_positive_to_negative():
    """When phase goes from near +π to near -π, that's a wrap."""
    assert phase_wrapped(-3.0, 3.0) is True


def test_phase_wrapped_ignores_normal_increment():
    """Phase going from 0.1 to 0.2 is not a wrap."""
    assert phase_wrapped(0.2, 0.1) is False


def test_phase_wrapped_handles_none():
    """First frame has no prior; no wrap."""
    assert phase_wrapped(0.5, None) is False


# ── OSC address matching ─────────────────────────────────────────

def test_osc_exact_match():
    assert _osc_address_matches("/voice/0/rhythm/phase",
                                 "/voice/0/rhythm/phase")


def test_osc_wildcard_matches_single_segment():
    assert _osc_address_matches("/voice/*/rhythm/phase",
                                 "/voice/17/rhythm/phase")


def test_osc_wildcard_does_not_cross_segment():
    assert not _osc_address_matches("/voice/*/phase",
                                     "/voice/0/rhythm/phase")


def test_osc_mismatch_length():
    assert not _osc_address_matches("/voice/0", "/voice/0/rhythm")


# ── Router dispatch with mock backends ───────────────────────────

def _make_router(mappings, *, cv_channels=4,
                 forward_targets=None) -> Router:
    """Construct a Router wired to mock backends. Skips the OSC
    server so tests can call handlers directly."""
    cfg = RouterConfig(
        cv_enabled=True,
        cv_channels=cv_channels,
        midi_enabled=True,
        midi_port="__nonexistent__",  # forces mock fallback
        mappings=mappings,
        osc_forward_targets=forward_targets or [],
    )
    cv = CVBackend(device=None, channels=cv_channels, sample_rate=48000,
                    mock=True)
    midi = MIDIBackend(port_name=None, mock=True)
    osc_fwd = (OSCForwardBackend(forward_targets)
               if forward_targets else None)
    return Router(cfg, cv=cv, midi=midi, osc_forward=osc_fwd,
                  start_server=False)


def test_router_routes_tonic_to_cv_channel_0():
    """/features/key with value 'A' at octave 4 → CV channel 0 at 4.75V."""
    router = _make_router([
        Mapping(source="/features/key", output="cv/0", scale="1V_per_oct"),
    ])
    router._on_osc_message("/features/key", "A")
    assert router.cv._voltages[0] == pytest.approx(4.75, abs=1e-6)


def test_router_routes_rhythm_phase_to_cv_sawtooth():
    router = _make_router([
        Mapping(source="/voice/0/rhythm/phase",
                output="cv/1", scale="sawtooth_0_5V"),
    ])
    router._on_osc_message("/voice/0/rhythm/phase", 0.0)
    assert router.cv._voltages[1] == pytest.approx(2.5, abs=1e-6)


def test_router_routes_wildcard_voice_phase():
    """Wildcard mapping hits multiple voice ids onto the same channel."""
    router = _make_router([
        Mapping(source="/voice/*/rhythm/phase",
                output="cv/0", scale="sawtooth_0_5V"),
    ])
    router._on_osc_message("/voice/3/rhythm/phase", float(np.pi) - 1e-9)
    assert router.cv._voltages[0] > 4.9


def test_router_dispatches_midi_cc():
    router = _make_router([
        Mapping(source="/features/consonance",
                output="midi/cc/7", scale="unit_0_5V"),
    ])
    # Value 0.5 → 2.5 V → rounded to MIDI integer 2 (floor of 2.5)
    router._on_osc_message("/features/consonance", 0.5)
    assert any(
        msg["type"] == "cc" and msg["cc"] == 7
        for msg in router.midi._sent
    )


def test_router_skips_unknown_scale():
    """A mapping with an unrecognized scale falls back to passthrough,
    so numeric sources still reach the backend."""
    router = _make_router([
        Mapping(source="/voice/0/amp",
                output="cv/2", scale="not_a_real_scale"),
    ])
    router._on_osc_message("/voice/0/amp", 0.73)
    assert router.cv._voltages[2] == pytest.approx(0.73, abs=1e-6)


def test_router_unmatched_address_is_noop():
    router = _make_router([
        Mapping(source="/voice/0/rhythm/phase",
                output="cv/0", scale="sawtooth_0_5V"),
    ])
    # Different address — shouldn't touch any channel.
    router._on_osc_message("/features/tempo", 120.0)
    assert np.all(router.cv._voltages == 0.0)


def test_router_osc_forward_rebroadcasts():
    """OSC forward re-emits every incoming message regardless of
    mapping table."""
    router = _make_router(
        [],
        forward_targets=[("127.0.0.1", 9999)],
    )
    router._on_osc_message("/anything/goes", 42)
    assert any(addr == "/anything/goes"
               for addr, _ in router.osc_forward._sent)


# ── End-to-end OSC round-trip ────────────────────────────────────

def test_end_to_end_udp_roundtrip():
    """Send a UDP OSC message from a real client; router receives
    it and writes to the CV backend."""
    import socket
    from pythonosc.udp_client import SimpleUDPClient

    # Find a free UDP port for the test (don't clobber 57120 which
    # the engine uses).
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    cfg = RouterConfig(
        cv_enabled=True, cv_channels=4,
        osc_listen_port=port,
        mappings=[
            Mapping(source="/voice/0/rhythm/phase",
                    output="cv/0", scale="sawtooth_0_5V"),
        ],
    )
    cv = CVBackend(device=None, channels=4, sample_rate=48000, mock=True)
    router = Router(cfg, cv=cv)
    try:
        client = SimpleUDPClient("127.0.0.1", port)
        client.send_message("/voice/0/rhythm/phase", 0.0)
        # Brief wait for the server thread to process
        time.sleep(0.05)
        assert cv._voltages[0] == pytest.approx(2.5, abs=0.01)
    finally:
        router.close()


# ── Config loading ────────────────────────────────────────────────

def test_router_config_from_toml(tmp_path):
    cfg_text = """
[router.cv]
enabled = true
channels = 8
sample_rate = 44100

[router.midi]
enabled = false

[router.osc]
listen_host = "127.0.0.1"
listen_port = 57125

[[router.osc.forward]]
host = "127.0.0.1"
port = 8000

[[router.mapping]]
source = "/voice/*/rhythm/phase"
output = "cv/0"
scale = "sawtooth_0_5V"

[[router.mapping]]
source = "/features/key"
output = "cv/1"
scale = "1V_per_oct"
"""
    path = tmp_path / "config.toml"
    path.write_text(cfg_text)
    cfg = RouterConfig.from_toml(path)
    assert cfg.cv_enabled
    assert cfg.cv_channels == 8
    assert cfg.cv_sample_rate == 44100
    assert not cfg.midi_enabled
    assert cfg.osc_listen_port == 57125
    assert cfg.osc_forward_targets == [("127.0.0.1", 8000)]
    assert len(cfg.mappings) == 2
    assert cfg.mappings[0].source == "/voice/*/rhythm/phase"
    assert cfg.mappings[1].scale == "1V_per_oct"


# ── Mock CV backend records writes ───────────────────────────────

def test_cv_mock_records_writes():
    cv = CVBackend(device=None, channels=3, sample_rate=48000, mock=True)
    cv.set_channel(0, 1.0)
    cv.set_channel(2, -0.5)
    cv.set_channel(5, 9.9)  # out-of-range, silently dropped
    assert cv._voltages[0] == pytest.approx(1.0)
    assert cv._voltages[2] == pytest.approx(-0.5)
    assert cv._writes == [(0, 1.0), (2, -0.5)]  # channel 5 not recorded
