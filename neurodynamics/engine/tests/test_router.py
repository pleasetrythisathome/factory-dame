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


# ── Hz → 1V/oct + gate scale functions ───────────────────────────

from neurodynamics.router import (
    scale_hz_to_1v_per_oct, scale_active_as_gate,
)


def test_hz_1v_per_oct_c4_is_zero():
    """The default anchor is C4 (261.63 Hz) = 0 V, matching the
    Eurorack standard and our Overtone/VCV CV encoding."""
    assert scale_hz_to_1v_per_oct(261.6256) == pytest.approx(0.0, abs=1e-6)


def test_hz_1v_per_oct_a4_is_0_75():
    assert scale_hz_to_1v_per_oct(440.0) == pytest.approx(0.75, abs=1e-3)


def test_hz_1v_per_oct_c2_is_negative():
    """Sub-C4 notes produce negative voltages. DC-coupled outs accept
    ±5–10 V; VCV and most Eurorack VCOs track fine."""
    c2 = scale_hz_to_1v_per_oct(65.4064)
    assert c2 == pytest.approx(-2.0, abs=1e-3)


def test_hz_1v_per_oct_handles_nonpositive():
    assert scale_hz_to_1v_per_oct(0.0) == 0.0
    assert scale_hz_to_1v_per_oct(-5.0) == 0.0


def test_hz_1v_per_oct_alternate_anchor():
    """Callers can anchor to A4 if they prefer."""
    assert scale_hz_to_1v_per_oct(440.0, ref_hz=440.0) == pytest.approx(0.0)
    assert scale_hz_to_1v_per_oct(880.0, ref_hz=440.0) == pytest.approx(1.0)


def test_active_as_gate():
    assert scale_active_as_gate(1) == 5.0
    assert scale_active_as_gate(0) == 0.0
    assert scale_active_as_gate(0.0) == 0.0
    assert scale_active_as_gate(0.51) == 5.0


def test_active_as_gate_custom_levels():
    assert scale_active_as_gate(1, on_v=8.0) == 8.0


# ── Per-voice routing round-trip ─────────────────────────────────
#
# Simulates the OSC messages a live engine emits for a voice cluster,
# and verifies that with voice-style mappings configured, the router
# produces the correct CV voltages for 1V/oct pitch + gate channels.
# This is the end-to-end test of the "engine voice → modular CV"
# pipeline that task-011 Phase 4 was scoped for.


def test_voice_center_freq_routed_to_cv_1v_per_oct():
    """Engine emits /voice/0/center_freq 440.0 (A4). With mapping
    /voice/0/center_freq → cv/0 (hz_1V_per_oct), the CV backend
    should land at +0.75 V (A4 above C4 anchor)."""
    cfg = RouterConfig(
        cv_enabled=False,   # we supply the mock below
        osc_listen_host="127.0.0.1",
        osc_listen_port=0,  # ephemeral port
        mappings=[
            Mapping(source="/voice/0/center_freq",
                    output="cv/0", scale="hz_1V_per_oct"),
        ],
    )
    cv = CVBackend(device=None, channels=4, sample_rate=48000, mock=True)
    r = Router(cfg, cv=cv, start_server=False)
    try:
        r._on_osc_message("/voice/0/center_freq", 440.0)
        assert cv._voltages[0] == pytest.approx(0.75, abs=1e-3)
    finally:
        r.close()


def test_voice_active_routed_to_gate():
    """Engine emits /voice/0/active 1 → cv/1 should be 5 V (gate
    high); subsequent /voice/0/active 0 → cv/1 should be 0 V."""
    cfg = RouterConfig(
        osc_listen_port=0,
        mappings=[
            Mapping(source="/voice/0/active",
                    output="cv/1", scale="gate_5V"),
        ],
    )
    cv = CVBackend(device=None, channels=4, sample_rate=48000, mock=True)
    r = Router(cfg, cv=cv, start_server=False)
    try:
        r._on_osc_message("/voice/0/active", 1)
        assert cv._voltages[1] == pytest.approx(5.0)
        r._on_osc_message("/voice/0/active", 0)
        assert cv._voltages[1] == pytest.approx(0.0)
    finally:
        r.close()


def test_multi_voice_routing_four_slots():
    """Four voices, each routed to its own pitch CV + gate CV pair
    (channels 0/1 for voice 0, 2/3 for voice 1, 4/5 for voice 2,
    6/7 for voice 3). Simulates a 4-voice modular polysynth."""
    mappings = []
    for i in range(4):
        mappings.append(Mapping(
            source=f"/voice/{i}/center_freq",
            output=f"cv/{2 * i}", scale="hz_1V_per_oct",
        ))
        mappings.append(Mapping(
            source=f"/voice/{i}/active",
            output=f"cv/{2 * i + 1}", scale="gate_5V",
        ))
    cfg = RouterConfig(osc_listen_port=0, mappings=mappings)
    cv = CVBackend(device=None, channels=8, sample_rate=48000, mock=True)
    r = Router(cfg, cv=cv, start_server=False)
    try:
        # 4 voices: C4, E4, G4, A4. All active.
        for i, freq in enumerate([261.63, 329.63, 392.0, 440.0]):
            r._on_osc_message(f"/voice/{i}/active", 1)
            r._on_osc_message(f"/voice/{i}/center_freq", freq)
        assert cv._voltages[0] == pytest.approx(0.0, abs=1e-3)   # C4
        assert cv._voltages[2] == pytest.approx(0.333, abs=1e-3) # E4
        assert cv._voltages[4] == pytest.approx(0.583, abs=1e-3) # G4
        assert cv._voltages[6] == pytest.approx(0.75, abs=1e-3)  # A4
        for gate_ch in (1, 3, 5, 7):
            assert cv._voltages[gate_ch] == pytest.approx(5.0)
        # Voice 2 goes silent.
        r._on_osc_message("/voice/2/active", 0)
        assert cv._voltages[5] == pytest.approx(0.0)
    finally:
        r.close()


def test_voice_pitch_tracks_sliding_freq():
    """A voice whose center_freq drifts (glissando) → CV tracks the
    drift continuously. The CV backend's per-channel voltage is
    updated on every OSC message, so the effective output is the
    latest value."""
    cfg = RouterConfig(
        osc_listen_port=0,
        mappings=[Mapping(source="/voice/0/center_freq",
                          output="cv/0", scale="hz_1V_per_oct")],
    )
    cv = CVBackend(device=None, channels=2, sample_rate=48000, mock=True)
    r = Router(cfg, cv=cv, start_server=False)
    try:
        # Sweep 220 Hz → 880 Hz (one octave up)
        freqs = [220.0, 311.13, 440.0, 622.25, 880.0]
        expected = [-0.25, -0.0833 + -0.5833, 0.75, 1.25, 1.75]  # log2 from C4
        for f in freqs:
            r._on_osc_message("/voice/0/center_freq", f)
        # Last value = 880 Hz → 1.75 V (log2(880/261.63) = 1.748)
        assert cv._voltages[0] == pytest.approx(1.748, abs=1e-2)
        # CV writes captured in order — voltages were monotonic up
        written = [v for ch, v in cv._writes if ch == 0]
        assert written == sorted(written)
    finally:
        r.close()
