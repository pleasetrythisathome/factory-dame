"""CV / MIDI / OSC router — consumes the engine's OSC stream and
emits control signals for modular hardware consumers.

This is Phase 4 of ``task-011`` and the product-side spine of the
live-NRT-to-modular vision. The engine publishes per-voice and
per-feature OSC (rhythm phase, motor phase, tonic, chord, ...);
this module subscribes, applies per-channel scaling, and drops
signals onto three backend families:

- **CV** via a DC-coupled USB audio interface (primary target
  Expert Sleepers ES-9, any audio-as-CV device works). Each output
  channel is one numerical stream; sounddevice writes them as
  "audio" and the DC-coupled hardware treats the samples as voltage.
- **MIDI** via ``mido`` (clock messages at detected tempo, MPE-style
  note events for chord changes, CC messages mirroring CV).
- **OSC** forwarding — re-broadcast selected paths to additional
  endpoints (TouchDesigner, Overtone, a second router, ...).

Design philosophy:

- Router is a **pure subscriber** to the engine's OSC stream.
  Engine is unchanged; router is an orthogonal process. You can
  start engine + viewer + router independently.
- **Config-driven routing**: ``[router]`` TOML block maps source
  OSC paths to output channels via typed scaling functions.
  Re-routing is a config edit, not a code change.
- **Backends are pluggable**: each backend implements a small
  protocol; disabled backends short-circuit at init. Running with
  only OSC forwarding (no hardware) works for development and CI.
- **Scale functions are pure** and unit-tested in isolation — the
  1V/oct math, sawtooth wrap, and gate-on-phase-wrap logic stay
  readable and regression-proof.

Hardware-in-the-loop tests (actual ES-9 voltage verification, real
MIDI downstream device sync) are flagged skip-when-missing; the
bulk of the test coverage validates dispatch correctness in
software.
"""

from __future__ import annotations

import argparse
import threading
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

__all__ = [
    "RouterConfig",
    "Mapping",
    "Router",
    "CVBackend",
    "MIDIBackend",
    "OSCForwardBackend",
    "scale_tonic_to_1v_per_oct",
    "scale_hz_to_1v_per_oct",
    "scale_phase_to_sawtooth",
    "scale_unit_to_5v",
    "scale_active_as_gate",
    "phase_wrapped",
]


# ── Pure scaling functions ────────────────────────────────────────

# 1V/oct reference: C0 = 0V. Chromatic offsets relative to C per
# pitch-class name. Consumers typically calibrate the zero-point
# with their quantizer — the router reports what it thinks C0 is
# and the user tunes.
_PITCH_CLASS_VOLTS: dict[str, float] = {
    "C": 0.0, "C#": 1 / 12, "D": 2 / 12, "D#": 3 / 12,
    "E": 4 / 12, "F": 5 / 12, "F#": 6 / 12, "G": 7 / 12,
    "G#": 8 / 12, "A": 9 / 12, "A#": 10 / 12, "B": 11 / 12,
}


def scale_tonic_to_1v_per_oct(tonic: str, octave: int = 4) -> float:
    """Map a pitch-class name (e.g. "F#") to a 1V/oct voltage.
    ``octave`` sets the absolute octave; default 4 = middle octave,
    so "C" at octave 4 returns 4.0 V. Consumers can shift via a
    calibration offset."""
    pc = _PITCH_CLASS_VOLTS.get(tonic)
    if pc is None:
        raise ValueError(f"unknown pitch class {tonic!r}")
    return float(octave) + pc


def scale_phase_to_sawtooth(phase_rad: float, v_min: float = 0.0,
                             v_max: float = 5.0) -> float:
    """Map phase in (-π, π] to a sawtooth in [v_min, v_max]. Phase
    wraparound produces a discontinuous jump — which is the right
    CV shape for a modular envelope generator expecting a ramp
    that resets every beat."""
    # Normalize phase to [0, 1)
    normalized = (float(phase_rad) + np.pi) / (2 * np.pi)
    normalized = normalized - np.floor(normalized)
    return float(v_min + (v_max - v_min) * normalized)


def scale_unit_to_5v(unit: float) -> float:
    """Map a [0, 1] scalar to a [0, 5] V signal, clipped."""
    return float(max(0.0, min(5.0, unit * 5.0)))


def scale_hz_to_1v_per_oct(hz: float, ref_hz: float = 261.6256) -> float:
    """Map a frequency in Hz to a 1V/oct voltage, anchored at
    ``ref_hz`` = 0 V. Default anchor is C4 (261.63 Hz), matching the
    VCV/Eurorack convention. Useful for routing voice.center_freq
    directly to a VCO's V/OCT input.

    Negative voltages for notes below the anchor (fine — Eurorack
    VCOs and DC-coupled audio interfaces handle ±5-10 V). Returns
    0.0 for non-positive ``hz`` to avoid log domain errors."""
    if hz <= 0 or ref_hz <= 0:
        return 0.0
    return float(np.log2(hz / ref_hz))


def scale_active_as_gate(active: float, on_v: float = 5.0,
                          off_v: float = 0.0) -> float:
    """Map a boolean-like value (0 or 1) to a gate voltage. Used for
    voice on/off signals that need to drive an ADSR gate."""
    return float(on_v if active > 0.5 else off_v)


def phase_wrapped(current_phase: float, prev_phase: float) -> bool:
    """True iff the phase wrapped from +π to -π between prev and
    current samples — the canonical "beat just landed" trigger."""
    if prev_phase is None:
        return False
    return bool(current_phase < prev_phase - np.pi / 2)


# ── Config ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Mapping:
    """One routing rule: OSC source path → output channel on a
    specific backend, with a named scale function applied."""

    source: str            # e.g. "/voice/0/rhythm/phase"
    output: str            # e.g. "cv/0", "midi/cc/7", "osc/forward/0"
    scale: str             # e.g. "sawtooth_0_5V", "1V_per_oct"


@dataclass
class RouterConfig:
    # CV backend (sounddevice output to DC-coupled interface)
    cv_enabled: bool = False
    cv_device: str | None = None  # audio device name; None → default
    cv_channels: int = 14          # ES-9 has 14 DC outs
    cv_sample_rate: int = 48000
    cv_block_size: int = 256

    # MIDI backend (mido)
    midi_enabled: bool = False
    midi_port: str | None = None   # mido port name

    # OSC subscribe side
    osc_listen_host: str = "127.0.0.1"
    osc_listen_port: int = 57120

    # OSC forward targets
    osc_forward_targets: list[tuple[str, int]] = field(default_factory=list)

    # Routing rules
    mappings: list[Mapping] = field(default_factory=list)

    @classmethod
    def from_toml(cls, path: Path) -> "RouterConfig":
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
        r = cfg.get("router", {})
        cv = r.get("cv", {})
        midi = r.get("midi", {})
        osc = r.get("osc", {})
        mappings = [
            Mapping(
                source=m["source"],
                output=m["output"],
                scale=m.get("scale", "passthrough"),
            )
            for m in r.get("mapping", [])
        ]
        return cls(
            cv_enabled=cv.get("enabled", False),
            cv_device=cv.get("device"),
            cv_channels=int(cv.get("channels", 14)),
            cv_sample_rate=int(cv.get("sample_rate", 48000)),
            cv_block_size=int(cv.get("block_size", 256)),
            midi_enabled=midi.get("enabled", False),
            midi_port=midi.get("port"),
            osc_listen_host=osc.get("listen_host", "127.0.0.1"),
            osc_listen_port=int(osc.get("listen_port", 57120)),
            osc_forward_targets=[
                (t["host"], int(t["port"]))
                for t in osc.get("forward", [])
            ],
            mappings=mappings,
        )


# ── Backends ──────────────────────────────────────────────────────

class CVBackend:
    """Write per-channel voltages to a DC-coupled audio interface.

    Holds an (channels,) voltage vector. A background sounddevice
    output stream samples that vector into the output buffer at
    ``sample_rate``. Voltages stay constant between updates; the
    hardware sees them as DC.

    When the sounddevice library isn't functional (no device, CI
    environment) the backend runs in mock mode and records writes
    for test assertions.
    """

    def __init__(self, device: str | None, channels: int,
                 sample_rate: int, block_size: int = 256,
                 mock: bool = False):
        self.channels = channels
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.mock = mock
        self._voltages = np.zeros(channels, dtype=np.float32)
        self._writes: list[tuple[int, float]] = []  # for mock mode
        self._stream = None
        if not mock:
            import sounddevice as sd
            self._stream = sd.OutputStream(
                samplerate=sample_rate,
                blocksize=block_size,
                channels=channels,
                dtype="float32",
                device=device,
                callback=self._callback,
            )
            self._stream.start()

    def _callback(self, outdata, frames, time_info, status):
        outdata[:] = np.broadcast_to(self._voltages, outdata.shape)

    def set_channel(self, idx: int, voltage: float) -> None:
        if 0 <= idx < self.channels:
            self._voltages[idx] = float(voltage)
            if self.mock:
                self._writes.append((idx, float(voltage)))

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass


class MIDIBackend:
    """Emit MIDI messages via ``mido``.

    Supports clock messages (24 ppq from an externally-computed
    BPM), note-on/off (for chord changes), and CC (for
    continuous-value mirrors of CV channels).

    When ``mock=True`` or when the named port doesn't exist, runs
    in record-only mode for test assertions.
    """

    def __init__(self, port_name: str | None, mock: bool = False):
        self.port_name = port_name
        self.mock = mock
        self._port = None
        self._sent: list[dict] = []  # record for mock / assertions
        if not mock and port_name:
            import mido
            available = mido.get_output_names()
            if port_name in available:
                self._port = mido.open_output(port_name)
            else:
                # Port not present — quietly fall back to mock so
                # the router still runs for non-MIDI channels.
                self.mock = True

    def send_clock(self) -> None:
        self._record({"type": "clock"})
        if self._port is not None:
            import mido
            self._port.send(mido.Message("clock"))

    def send_note_on(self, channel: int, note: int, velocity: int = 96) -> None:
        self._record({"type": "note_on", "channel": channel,
                       "note": note, "velocity": velocity})
        if self._port is not None:
            import mido
            self._port.send(mido.Message(
                "note_on", channel=channel, note=note, velocity=velocity,
            ))

    def send_note_off(self, channel: int, note: int) -> None:
        self._record({"type": "note_off", "channel": channel, "note": note})
        if self._port is not None:
            import mido
            self._port.send(mido.Message(
                "note_off", channel=channel, note=note, velocity=0,
            ))

    def send_cc(self, channel: int, cc: int, value: int) -> None:
        self._record({"type": "cc", "channel": channel,
                       "cc": cc, "value": value})
        if self._port is not None:
            import mido
            self._port.send(mido.Message(
                "control_change", channel=channel, control=cc, value=value,
            ))

    def _record(self, msg: dict) -> None:
        if self.mock:
            self._sent.append(msg)

    def close(self) -> None:
        if self._port is not None:
            try:
                self._port.close()
            except Exception:
                pass


class OSCForwardBackend:
    """Re-broadcast selected OSC paths to additional endpoints.
    Useful for chaining a router into TouchDesigner or Overtone
    without modifying the engine."""

    def __init__(self, targets: list[tuple[str, int]]):
        self.targets = targets
        self._clients = [SimpleUDPClient(h, p) for h, p in targets]
        self._sent: list[tuple[str, tuple]] = []  # record for tests

    def forward(self, address: str, args: tuple) -> None:
        self._sent.append((address, args))
        for c in self._clients:
            try:
                c.send_message(address, list(args) if len(args) != 1 else args[0])
            except Exception:
                pass


# ── Scale function registry ───────────────────────────────────────

ScaleFn = Callable[[tuple], float | None]


def _scale_1v_per_oct(args: tuple) -> float | None:
    """args = (pitch_class_str,) → volts."""
    if not args:
        return None
    try:
        return scale_tonic_to_1v_per_oct(str(args[0]))
    except ValueError:
        return None


def _scale_sawtooth_0_5V(args: tuple) -> float | None:
    """args = (phase_rad,) → volts in [0, 5]."""
    if not args:
        return None
    return scale_phase_to_sawtooth(float(args[0]), 0.0, 5.0)


def _scale_unit_0_5V(args: tuple) -> float | None:
    """args = (unit_scalar,) → volts in [0, 5]."""
    if not args:
        return None
    return scale_unit_to_5v(float(args[0]))


def _scale_passthrough(args: tuple) -> float | None:
    if not args:
        return None
    try:
        return float(args[0])
    except (TypeError, ValueError):
        return None


def _scale_hz_1v_per_oct(args: tuple) -> float | None:
    """args = (hz_float,) → volts. Used for /voice/*/center_freq
    routed to a V/OCT CV channel."""
    if not args:
        return None
    try:
        return scale_hz_to_1v_per_oct(float(args[0]))
    except (TypeError, ValueError):
        return None


def _scale_gate_5v(args: tuple) -> float | None:
    """args = (active_0_or_1,) → 5 V or 0 V. Used for
    /voice/*/active routed to a gate CV channel."""
    if not args:
        return None
    try:
        return scale_active_as_gate(float(args[0]))
    except (TypeError, ValueError):
        return None


_SCALE_REGISTRY: dict[str, ScaleFn] = {
    "1V_per_oct": _scale_1v_per_oct,
    "hz_1V_per_oct": _scale_hz_1v_per_oct,
    "sawtooth_0_5V": _scale_sawtooth_0_5V,
    "unit_0_5V": _scale_unit_0_5V,
    "gate_5V": _scale_gate_5v,
    "passthrough": _scale_passthrough,
}


# ── Router ────────────────────────────────────────────────────────

class Router:
    """Coordinates the three backends against a routing table.

    Owns the OSC listener thread, dispatches incoming messages to
    mappings, and applies scale functions before dispatching to
    backends. Starts in constructor; call ``close()`` to tear down.
    """

    def __init__(self, config: RouterConfig, *,
                 cv: CVBackend | None = None,
                 midi: MIDIBackend | None = None,
                 osc_forward: OSCForwardBackend | None = None,
                 start_server: bool = True):
        self.config = config
        self.cv = cv
        self.midi = midi
        self.osc_forward = osc_forward
        # Phase wrap state per beat-gate mapping, keyed by source path.
        self._prev_phase: dict[str, float] = {}
        self._dispatcher = Dispatcher()
        # Register a default handler that runs mapping logic for every
        # incoming OSC address. Wildcarding in pythonosc dispatcher is
        # limited, so we handle matching manually.
        self._dispatcher.set_default_handler(self._on_osc_message)
        self._server: ThreadingOSCUDPServer | None = None
        self._server_thread: threading.Thread | None = None
        if start_server:
            self._server = ThreadingOSCUDPServer(
                (config.osc_listen_host, config.osc_listen_port),
                self._dispatcher,
            )
            self._server_thread = threading.Thread(
                target=self._server.serve_forever, daemon=True,
            )
            self._server_thread.start()

    # OSC handler ——————————————————————————————————————————————————

    def _on_osc_message(self, address: str, *args) -> None:
        """Dispatch one OSC message to every matching mapping."""
        if self.osc_forward is not None:
            self.osc_forward.forward(address, args)
        for mapping in self.config.mappings:
            if _osc_address_matches(mapping.source, address):
                self._apply_mapping(mapping, address, args)

    def _apply_mapping(self, mapping: Mapping, address: str,
                       args: tuple) -> None:
        scale_fn = _SCALE_REGISTRY.get(mapping.scale, _scale_passthrough)
        value = scale_fn(args)
        if value is None:
            return
        kind, _, channel = mapping.output.partition("/")
        if kind == "cv" and self.cv is not None:
            try:
                idx = int(channel)
            except ValueError:
                return
            self.cv.set_channel(idx, float(value))
        elif kind == "midi" and self.midi is not None:
            self._dispatch_midi(channel, float(value), address, args, mapping)
        elif kind == "osc" and self.osc_forward is not None:
            # already forwarded at top of handler; nothing more to do
            return

    def _dispatch_midi(self, channel_spec: str, value: float,
                       address: str, args: tuple, mapping: Mapping) -> None:
        if channel_spec == "clock":
            # Emit a single clock tick — the router doesn't drive
            # 24 ppq itself; the mapping source is already at ppq rate.
            self.midi.send_clock()
            return
        if channel_spec.startswith("cc/"):
            try:
                cc_num = int(channel_spec.split("/", 1)[1])
            except (ValueError, IndexError):
                return
            midi_value = max(0, min(127, int(value)))
            self.midi.send_cc(0, cc_num, midi_value)
            return

    def close(self) -> None:
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
        if self.cv is not None:
            self.cv.close()
        if self.midi is not None:
            self.midi.close()


def _osc_address_matches(pattern: str, address: str) -> bool:
    """Minimal glob matching for mapping source patterns. Supports
    ``*`` as a single-segment wildcard; other glob features aren't
    wired yet."""
    pat_parts = pattern.strip("/").split("/")
    addr_parts = address.strip("/").split("/")
    if len(pat_parts) != len(addr_parts):
        return False
    return all(p == "*" or p == a for p, a in zip(pat_parts, addr_parts))


# ── CLI ────────────────────────────────────────────────────────────

def build_from_config(config: RouterConfig) -> Router:
    """Construct a Router with all backends enabled per config.
    Missing hardware is handled gracefully (backend falls to mock)."""
    cv = None
    if config.cv_enabled:
        try:
            cv = CVBackend(
                device=config.cv_device,
                channels=config.cv_channels,
                sample_rate=config.cv_sample_rate,
                block_size=config.cv_block_size,
            )
        except Exception as e:
            print(f"[router] CV backend unavailable ({e}); using mock")
            cv = CVBackend(
                device=None, channels=config.cv_channels,
                sample_rate=config.cv_sample_rate,
                block_size=config.cv_block_size, mock=True,
            )
    midi = None
    if config.midi_enabled:
        midi = MIDIBackend(port_name=config.midi_port)
    osc_fwd = None
    if config.osc_forward_targets:
        osc_fwd = OSCForwardBackend(config.osc_forward_targets)
    return Router(config, cv=cv, midi=midi, osc_forward=osc_fwd)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="CV/MIDI/OSC router for the NRT engine",
    )
    ap.add_argument("--config", type=Path, default=Path("config.toml"),
                    help="engine + router TOML config (default: config.toml)")
    args = ap.parse_args()
    cfg = RouterConfig.from_toml(args.config)
    print(f"[router] listening on {cfg.osc_listen_host}:{cfg.osc_listen_port}")
    print(f"[router] CV: {'enabled' if cfg.cv_enabled else 'disabled'}")
    print(f"[router] MIDI: {'enabled' if cfg.midi_enabled else 'disabled'}")
    print(f"[router] OSC forward targets: {len(cfg.osc_forward_targets)}")
    print(f"[router] mappings: {len(cfg.mappings)}")
    router = build_from_config(cfg)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[router] shutting down")
    finally:
        router.close()


if __name__ == "__main__":
    main()
