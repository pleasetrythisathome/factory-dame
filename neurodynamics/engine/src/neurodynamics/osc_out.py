"""OSC broadcaster. Language-agnostic bridge to SuperCollider, TouchDesigner,
Overtone/Quil, browsers via osc.js, etc.

Fans out every message to ALL configured endpoints, so the router,
live viewer, TouchDesigner, and any other consumer can each bind
their own UDP port and receive the full stream independently.
UDP doesn't share ports across listeners, so "broadcast to many,
receive on one each" is the right topology.
"""

from __future__ import annotations

import numpy as np
from pythonosc.udp_client import SimpleUDPClient


class OSCBroadcaster:
    """One broadcaster, many destinations.

    Constructor accepts either the legacy single-destination
    ``host``/``port`` form (kept for backward compatibility with the
    original ``[osc]`` config block) OR a list of ``(host, port)``
    endpoints. Internally the broadcaster holds a list of UDP clients
    and forwards every message to each one.
    """

    def __init__(self, host: str | None = None, port: int | None = None,
                 enabled: bool = True,
                 endpoints: list[tuple[str, int]] | None = None):
        self.enabled = enabled
        targets: list[tuple[str, int]] = []
        if endpoints:
            targets.extend(endpoints)
        if host is not None and port is not None:
            targets.append((host, int(port)))
        # Deduplicate while preserving order.
        seen: set[tuple[str, int]] = set()
        self.endpoints: list[tuple[str, int]] = []
        for t in targets:
            if t in seen:
                continue
            seen.add(t)
            self.endpoints.append(t)
        self._clients: list[SimpleUDPClient] = (
            [SimpleUDPClient(h, p) for h, p in self.endpoints]
            if enabled else []
        )
        # Legacy attribute, retained for any downstream code that
        # was reading ``.client`` directly.
        self.client = self._clients[0] if self._clients else None

    def _send(self, address: str, value) -> None:
        """Fan-out: send ``value`` to every configured endpoint."""
        if not self.enabled:
            return
        for client in self._clients:
            client.send_message(address, value)

    def send_layer(self, layer: str, z: np.ndarray,
                   phantom: np.ndarray, drive: np.ndarray,
                   residual: np.ndarray) -> None:
        if not self.enabled:
            return
        amp = np.abs(z).astype(np.float32).tolist()
        phase = np.angle(z).astype(np.float32).tolist()
        self._send(f"/{layer}/amp", amp)
        self._send(f"/{layer}/phase", phase)
        self._send(f"/{layer}/phantom",
                                 phantom.astype(np.int32).tolist())
        self._send(f"/{layer}/drive",
                                 drive.astype(np.float32).tolist())
        self._send(f"/{layer}/residual",
                                 residual.astype(np.float32).tolist())

    def send_features(self, features: dict) -> None:
        """Emit perceptual rollups on the /features/* namespace.

        Expects a dict with keys: tempo (float BPM), tempo_conf (0-1),
        tonic (str pitch class), mode ("major"/"minor"), key_conf (0-1),
        consonance (0-1). Missing keys are skipped.
        """
        if not self.enabled:
            return
        if "tempo" in features:
            self._send("/features/tempo",
                                     float(features["tempo"]))
        if "tempo_conf" in features:
            self._send("/features/tempo_confidence",
                                     float(features["tempo_conf"]))
        if "tonic" in features:
            self._send("/features/key",
                                     str(features["tonic"]))
        if "mode" in features:
            self._send("/features/mode",
                                     str(features["mode"]))
        if "key_conf" in features:
            self._send("/features/key_confidence",
                                     float(features["key_conf"]))
        if "chord" in features:
            self._send("/features/chord",
                                     str(features["chord"]))
        if "chord_quality" in features:
            self._send("/features/chord_quality",
                                     str(features["chord_quality"]))
        if "chord_conf" in features:
            self._send("/features/chord_confidence",
                                     float(features["chord_conf"]))
        if "consonance" in features:
            self._send("/features/consonance",
                                     float(features["consonance"]))

    def send_rhythm_structure(self, rhythm: dict) -> None:
        """Emit the multi-clock rhythm primitives on /rhythm/*.

        The peak oscillator carries the main beat; companions carry
        phase-locked subdivisions (triplets, half-time, polyrhythms).
        Downstream consumers (router, TouchDesigner) turn the phases
        into gates, CV, or visual triggers.

        Messages:
            /rhythm/peak/freq   float Hz
            /rhythm/peak/bpm    float BPM
            /rhythm/peak/phase  float radians, (-π, π]
            /rhythm/peak/idx    int   oscillator index
            For each companion i:
                /rhythm/companion/<i>/ratio  [p, q] ints
                /rhythm/companion/<i>/freq   float Hz
                /rhythm/companion/<i>/phase  float radians
                /rhythm/companion/<i>/plv    float 0-1
            /rhythm/companion/count  int total number of companions
        """
        if not self.enabled:
            return
        peak = rhythm.get("peak", {})
        if peak:
            self._send("/rhythm/peak/freq",
                                     float(peak.get("freq", 0.0)))
            self._send("/rhythm/peak/bpm",
                                     float(peak.get("bpm", 0.0)))
            self._send("/rhythm/peak/phase",
                                     float(peak.get("phase", 0.0)))
            self._send("/rhythm/peak/idx",
                                     int(peak.get("idx", 0)))
        companions = rhythm.get("companions", [])
        self._send("/rhythm/companion/count", len(companions))
        for i, comp in enumerate(companions):
            base = f"/rhythm/companion/{i}"
            self._send(f"{base}/ratio",
                                     [int(comp["ratio_p"]),
                                      int(comp["ratio_q"])])
            self._send(f"{base}/freq",
                                     float(comp["freq"]))
            self._send(f"{base}/phase",
                                     float(comp["phase"]))
            self._send(f"{base}/plv",
                                     float(comp["plv"]))

    def send_voices(self, voice_state) -> None:
        """Emit per-voice identity state on /voice/*.

        A voice is a phase-coherent, amplitude-correlated cluster of
        pitch oscillators. IDs persist across frames — a consumer
        watching ``/voice/<id>/active`` sees a clean on/off signal
        even across brief silences.

        Messages (per active voice):
            /voice/<id>/active       int (0/1)
            /voice/<id>/center_freq  float Hz
            /voice/<id>/amp          float 0-1ish
            /voice/<id>/phase        float radians, (-π, π]
            /voice/<id>/confidence   float 0-1
            /voice/<id>/age_frames   int
        Global:
            /voice/active_count      int
            /voice/active_ids        [int, int, ...]
        """
        if not self.enabled:
            return
        active = [v for v in voice_state.voices if v.active]
        self._send("/voice/active_count", len(active))
        self._send(
            "/voice/active_ids",
            [int(v.id) for v in active] if active else [-1],
        )
        for v in active:
            base = f"/voice/{int(v.id)}"
            self._send(f"{base}/active", 1)
            self._send(f"{base}/center_freq",
                                     float(v.center_freq))
            self._send(f"{base}/amp", float(v.amp))
            self._send(f"{base}/phase",
                                     float(v.phase_centroid))
            self._send(f"{base}/confidence",
                                     float(v.confidence))
            self._send(f"{base}/age_frames",
                                     int(v.age_frames))
            # Per-voice rhythm (Phase 2). The OSC consumer uses the
            # rhythm osc_idx + phase to generate per-voice triggers;
            # bpm and freq are convenience fields. When rhythm is
            # None (voice has no discernible tempo), these messages
            # are skipped so stale values don't get latched.
            if v.rhythm is not None:
                self._send(f"{base}/rhythm/freq",
                                         float(v.rhythm.freq))
                self._send(f"{base}/rhythm/bpm",
                                         float(v.rhythm.bpm))
                self._send(f"{base}/rhythm/phase",
                                         float(v.rhythm.phase))
                self._send(f"{base}/rhythm/osc_idx",
                                         int(v.rhythm.osc_idx))
                self._send(f"{base}/rhythm/confidence",
                                         float(v.rhythm.confidence))
            # Per-voice motor (Phase 3) — predictive beat phase. The
            # semantic distinction from rhythm: motor carries forward-
            # prediction through bidirectional sensory↔motor coupling,
            # so this phase is the anticipated next beat rather than
            # the current sensory one.
            if v.motor is not None:
                self._send(f"{base}/motor/freq",
                                         float(v.motor.freq))
                self._send(f"{base}/motor/bpm",
                                         float(v.motor.bpm))
                self._send(f"{base}/motor/phase",
                                         float(v.motor.phase))
                self._send(f"{base}/motor/osc_idx",
                                         int(v.motor.osc_idx))
                self._send(f"{base}/motor/confidence",
                                         float(v.motor.confidence))
        # Signal deactivation for voices that went silent this frame.
        for v in voice_state.voices:
            if not v.active and v.silent_frames == 1:
                self._send(f"/voice/{int(v.id)}/active", 0)
