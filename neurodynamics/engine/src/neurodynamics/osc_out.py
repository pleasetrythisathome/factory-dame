"""OSC broadcaster. Language-agnostic bridge to SuperCollider, TouchDesigner,
Overtone/Quil, browsers via osc.js, etc."""

from __future__ import annotations

import numpy as np
from pythonosc.udp_client import SimpleUDPClient


class OSCBroadcaster:
    def __init__(self, host: str, port: int, enabled: bool = True):
        self.enabled = enabled
        self.client = SimpleUDPClient(host, port) if enabled else None

    def send_layer(self, layer: str, z: np.ndarray,
                   phantom: np.ndarray, drive: np.ndarray,
                   residual: np.ndarray) -> None:
        if not self.enabled:
            return
        amp = np.abs(z).astype(np.float32).tolist()
        phase = np.angle(z).astype(np.float32).tolist()
        self.client.send_message(f"/{layer}/amp", amp)
        self.client.send_message(f"/{layer}/phase", phase)
        self.client.send_message(f"/{layer}/phantom",
                                 phantom.astype(np.int32).tolist())
        self.client.send_message(f"/{layer}/drive",
                                 drive.astype(np.float32).tolist())
        self.client.send_message(f"/{layer}/residual",
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
            self.client.send_message("/features/tempo",
                                     float(features["tempo"]))
        if "tempo_conf" in features:
            self.client.send_message("/features/tempo_confidence",
                                     float(features["tempo_conf"]))
        if "tonic" in features:
            self.client.send_message("/features/key",
                                     str(features["tonic"]))
        if "mode" in features:
            self.client.send_message("/features/mode",
                                     str(features["mode"]))
        if "key_conf" in features:
            self.client.send_message("/features/key_confidence",
                                     float(features["key_conf"]))
        if "chord" in features:
            self.client.send_message("/features/chord",
                                     str(features["chord"]))
        if "chord_quality" in features:
            self.client.send_message("/features/chord_quality",
                                     str(features["chord_quality"]))
        if "chord_conf" in features:
            self.client.send_message("/features/chord_confidence",
                                     float(features["chord_conf"]))
        if "consonance" in features:
            self.client.send_message("/features/consonance",
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
            self.client.send_message("/rhythm/peak/freq",
                                     float(peak.get("freq", 0.0)))
            self.client.send_message("/rhythm/peak/bpm",
                                     float(peak.get("bpm", 0.0)))
            self.client.send_message("/rhythm/peak/phase",
                                     float(peak.get("phase", 0.0)))
            self.client.send_message("/rhythm/peak/idx",
                                     int(peak.get("idx", 0)))
        companions = rhythm.get("companions", [])
        self.client.send_message("/rhythm/companion/count", len(companions))
        for i, comp in enumerate(companions):
            base = f"/rhythm/companion/{i}"
            self.client.send_message(f"{base}/ratio",
                                     [int(comp["ratio_p"]),
                                      int(comp["ratio_q"])])
            self.client.send_message(f"{base}/freq",
                                     float(comp["freq"]))
            self.client.send_message(f"{base}/phase",
                                     float(comp["phase"]))
            self.client.send_message(f"{base}/plv",
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
        self.client.send_message("/voice/active_count", len(active))
        self.client.send_message(
            "/voice/active_ids",
            [int(v.id) for v in active] if active else [-1],
        )
        for v in active:
            base = f"/voice/{int(v.id)}"
            self.client.send_message(f"{base}/active", 1)
            self.client.send_message(f"{base}/center_freq",
                                     float(v.center_freq))
            self.client.send_message(f"{base}/amp", float(v.amp))
            self.client.send_message(f"{base}/phase",
                                     float(v.phase_centroid))
            self.client.send_message(f"{base}/confidence",
                                     float(v.confidence))
            self.client.send_message(f"{base}/age_frames",
                                     int(v.age_frames))
            # Per-voice rhythm (Phase 2). The OSC consumer uses the
            # rhythm osc_idx + phase to generate per-voice triggers;
            # bpm and freq are convenience fields. When rhythm is
            # None (voice has no discernible tempo), these messages
            # are skipped so stale values don't get latched.
            if v.rhythm is not None:
                self.client.send_message(f"{base}/rhythm/freq",
                                         float(v.rhythm.freq))
                self.client.send_message(f"{base}/rhythm/bpm",
                                         float(v.rhythm.bpm))
                self.client.send_message(f"{base}/rhythm/phase",
                                         float(v.rhythm.phase))
                self.client.send_message(f"{base}/rhythm/osc_idx",
                                         int(v.rhythm.osc_idx))
                self.client.send_message(f"{base}/rhythm/confidence",
                                         float(v.rhythm.confidence))
            # Per-voice motor (Phase 3) — predictive beat phase. The
            # semantic distinction from rhythm: motor carries forward-
            # prediction through bidirectional sensory↔motor coupling,
            # so this phase is the anticipated next beat rather than
            # the current sensory one.
            if v.motor is not None:
                self.client.send_message(f"{base}/motor/freq",
                                         float(v.motor.freq))
                self.client.send_message(f"{base}/motor/bpm",
                                         float(v.motor.bpm))
                self.client.send_message(f"{base}/motor/phase",
                                         float(v.motor.phase))
                self.client.send_message(f"{base}/motor/osc_idx",
                                         int(v.motor.osc_idx))
                self.client.send_message(f"{base}/motor/confidence",
                                         float(v.motor.confidence))
        # Signal deactivation for voices that went silent this frame.
        for v in voice_state.voices:
            if not v.active and v.silent_frames == 1:
                self.client.send_message(f"/voice/{int(v.id)}/active", 0)
