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
