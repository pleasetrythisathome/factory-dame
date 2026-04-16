"""Cochlear front-end: gammatone filterbank → per-channel Hilbert envelope.

This is the x(t) driver for the GrFNN networks. Rhythm GrFNN takes the sum
of envelopes (a single onset-sensitive signal). Pitch GrFNN takes the full
band-limited signals directly — each pitch oscillator resonates with the
channels near its natural frequency.

Uses scipy.signal.gammatone for a proper 4th-order IIR gammatone (Slaney 1993).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import freqz, gammatone, hilbert, lfilter


def erb_space(low_hz: float, high_hz: float, n: int) -> np.ndarray:
    """Center frequencies log-spaced on the ERB scale, low → high.

    The ERB scale matches cochlear frequency resolution — more channels at
    low frequencies, fewer at high.
    """
    ear_q = 9.26449
    min_bw = 24.7
    erb_low = ear_q * np.log(1 + low_hz / (ear_q * min_bw))
    erb_high = ear_q * np.log(1 + high_hz / (ear_q * min_bw))
    erb = np.linspace(erb_low, erb_high, n)
    return ear_q * min_bw * (np.exp(erb / ear_q) - 1)


def _normalized_fir_gammatone(fc: float, fs: float) -> np.ndarray:
    """FIR gammatone impulse response, normalized so |H(fc)| == 1.

    We use the FIR form exclusively because scipy's IIR gammatone can put
    poles outside the unit circle at low fc/fs ratios (e.g. 30 Hz at 16 kHz
    fs), producing an unstable filter. The FIR form is unconditionally stable.
    """
    b, _ = gammatone(float(fc), "fir", fs=fs)
    _w, h = freqz(b, 1.0, worN=[fc], fs=fs)
    gain = float(np.abs(h[0]))
    if gain > 0:
        b = b / gain
    return np.asarray(b, dtype=np.float64)


class GammatoneFilterbank:
    """Bank of 4th-order FIR gammatone filters + Hilbert envelope."""

    def __init__(self, n_channels: int, low_hz: float, high_hz: float, fs: float):
        self.fs = fs
        self.fc = erb_space(low_hz, high_hz, n_channels)
        # List of FIR taps, one per channel. Lengths vary (longer at low fc).
        self.taps = [_normalized_fir_gammatone(float(f), fs) for f in self.fc]

    def filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply the filterbank. Returns (n_channels, n_samples) band-limited signals."""
        # float64 internal compute avoids underflow/overflow for wide dynamic
        # range (low-freq taps span 6+ orders of magnitude in amplitude).
        audio64 = audio.astype(np.float64)
        out = np.zeros((len(self.fc), len(audio)), dtype=np.float32)
        for i, b in enumerate(self.taps):
            out[i] = lfilter(b, 1.0, audio64).astype(np.float32)
        return out

    def envelope(self, audio: np.ndarray) -> np.ndarray:
        """Hilbert envelope per channel. Returns (n_channels, n_samples)."""
        bands = self.filter(audio)
        env = np.abs(hilbert(bands, axis=1)).astype(np.float32)
        return env
