"""Twelve-tone equal temperament grid for the pitch GrFNN.

A pitch oscillator bank laid out on a log-uniform grid (``geomspace``)
does not land on named notes. A4=440 Hz falls between bins, every 2f
harmonic lands between bins, and the Hopf oscillator's finite
bandwidth means detuned drive produces weaker response. That cascades:
near-noise-floor amplitudes, spurious correlations in the voice
clusterer, voice centroids at arbitrary Hz instead of note
frequencies.

This module produces a bank whose bins sit *on* the 12-TET grid —
each bin at ``a4_hz × 2 ** (semitones_from_a4 / 12)``. Harmonics then
land on integer bin offsets (2f = +12×bins_per_semitone, 4f = +24×,
etc.), the harmonic boost in ``voices.py`` sees exact ratios, and the
UI reports note names.

The ``a4_hz`` anchor is configurable because a lot of music — 432 Hz
ambient, 415 Hz baroque, DJ pitch-shifted tracks — isn't at the
standard reference. At 3 bins per semitone (33 cent spacing), common
anchors land within ~1.5 cents of a bin (432 → bin at 431.65, 415 →
bin at 415.3), so even a fixed 440 anchor absorbs most real-world
variation; setting the anchor explicitly just pushes worst-case
detune to zero.
"""

from __future__ import annotations

import math

import numpy as np


def twelve_tet_freqs(
    low_hz: float,
    high_hz: float,
    *,
    a4_hz: float = 440.0,
    bins_per_semitone: int = 3,
) -> np.ndarray:
    """Return a frequency array on a 12-TET grid covering [low_hz, high_hz].

    The grid is anchored at ``a4_hz`` (the pitch of bin at semitone
    offset 0). Bin spacing is ``12 * bins_per_semitone`` per octave.
    The first bin is the largest bin ≤ ``low_hz``, the last is the
    smallest bin ≥ ``high_hz`` — so the returned range always covers
    the requested interval.

    With ``bins_per_semitone=3`` and the default 440 Hz anchor, bins
    fall at (e.g.) A4=440.00, A4+33c=448.55, A4+67c=457.27,
    A#4=466.16, etc. Worst-case detune of an arbitrary input tone
    from its nearest bin is 16.67 cents — within Hopf bandwidth and
    well within the 50-cent harmonic-ratio tolerance used by the
    voice clusterer.
    """
    if low_hz <= 0 or high_hz <= low_hz:
        raise ValueError(
            f"invalid range: low_hz={low_hz}, high_hz={high_hz}"
        )
    if bins_per_semitone < 1:
        raise ValueError(
            f"bins_per_semitone must be ≥ 1, got {bins_per_semitone}"
        )
    # Semitones (continuous) from A4 at the range boundaries.
    low_semi = 12.0 * math.log2(low_hz / a4_hz)
    high_semi = 12.0 * math.log2(high_hz / a4_hz)
    # Snap to bin indices. Each bin sits at semitone offset
    # k / bins_per_semitone relative to the anchor.
    k_low = math.floor(low_semi * bins_per_semitone)
    k_high = math.ceil(high_semi * bins_per_semitone)
    ks = np.arange(k_low, k_high + 1, dtype=np.float64)
    semis_from_a4 = ks / bins_per_semitone
    freqs = a4_hz * np.power(2.0, semis_from_a4 / 12.0)
    return freqs


def note_name(freq_hz: float, *, a4_hz: float = 440.0) -> str:
    """Nearest 12-TET note name with cents offset, e.g. ``"A4+0"`` or
    ``"C#5-12"``. Useful for diagnostic output."""
    if freq_hz <= 0:
        return "—"
    semis_from_a4 = 12.0 * math.log2(freq_hz / a4_hz)
    nearest = round(semis_from_a4)
    cents_off = int(round((semis_from_a4 - nearest) * 100))
    # Note index: A4 is MIDI 69. Semis above A4 → MIDI = 69 + nearest.
    midi = 69 + nearest
    pcs = ["C", "C#", "D", "D#", "E", "F",
            "F#", "G", "G#", "A", "A#", "B"]
    pc = pcs[midi % 12]
    octv = midi // 12 - 1
    sign = "+" if cents_off >= 0 else ""
    return f"{pc}{octv}{sign}{cents_off}"
