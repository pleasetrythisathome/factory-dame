"""Generate a test audio file: a simple groove with a missing-pulse rhythm
and a few melodic pitches. Useful for sanity-checking the engine."""

import numpy as np
import soundfile as sf

sr = 16000
duration = 6.0
n = int(sr * duration)
t = np.arange(n) / sr

audio = np.zeros(n, dtype=np.float32)

# Rhythmic onset pattern: on-beats at 2 Hz (120 BPM) but SKIP every 4th beat.
# This creates a missing-pulse rhythm — the brain (and our GrFNN) should still
# resonate at 2 Hz even though there's no energy at that exact period.
beat_period = 0.5  # seconds
skip_every = 4
for k in range(int(duration / beat_period)):
    if k % skip_every == 0:
        continue
    onset = int(k * beat_period * sr)
    # Short noise burst as "drum".
    burst_len = int(0.02 * sr)
    env = np.exp(-np.linspace(0, 8, burst_len))
    noise = np.random.randn(burst_len) * env
    audio[onset:onset + burst_len] += noise * 0.6

# Melodic pitches: a simple motif over the rhythm. C4 - E4 - G4 - C5.
pitches_hz = [261.63, 329.63, 392.00, 523.25]
note_dur = 1.0
for k, f in enumerate(pitches_hz):
    start = int(k * note_dur * sr)
    end = int((k * note_dur + 0.8) * note_dur * sr / note_dur)
    if end > n:
        end = n
    segment = t[start:end] - t[start]
    note = np.sin(2 * np.pi * f * segment) * 0.3
    # Small AD envelope.
    env = np.ones_like(note)
    attack = int(0.02 * sr)
    release = int(0.1 * sr)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    audio[start:end] += (note * env).astype(np.float32)

# Normalize.
audio /= max(np.abs(audio).max(), 1e-6)
audio *= 0.8

sf.write("test.wav", audio, sr)
print(f"Wrote test.wav  {duration}s @ {sr} Hz")
