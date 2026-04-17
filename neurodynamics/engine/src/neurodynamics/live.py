"""Live audio mode — real-time streaming from sounddevice into the engine.

Reads an input device (mic, Loopback, any CoreAudio source) in chunks,
advances the engine through each chunk via ``GrFNN.step_many``, and
emits OSC + per-voice state on snapshot ticks. No parquet log —
this path is for streaming into the router, not for offline playback.

Usage:
    uv run nd-live                              # default input
    uv run nd-live --device "Loopback Audio"    # named device
    uv run nd-live --config config.toml --record out.parquet

Architecture: audio callback pushes chunks onto a thread-safe queue;
processing thread dequeues, runs the engine, emits OSC. The audio
callback stays short so the CoreAudio thread never misses its
deadline — Python GIL contention is the main risk, bounded by
keeping everything in the callback to numpy copies and queue puts.

The engine needs ≤1× realtime to keep up. With the numba JIT landed
in the prior commit (step_many at ~10× realtime on M-series), there
is comfortable headroom. If the queue backs up the processing thread
catches up or drops chunks — failure mode is glitchy CV, not
silent corruption.
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
import tomllib
from pathlib import Path

import numpy as np
import sounddevice as sd

from .cochlea import GammatoneFilterbank
from .grfnn import GrFNN, GrFNNParams, channel_to_oscillator_weights
from .osc_out import OSCBroadcaster
from .perceptual import (
    StateWindow,
    extract_chord,
    extract_consonance,
    extract_key,
    extract_rhythm_structure,
)
from .voices import (
    VoiceClusteringConfig,
    VoiceState,
    extract_voice_motor,
    extract_voice_rhythms,
    extract_voices,
)


def _grfnn_params(section: dict) -> GrFNNParams:
    return GrFNNParams(
        alpha=section["alpha"],
        beta1=section["beta1"],
        beta2=section["beta2"],
        delta1=section["delta1"],
        delta2=section["delta2"],
        epsilon=section["epsilon"],
        input_gain=section["input_gain"],
    )


def _build_grfnn(section: dict) -> GrFNN:
    hb = section.get("hebbian", {})
    dl = section.get("delay", {})
    nz = section.get("noise", {})
    return GrFNN(
        n_oscillators=section["n_oscillators"],
        low_hz=section["low_hz"],
        high_hz=section["high_hz"],
        dt=section["dt"],
        params=_grfnn_params(section),
        hebbian=bool(hb.get("enabled", False)),
        learn_rate=float(hb.get("learn_rate", 0.0)),
        weight_decay=float(hb.get("weight_decay", 0.0)),
        delay_tau=float(dl.get("tau", 0.0)),
        delay_gain=float(dl.get("gain", 0.0)),
        noise_amp=float(nz.get("amp", 0.0)),
        noise_seed=int(nz.get("seed", 0)),
    )


class LiveEngine:
    """Streaming engine: one processing step per audio chunk.

    Held as a class so the processing loop can carry state (rolling
    pitch buffer for voice extraction, previous rhythm peak for
    tempo persistence, etc.) across chunks.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        fs = int(cfg["audio"]["sample_rate"])
        self.fs = fs
        self.fb = GammatoneFilterbank(
            n_channels=cfg["cochlea"]["n_channels"],
            low_hz=cfg["cochlea"]["low_hz"],
            high_hz=cfg["cochlea"]["high_hz"],
            fs=fs,
        )
        self.pitch = _build_grfnn(cfg["pitch_grfnn"])
        self.rhythm = _build_grfnn(cfg["rhythm_grfnn"])
        m_cfg = cfg.get("motor_grfnn", {})
        self.motor = (_build_grfnn(m_cfg)
                      if m_cfg.get("enabled", False) else None)
        self.motor_forward = float(m_cfg.get("forward_gain", 0.0))
        self.motor_backward = float(m_cfg.get("backward_gain", 0.0))
        self.W_pitch = channel_to_oscillator_weights(self.fb.fc, self.pitch.f)
        osc_cfg = cfg["osc"]
        endpoints = [
            (e["host"], int(e["port"]))
            for e in osc_cfg.get("endpoints", [])
        ]
        self.osc = OSCBroadcaster(
            host=osc_cfg.get("host"), port=osc_cfg.get("port"),
            enabled=osc_cfg["enabled"],
            endpoints=endpoints,
        )
        self.snap_hz = int(cfg["state_log"]["snapshot_hz"])
        self.snap_interval_samples = fs // self.snap_hz
        self.phantom_cfg = cfg["phantom"]

        # Rhythm clock: pitch runs at fs, rhythm at 1/rhythm_dt.
        self.rhythm_dt = cfg["rhythm_grfnn"]["dt"]
        self.rhythm_step_samples = int(fs * self.rhythm_dt)

        # Rolling state for per-chunk processing.
        self._sample_count = 0
        self._next_snap = 0
        self._prev_peak_idx: int | None = None
        self._rhythm_drive_residual = 0.0  # carried rhythm drive across chunks
        # Running-peak normalizer for rhythm drive. Offline mode
        # normalizes by the whole-track max (one-shot); live can't
        # see the future, so we use a compressor-style running peak
        # that grows instantly on a new loud onset and decays slowly.
        # Decay per chunk: (1 - 1/HALF_LIFE_CHUNKS)^N. With 31
        # chunks/sec at 16 kHz / 512-block, half_life=150 chunks ≈ 5s.
        self._rhythm_peak = 0.1  # initial; grows as audio arrives
        self._rhythm_peak_decay = 0.995

        # Voice extraction: rolling 2.5 s pitch buffer (at snapshot rate)
        voice_buf_len = max(int(self.snap_hz * 2.5), 16)
        self._pitch_snap_buf: list[np.ndarray] = []
        self._motor_snap_buf: list[np.ndarray] = []
        self._voice_buf_len = voice_buf_len
        self._voice_state = VoiceState()
        self._voice_cfg = VoiceClusteringConfig()
        self._voice_stride = max(1, self.snap_hz // 20)
        self._snap_count = 0

    def process(self, audio_chunk: np.ndarray) -> None:
        """Advance the engine through one chunk of audio. Emits OSC
        + voice updates at snapshot boundaries within the chunk."""
        audio_chunk = audio_chunk.astype(np.float32)

        # 1. Cochlea: full chunk → envelope + band signals
        env = self.fb.envelope(audio_chunk)          # (n_channels, n_samples)
        pitch_bands = self.fb.filter(audio_chunk)    # (n_channels, n_samples)

        # 2. Pitch drives: per-sample complex input to the pitch GrFNN.
        pitch_drives = (self.W_pitch @ pitch_bands.astype(np.float64)).T
        pitch_drives = pitch_drives.astype(np.complex128)

        # 3. Rhythm drive: onset signal = half-wave-rectified derivative
        #    of summed envelopes.
        rhythm_drive = env.sum(axis=0)
        # Prepend the residual from the previous chunk so the diff is
        # continuous across chunk boundaries.
        prev = np.array([self._rhythm_drive_residual], dtype=np.float32)
        rhythm_drive = np.diff(np.concatenate([prev, rhythm_drive]))
        rhythm_drive = np.maximum(rhythm_drive, 0.0)
        # Running-peak normalization instead of per-chunk. Without
        # this, quiet chunks get amplified and pump noise into the
        # rhythm GrFNN; the engine then locks onto random oscillators.
        # Compressor-style: fast attack on new peaks, slow release.
        chunk_peak = float(rhythm_drive.max())
        if chunk_peak > self._rhythm_peak:
            self._rhythm_peak = chunk_peak
        else:
            self._rhythm_peak = self._rhythm_peak * self._rhythm_peak_decay
        rhythm_drive = rhythm_drive / (self._rhythm_peak + 1e-9)
        self._rhythm_drive_residual = float(env.sum(axis=0)[-1])

        # 4. Snapshot-aware stepping. The pitch bank advances at fs;
        #    we step_many across contiguous spans between snapshots,
        #    then emit OSC at each snapshot boundary.
        sample = 0
        n_samples = pitch_drives.shape[0]
        while sample < n_samples:
            # Samples until the next snapshot tick.
            samples_until_snap = self._next_snap - self._sample_count
            chunk_end = sample + samples_until_snap
            if chunk_end > n_samples:
                chunk_end = n_samples
            span = chunk_end - sample
            if span > 0:
                self.pitch.step_many(pitch_drives[sample:chunk_end])
                # Advance rhythm in sub-steps of rhythm_step_samples.
                self._advance_rhythm(
                    rhythm_drive, sample, chunk_end,
                )
                self._sample_count += span
                sample = chunk_end
            # If we landed on a snapshot tick, emit state.
            if self._sample_count >= self._next_snap:
                self._emit_snapshot()
                self._next_snap += self.snap_interval_samples

    def _advance_rhythm(self, rhythm_drive: np.ndarray,
                         lo: int, hi: int) -> None:
        """Step the rhythm network through the portion of this chunk
        spanning audio-sample indices [lo, hi). The rhythm network
        advances on its own dt; we take samples from ``rhythm_drive``
        at its native sample rate (same as audio for simplicity)."""
        if hi <= lo:
            return
        # Rhythm steps correspond to audio samples at
        # rhythm_step_samples stride.
        rhythm_samples: list[float] = []
        # The rhythm network's internal clock measures sample indices,
        # not real time. Pick drive values at every rhythm_step_samples.
        start = lo + (self.rhythm_step_samples
                      - lo % self.rhythm_step_samples) % self.rhythm_step_samples
        for s in range(start, hi, self.rhythm_step_samples):
            rhythm_samples.append(float(rhythm_drive[s]))
        if not rhythm_samples:
            return
        arr = np.array(rhythm_samples, dtype=np.complex128)
        # Broadcast the scalar drive across all rhythm oscillators.
        drives = np.tile(arr[:, None], (1, self.rhythm.n))
        # Optional motor coupling: backward from motor to rhythm.
        if self.motor is not None and self.motor_backward != 0.0:
            drives = drives + self.motor_backward * self.motor.z
        self.rhythm.step_many(drives)
        if self.motor is not None:
            motor_drives = self.motor_forward * np.tile(
                self.rhythm.z[None, :], (len(rhythm_samples), 1),
            )
            self.motor.step_many(motor_drives)

    def _emit_snapshot(self) -> None:
        """Emit OSC messages for the current engine state."""
        rp = self.pitch.phantom_mask(
            self.phantom_cfg["amp_thresh"],
            self.phantom_cfg["drive_thresh"])
        rr = self.rhythm.phantom_mask(
            self.phantom_cfg["amp_thresh"],
            self.phantom_cfg["drive_thresh"])
        self.osc.send_layer("rhythm", self.rhythm.z, rr,
                             self.rhythm.last_input_mag,
                             self.rhythm.last_residual)
        self.osc.send_layer("pitch", self.pitch.z, rp,
                             self.pitch.last_input_mag,
                             self.pitch.last_residual)
        if self.motor is not None:
            mp = self.motor.phantom_mask(
                self.phantom_cfg["amp_thresh"],
                self.phantom_cfg["drive_thresh"])
            self.osc.send_layer("motor", self.motor.z, mp,
                                 self.motor.last_input_mag,
                                 self.motor.last_residual)
        # Perceptual rollups on instantaneous state (no windowing).
        sw_instant = StateWindow(
            pitch_z=self.pitch.z, pitch_freqs=self.pitch.f,
            rhythm_z=self.rhythm.z, rhythm_freqs=self.rhythm.f,
            frame_hz=float(self.snap_hz),
            w_pitch=self.pitch.W,
        )
        rhythm_struct = extract_rhythm_structure(
            sw_instant, prev_peak_idx=self._prev_peak_idx,
        )
        self._prev_peak_idx = rhythm_struct["peak"]["idx"]
        key = extract_key(sw_instant)
        chord = extract_chord(sw_instant)
        self.osc.send_features({
            "tempo": rhythm_struct["peak"]["bpm"],
            "tonic": key["tonic"],
            "mode": key["mode"],
            "key_conf": key["confidence"],
            "chord": chord["name"],
            "chord_quality": chord["quality"],
            "chord_conf": chord["confidence"],
            "consonance": extract_consonance(sw_instant),
        })
        self.osc.send_rhythm_structure(rhythm_struct)

        # Voice extraction: roll the pitch snapshot buffer and run
        # the extractor every ``voice_stride`` snapshots.
        self._pitch_snap_buf.append(self.pitch.z.copy())
        if len(self._pitch_snap_buf) > self._voice_buf_len:
            self._pitch_snap_buf.pop(0)
        if self.motor is not None:
            self._motor_snap_buf.append(self.motor.z.copy())
            if len(self._motor_snap_buf) > self._voice_buf_len:
                self._motor_snap_buf.pop(0)
        self._snap_count += 1
        if (self._snap_count % self._voice_stride == 0
                and len(self._pitch_snap_buf) >= 8):
            pitch_hist = np.stack(self._pitch_snap_buf)
            motor_hist = None
            motor_freqs = None
            if (self.motor is not None
                    and len(self._motor_snap_buf) >= 8):
                motor_hist = np.stack(self._motor_snap_buf)
                motor_freqs = self.motor.f
            sw_voice = StateWindow(
                pitch_z=pitch_hist, pitch_freqs=self.pitch.f,
                rhythm_z=self.rhythm.z, rhythm_freqs=self.rhythm.f,
                frame_hz=float(self.snap_hz),
                w_pitch=self.pitch.W,
                motor_z=motor_hist, motor_freqs=motor_freqs,
            )
            self._voice_state = extract_voices(
                sw_voice, prev_state=self._voice_state,
                config=self._voice_cfg,
            )
            self._voice_state = extract_voice_rhythms(
                sw_voice, self._voice_state,
            )
            if self.motor is not None:
                self._voice_state = extract_voice_motor(
                    sw_voice, self._voice_state,
                )
            self.osc.send_voices(self._voice_state)


def run(config_path: Path, device: str | None = None,
        block_size: int = 512) -> None:
    """Open the input stream, run the engine until Ctrl-C.

    When the device's native sample rate doesn't match the engine's
    target rate (16 kHz by default), opens the stream at the native
    rate and resamples each chunk via ``scipy.signal.resample_poly``
    before handing it to the engine. Leaving sounddevice / CoreAudio
    to do the conversion gives noticeably worse quality — aliasing
    artifacts and pitch drift in the cochlear bank — so we pay the
    small CPU cost of explicit resampling.
    """
    from scipy.signal import resample_poly
    from math import gcd
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    engine = LiveEngine(cfg)
    fs_engine = engine.fs

    # Resolve the device's native sample rate so we can match it
    # instead of asking sounddevice to silently convert.
    devices = sd.query_devices()
    if device is None:
        device_info = sd.query_devices(kind="input")
    else:
        # Accept device name or index
        try:
            device_info = sd.query_devices(device, kind="input")
        except Exception:
            device_info = sd.query_devices(device)
    native_rate = int(device_info["default_samplerate"])
    device_name = device_info["name"]

    # Resampling factors. If rates already match, resample_poly is
    # a no-op pass-through (up=down=1), which is cheap enough not
    # to branch on.
    r_gcd = gcd(fs_engine, native_rate)
    up = fs_engine // r_gcd
    down = native_rate // r_gcd
    resampling = up != 1 or down != 1

    print(f"[nd-live] input device: {device_name!r} @ {native_rate} Hz")
    print(f"[nd-live] engine rate:  {fs_engine} Hz")
    if resampling:
        print(f"[nd-live] resampling:   up={up}, down={down} "
              f"(scipy resample_poly)")
    print(f"[nd-live] block size:   {block_size} samples "
          f"({block_size / native_rate * 1000:.1f} ms at device rate)")

    # Bounded queue. If processing falls behind, chunks drop rather
    # than blocking the audio callback.
    q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=20)

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[nd-live] {status}")
        # Copy first so the stream's internal buffer can advance.
        chunk = indata[:, 0].copy()
        try:
            q.put_nowait(chunk)
        except queue.Full:
            pass  # drop chunk

    stop = threading.Event()

    def worker() -> None:
        while not stop.is_set():
            try:
                chunk = q.get(timeout=0.5)
            except queue.Empty:
                continue
            if resampling:
                # resample_poly applies an anti-aliasing FIR
                # automatically — high-quality downsampling for
                # 48→16 kHz or whatever ratio the device lands at.
                chunk = resample_poly(chunk, up, down).astype(np.float32)
            try:
                engine.process(chunk)
            except Exception as e:
                print(f"[nd-live] processing error: {e!r}")

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    with sd.InputStream(
        samplerate=native_rate, channels=1, dtype="float32",
        blocksize=block_size, device=device,
        callback=audio_callback,
    ):
        print(f"[nd-live] running — Ctrl-C to stop")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n[nd-live] stopping")
    stop.set()
    t.join(timeout=2.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("config.toml"))
    ap.add_argument("--device", type=str, default=None,
                    help="sounddevice input device name")
    ap.add_argument("--block-size", type=int, default=512,
                    help="audio callback block size (samples)")
    args = ap.parse_args()
    run(args.config, device=args.device, block_size=args.block_size)


if __name__ == "__main__":
    main()
