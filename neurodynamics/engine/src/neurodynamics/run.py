"""Main pipeline: audio file → cochlea → dual GrFNN → state log + OSC.

Usage:
    uv run nd-run --config config.toml
"""

from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import tomllib
from collections import deque
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .cochlea import GammatoneFilterbank
from .grfnn import GrFNN, GrFNNParams, channel_to_oscillator_weights
from .osc_out import OSCBroadcaster
from .perceptual import (
    StateWindow,
    extract_chord,
    extract_consonance,
    extract_key,
    extract_rhythm_structure,
    extract_tempo,
)
from .state_log import StateLog
from .voices import (
    VoiceClusteringConfig,
    VoiceState,
    extract_voice_rhythms,
    extract_voices,
)

# Formats libsndfile handles directly. Everything else (mp3, m4a, opus, webm…)
# is routed through ffmpeg.
_NATIVE_EXTS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".au", ".raw",
                ".w64", ".caf"}


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
    """Construct a GrFNN from a config section, including optional Hebbian
    plasticity, delay coupling, and stochastic noise."""
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


def _decode_via_ffmpeg(path: Path, target_sr: int) -> np.ndarray:
    """Pipe any format ffmpeg can read → float32 mono PCM @ target_sr."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"ffmpeg not found but required for {path.suffix} files. "
            "Install with: brew install ffmpeg"
        )
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error",
         "-i", str(path),
         "-f", "f32le", "-acodec", "pcm_f32le",
         "-ac", "1", "-ar", str(target_sr),
         "pipe:1"],
        capture_output=True, check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def load_audio(path: Path, target_sr: int) -> np.ndarray:
    """Load audio as mono float32 at target_sr. Handles wav/flac/aiff natively,
    mp3/m4a/opus/etc. via ffmpeg pipe."""
    if path.suffix.lower() in _NATIVE_EXTS:
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            from math import gcd
            g = gcd(sr, target_sr)
            audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
        return audio
    # ffmpeg handles decode + resample + mono downmix in one pass.
    return _decode_via_ffmpeg(path, target_sr)


def state_path_for(cfg: dict, cfg_dir: Path, audio_path: Path) -> Path:
    """Derive the per-audio-file state log path, e.g. output/<stem>.parquet."""
    out_dir = cfg_dir / cfg["state_log"].get("output_dir", "output")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{audio_path.stem}.parquet"


def run(config_path: Path, audio_override: Path | None = None,
        output_override: Path | None = None) -> None:
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    cfg_dir = config_path.parent
    if audio_override is not None:
        audio_path = audio_override
    else:
        audio_path = cfg_dir / cfg["audio"]["input_file"]
    fs = cfg["audio"]["sample_rate"]

    print(f"Loading {audio_path} at {fs} Hz")
    audio = load_audio(audio_path, fs)
    duration = len(audio) / fs
    print(f"  {duration:.2f} s, {len(audio)} samples")

    # Cochlear front-end.
    print("Building cochlear filterbank…")
    fb = GammatoneFilterbank(
        n_channels=cfg["cochlea"]["n_channels"],
        low_hz=cfg["cochlea"]["low_hz"],
        high_hz=cfg["cochlea"]["high_hz"],
        fs=fs,
    )
    env = fb.envelope(audio)  # (n_channels, n_samples), float32

    # Rhythm drive: single broadband onset signal (sum of envelopes, then
    # differentiated and half-wave rectified to emphasize onsets).
    rhythm_drive = env.sum(axis=0)
    rhythm_drive = np.diff(rhythm_drive, prepend=rhythm_drive[0])
    rhythm_drive = np.maximum(rhythm_drive, 0.0)
    # Normalize so input_gain in config has predictable effect.
    m = rhythm_drive.max() + 1e-9
    rhythm_drive = rhythm_drive / m

    # Rhythm GrFNN runs at its own (larger) dt. Precompute drive resampled
    # to the rhythm-network step rate.
    r_cfg = cfg["rhythm_grfnn"]
    rhythm_dt = r_cfg["dt"]
    rhythm_fs = 1.0 / rhythm_dt
    n_rhythm_steps = int(duration * rhythm_fs)
    idx = (np.arange(n_rhythm_steps) * (fs * rhythm_dt)).astype(np.int64)
    idx = np.clip(idx, 0, len(rhythm_drive) - 1)
    rhythm_drive_stepped = rhythm_drive[idx]

    rhythm_net = _build_grfnn(r_cfg)

    # Optional motor-cortex layer — a second rhythm-scale GrFNN with
    # bidirectional coupling to the sensory rhythm network. Enables the
    # full "felt pulse" / missing-pulse effect from NRT Fig. 4.
    m_cfg = cfg.get("motor_grfnn", {})
    motor_enabled = bool(m_cfg.get("enabled", False))
    motor_net = _build_grfnn(m_cfg) if motor_enabled else None
    forward_gain = float(m_cfg.get("forward_gain", 0.0))
    backward_gain = float(m_cfg.get("backward_gain", 0.0))

    # Pitch GrFNN runs at audio rate. Each oscillator is driven by nearby
    # cochlear channels weighted by frequency proximity.
    p_cfg = cfg["pitch_grfnn"]
    pitch_net = _build_grfnn(p_cfg)
    # Band-limited signals, not envelopes, for pitch — oscillators need
    # oscillatory drive, not slow amplitude contours.
    pitch_bands = fb.filter(audio)  # (n_channels, n_samples)
    W_pitch = channel_to_oscillator_weights(fb.fc, pitch_net.f)

    # Output sinks. Per-file output so repeated runs don't stomp.
    snap_hz = cfg["state_log"]["snapshot_hz"]
    snap_interval = 1.0 / snap_hz
    if output_override is not None:
        out_path = output_override
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = state_path_for(cfg, cfg_dir, audio_path)
    layers_meta: dict = {"rhythm": {"f": rhythm_net.f},
                          "pitch": {"f": pitch_net.f}}
    if motor_net is not None:
        layers_meta["motor"] = {"f": motor_net.f}
    state = StateLog(out_path, layers=layers_meta)
    osc_cfg = cfg["osc"]
    osc = OSCBroadcaster(osc_cfg["host"], osc_cfg["port"], osc_cfg["enabled"])
    # Mutable state threaded through the snapshot loop. prev_peak_idx
    # carries the persistence hint for the rhythm structure extractor
    # so the main-beat BPM doesn't flip between adjacent oscillators.
    # Voice state carries tracked voice identities across frames.
    # pitch_buf holds a rolling 2.5 s window of pitch snapshots so
    # voice extraction has amplitude-envelope history to correlate
    # over — a single-snapshot instantaneous read can't identify
    # voices.
    voice_cfg = VoiceClusteringConfig()
    voice_buf_len = max(int(snap_hz * 2.5), 16)
    osc_state: dict = {
        "prev_peak_idx": None,
        "voice_state": VoiceState(),
        "pitch_buf": deque(maxlen=voice_buf_len),
        "voice_stride": max(1, int(snap_hz / 20)),  # emit voices at ~20 Hz
        "snap_count": 0,
    }
    phantom_cfg = cfg["phantom"]

    # Optional W history. Logged only when w_snapshot_hz > 0 AND a layer
    # actually has Hebbian on (otherwise W is None and there's nothing to
    # snapshot). Each layer accumulates its own (T, n, n) tensor + times.
    w_snap_hz = float(cfg["state_log"].get("w_snapshot_hz", 0.0))
    w_snap_interval = (1.0 / w_snap_hz) if w_snap_hz > 0 else None
    w_history = {
        "rhythm": {"snaps": [], "times": []} if rhythm_net.W is not None else None,
        "pitch": {"snaps": [], "times": []} if pitch_net.W is not None else None,
    }
    next_w_snap = 0.0

    # Time-stepped simulation. We advance both networks on their own clocks
    # and snapshot at snap_hz.
    n_audio_steps = len(audio)
    rhythm_step = 0
    next_snap = 0.0
    print("Running networks…")
    log_every = max(1, n_audio_steps // 20)
    for i in range(n_audio_steps):
        t = i / fs

        # Pitch network: one step per audio sample.
        drive_pitch = W_pitch @ pitch_bands[:, i].astype(np.float64)
        # Cast to complex — real drive injects to real component.
        pitch_net.step(drive_pitch.astype(np.complex128))

        # Rhythm network: advance when its own clock catches up. If the
        # motor layer is active, we advance both in lockstep and wire up
        # bidirectional coupling inline.
        while rhythm_step * rhythm_dt <= t and rhythm_step < n_rhythm_steps:
            drive_r = np.full(
                rhythm_net.n,
                rhythm_drive_stepped[rhythm_step],
                dtype=np.complex128,
            )
            if motor_net is not None and backward_gain != 0.0:
                drive_r = drive_r + backward_gain * motor_net.z
            rhythm_net.step(drive_r)
            if motor_net is not None:
                motor_drive = forward_gain * rhythm_net.z
                motor_net.step(motor_drive)
            rhythm_step += 1

        # Snapshot.
        if t >= next_snap:
            rp = rhythm_net.phantom_mask(
                phantom_cfg["amp_thresh"], phantom_cfg["drive_thresh"])
            pp = pitch_net.phantom_mask(
                phantom_cfg["amp_thresh"], phantom_cfg["drive_thresh"])
            state.snapshot(t, "rhythm", rhythm_net.z.copy(), rp,
                           rhythm_net.last_input_mag,
                           rhythm_net.last_residual)
            state.snapshot(t, "pitch", pitch_net.z.copy(), pp,
                           pitch_net.last_input_mag,
                           pitch_net.last_residual)
            osc.send_layer("rhythm", rhythm_net.z, rp,
                           rhythm_net.last_input_mag,
                           rhythm_net.last_residual)
            osc.send_layer("pitch", pitch_net.z, pp,
                           pitch_net.last_input_mag,
                           pitch_net.last_residual)
            if motor_net is not None:
                mp = motor_net.phantom_mask(
                    phantom_cfg["amp_thresh"], phantom_cfg["drive_thresh"])
                state.snapshot(t, "motor", motor_net.z.copy(), mp,
                               motor_net.last_input_mag,
                               motor_net.last_residual)
                osc.send_layer("motor", motor_net.z, mp,
                               motor_net.last_input_mag,
                               motor_net.last_residual)
            # Perceptual rollups on the instantaneous state. Cheap.
            sw = StateWindow(
                pitch_z=pitch_net.z,
                pitch_freqs=pitch_net.f,
                rhythm_z=rhythm_net.z,
                rhythm_freqs=rhythm_net.f,
                frame_hz=snap_hz,
                w_pitch=pitch_net.W,
            )
            rhythm = extract_rhythm_structure(
                sw, prev_peak_idx=osc_state["prev_peak_idx"]
            )
            osc_state["prev_peak_idx"] = rhythm["peak"]["idx"]
            key = extract_key(sw)
            chord = extract_chord(sw)
            osc.send_features({
                "tempo": rhythm["peak"]["bpm"],
                "tonic": key["tonic"],
                "mode": key["mode"],
                "key_conf": key["confidence"],
                "chord": chord["name"],
                "chord_quality": chord["quality"],
                "chord_conf": chord["confidence"],
                "consonance": extract_consonance(sw),
            })
            osc.send_rhythm_structure(rhythm)

            # Voice extraction + broadcast. Maintains a rolling buffer
            # of pitch_z snapshots so envelope-correlation clustering
            # has history to work with. Emitted at 1/voice_stride the
            # snapshot rate — 20 Hz is plenty for modular consumers.
            osc_state["pitch_buf"].append(pitch_net.z.copy())
            osc_state["snap_count"] += 1
            if (osc_state["snap_count"] % osc_state["voice_stride"] == 0
                    and len(osc_state["pitch_buf"]) >= 8):
                pitch_hist = np.stack(list(osc_state["pitch_buf"]))
                voice_sw = StateWindow(
                    pitch_z=pitch_hist,
                    pitch_freqs=pitch_net.f,
                    rhythm_z=rhythm_net.z,
                    rhythm_freqs=rhythm_net.f,
                    frame_hz=float(snap_hz),
                    w_pitch=pitch_net.W,
                )
                voice_state = extract_voices(
                    voice_sw,
                    prev_state=osc_state["voice_state"],
                    config=voice_cfg,
                )
                # Phase 2 — per-voice rhythm association. Reads from
                # the same rolling pitch buffer + current rhythm state.
                voice_state = extract_voice_rhythms(voice_sw, voice_state)
                osc_state["voice_state"] = voice_state
                osc.send_voices(voice_state)

            next_snap += snap_interval

        # Hebbian weight snapshot — independent cadence from the state log.
        if w_snap_interval is not None and t >= next_w_snap:
            for name, net in (("rhythm", rhythm_net), ("pitch", pitch_net)):
                slot = w_history[name]
                if slot is not None:
                    slot["snaps"].append(net.W.copy())
                    slot["times"].append(t)
            next_w_snap += w_snap_interval

        if i % log_every == 0:
            pct = 100.0 * i / n_audio_steps
            print(f"  {pct:5.1f}%  t={t:6.2f}s  "
                  f"|z_r|={np.abs(rhythm_net.z).max():.3f}  "
                  f"|z_p|={np.abs(pitch_net.z).max():.3f}")

    print("Flushing state log…")
    state.flush()
    print(f"Wrote {state.path}")

    # End-of-run W snapshot so the final history entry matches rhythm_W /
    # pitch_W exactly. Without this the test for "history[-1] == final W"
    # races against where the snapshot cadence happened to land.
    if w_snap_interval is not None:
        for name, net in (("rhythm", rhythm_net), ("pitch", pitch_net)):
            slot = w_history[name]
            if slot is not None and slot["snaps"]:
                slot["snaps"].append(net.W.copy())
                slot["times"].append(duration)

    # Persist any learned Hebbian weights alongside the state log. If
    # w_snapshot_hz was set, also persist the W history tensor + times so
    # the viewer can animate the learning evolution.
    learned: dict = {}
    if rhythm_net.W is not None:
        learned["rhythm_W"] = rhythm_net.W
        learned["rhythm_f"] = rhythm_net.f
        slot = w_history["rhythm"]
        if slot is not None and slot["snaps"]:
            learned["rhythm_W_history"] = np.stack(slot["snaps"])
            learned["rhythm_W_times"] = np.array(slot["times"], dtype=np.float64)
    if pitch_net.W is not None:
        learned["pitch_W"] = pitch_net.W
        learned["pitch_f"] = pitch_net.f
        slot = w_history["pitch"]
        if slot is not None and slot["snaps"]:
            learned["pitch_W_history"] = np.stack(slot["snaps"])
            learned["pitch_W_times"] = np.array(slot["times"], dtype=np.float64)
    if learned:
        weights_path = out_path.with_suffix(".weights.npz")
        np.savez(weights_path, **learned)
        print(f"Wrote {weights_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("config.toml"))
    ap.add_argument("--audio", type=Path, default=None,
                    help="Audio file to process (overrides config)")
    ap.add_argument("--output", type=Path, default=None,
                    help="State output path (default: output/<audio_stem>.parquet)")
    args = ap.parse_args()
    run(args.config, audio_override=args.audio, output_override=args.output)


if __name__ == "__main__":
    main()
