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
    extract_voice_motor,
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
    plasticity, delay coupling, and stochastic noise.

    If the section has a ``[tuning]`` subtable with ``a4_hz`` and
    ``bins_per_semitone``, the oscillator frequencies are laid out on
    a 12-TET grid (each bin on a named note) instead of the default
    log-uniform ``geomspace``. See ``tuning.py``.
    """
    hb = section.get("hebbian", {})
    dl = section.get("delay", {})
    nz = section.get("noise", {})
    tn = section.get("tuning", {})
    cp = section.get("coupling", {})
    freqs = None
    if tn:
        from .tuning import twelve_tet_freqs
        freqs = twelve_tet_freqs(
            low_hz=section["low_hz"],
            high_hz=section["high_hz"],
            a4_hz=float(tn.get("a4_hz", 440.0)),
            bins_per_semitone=int(tn.get("bins_per_semitone", 3)),
        )
    coupling_kernel = None
    coupling_gain = 0.0
    if cp.get("enabled", False):
        from .grfnn import build_integer_ratio_coupling
        if freqs is None:
            from numpy import geomspace
            freqs_for_kernel = geomspace(
                section["low_hz"], section["high_hz"],
                int(section["n_oscillators"]),
            )
        else:
            freqs_for_kernel = freqs
        coupling_kernel = build_integer_ratio_coupling(
            freqs_for_kernel,
            tolerance_semitones=float(cp.get("tolerance_semitones", 0.4)),
            max_octave_distance=float(cp.get("max_octave_distance", 2.5)),
        )
        coupling_gain = float(cp.get("gain", 0.05))
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
        freqs=freqs,
        coupling_kernel=coupling_kernel,
        coupling_gain=coupling_gain,
        per_oscillator_tau=bool(section.get("per_oscillator_tau", False)),
        tau_reference_hz=float(section.get("tau_reference_hz", 1.0)),
        nonlinear_input=bool(section.get("nonlinear_input", False)),
        adaptive_frequency=bool(section.get("adaptive_frequency", False)),
        adaptive_lambda_freq=float(
            section.get("adaptive_lambda_freq", 4.0)
        ),
        adaptive_lambda_elastic=float(
            section.get("adaptive_lambda_elastic", 2.0)
        ),
        adaptive_freq_min_ratio=float(
            section.get("adaptive_freq_min_ratio", 0.5)
        ),
        adaptive_freq_max_ratio=float(
            section.get("adaptive_freq_max_ratio", 2.0)
        ),
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
        output_override: Path | None = None,
        disable_osc: bool = False) -> None:
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    if disable_osc:
        cfg.setdefault("osc", {})["enabled"] = False

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

    # Phase 3 brainstem cascade (per Lerud 2014). Optional layer
    # between gammatone and pitch. When enabled, gammatone bands
    # drive the brainstem (which generates harmonics + combination
    # tones via canonical input + integer-ratio coupling), and the
    # brainstem's state drives the pitch (cortex) layer.
    bs_cfg = cfg.get("brainstem_grfnn", {})
    brainstem_enabled = bool(bs_cfg.get("enabled", False))
    brainstem_net = _build_grfnn(bs_cfg) if brainstem_enabled else None
    if brainstem_net is not None:
        W_brainstem = channel_to_oscillator_weights(fb.fc, brainstem_net.f)
        if (brainstem_net.f.shape != pitch_net.f.shape
                or not np.allclose(brainstem_net.f, pitch_net.f)):
            raise ValueError(
                "brainstem_grfnn frequency layout must match pitch_grfnn "
                "for cascaded multi-layer mode (Lerud 2014 tonotopic "
                "preservation)"
            )
    else:
        W_brainstem = None

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
    if brainstem_net is not None:
        layers_meta["brainstem"] = {"f": brainstem_net.f}
    state = StateLog(out_path, layers=layers_meta)
    osc_cfg = cfg["osc"]
    endpoints = [
        (e["host"], int(e["port"]))
        for e in osc_cfg.get("endpoints", [])
    ]
    osc = OSCBroadcaster(
        host=osc_cfg.get("host"), port=osc_cfg.get("port"),
        enabled=osc_cfg["enabled"],
        endpoints=endpoints,
    )
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
        # Motor buffer mirrors the pitch buffer so per-voice motor
        # coupling (Phase 3) has the same 2.5 s window of motor state
        # to DFT the voice envelope against. Only accumulated when
        # the motor layer is enabled.
        "motor_buf": (deque(maxlen=voice_buf_len)
                       if motor_net is not None else None),
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

    # Time-stepped simulation, chunked by snap_interval so the
    # step_many JIT path can amortize Python overhead across the
    # ~1/snap_hz seconds between snapshots. Previously this used
    # per-sample step() calls, which made the cascade pipeline run
    # ~10× slower than necessary (each Python-level step pays the
    # full per-call overhead, even with numba-compiled inner steps).
    n_audio_steps = len(audio)
    rhythm_step = 0
    next_snap = 0.0
    snap_chunk_samples = max(1, int(round(snap_interval * fs)))
    rhythm_step_samples = int(round(rhythm_dt * fs))
    print("Running networks…")
    log_every = max(1, n_audio_steps // 20)
    # Pre-compute brainstem drives if cascade is enabled.
    if brainstem_net is not None:
        bs_drives_all = (W_brainstem @ pitch_bands.astype(np.float64)
                         ).T.astype(np.complex128)  # (n_samples, n_brainstem)
    else:
        pitch_drives_all = (W_pitch @ pitch_bands.astype(np.float64)
                            ).T.astype(np.complex128)  # (n_samples, n_pitch)
    cursor = 0
    while cursor < n_audio_steps:
        chunk_end = min(cursor + snap_chunk_samples, n_audio_steps)
        span = chunk_end - cursor
        t = (chunk_end - 1) / fs  # snapshot moment is end of chunk

        # Pitch / cascade
        if brainstem_net is not None:
            bs_traj = brainstem_net.step_many_record(
                bs_drives_all[cursor:chunk_end].copy()
            )
            pitch_net.step_many(bs_traj)
        else:
            pitch_net.step_many(pitch_drives_all[cursor:chunk_end].copy())

        # Rhythm + motor — chunked at rhythm_dt rate, may span multiple
        # rhythm steps within this snap-chunk.
        # Collect rhythm drive samples for this chunk
        rhythm_samples_for_chunk: list[float] = []
        target_rhythm_step = int((t + rhythm_dt) / rhythm_dt)
        while rhythm_step < min(target_rhythm_step, n_rhythm_steps):
            rhythm_samples_for_chunk.append(
                float(rhythm_drive_stepped[rhythm_step])
            )
            rhythm_step += 1
        if rhythm_samples_for_chunk:
            r_arr = np.array(rhythm_samples_for_chunk, dtype=np.complex128)
            drives_r = np.tile(r_arr[:, None], (1, rhythm_net.n))
            if motor_net is not None and backward_gain != 0.0:
                drives_r = drives_r + backward_gain * motor_net.z
            rhythm_net.step_many(drives_r)
            if motor_net is not None:
                motor_drives = (forward_gain
                                * np.tile(rhythm_net.z[None, :],
                                          (len(r_arr), 1)))
                motor_net.step_many(motor_drives)

        cursor = chunk_end

        # Snapshot at end of every chunk (chunk size = snap_interval)
        if True:
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
            if brainstem_net is not None:
                bp = brainstem_net.phantom_mask(
                    phantom_cfg["amp_thresh"], phantom_cfg["drive_thresh"])
                state.snapshot(t, "brainstem", brainstem_net.z.copy(), bp,
                               brainstem_net.last_input_mag,
                               brainstem_net.last_residual)
                osc.send_layer("brainstem", brainstem_net.z, bp,
                               brainstem_net.last_input_mag,
                               brainstem_net.last_residual)
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
            if osc_state["motor_buf"] is not None and motor_net is not None:
                osc_state["motor_buf"].append(motor_net.z.copy())
            osc_state["snap_count"] += 1
            if (osc_state["snap_count"] % osc_state["voice_stride"] == 0
                    and len(osc_state["pitch_buf"]) >= 8):
                pitch_hist = np.stack(list(osc_state["pitch_buf"]))
                motor_hist = None
                motor_freqs = None
                if (osc_state["motor_buf"] is not None
                        and motor_net is not None
                        and len(osc_state["motor_buf"]) >= 8):
                    motor_hist = np.stack(list(osc_state["motor_buf"]))
                    motor_freqs = motor_net.f
                voice_sw = StateWindow(
                    pitch_z=pitch_hist,
                    pitch_freqs=pitch_net.f,
                    rhythm_z=rhythm_net.z,
                    rhythm_freqs=rhythm_net.f,
                    frame_hz=float(snap_hz),
                    w_pitch=pitch_net.W,
                    motor_z=motor_hist,
                    motor_freqs=motor_freqs,
                )
                voice_state = extract_voices(
                    voice_sw,
                    prev_state=osc_state["voice_state"],
                    config=voice_cfg,
                )
                # Phase 2 — per-voice rhythm association.
                voice_state = extract_voice_rhythms(voice_sw, voice_state)
                # Phase 3 — per-voice motor coupling. No-op when the
                # motor layer is disabled.
                voice_state = extract_voice_motor(voice_sw, voice_state)
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

        if cursor % (log_every * snap_chunk_samples // max(snap_chunk_samples, 1)) < snap_chunk_samples:
            pct = 100.0 * cursor / n_audio_steps
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
    ap.add_argument("--no-osc", action="store_true",
                    help="Disable OSC broadcasting (batch mode). Saves ~30%% "
                         "wall time on cascade-enabled pipelines by skipping "
                         "pythonosc serialization. No effect on parquet output.")
    args = ap.parse_args()
    run(args.config, audio_override=args.audio,
        output_override=args.output, disable_osc=args.no_osc)


if __name__ == "__main__":
    main()
