"""Live VCV Rack roundtrip via two virtual audio devices.

Instead of the offline file-recorder workflow (Audio File → Recorder
→ rendered WAV), this treats VCV Rack as a live synth in the
processing chain:

    Python CV streamer  →  [Loopback CV]  →  VCV Audio in
                                              ↓
                                            VCV patch
                                              ↓
                            VCV Audio out  →  [Loopback Audio]  →  capture here

Both loopbacks are virtual CoreAudio devices — BlackHole (free,
https://existential.audio/blackhole/) works well. Install two
instances (or one 16ch and pick a channel pair for each role) and
name them something like ``BlackHole-CV`` and ``BlackHole-Audio``.

This is the architecture the real product uses too — the router's
CV backend currently talks to the ES-9 via sounddevice, but you can
repoint it at a loopback device and drop VCV Rack into the chain as
a software stand-in for hardware modular.

Usage:
    # One-off: render one pattern via live VCV
    uv run python -m tests.vcv_live_roundtrip --pattern single \\
        --cv-device "BlackHole-CV" --audio-device "BlackHole-Audio"

    # Default devices (read from config if present)
    uv run python -m tests.vcv_live_roundtrip --pattern all
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import sounddevice as sd

from tests.trigger_roundtrip import (
    ALL_PATTERNS, CV_FS, ENGINE_DIR, OUT_DIR,
    compare, extract_voice_timeline, run_engine,
    write_cv_wav,
)

CAPTURE_FS = 48000    # record audio at VCV's native rate; nd-run resamples


def _resolve_device(name_or_idx: str | int | None, kind: str) -> int:
    """Resolve a device name substring to a device index. ``kind`` is
    'input' or 'output'. Raises if nothing matches."""
    if name_or_idx is None:
        if kind == "input":
            return sd.default.device[0]
        return sd.default.device[1]
    if isinstance(name_or_idx, int) or str(name_or_idx).isdigit():
        return int(name_or_idx)
    substr = str(name_or_idx).lower()
    for i, dev in enumerate(sd.query_devices()):
        if substr in dev["name"].lower():
            has_cap = (dev["max_input_channels"] > 0 if kind == "input"
                        else dev["max_output_channels"] > 0)
            if has_cap:
                return i
    available = [d["name"] for d in sd.query_devices()
                 if (d["max_input_channels"] if kind == "input"
                     else d["max_output_channels"]) > 0]
    raise RuntimeError(
        f"No {kind} device matching '{name_or_idx}'. Available:\n  "
        + "\n  ".join(available)
    )


def live_render(
    pattern_name: str,
    cv_device: str | None,
    audio_device: str | None,
    lead_s: float = 0.5,
    trail_s: float = 0.5,
) -> Path:
    """Stream the pattern's CV WAV to ``cv_device`` while recording
    audio from ``audio_device``. Writes the captured audio to
    ``test_audio/triggers/<name>.vcv.wav`` and returns its path.

    ``lead_s`` buffers capture start before CV playback so envelope
    attacks at t=0 aren't clipped; ``trail_s`` lets the ADSR release
    tail finish after the last gate-off.
    """
    if pattern_name not in ALL_PATTERNS:
        raise ValueError(f"unknown pattern: {pattern_name}")
    pattern = ALL_PATTERNS[pattern_name]()

    # Re-emit CV WAV in case patterns changed since last setup.
    cv_path = write_cv_wav(pattern, OUT_DIR / "cv")
    cv, cv_sr = sf.read(str(cv_path), dtype="float32", always_2d=True)
    assert cv_sr == CV_FS, f"CV sample rate mismatch: {cv_sr}"
    assert cv.shape[1] == 2, "CV WAV must be stereo (L=gate, R=pitch)"

    cv_dev_idx = _resolve_device(cv_device, "output")
    audio_dev_idx = _resolve_device(audio_device, "input")
    cv_dev = sd.query_devices(cv_dev_idx)
    audio_dev = sd.query_devices(audio_dev_idx)
    print(f"  CV out → [{cv_dev_idx}] {cv_dev['name']} "
          f"({cv_dev['max_output_channels']}ch, {cv_dev['default_samplerate']:.0f} Hz)")
    print(f"  Audio in ← [{audio_dev_idx}] {audio_dev['name']} "
          f"({audio_dev['max_input_channels']}ch, {audio_dev['default_samplerate']:.0f} Hz)")

    total_s = lead_s + pattern.duration_s + trail_s
    n_capture = int(total_s * CAPTURE_FS)
    captured = np.zeros((n_capture, 1), dtype=np.float32)
    capture_cursor = [0]
    capture_started = threading.Event()

    def capture_cb(indata, frames, time_info, status):
        if status:
            print(f"    [capture] {status}")
        c = capture_cursor[0]
        remaining = n_capture - c
        if remaining <= 0:
            raise sd.CallbackStop
        # Sum to mono if multi-channel.
        mono = indata.mean(axis=1, keepdims=True).astype(np.float32)
        take = min(remaining, frames)
        captured[c:c + take, :] = mono[:take]
        capture_cursor[0] += take
        capture_started.set()

    cv_stream = sd.OutputStream(
        device=cv_dev_idx, samplerate=CV_FS, channels=2,
        dtype="float32",
    )
    audio_stream = sd.InputStream(
        device=audio_dev_idx, samplerate=CAPTURE_FS, channels=1,
        dtype="float32", callback=capture_cb,
    )

    audio_stream.start()
    capture_started.wait(timeout=2.0)   # wait for first capture block
    cv_stream.start()

    # Silent lead-in on CV while capture primes.
    cv_stream.write(np.zeros((int(lead_s * CV_FS), 2), dtype=np.float32))
    # Play the CV pattern.
    cv_stream.write(cv)
    # Silent tail so ADSR release finishes; capture keeps filling.
    cv_stream.write(np.zeros((int(trail_s * CV_FS), 2), dtype=np.float32))
    cv_stream.stop()
    cv_stream.close()

    # Wait for capture to fill the rest of the buffer.
    import time
    t0 = time.monotonic()
    while capture_cursor[0] < n_capture and time.monotonic() - t0 < total_s + 2.0:
        time.sleep(0.05)
    audio_stream.stop()
    audio_stream.close()

    out_path = OUT_DIR / f"{pattern.name}.vcv.wav"
    sf.write(out_path, captured[:capture_cursor[0]], CAPTURE_FS,
             subtype="FLOAT")
    peak = float(np.max(np.abs(captured[:capture_cursor[0]])))
    print(f"  captured {capture_cursor[0] / CAPTURE_FS:.2f}s, "
          f"peak {peak:.3f} → {out_path.name}")
    if peak < 0.001:
        print("  [warning] captured audio is near-silent — VCV patch "
              "may not be producing output, or loopback routing is wrong")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", type=str, default="single",
                    help=f"one of {', '.join(ALL_PATTERNS)} or 'all'")
    ap.add_argument("--cv-device", type=str, default="Loopback CV",
                    help="substring of the output device name that feeds "
                         "VCV's CV input (default: 'Loopback CV')")
    ap.add_argument("--audio-device", type=str, default="Loopback Audio",
                    help="substring of the input device name that receives "
                         "VCV's audio output (default: 'Loopback Audio')")
    ap.add_argument("--list-devices", action="store_true",
                    help="Print all CoreAudio devices and exit")
    ap.add_argument("--config", type=Path,
                    default=ENGINE_DIR / "config.toml")
    args = ap.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    names = ([args.pattern] if args.pattern != "all"
             else list(ALL_PATTERNS.keys()))
    for name in names:
        print(f"\n=== {name} ===")
        wav = live_render(name, args.cv_device, args.audio_device)
        parquet = ENGINE_DIR / "output" / f"trigger_vcv_{name}.parquet"
        run_engine(wav, parquet, args.config)
        timeline = extract_voice_timeline(parquet)
        pattern = ALL_PATTERNS[name]()
        report = compare(pattern, timeline)
        print(f"  coverage:       {report['coverage_pct']:5.1f}% "
              f"({report['triggers_matched']}/{report['triggers_total']})")
        print(f"  phantom voices: {report['phantom_pct']:5.1f}% "
              f"({report['phantom_voice_frames']}/{report['total_voice_frames']})")
        cents = [r["cents_off"] for r in report["per_trigger"]
                 if r.get("cents_off") is not None]
        if cents:
            print(f"  pitch error:    median {float(np.median(cents)):.0f} cents")


if __name__ == "__main__":
    main()
