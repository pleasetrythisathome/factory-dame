"""Diagnostic: compare the offline (nd-run) and live (LiveEngine.process)
engine paths on the same audio file. If both produce the same per-snapshot
state, any live/offline visual drift Scarlet sees is plumbing (Loopback
latency, resampling, viewer refresh interval) rather than an engine bug.

Usage:
    uv run python -m tests.compare_offline_vs_live --audio PATH

Writes:
    /tmp/compare_offline.csv   — features per snapshot, offline path
    /tmp/compare_live.csv      — features per snapshot, live path
    /tmp/compare_summary.txt   — diff summary
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from neurodynamics.cochlea import GammatoneFilterbank
from neurodynamics.grfnn import GrFNN, GrFNNParams, channel_to_oscillator_weights
from neurodynamics.live import LiveEngine
from neurodynamics.perceptual import (
    StateWindow,
    extract_chord,
    extract_consonance,
    extract_key,
    extract_rhythm_structure,
)
from neurodynamics.run import load_audio


def _grfnn_params(section: dict) -> GrFNNParams:
    return GrFNNParams(
        alpha=section["alpha"], beta1=section["beta1"], beta2=section["beta2"],
        delta1=section["delta1"], delta2=section["delta2"],
        epsilon=section["epsilon"], input_gain=section["input_gain"],
    )


def run_offline_capture(cfg: dict, audio: np.ndarray, fs: int) -> list[dict]:
    """Replicate nd-run's snapshot path but capture per-snapshot features
    into memory instead of writing parquet + OSC."""
    fb = GammatoneFilterbank(
        n_channels=cfg["cochlea"]["n_channels"],
        low_hz=cfg["cochlea"]["low_hz"],
        high_hz=cfg["cochlea"]["high_hz"],
        fs=fs,
    )
    env = fb.envelope(audio)
    pitch_bands = fb.filter(audio)
    r_cfg = cfg["rhythm_grfnn"]
    p_cfg = cfg["pitch_grfnn"]
    rhythm = GrFNN(
        n_oscillators=r_cfg["n_oscillators"],
        low_hz=r_cfg["low_hz"], high_hz=r_cfg["high_hz"],
        dt=r_cfg["dt"], params=_grfnn_params(r_cfg),
        hebbian=bool(r_cfg.get("hebbian", {}).get("enabled", False)),
        learn_rate=float(r_cfg.get("hebbian", {}).get("learn_rate", 0.0)),
        weight_decay=float(r_cfg.get("hebbian", {}).get("weight_decay", 0.0)),
        delay_tau=float(r_cfg.get("delay", {}).get("tau", 0.0)),
        delay_gain=float(r_cfg.get("delay", {}).get("gain", 0.0)),
        noise_amp=float(r_cfg.get("noise", {}).get("amp", 0.0)),
        noise_seed=int(r_cfg.get("noise", {}).get("seed", 0)),
    )
    pitch = GrFNN(
        n_oscillators=p_cfg["n_oscillators"],
        low_hz=p_cfg["low_hz"], high_hz=p_cfg["high_hz"],
        dt=p_cfg["dt"], params=_grfnn_params(p_cfg),
        hebbian=bool(p_cfg.get("hebbian", {}).get("enabled", False)),
        learn_rate=float(p_cfg.get("hebbian", {}).get("learn_rate", 0.0)),
        weight_decay=float(p_cfg.get("hebbian", {}).get("weight_decay", 0.0)),
        delay_tau=float(p_cfg.get("delay", {}).get("tau", 0.0)),
        delay_gain=float(p_cfg.get("delay", {}).get("gain", 0.0)),
        noise_amp=float(p_cfg.get("noise", {}).get("amp", 0.0)),
        noise_seed=int(p_cfg.get("noise", {}).get("seed", 0)),
    )
    W_pitch = channel_to_oscillator_weights(fb.fc, pitch.f)
    rhythm_dt = r_cfg["dt"]
    rhythm_fs = 1.0 / rhythm_dt
    n_rhythm_steps = int(len(audio) / fs * rhythm_fs)
    idx = (np.arange(n_rhythm_steps) * (fs * rhythm_dt)).astype(np.int64)
    idx = np.clip(idx, 0, env.shape[1] - 1)
    rhythm_drive = env.sum(axis=0)
    rhythm_drive = np.diff(rhythm_drive, prepend=rhythm_drive[0])
    rhythm_drive = np.maximum(rhythm_drive, 0.0)
    rhythm_drive /= rhythm_drive.max() + 1e-9
    rhythm_drive_stepped = rhythm_drive[idx]

    snap_hz = cfg["state_log"]["snapshot_hz"]
    snap_interval = 1.0 / snap_hz
    prev_peak_idx: int | None = None

    records: list[dict] = []
    rhythm_step = 0
    next_snap = 0.0
    for i in range(len(audio)):
        t = i / fs
        drive_pitch = W_pitch @ pitch_bands[:, i].astype(np.float64)
        pitch.step(drive_pitch.astype(np.complex128))
        while rhythm_step * rhythm_dt <= t and rhythm_step < n_rhythm_steps:
            drive_r = np.full(
                rhythm.n, rhythm_drive_stepped[rhythm_step],
                dtype=np.complex128,
            )
            rhythm.step(drive_r)
            rhythm_step += 1
        if t >= next_snap:
            sw = StateWindow(
                pitch_z=pitch.z, pitch_freqs=pitch.f,
                rhythm_z=rhythm.z, rhythm_freqs=rhythm.f,
                frame_hz=snap_hz, w_pitch=pitch.W,
            )
            rs = extract_rhythm_structure(sw, prev_peak_idx=prev_peak_idx)
            prev_peak_idx = rs["peak"]["idx"]
            key = extract_key(sw)
            chord = extract_chord(sw)
            records.append({
                "t": t,
                "pitch_amp_sum": float(np.abs(pitch.z).sum()),
                "rhythm_amp_sum": float(np.abs(rhythm.z).sum()),
                "tempo": float(rs["peak"]["bpm"]),
                "tonic": key["tonic"],
                "mode": key["mode"],
                "chord": chord["name"],
                "consonance": float(extract_consonance(sw)),
            })
            next_snap += snap_interval
    return records


def run_live_capture(cfg: dict, audio: np.ndarray, fs: int,
                     chunk_size: int = 512) -> list[dict]:
    """Replicate nd-live's snapshot path by driving LiveEngine.process
    chunk by chunk. Capture each snapshot via a patched emit."""
    engine = LiveEngine(cfg)
    records: list[dict] = []
    original_emit = engine._emit_snapshot

    def capturing_emit():
        # Let the engine broadcast as usual; we intercept after to
        # capture the state at the same moment.
        original_emit()
        rs_peak_bpm = 0.0
        try:
            sw = StateWindow(
                pitch_z=engine.pitch.z, pitch_freqs=engine.pitch.f,
                rhythm_z=engine.rhythm.z, rhythm_freqs=engine.rhythm.f,
                frame_hz=engine.snap_hz, w_pitch=engine.pitch.W,
            )
            rs = extract_rhythm_structure(sw, prev_peak_idx=engine._prev_peak_idx)
            key = extract_key(sw)
            chord = extract_chord(sw)
            rs_peak_bpm = float(rs["peak"]["bpm"])
            records.append({
                "t": engine._sample_count / fs,
                "pitch_amp_sum": float(np.abs(engine.pitch.z).sum()),
                "rhythm_amp_sum": float(np.abs(engine.rhythm.z).sum()),
                "tempo": rs_peak_bpm,
                "tonic": key["tonic"],
                "mode": key["mode"],
                "chord": chord["name"],
                "consonance": float(extract_consonance(sw)),
            })
        except Exception as e:
            print(f"[compare] capture error: {e!r}")

    engine._emit_snapshot = capturing_emit
    # Disable OSC to keep the diagnostic quiet.
    engine.osc.enabled = False

    for lo in range(0, len(audio), chunk_size):
        engine.process(audio[lo:lo + chunk_size])
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("config.toml"))
    ap.add_argument("--duration", type=float, default=15.0,
                    help="seconds of audio to analyze (default 15)")
    ap.add_argument("--chunk-size", type=int, default=512)
    args = ap.parse_args()

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    fs = int(cfg["audio"]["sample_rate"])

    print(f"[compare] loading {args.audio} @ {fs} Hz "
          f"(first {args.duration:.0f}s)")
    audio = load_audio(args.audio, fs)
    audio = audio[:int(args.duration * fs)]
    print(f"[compare] audio length: {len(audio)} samples "
          f"({len(audio) / fs:.1f} s)")

    print("[compare] running offline capture…")
    offline_records = run_offline_capture(cfg, audio, fs)
    print(f"[compare]   {len(offline_records)} snapshots")

    print("[compare] running live capture…")
    live_records = run_live_capture(cfg, audio, fs,
                                      chunk_size=args.chunk_size)
    print(f"[compare]   {len(live_records)} snapshots")

    # Write CSVs
    def write_csv(records: list[dict], path: str) -> None:
        with open(path, "w") as f:
            cols = list(records[0].keys())
            f.write(",".join(cols) + "\n")
            for r in records:
                f.write(",".join(str(r[c]) for c in cols) + "\n")

    write_csv(offline_records, "/tmp/compare_offline.csv")
    write_csv(live_records, "/tmp/compare_live.csv")

    # Compute the diff across overlapping snapshots. Offline and
    # live should produce roughly the same count; align by index.
    n = min(len(offline_records), len(live_records))
    summary_lines: list[str] = []
    summary_lines.append(f"snapshots compared: {n}")

    def stats(field: str) -> tuple[float, float, float]:
        diffs = []
        for i in range(n):
            a = offline_records[i][field]
            b = live_records[i][field]
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                diffs.append(abs(a - b))
        if not diffs:
            return 0.0, 0.0, 0.0
        return float(np.mean(diffs)), float(np.max(diffs)), float(np.median(diffs))

    def match_ratio(field: str) -> float:
        same = sum(1 for i in range(n)
                    if offline_records[i][field] == live_records[i][field])
        return same / n if n else 0.0

    summary_lines.append("")
    summary_lines.append("Numeric fields (mean | max | median absolute diff):")
    for fld in ("pitch_amp_sum", "rhythm_amp_sum",
                "tempo", "consonance"):
        m, mx, med = stats(fld)
        summary_lines.append(f"  {fld:<20}  mean={m:10.4f}  max={mx:10.4f}"
                              f"  median={med:10.4f}")

    summary_lines.append("")
    summary_lines.append("Categorical fields (fraction of snapshots that agree):")
    for fld in ("tonic", "mode", "chord"):
        r = match_ratio(fld)
        summary_lines.append(f"  {fld:<20}  agreement={r*100:.1f}%")

    # Also report the top-5 snapshots with largest tempo diff — these
    # point to specific moments where the two paths diverge.
    tempo_diffs = sorted(
        ((i, abs(offline_records[i]["tempo"] - live_records[i]["tempo"]))
         for i in range(n)),
        key=lambda x: -x[1],
    )[:5]
    summary_lines.append("")
    summary_lines.append("Top tempo disagreements (offline vs live BPM):")
    for i, diff in tempo_diffs:
        summary_lines.append(
            f"  snap {i:4d} @ t={offline_records[i]['t']:6.2f}s  "
            f"offline={offline_records[i]['tempo']:7.2f}  "
            f"live={live_records[i]['tempo']:7.2f}  "
            f"|Δ|={diff:6.2f}"
        )

    summary = "\n".join(summary_lines)
    print()
    print(summary)
    with open("/tmp/compare_summary.txt", "w") as f:
        f.write(summary)
    print()
    print("Wrote /tmp/compare_offline.csv, /tmp/compare_live.csv, "
          "/tmp/compare_summary.txt")


if __name__ == "__main__":
    main()
