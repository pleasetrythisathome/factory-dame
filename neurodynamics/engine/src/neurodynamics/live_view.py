"""Live viewer — subscribes to ``nd-live``'s OSC stream and animates
the engine state from a rolling ring buffer. Shares aesthetics with
``viewer.py`` (the offline parquet path) but runs on streaming data
instead of precomputed arrays.

Scope deliberately narrower than the offline viewer:
- Pitch / rhythm / motor heatmaps (scroll right-to-left in real time)
- Banner with key, chord, BPM, voice count
- Voice ticks on the pitch right edge

Dropped for live:
- W matrix panels (the engine doesn't broadcast W; requires either
  OSC extension or a side channel)
- Mode-lock constellation overlay (needs phase history across the
  window, computable but not in v1)
- Phase coherence / phantom+residual strips (not critical for
  sanity-check use)

The live viewer is for confirming the live engine sees what you
think it sees before routing CV to a modular — it doesn't need the
offline viewer's full diagnostic surface.
"""

from __future__ import annotations

import argparse
import threading
import tomllib
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import tkinter as tk             # noqa: E402

from matplotlib.animation import FuncAnimation                     # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg    # noqa: E402
from matplotlib.collections import LineCollection                  # noqa: E402
from matplotlib.colors import Normalize, PowerNorm                 # noqa: E402
from matplotlib.gridspec import GridSpec                           # noqa: E402
from pythonosc.dispatcher import Dispatcher                        # noqa: E402
from pythonosc.osc_server import ThreadingOSCUDPServer             # noqa: E402

from .viewer import (
    _CMAP_PITCH, _CMAP_RHYTHM,
    _GHOST_INK, _INK, _INK_SOFT, _PAPER, _RULE,
    _VOICE_PALETTE,
)

WINDOW_S = 12.0


class LiveStateBuffer:
    """Thread-safe rolling buffer of engine state — OSC handlers push
    new snapshots, animation reads the most-recent window.

    Layout: ring buffer of depth ``buf_depth`` × ``n_osc`` for each
    layer. ``write_idx`` points at the next slot to overwrite.
    Reading assembles a contiguous view of the last ``window_depth``
    samples by unwrapping the ring.
    """

    def __init__(self, n_pitch: int, n_rhythm: int, n_motor: int,
                 snap_hz: int, window_s: float = WINDOW_S):
        self.snap_hz = snap_hz
        self.window_s = window_s
        # 1.5× the displayed window so we have headroom even if the
        # animation is slightly behind.
        self.buf_depth = max(16, int(snap_hz * window_s * 1.5))
        self.n_pitch = n_pitch
        self.n_rhythm = n_rhythm
        self.n_motor = n_motor
        self.pamp = np.zeros((self.buf_depth, n_pitch), dtype=np.float32)
        self.pph = np.zeros((self.buf_depth, n_pitch), dtype=np.float32)
        self.ramp = np.zeros((self.buf_depth, n_rhythm), dtype=np.float32)
        self.rph = np.zeros((self.buf_depth, n_rhythm), dtype=np.float32)
        self.mamp = (np.zeros((self.buf_depth, n_motor), dtype=np.float32)
                     if n_motor else None)
        self.mph = (np.zeros((self.buf_depth, n_motor), dtype=np.float32)
                     if n_motor else None)
        self._write_idx = 0
        self._samples_written = 0
        self._lock = threading.Lock()

        # Latest feature dict + voices — updated on /features/*, /voice/*
        self.features: dict = {}
        self.voices: list[dict] = []

    def push_pitch(self, amp: np.ndarray, phase: np.ndarray) -> None:
        with self._lock:
            idx = self._write_idx
            # Ring buffer writes advance only once all four layer
            # messages for this tick have arrived — simpler: advance
            # on every push. Mismatch is bounded to ~1 frame.
            self.pamp[idx] = amp[:self.n_pitch]
            self.pph[idx] = phase[:self.n_pitch]

    def push_rhythm(self, amp: np.ndarray, phase: np.ndarray) -> None:
        with self._lock:
            idx = self._write_idx
            self.ramp[idx] = amp[:self.n_rhythm]
            self.rph[idx] = phase[:self.n_rhythm]

    def push_motor(self, amp: np.ndarray, phase: np.ndarray) -> None:
        if self.mamp is None:
            return
        with self._lock:
            idx = self._write_idx
            self.mamp[idx] = amp[:self.n_motor]
            self.mph[idx] = phase[:self.n_motor]

    def advance(self) -> None:
        """Advance the ring-buffer write pointer. Called after a
        complete snapshot (all layers pushed)."""
        with self._lock:
            self._write_idx = (self._write_idx + 1) % self.buf_depth
            self._samples_written += 1

    def latest_window(self, window_depth: int) -> tuple:
        """Return the last ``window_depth`` samples of each buffer as
        a contiguous array. Oldest slot first, newest last."""
        with self._lock:
            depth = min(window_depth, self._samples_written, self.buf_depth)
            if depth == 0:
                return None, None, None, None, None, None
            start = (self._write_idx - depth) % self.buf_depth
            if start + depth <= self.buf_depth:
                pa = self.pamp[start:start + depth].copy()
                pp = self.pph[start:start + depth].copy()
                ra = self.ramp[start:start + depth].copy()
                rp = self.rph[start:start + depth].copy()
                ma = (self.mamp[start:start + depth].copy()
                      if self.mamp is not None else None)
                mp = (self.mph[start:start + depth].copy()
                      if self.mph is not None else None)
            else:
                # Wraps around the ring end.
                first = self.buf_depth - start
                pa = np.concatenate([self.pamp[start:], self.pamp[:depth - first]])
                pp = np.concatenate([self.pph[start:], self.pph[:depth - first]])
                ra = np.concatenate([self.ramp[start:], self.ramp[:depth - first]])
                rp = np.concatenate([self.rph[start:], self.rph[:depth - first]])
                if self.mamp is not None:
                    ma = np.concatenate([self.mamp[start:],
                                         self.mamp[:depth - first]])
                    mp = np.concatenate([self.mph[start:],
                                         self.mph[:depth - first]])
                else:
                    ma, mp = None, None
            return pa, pp, ra, rp, ma, mp


def _build_server(buffer: LiveStateBuffer, host: str, port: int) -> ThreadingOSCUDPServer:
    """Wire an OSC server whose handlers populate the ring buffer."""
    dispatcher = Dispatcher()

    pitch_state: dict = {"amp": None, "phase": None}
    rhythm_state: dict = {"amp": None, "phase": None}
    motor_state: dict = {"amp": None, "phase": None}

    def _handle_layer(kind, field, *args):
        vals = np.array(list(args), dtype=np.float32)
        if kind == "pitch":
            pitch_state[field] = vals
            if pitch_state["amp"] is not None and pitch_state["phase"] is not None:
                buffer.push_pitch(pitch_state["amp"], pitch_state["phase"])
        elif kind == "rhythm":
            rhythm_state[field] = vals
            if rhythm_state["amp"] is not None and rhythm_state["phase"] is not None:
                buffer.push_rhythm(rhythm_state["amp"], rhythm_state["phase"])
        elif kind == "motor":
            motor_state[field] = vals
            if motor_state["amp"] is not None and motor_state["phase"] is not None:
                buffer.push_motor(motor_state["amp"], motor_state["phase"])

    def _on_pitch_amp(_addr, *args):
        _handle_layer("pitch", "amp", *args)
    def _on_pitch_phase(_addr, *args):
        _handle_layer("pitch", "phase", *args)
    def _on_rhythm_amp(_addr, *args):
        _handle_layer("rhythm", "amp", *args)
    def _on_rhythm_phase(_addr, *args):
        _handle_layer("rhythm", "phase", *args)
    def _on_motor_amp(_addr, *args):
        _handle_layer("motor", "amp", *args)
    def _on_motor_phase(_addr, *args):
        _handle_layer("motor", "phase", *args)

    def _on_features_tempo(_addr, *args):
        if args:
            buffer.features["tempo"] = float(args[0])
    def _on_features_key(_addr, *args):
        if args:
            buffer.features["tonic"] = str(args[0])
    def _on_features_mode(_addr, *args):
        if args:
            buffer.features["mode"] = str(args[0])
    def _on_features_chord(_addr, *args):
        if args:
            buffer.features["chord"] = str(args[0])
    def _on_features_consonance(_addr, *args):
        if args:
            buffer.features["consonance"] = float(args[0])

    def _on_voice_count(_addr, *args):
        if args:
            buffer.features["voice_count"] = int(args[0])

    # The /pitch/phase message arriving last (after /pitch/amp) is
    # the natural trigger to advance the write pointer.
    def _on_layer_residual(_addr, *args):
        # After all three layers have emitted their residual (last
        # message in send_layer), advance the ring buffer. We key on
        # pitch residual specifically since it's emitted last by
        # nd-live's snapshot block for the pitch layer.
        buffer.advance()

    dispatcher.map("/pitch/amp", _on_pitch_amp)
    dispatcher.map("/pitch/phase", _on_pitch_phase)
    dispatcher.map("/rhythm/amp", _on_rhythm_amp)
    dispatcher.map("/rhythm/phase", _on_rhythm_phase)
    dispatcher.map("/motor/amp", _on_motor_amp)
    dispatcher.map("/motor/phase", _on_motor_phase)
    dispatcher.map("/features/tempo", _on_features_tempo)
    dispatcher.map("/features/key", _on_features_key)
    dispatcher.map("/features/mode", _on_features_mode)
    dispatcher.map("/features/chord", _on_features_chord)
    dispatcher.map("/features/consonance", _on_features_consonance)
    dispatcher.map("/voice/active_count", _on_voice_count)
    # Simpler advance rule: advance whenever /pitch/residual fires
    # (it's the last pitch-layer message nd-live sends per snapshot).
    dispatcher.map("/pitch/residual", _on_layer_residual)

    return ThreadingOSCUDPServer((host, port), dispatcher)


def run_live(config_path: Path, osc_port: int | None = None) -> None:
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    snap_hz = int(cfg["state_log"]["snapshot_hz"])
    pf = np.geomspace(
        cfg["pitch_grfnn"]["low_hz"], cfg["pitch_grfnn"]["high_hz"],
        int(cfg["pitch_grfnn"]["n_oscillators"]),
    )
    rf = np.geomspace(
        cfg["rhythm_grfnn"]["low_hz"], cfg["rhythm_grfnn"]["high_hz"],
        int(cfg["rhythm_grfnn"]["n_oscillators"]),
    )
    m_cfg = cfg.get("motor_grfnn", {})
    motor_enabled = bool(m_cfg.get("enabled", False))
    mf = (np.geomspace(m_cfg["low_hz"], m_cfg["high_hz"],
                       int(m_cfg["n_oscillators"]))
          if motor_enabled else np.zeros(0))

    buffer = LiveStateBuffer(len(pf), len(rf), len(mf), snap_hz)

    # Default viewer port: the second endpoint in [osc.endpoints]
    # if one is configured, else fall back to the primary osc.port.
    osc_cfg = cfg.get("osc", {})
    host = osc_cfg.get("host", "127.0.0.1")
    default_port = 57121
    endpoints = osc_cfg.get("endpoints", [])
    if endpoints:
        default_port = int(endpoints[0].get("port", default_port))
        host = endpoints[0].get("host", host)
    elif "port" in osc_cfg:
        default_port = int(osc_cfg["port"])
    port = osc_port if osc_port is not None else default_port
    server = _build_server(buffer, host, port)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"[nd-view-live] listening on {host}:{port}")

    window_depth = int(snap_hz * WINDOW_S)

    fig = plt.figure(figsize=(14, 10), constrained_layout=True,
                     facecolor=_PAPER)
    rows = [1.0, 3, 3]
    if motor_enabled:
        rows.append(2)
    gs = GridSpec(len(rows), 1, figure=fig, height_ratios=rows)
    ax_banner = fig.add_subplot(gs[0])
    ax_banner.set_axis_off()
    ax_p = fig.add_subplot(gs[1])
    ax_r = fig.add_subplot(gs[2])
    ax_m = fig.add_subplot(gs[3]) if motor_enabled else None

    ax_banner.text(
        0.5, 0.85, "NEURAL RESONANCE  ·  LIVE",
        transform=ax_banner.transAxes,
        ha="center", va="top", color=_INK, fontsize=12,
    )
    features_text = ax_banner.text(
        0.5, 0.25, "(waiting for nd-live …)",
        transform=ax_banner.transAxes,
        ha="center", va="center", color=_INK, fontsize=14,
        animated=True,
    )

    init_p = np.zeros((len(pf), window_depth), dtype=np.float32)
    im_p = ax_p.imshow(
        init_p, aspect="auto", origin="lower",
        extent=(-WINDOW_S, 0.0, 0, len(pf)),
        cmap=_CMAP_PITCH, norm=PowerNorm(gamma=0.35, vmin=0, vmax=0.2),
        interpolation="nearest",
    )
    ax_p.set_title(f"PITCH  ·  {len(pf)} oscillators")
    ax_p.set_ylabel("freq")
    p_ticks = np.linspace(0, len(pf) - 1, 6).astype(int)
    ax_p.set_yticks(p_ticks)
    ax_p.set_yticklabels([f"{pf[i]:.0f}" for i in p_ticks])
    ax_p.axvline(0, color=_INK, linewidth=2.0)

    init_r = np.zeros((len(rf), window_depth), dtype=np.float32)
    im_r = ax_r.imshow(
        init_r, aspect="auto", origin="lower",
        extent=(-WINDOW_S, 0.0, 0, len(rf)),
        cmap=_CMAP_RHYTHM, norm=PowerNorm(gamma=0.6, vmin=0, vmax=0.8),
        interpolation="nearest",
    )
    ax_r.set_title(f"RHYTHM  ·  {len(rf)} oscillators  ·  {rf[0]:.2f}-{rf[-1]:.2f} Hz")
    ax_r.set_ylabel("freq")
    r_ticks = np.linspace(0, len(rf) - 1, 6).astype(int)
    ax_r.set_yticks(r_ticks)
    ax_r.set_yticklabels([f"{rf[i]:.2f}" for i in r_ticks])
    ax_r.set_xlabel("seconds ago  ·  live")
    ax_r.axvline(0, color=_INK, linewidth=2.0)

    im_m = None
    if ax_m is not None:
        init_m = np.zeros((len(mf), window_depth), dtype=np.float32)
        im_m = ax_m.imshow(
            init_m, aspect="auto", origin="lower",
            extent=(-WINDOW_S, 0.0, 0, len(mf)),
            cmap=_CMAP_RHYTHM,
            norm=Normalize(vmin=0.0, vmax=0.5),
            interpolation="nearest",
        )
        ax_m.set_title(f"MOTOR  ·  {len(mf)} oscillators")
        ax_m.set_ylabel("freq")
        m_ticks = np.linspace(0, len(mf) - 1, 6).astype(int)
        ax_m.set_yticks(m_ticks)
        ax_m.set_yticklabels([f"{mf[i]:.2f}" for i in m_ticks])
        ax_m.axvline(0, color=_INK, linewidth=2.0)

    voice_lines_p = LineCollection([], colors=[], linewidths=4.5,
                                    alpha=0.95, zorder=6, capstyle="butt")
    ax_p.add_collection(voice_lines_p)

    def update(_frame):
        pa, _pp, ra, _rp, ma, _mp = buffer.latest_window(window_depth)
        if pa is None:
            features_text.set_text("(waiting for nd-live …)")
            return (features_text, im_p, im_r, voice_lines_p) + (
                (im_m,) if im_m is not None else ())
        # Pad left with zeros if we haven't filled the window yet.
        if pa.shape[0] < window_depth:
            pad = window_depth - pa.shape[0]
            pa = np.concatenate([np.zeros((pad, pa.shape[1]), dtype=np.float32), pa])
            ra = np.concatenate([np.zeros((pad, ra.shape[1]), dtype=np.float32), ra])
            if ma is not None:
                ma = np.concatenate([np.zeros((pad, ma.shape[1]), dtype=np.float32), ma])
        # imshow expects (n_osc, time)
        im_p.set_data(pa.T)
        # Rescale pitch vmax to keep detail visible as activity shifts.
        p_vmax = max(np.percentile(pa, 80), 0.05)
        im_p.set_clim(vmin=0, vmax=p_vmax)
        im_r.set_data(ra.T)
        r_vmax = max(np.percentile(ra, 90), 0.05)
        im_r.set_clim(vmin=0, vmax=r_vmax)
        if im_m is not None and ma is not None:
            im_m.set_data(ma.T)
            m_vmin = np.percentile(ma, 5)
            m_vmax = max(np.percentile(ma, 99), m_vmin + 1e-3)
            im_m.set_clim(vmin=m_vmin, vmax=m_vmax)
        # Banner
        f = buffer.features
        tonic = f.get("tonic", "—")
        mode = f.get("mode", "").upper()
        chord = f.get("chord", "—")
        bpm = f.get("tempo", 0.0)
        conson = f.get("consonance", 0.0)
        voices = f.get("voice_count", 0)
        features_text.set_text(
            f"{tonic} {mode} · {chord} · {bpm:.0f} BPM · "
            f"CONSONANCE {conson:.2f} · VOICES {voices}"
        )
        artists = [features_text, im_p, im_r, voice_lines_p]
        if im_m is not None:
            artists.append(im_m)
        return tuple(artists)

    root = tk.Tk()
    root.title("nd-view-live")
    root.geometry("1200x800")
    root.configure(bg=_PAPER)
    fig_canvas = FigureCanvasTkAgg(fig, master=root)
    fig_canvas.draw()
    fig_canvas.get_tk_widget().pack(fill="both", expand=True)

    anim = FuncAnimation(fig, update, interval=100, blit=True,
                          cache_frame_data=False)

    def on_close():
        server.shutdown()
        server.server_close()
        root.quit()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)

    try:
        root.mainloop()
    finally:
        try:
            server.shutdown()
            server.server_close()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("config.toml"))
    ap.add_argument("--port", type=int, default=None,
                    help="OSC port (overrides config osc.port)")
    args = ap.parse_args()
    run_live(args.config, osc_port=args.port)


if __name__ == "__main__":
    main()
