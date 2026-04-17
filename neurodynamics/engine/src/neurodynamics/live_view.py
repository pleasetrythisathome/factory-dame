"""Live viewer — subscribes to ``nd-live``'s OSC stream and animates
the engine state from a rolling ring buffer. Matches the offline
viewer's diagnostic surface (heatmaps, scopes, coherence strips,
phantom+residual) with a streaming data source.

Shares helpers with ``viewer.py`` so the aesthetics, rendering math
and scope behavior stay identical — only the data source differs
(OSC → ring buffer vs. parquet → preloaded arrays).
"""

from __future__ import annotations

import argparse
import threading
import time
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
    WATERFALL_N,
    _CMAP_COHERENCE, _CMAP_PITCH, _CMAP_RHYTHM,
    _INK, _INK_SOFT, _PAPER, _PHANTOM_INK, _PITCH_INK,
    _RESIDUAL_INK, _RHYTHM_INK, _VOICE_PALETTE,
    _composite, _make_scope_trace, _rolling_coherence,
    _update_scope_trace,
)

WINDOW_S = 12.0


class LiveStateBuffer:
    """Thread-safe rolling buffer. OSC handlers push, animation reads.

    Carries per-layer amp, phase, phantom, drive, residual at snapshot
    granularity. Ring buffer depth is 1.5× the displayed window so the
    animation never starves if the render thread lags a frame.
    """

    def __init__(self, n_pitch: int, n_rhythm: int, n_motor: int,
                 snap_hz: int, window_s: float = WINDOW_S):
        self.snap_hz = snap_hz
        self.window_s = window_s
        self.buf_depth = max(16, int(snap_hz * window_s * 1.5))
        self.n_pitch = n_pitch
        self.n_rhythm = n_rhythm
        self.n_motor = n_motor
        # Per-layer arrays, all (buf_depth, n_osc) at float32.
        def _buf(n): return np.zeros((self.buf_depth, n), dtype=np.float32)
        self.pamp = _buf(n_pitch)
        self.pph = _buf(n_pitch)
        self.pphan = _buf(n_pitch)
        self.pdrv = _buf(n_pitch)
        self.pres = _buf(n_pitch)
        self.ramp = _buf(n_rhythm)
        self.rph = _buf(n_rhythm)
        self.rphan = _buf(n_rhythm)
        self.rdrv = _buf(n_rhythm)
        self.rres = _buf(n_rhythm)
        if n_motor:
            self.mamp = _buf(n_motor)
            self.mph = _buf(n_motor)
        else:
            self.mamp = None
            self.mph = None
        self._write_idx = 0
        self._samples_written = 0
        self._lock = threading.Lock()
        # Per-layer scratch: receive per-field OSC messages out of
        # order, flush when /pitch/residual lands (last message in
        # the engine's layer broadcast).
        self._scratch: dict = {
            "pitch": {}, "rhythm": {}, "motor": {},
        }
        self.features: dict = {}
        self.voices: list[dict] = []
        # Per-voice rolling state for the VOICES panel. Each entry
        # tracks the voice's recent amp + center_freq history so the
        # panel can draw a sparkline over the same ~12s window as the
        # heatmaps. Voices that stop being updated go stale and fall
        # out of the panel.
        self._voice_hist_depth = int(snap_hz * window_s)
        self.voice_state: dict[int, dict] = {}  # id → latest fields
        self.voice_amp_hist: dict[int, np.ndarray] = {}   # id → (depth,)
        self.voice_last_update: dict[int, int] = {}       # id → sample idx
        self._voice_snap_idx = 0

    def _push_frame(self, idx: int, layer: str, fields: dict) -> None:
        """Commit a fully-accumulated layer snapshot into the ring."""
        def _set(arr, key, n):
            if arr is None:
                return
            v = fields.get(key)
            if v is None:
                return
            vv = np.asarray(v, dtype=np.float32)
            if len(vv) < n:
                vv = np.pad(vv, (0, n - len(vv)))
            arr[idx] = vv[:n]
        if layer == "pitch":
            _set(self.pamp, "amp", self.n_pitch)
            _set(self.pph, "phase", self.n_pitch)
            _set(self.pphan, "phantom", self.n_pitch)
            _set(self.pdrv, "drive", self.n_pitch)
            _set(self.pres, "residual", self.n_pitch)
        elif layer == "rhythm":
            _set(self.ramp, "amp", self.n_rhythm)
            _set(self.rph, "phase", self.n_rhythm)
            _set(self.rphan, "phantom", self.n_rhythm)
            _set(self.rdrv, "drive", self.n_rhythm)
            _set(self.rres, "residual", self.n_rhythm)
        elif layer == "motor":
            if self.mamp is not None:
                _set(self.mamp, "amp", self.n_motor)
                _set(self.mph, "phase", self.n_motor)

    def push_field(self, layer: str, field: str, values) -> None:
        """OSC handler entry point. Accumulates until a layer's
        ``residual`` field arrives (engine emits it last per snapshot),
        then commits the whole layer to the ring."""
        with self._lock:
            self._scratch[layer][field] = values
            if field == "residual":
                idx = self._write_idx
                self._push_frame(idx, layer, self._scratch[layer])
                self._scratch[layer] = {}
                # Advance after the pitch layer since that's the last
                # fully-populated layer per snapshot (engine emits
                # pitch last).
                if layer == "pitch":
                    self._write_idx = (self._write_idx + 1) % self.buf_depth
                    self._samples_written += 1

    def update_voice_field(self, voice_id: int, field: str, value) -> None:
        """Record a /voice/<id>/<field> OSC message. Voice amp drives
        the per-voice sparkline panel; center_freq labels each row;
        active toggles fade when a voice goes silent.

        Each active voice generates amp messages at ~20 Hz while
        alive, so a 240-sample depth at that rate holds ~12 s of
        history — matching the heatmap window."""
        with self._lock:
            self._voice_snap_idx += 1
            self.voice_last_update[voice_id] = self._voice_snap_idx
            entry = self.voice_state.setdefault(
                voice_id,
                {"amp": 0.0, "center_freq": 0.0, "active": 1},
            )
            entry[field] = value
            if field == "amp":
                hist = self.voice_amp_hist.get(voice_id)
                if hist is None:
                    hist = np.zeros(self._voice_hist_depth,
                                    dtype=np.float32)
                    self.voice_amp_hist[voice_id] = hist
                hist[:-1] = hist[1:]
                hist[-1] = float(value)

    def prune_stale_voices(self, stale_after: int = 300) -> None:
        """Drop voices that haven't had an update in ``stale_after``
        buffer advances (≈ 5 s at 60 Hz). Called from the animation
        so the panel doesn't clog with retired voices."""
        with self._lock:
            cutoff = self._voice_snap_idx - stale_after
            stale = [
                vid for vid, last in self.voice_last_update.items()
                if last < cutoff
            ]
            for vid in stale:
                self.voice_state.pop(vid, None)
                self.voice_amp_hist.pop(vid, None)
                self.voice_last_update.pop(vid, None)

    def active_voice_rows(self, limit: int) -> list[tuple[int, dict, np.ndarray]]:
        """Return up to ``limit`` currently-tracked voices as
        (voice_id, state_dict, amp_history) tuples, sorted by
        center_freq ascending so low→high reads naturally."""
        with self._lock:
            items = []
            for vid, state in self.voice_state.items():
                hist = self.voice_amp_hist.get(vid)
                if hist is None:
                    continue
                items.append((vid, dict(state), hist.copy()))
            items.sort(key=lambda x: x[1].get("center_freq", 0.0))
            return items[:limit]

    def latest(self, depth: int) -> dict:
        """Return a dict of (depth, n_osc) tails for every buffer.
        Oldest slot first, newest last."""
        with self._lock:
            d = min(depth, self._samples_written, self.buf_depth)
            if d == 0:
                return {}
            start = (self._write_idx - d) % self.buf_depth

            def _take(arr):
                if arr is None:
                    return None
                if start + d <= self.buf_depth:
                    return arr[start:start + d].copy()
                first = self.buf_depth - start
                return np.concatenate([arr[start:], arr[:d - first]])

            return {
                "pamp": _take(self.pamp),
                "pph": _take(self.pph),
                "pphan": _take(self.pphan),
                "pdrv": _take(self.pdrv),
                "pres": _take(self.pres),
                "ramp": _take(self.ramp),
                "rph": _take(self.rph),
                "rphan": _take(self.rphan),
                "rdrv": _take(self.rdrv),
                "rres": _take(self.rres),
                "mamp": _take(self.mamp),
                "mph": _take(self.mph),
            }


def _build_server(buffer: LiveStateBuffer, host: str, port: int) -> ThreadingOSCUDPServer:
    dispatcher = Dispatcher()

    def handler(layer: str, field: str):
        def _h(_addr, *args):
            buffer.push_field(layer, field, args)
        return _h

    for layer in ("pitch", "rhythm", "motor"):
        for field in ("amp", "phase", "phantom", "drive", "residual"):
            dispatcher.map(f"/{layer}/{field}", handler(layer, field))

    def _on_tempo(_addr, *args):
        if args:
            buffer.features["tempo"] = float(args[0])
    def _on_key(_addr, *args):
        if args:
            buffer.features["tonic"] = str(args[0])
    def _on_mode(_addr, *args):
        if args:
            buffer.features["mode"] = str(args[0])
    def _on_chord(_addr, *args):
        if args:
            buffer.features["chord"] = str(args[0])
    def _on_consonance(_addr, *args):
        if args:
            buffer.features["consonance"] = float(args[0])
    def _on_voice_count(_addr, *args):
        if args:
            buffer.features["voice_count"] = int(args[0])

    dispatcher.map("/features/tempo", _on_tempo)
    dispatcher.map("/features/key", _on_key)
    dispatcher.map("/features/mode", _on_mode)
    dispatcher.map("/features/chord", _on_chord)
    dispatcher.map("/features/consonance", _on_consonance)
    dispatcher.map("/voice/active_count", _on_voice_count)

    # Per-voice fields — dynamic ids, so we catch them in the default
    # handler and parse the id from the OSC address.
    def _on_default(addr, *args):
        # Expecting /voice/<id>/<field> for the fields we care about.
        if not addr.startswith("/voice/"):
            return
        parts = addr.strip("/").split("/")
        if len(parts) < 3:
            return
        try:
            voice_id = int(parts[1])
        except ValueError:
            return
        field = parts[2]
        if field in ("amp", "center_freq", "active") and args:
            buffer.update_voice_field(voice_id, field, args[0])

    dispatcher.set_default_handler(_on_default)

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

    # ── Figure layout (matches offline viewer + per-voice panel) ─
    # Per-voice sparklines get their own full-width row, sized by
    # expected voice count on real music (up to ~8 simultaneous).
    VOICE_ROWS = 8
    row_h = [1.0, 3, 3, 3, 2, 1, 3]  # banner, pitch heat, rhythm heat,
                                     # scopes, coherence, phantom, voices
    if motor_enabled:
        row_h.append(2)
    fig = plt.figure(
        figsize=(14, 16 + (2 if motor_enabled else 0)),
        constrained_layout=True, facecolor=_PAPER,
    )
    gs = GridSpec(len(row_h), 2, figure=fig, height_ratios=row_h,
                   width_ratios=[len(pf), len(rf)])
    ax_banner = fig.add_subplot(gs[0, :]); ax_banner.set_axis_off()
    ax_p = fig.add_subplot(gs[1, :])
    ax_r = fig.add_subplot(gs[2, :])
    ax_sp = fig.add_subplot(gs[3, 0])
    ax_sr = fig.add_subplot(gs[3, 1])
    ax_psp = fig.add_subplot(gs[4, 0])
    ax_prs = fig.add_subplot(gs[4, 1])
    ax_php = fig.add_subplot(gs[5, 0])
    ax_phr = fig.add_subplot(gs[5, 1])
    ax_voices = fig.add_subplot(gs[6, :])
    ax_m = fig.add_subplot(gs[7, :]) if motor_enabled else None

    # Banner
    ax_banner.text(
        0.5, 0.85, "NEURAL RESONANCE  ·  LIVE",
        transform=ax_banner.transAxes, ha="center", va="top",
        color=_INK, fontsize=12,
    )
    features_text = ax_banner.text(
        0.5, 0.25, "(waiting for nd-live …)",
        transform=ax_banner.transAxes, ha="center", va="center",
        color=_INK, fontsize=14, animated=True,
    )

    # Pitch heatmap
    init_p = np.zeros((len(pf), window_depth), dtype=np.float32)
    im_p = ax_p.imshow(
        init_p, aspect="auto", origin="lower",
        extent=(-WINDOW_S, 0.0, 0, len(pf)),
        cmap=_CMAP_PITCH, norm=PowerNorm(gamma=0.35, vmin=0, vmax=0.2),
        interpolation="nearest",
    )
    ax_p.set_title(f"PITCH  ·  {len(pf)} oscillators  ·  "
                   f"{pf[0]:.0f}–{pf[-1]:.0f} Hz")
    ax_p.set_ylabel("freq")
    p_ticks = np.linspace(0, len(pf) - 1, 6).astype(int)
    ax_p.set_yticks(p_ticks)
    ax_p.set_yticklabels([f"{pf[i]:.0f}" for i in p_ticks])
    ax_p.axvline(0, color=_INK, linewidth=2.0)

    # Rhythm heatmap
    init_r = np.zeros((len(rf), window_depth), dtype=np.float32)
    im_r = ax_r.imshow(
        init_r, aspect="auto", origin="lower",
        extent=(-WINDOW_S, 0.0, 0, len(rf)),
        cmap=_CMAP_RHYTHM, norm=PowerNorm(gamma=0.6, vmin=0, vmax=0.8),
        interpolation="nearest",
    )
    ax_r.set_title(f"RHYTHM  ·  {len(rf)} oscillators  ·  "
                    f"{rf[0]:.2f}–{rf[-1]:.2f} Hz")
    ax_r.set_ylabel("freq")
    r_ticks = np.linspace(0, len(rf) - 1, 6).astype(int)
    ax_r.set_yticks(r_ticks)
    ax_r.set_yticklabels([f"{rf[i]:.2f}" for i in r_ticks])
    ax_r.set_xlabel("seconds ago  ·  live")
    ax_r.axvline(0, color=_INK, linewidth=2.0)

    # Scope traces — oscilloscope-style amplitude profiles
    scope_p = _make_scope_trace(
        ax_sp, n_osc=len(pf), n_history=WATERFALL_N,
        max_amp=0.2, ink_color=_PITCH_INK,
    )
    scope_r = _make_scope_trace(
        ax_sr, n_osc=len(rf), n_history=WATERFALL_N,
        max_amp=0.8, ink_color=_RHYTHM_INK,
    )
    ax_sp.set_title(f"PITCH PROFILE  ·  last {WATERFALL_N / snap_hz:.2f}s")
    ax_sr.set_title(f"RHYTHM PROFILE  ·  last {WATERFALL_N / snap_hz:.2f}s")

    # Coherence strips
    init_psp = np.zeros((WATERFALL_N, len(pf)), dtype=np.float32)
    init_prs = np.zeros((WATERFALL_N, len(rf)), dtype=np.float32)
    im_psp = ax_psp.imshow(
        init_psp.T, aspect="auto", origin="lower",
        cmap=_CMAP_COHERENCE,
        norm=Normalize(vmin=0, vmax=1),
        interpolation="nearest",
    )
    ax_psp.set_title(f"PITCH COHERENCE  ·  phase stability over "
                      f"{WATERFALL_N / snap_hz:.2f}s")
    ax_psp.set_yticks([])
    ax_psp.set_xticks([])
    im_prs = ax_prs.imshow(
        init_prs.T, aspect="auto", origin="lower",
        cmap=_CMAP_COHERENCE,
        norm=Normalize(vmin=0, vmax=1),
        interpolation="nearest",
    )
    ax_prs.set_title(f"RHYTHM COHERENCE  ·  phase stability over "
                      f"{WATERFALL_N / snap_hz:.2f}s")
    ax_prs.set_yticks([])
    ax_prs.set_xticks([])

    # Phantom + residual composite
    init_php = np.zeros((WATERFALL_N, len(pf), 3), dtype=np.float32)
    init_phr = np.zeros((WATERFALL_N, len(rf), 3), dtype=np.float32)
    im_php = ax_php.imshow(
        np.swapaxes(init_php, 0, 1),
        aspect="auto", origin="lower", interpolation="nearest",
    )
    ax_php.set_title("PITCH  ·  rose = phantom (inner voice)  ·  red flash = surprise")
    ax_php.set_yticks([])
    ax_php.set_xticks([])
    im_phr = ax_phr.imshow(
        np.swapaxes(init_phr, 0, 1),
        aspect="auto", origin="lower", interpolation="nearest",
    )
    ax_phr.set_title("RHYTHM  ·  same encoding")
    ax_phr.set_yticks([])
    ax_phr.set_xticks([])

    # Motor heatmap
    im_m = None
    if ax_m is not None:
        init_m = np.zeros((len(mf), window_depth), dtype=np.float32)
        im_m = ax_m.imshow(
            init_m, aspect="auto", origin="lower",
            extent=(-WINDOW_S, 0.0, 0, len(mf)),
            cmap=_CMAP_RHYTHM, norm=Normalize(vmin=0.0, vmax=0.5),
            interpolation="nearest",
        )
        ax_m.set_title(f"MOTOR  ·  {len(mf)} oscillators  ·  "
                       f"{mf[0]:.2f}–{mf[-1]:.2f} Hz")
        ax_m.set_ylabel("freq")
        ax_m.axvline(0, color=_INK, linewidth=2.0)

    # Voice ticks on the pitch heatmap
    voice_lines_p = LineCollection([], colors=[], linewidths=4.5,
                                    alpha=0.95, zorder=6, capstyle="butt")
    ax_p.add_collection(voice_lines_p)

    # ── Per-voice sparklines panel ────────────────────────────────
    # Each tracked voice gets one stacked row showing its amplitude
    # envelope over the 12 s window, colored to match its voice-id
    # palette entry. Empty rows stay blank for inactive slots.
    ax_voices.set_xlim(-WINDOW_S, 0.0)
    ax_voices.set_ylim(-0.5, VOICE_ROWS - 0.5)
    ax_voices.invert_yaxis()
    ax_voices.set_yticks([])
    ax_voices.set_xlabel("seconds ago")
    ax_voices.set_title(f"VOICES  ·  per-voice amplitude over "
                         f"{int(WINDOW_S)}s  ·  up to {VOICE_ROWS} "
                         f"shown, sorted low→high freq")
    ax_voices.grid(axis="x", color="#E4DBC6", linewidth=0.5)
    voice_lines_v: list = []
    voice_fills_v: list = []
    voice_labels_v: list = []
    for i in range(VOICE_ROWS):
        (ln,) = ax_voices.plot(
            [], [], color=_INK_SOFT, linewidth=1.4,
            solid_joinstyle="round", animated=True,
        )
        fill = ax_voices.fill_between(
            [0, 1], [i, i], [i, i],
            color=_INK_SOFT, alpha=0.0, linewidth=0, animated=True,
        )
        # Thin guide line separating rows.
        ax_voices.axhline(i + 0.5, color="#E4DBC6",
                           linewidth=0.4, zorder=1)
        label = ax_voices.text(
            -WINDOW_S + 0.1, i - 0.1, "",
            ha="left", va="center", color=_INK_SOFT, fontsize=8,
            animated=True,
        )
        voice_lines_v.append(ln)
        voice_fills_v.append(fill)
        voice_labels_v.append(label)

    # Residual vmax for composite — kept small so short bursts flash.
    pres_vmax = 0.1
    rres_vmax = 0.1

    def update(_frame):
        snap = buffer.latest(window_depth)
        if not snap:
            features_text.set_text("(waiting for nd-live …)")
            artists = [features_text, im_p, im_r,
                       scope_p["current"], *scope_p["history"], scope_p["fill"],
                       scope_r["current"], *scope_r["history"], scope_r["fill"],
                       im_psp, im_prs, im_php, im_phr, voice_lines_p]
            if im_m is not None:
                artists.append(im_m)
            return tuple(artists)

        # Pad left with zeros if we haven't filled the window yet.
        def _pad(arr, depth):
            if arr is None:
                return None
            if arr.shape[0] < depth:
                pad = depth - arr.shape[0]
                return np.concatenate([np.zeros((pad, arr.shape[1]),
                                                dtype=arr.dtype), arr])
            return arr

        pa = _pad(snap["pamp"], window_depth)
        ra = _pad(snap["ramp"], window_depth)
        ma = _pad(snap["mamp"], window_depth) if snap.get("mamp") is not None else None

        im_p.set_data(pa.T)
        im_p.set_clim(vmin=0, vmax=max(np.percentile(pa, 80), 0.05))
        im_r.set_data(ra.T)
        im_r.set_clim(vmin=0, vmax=max(np.percentile(ra, 90), 0.05))
        if im_m is not None and ma is not None:
            im_m.set_data(ma.T)
            m_vmin = np.percentile(ma, 5)
            m_vmax = max(np.percentile(ma, 99), m_vmin + 1e-3)
            im_m.set_clim(vmin=m_vmin, vmax=m_vmax)

        # Scopes — last WATERFALL_N samples.
        depth = min(WATERFALL_N, snap["pamp"].shape[0])
        if depth > 0:
            p_tail = np.zeros((WATERFALL_N, len(pf)), dtype=np.float32)
            r_tail = np.zeros((WATERFALL_N, len(rf)), dtype=np.float32)
            p_tail[-depth:] = snap["pamp"][-depth:]
            r_tail[-depth:] = snap["ramp"][-depth:]
            _update_scope_trace(scope_p, p_tail)
            _update_scope_trace(scope_r, r_tail)
            # Refresh max-amp limits so strong sections aren't clipped.
            scope_p["ax"].set_ylim(0, max(np.max(p_tail) * 1.2, 0.05))
            scope_r["ax"].set_ylim(0, max(np.max(r_tail) * 1.2, 0.1))

        # Coherence strips — over the scope window.
        if depth >= 4:
            pcoh = _rolling_coherence(
                snap["pph"][-depth:].astype(np.float32), window=min(12, depth),
            )
            rcoh = _rolling_coherence(
                snap["rph"][-depth:].astype(np.float32), window=min(12, depth),
            )
            pcoh_padded = np.zeros((WATERFALL_N, len(pf)), dtype=np.float32)
            rcoh_padded = np.zeros((WATERFALL_N, len(rf)), dtype=np.float32)
            pcoh_padded[-depth:] = pcoh
            rcoh_padded[-depth:] = rcoh
            im_psp.set_data(pcoh_padded.T)
            im_prs.set_data(rcoh_padded.T)

        # Phantom + residual composite.
        if depth > 0:
            pph_tail = np.zeros((WATERFALL_N, len(pf)), dtype=np.float32)
            pres_tail = np.zeros((WATERFALL_N, len(pf)), dtype=np.float32)
            rph_tail = np.zeros((WATERFALL_N, len(rf)), dtype=np.float32)
            rres_tail = np.zeros((WATERFALL_N, len(rf)), dtype=np.float32)
            pph_tail[-depth:] = snap["pphan"][-depth:]
            pres_tail[-depth:] = snap["pres"][-depth:]
            rph_tail[-depth:] = snap["rphan"][-depth:]
            rres_tail[-depth:] = snap["rres"][-depth:]
            comp_p = _composite(pph_tail, pres_tail, pres_vmax)
            comp_r = _composite(rph_tail, rres_tail, rres_vmax)
            im_php.set_data(np.swapaxes(comp_p, 0, 1))
            im_phr.set_data(np.swapaxes(comp_r, 0, 1))

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

        # Per-voice sparklines. Stale voices (no recent /voice/<id>
        # updates) are dropped so the panel doesn't fossilize.
        buffer.prune_stale_voices(stale_after=300)
        rows = buffer.active_voice_rows(VOICE_ROWS)
        # Compute a shared amp scale so spikes don't visually dominate.
        max_amp = 0.0
        for _, _, hist in rows:
            if len(hist):
                m = float(hist.max())
                if m > max_amp:
                    max_amp = m
        max_amp = max(max_amp, 0.05)
        x_axis = np.linspace(-WINDOW_S, 0.0, buffer._voice_hist_depth)
        for slot in range(VOICE_ROWS):
            ln = voice_lines_v[slot]
            fill = voice_fills_v[slot]
            label = voice_labels_v[slot]
            # Remove previous fill polygon and clear line/label.
            try:
                fill.remove()
            except Exception:
                pass
            if slot < len(rows):
                vid, state, hist = rows[slot]
                color = _VOICE_PALETTE[vid % len(_VOICE_PALETTE)]
                # Map amp [0, max_amp] → vertical offset inside row.
                normalized = np.clip(hist / max_amp, 0.0, 1.0) * 0.42
                y = slot + 0.3 - normalized
                ln.set_data(x_axis, y)
                ln.set_color(color)
                new_fill = ax_voices.fill_between(
                    x_axis, slot + 0.3, y, color=color, alpha=0.18,
                    linewidth=0, animated=True,
                )
                voice_fills_v[slot] = new_fill
                freq = state.get("center_freq", 0.0)
                label.set_text(f"V{vid} · {freq:.0f} Hz")
                label.set_color(color)
            else:
                ln.set_data([], [])
                new_fill = ax_voices.fill_between(
                    [0, 1], [slot, slot], [slot, slot],
                    color=_INK_SOFT, alpha=0.0, linewidth=0,
                    animated=True,
                )
                voice_fills_v[slot] = new_fill
                label.set_text("")

        artists = [features_text, im_p, im_r,
                   scope_p["current"], *scope_p["history"], scope_p["fill"],
                   scope_r["current"], *scope_r["history"], scope_r["fill"],
                   im_psp, im_prs, im_php, im_phr, voice_lines_p,
                   *voice_lines_v, *voice_fills_v, *voice_labels_v]
        if im_m is not None:
            artists.append(im_m)
        return tuple(artists)

    # Scrollable Tk canvas so the tall panel stack fits any window.
    root = tk.Tk()
    root.title("nd-view-live")
    root.geometry("1200x900")
    root.configure(bg=_PAPER)

    outer = tk.Frame(root, bg=_PAPER)
    outer.pack(fill="both", expand=True)
    scroll_canvas = tk.Canvas(outer, borderwidth=0, highlightthickness=0,
                               bg=_PAPER)
    vscroll = tk.Scrollbar(outer, orient="vertical",
                            command=scroll_canvas.yview)
    scroll_canvas.configure(yscrollcommand=vscroll.set)
    vscroll.pack(side="right", fill="y")
    scroll_canvas.pack(side="left", fill="both", expand=True)
    inner = tk.Frame(scroll_canvas, bg=_PAPER)
    inner_id = scroll_canvas.create_window((0, 0), window=inner, anchor="nw")
    fig_canvas = FigureCanvasTkAgg(fig, master=inner)
    fig_canvas.draw()
    fig_canvas.get_tk_widget().pack(fill="both", expand=True)

    def on_inner_configure(_event):
        scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

    def on_scroll_configure(event):
        scroll_canvas.itemconfigure(inner_id, width=event.width)

    inner.bind("<Configure>", on_inner_configure)
    scroll_canvas.bind("<Configure>", on_scroll_configure)

    def on_mousewheel(event):
        if abs(event.delta) >= 120:
            step = -int(event.delta / 120)
        else:
            step = -int(event.delta) if event.delta else 0
        scroll_canvas.yview_scroll(step, "units")

    scroll_canvas.bind_all("<MouseWheel>", on_mousewheel)
    scroll_canvas.bind_all("<Button-4>",
                            lambda _e: scroll_canvas.yview_scroll(-3, "units"))
    scroll_canvas.bind_all("<Button-5>",
                            lambda _e: scroll_canvas.yview_scroll(3, "units"))

    # 60 fps target (interval=16 ms). Blit keeps per-frame cost small
    # — only animated artists repaint each tick — and an M-series
    # machine has headroom to spare. Drop toward 33 ms if the render
    # loop starts chewing CPU.
    anim = FuncAnimation(fig, update, interval=16, blit=True,
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
                    help="OSC port (overrides the first configured endpoint)")
    args = ap.parse_args()
    run_live(args.config, osc_port=args.port)


if __name__ == "__main__":
    main()
