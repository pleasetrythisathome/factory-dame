"""Time-synced viewer with scrolling horizontal heatmaps + live bar readout.

Usage:
    uv run nd-view --config config.toml --audio some_song.wav

Layout (top → bottom):
    pitch   heatmap — time on x-axis (±6 s around playhead), freq on y-axis
    rhythm  heatmap — same orientation
    bars    current per-oscillator |z| (pitch gray, rhythm blue, phantom red)

Playhead is the vertical white line at x=0. Data scrolls left-to-right through it.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

# TkAgg backend required for the scrollable Tk wrapper below.
# Must be set before pyplot is imported.
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt  # noqa: E402

# Ink-on-paper Feltron aesthetic with a Futura flair: cream paper, dark
# ink, restrained editorial palette, geometric sans-serif throughout.
_PAPER = "#F5F0E6"             # cream paper
_PAPER_DEEP = "#ECE5D3"        # subtle panel tint
_INK = "#1A1714"               # ink (not pure black)
_INK_SOFT = "#504A42"          # secondary text
_RULE = "#C8BFAE"              # thin rules / axes
_GRID = "#E4DBC6"              # very faint grid

_PITCH_INK = "#1F3B5E"         # pitch accent — navy
_RHYTHM_INK = "#A7441E"        # rhythm accent — terracotta / deep orange
_PHANTOM_INK = "#7A2B2B"       # phantom tint — oxidized red
_RESIDUAL_INK = "#C43C1D"      # residual flash — editorial red
_GHOST_INK = "#2D5F4C"         # ghost playhead — muted green, stands on cream

# Voice-identity palette — cycled by voice id mod len. Feltron-adjacent,
# intentionally distinct on cream. Keep density moderate so overlays
# never outshine the heatmap data beneath.
_VOICE_PALETTE = [
    "#2A3F66",  # indigo
    "#C08E42",  # ochre
    "#4A6B3A",  # forest
    "#B05F2A",  # terracotta
    "#5B3A5E",  # eggplant
    "#2E6E6A",  # teal
    "#8C4A2A",  # sienna
    "#AB6272",  # dusty rose
    "#6F5E2A",  # olive
    "#3D5878",  # slate blue
]

from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Paper-to-ink sequential cmaps per layer — narrow hue, rises from the
# cream through a deepening wash to the accent ink.
_CMAP_PITCH = LinearSegmentedColormap.from_list(
    "nd_pitch_ink",
    [_PAPER, "#D7D2C5", "#8E9FB4", "#44638A", _PITCH_INK, "#0D1C30"],
)
_CMAP_RHYTHM = LinearSegmentedColormap.from_list(
    "nd_rhythm_ink",
    [_PAPER, "#E5D3C3", "#D0A17A", "#B86B42", _RHYTHM_INK, "#4E1F0A"],
)

# Narrow cyclic-ish phase cmap (monochrome near-cream through ink and back).
_CMAP_COHERENCE = LinearSegmentedColormap.from_list(
    "nd_coh", [_PAPER, "#BFB08C", "#736851", _INK],
)

plt.rcParams.update({
    "figure.facecolor": _PAPER,
    "axes.facecolor": _PAPER,
    "axes.edgecolor": _RULE,
    "axes.labelcolor": _INK_SOFT,
    "axes.titlecolor": _INK,
    "axes.titlesize": 10,
    "axes.titleweight": "regular",
    "axes.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.color": _INK_SOFT,
    "ytick.color": _INK_SOFT,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "text.color": _INK,
    "font.family": "sans-serif",
    "font.sans-serif": ["Futura", "Avenir Next", "Avenir", "Century Gothic",
                         "ITC Avant Garde Gothic Std", "Helvetica Neue",
                         "Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
    "figure.titlesize": 12,
    "figure.titleweight": "regular",
    "savefig.facecolor": _PAPER,
    "hatch.color": _GHOST_INK,
    "hatch.linewidth": 0.6,
})
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import soundfile as sf  # noqa: E402
import tkinter as tk  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patheffects import Normal, Stroke  # noqa: E402

from .modelock import detect_mode_locks
from .perceptual import (
    StateWindow,
    extract_chord,
    extract_consonance,
    extract_key,
    extract_rhythm_structure,
    extract_tempo,
)
from .run import _NATIVE_EXTS, state_path_for
from .voices import (
    VoiceClusteringConfig,
    VoiceState,
    extract_voice_rhythms,
    extract_voices,
)

WINDOW_S = 12.0       # heatmap time window (±6 s around playhead)
WATERFALL_N = 30      # number of stacked slices in the waterfall ridges


def _parse_layer_f(metadata: dict, layer: str) -> np.ndarray:
    key = f"layer.{layer}.f".encode()
    if metadata is None or key not in metadata:
        return np.array([])
    raw = metadata[key].decode()
    return np.array([float(x) for x in raw.split(",")])


def _extract_layer(table, layer: str):
    """Return (times, amps, phases, phantoms, residuals, freqs) for one layer.

    Older state files may not have residual; we fall back to zeros.
    """
    mask = np.array(table.column("layer").to_pylist()) == layer
    t = np.array(table.column("t").to_pylist())[mask]
    amp_lists = [np.asarray(x) for x in table.column("amp").to_pylist()]
    amp = np.stack([a for a, m in zip(amp_lists, mask) if m])
    phase_lists = [np.asarray(x) for x in table.column("phase").to_pylist()]
    phase = np.stack([a for a, m in zip(phase_lists, mask) if m])
    phantom_lists = [np.asarray(x) for x in table.column("phantom").to_pylist()]
    phantom = np.stack([a for a, m in zip(phantom_lists, mask) if m])
    if "residual" in table.column_names:
        res_lists = [np.asarray(x) for x in table.column("residual").to_pylist()]
        residual = np.stack([a for a, m in zip(res_lists, mask) if m])
    else:
        residual = np.zeros_like(amp)
    freqs = _parse_layer_f(table.schema.metadata, layer)
    order = np.argsort(t)
    return (t[order], amp[order], phase[order],
            phantom[order], residual[order], freqs)


def _pad_for_window(amp: np.ndarray, half_samples: int) -> np.ndarray:
    """Zero-pad time axis so window slices around any playhead index stay
    in bounds. amp is (T, N_osc) — we pad the T axis."""
    pad = np.zeros((half_samples, amp.shape[1]), dtype=amp.dtype)
    return np.vstack([pad, amp, pad])


def _hex_to_rgb(hex_str: str) -> tuple[float, float, float]:
    s = hex_str.lstrip("#")
    return tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _composite(phantom: np.ndarray, residual: np.ndarray,
               residual_vmax: float) -> np.ndarray:
    """Phantom = oxidized-red tint on paper, positive residual = editorial-red
    flash. Both live in the ink palette: the strip reads as a sparse field of
    red marks over cream paper."""
    pos_res = np.clip(residual, 0.0, None) / max(residual_vmax, 1e-9)
    pos_res = np.clip(pos_res, 0.0, 1.0).astype(np.float32)
    phantom_f = phantom.astype(np.float32)

    bg = _hex_to_rgb(_PAPER)
    phantom_col = _hex_to_rgb(_PHANTOM_INK)
    residual_col = _hex_to_rgb(_RESIDUAL_INK)

    rgb = np.zeros((*phantom.shape, 3), dtype=np.float32)
    rgb[..., 0] = bg[0]
    rgb[..., 1] = bg[1]
    rgb[..., 2] = bg[2]

    phantom_a = phantom_f * 0.85
    for c in range(3):
        rgb[..., c] = (phantom_col[c] * phantom_a
                      + rgb[..., c] * (1.0 - phantom_a))
    a = pos_res
    for c in range(3):
        rgb[..., c] = residual_col[c] * a + rgb[..., c] * (1.0 - a)
    return rgb


def _make_scope_trace(ax, n_osc: int, n_history: int, max_amp: float,
                      ink_color: str) -> dict:
    """Oscilloscope-style amplitude profile. One solid ink line for the
    current instant, n_history faded traces behind it for recent memory."""
    x = np.arange(n_osc)
    ax.set_xlim(-0.5, n_osc - 0.5)
    ax.set_ylim(0, max_amp * 1.2)
    ax.set_yticks([])
    ax.grid(axis="x", color=_GRID, linewidth=0.5)
    history_lines = []
    for i in range(n_history):
        alpha = 0.06 + 0.20 * (i / max(n_history - 1, 1))
        (ln,) = ax.plot(x, np.zeros(n_osc), color=ink_color,
                        linewidth=0.7, alpha=alpha,
                        solid_joinstyle="round")
        history_lines.append(ln)
    (current_line,) = ax.plot(
        x, np.zeros(n_osc), color=ink_color, linewidth=1.6,
        solid_joinstyle="round",
    )
    fill = ax.fill_between(x, 0, 0, color=ink_color, alpha=0.10,
                            linewidth=0)
    return {"history": history_lines, "current": current_line,
            "fill": fill, "ax": ax, "ink": ink_color}


def _update_scope_trace(scope: dict, amp_history: np.ndarray) -> None:
    """amp_history: (n_history, n_osc) oldest-first. Last row is current."""
    n_history, n_osc = amp_history.shape
    x = np.arange(n_osc)
    for i, ln in enumerate(scope["history"]):
        ln.set_ydata(amp_history[i])
    current = amp_history[-1]
    scope["current"].set_ydata(current)
    scope["fill"].remove()
    scope["fill"] = scope["ax"].fill_between(
        x, 0, current, color=scope["ink"], alpha=0.10, linewidth=0,
    )


# Ratio → ink color for the mode-lock constellation. Restrained palette so
# different ratios are distinguishable but the overlay stays editorial.
_LOCK_COLORS = {
    (1, 1): "#3D6B47",   # green — unison
    (2, 1): "#1F3B5E",   # navy — octave / 2:1
    (3, 2): "#A7441E",   # terracotta — fifth / 3:2
    (4, 3): "#7A5C2E",   # ochre — fourth
    (5, 4): "#6B2E5E",   # plum — major third
    (7, 4): "#7A2B2B",   # oxidized red — minor seventh
}
_LOCK_RATIOS = list(_LOCK_COLORS.keys())


def _detect_locks_filtered(
    phases: np.ndarray, amps: np.ndarray, *,
    threshold: float = 0.85, amp_threshold: float = 0.05,
) -> list[dict]:
    """Vectorized mode-lock scan over the phase window, filtered to
    actively-amped oscillators. The reference module function is correct
    but Python-looped — at 100 oscillators × 6 ratios that loop blocks
    the render thread for ~250 ms per refresh. This implementation does
    one numpy reduction per ratio over all upper-triangle pairs at once."""
    mean_amp = amps.mean(axis=0)
    active = np.where(mean_amp > amp_threshold)[0]
    n_active = len(active)
    if n_active < 2:
        return []
    sub = phases[:, active].astype(np.float64)  # (T, n_active)
    iu = np.triu_indices(n_active, k=1)
    phi_a = sub[:, iu[0]]  # (T, npairs)
    phi_b = sub[:, iu[1]]
    locks = []
    for p, q in _LOCK_RATIOS:
        rel = q * phi_a - p * phi_b
        plv = np.abs(np.mean(np.exp(1j * rel), axis=0))
        for k in np.where(plv >= threshold)[0]:
            locks.append({
                "i": int(active[iu[0][k]]),
                "j": int(active[iu[1][k]]),
                "p": p, "q": q, "plv": float(plv[k]),
            })
    return locks


def _rolling_coherence(phase_history: np.ndarray,
                       window: int = 12) -> np.ndarray:
    """Per-oscillator phase coherence over a rolling window.

    phase_history: (T, n_osc). Returns (T, n_osc) coherence values in [0, 1]
    computed as |⟨exp(iφ)⟩| over the trailing `window` samples. High = phase
    is stable over that span; low = phase is tumbling. Readable as a simple
    monochrome strip ("how in-sync has this oscillator been lately").
    """
    T, n = phase_history.shape
    if window <= 1 or T == 0:
        return np.ones_like(phase_history)
    e = np.exp(1j * phase_history.astype(np.float64))
    # Cumulative mean via cumsum for O(T·n).
    cs = np.cumsum(e, axis=0)
    windowed = np.empty_like(cs)
    windowed[:window] = cs[:window] / np.arange(1, window + 1)[:, None]
    windowed[window:] = (cs[window:] - cs[:-window]) / window
    return np.abs(windowed).astype(np.float32)


def _playback_start(audio_path: Path, audio_arr: np.ndarray, sr: int):
    """Kick off audio playback and return a (stop_callable, started_at) tuple.

    On macOS uses afplay subprocess (bulletproof, handles any codec/rate with
    zero Core Audio fuss). Elsewhere falls back to sounddevice.
    """
    started = time.monotonic()
    if platform.system() == "Darwin" and shutil.which("afplay") is not None:
        # afplay decodes and plays at device rate natively — no PaMacCore
        # paramErr spam, no resampling logic, no channel-count gotchas.
        proc = subprocess.Popen(
            ["afplay", str(audio_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return proc.terminate, started
    # Non-macOS fallback: sounddevice.
    import sounddevice as sd
    def worker():
        try:
            sd.play(audio_arr, sr)
            sd.wait()
        except Exception as e:  # pragma: no cover
            print(f"audio playback failed: {e}", file=sys.stderr)
    threading.Thread(target=worker, daemon=True).start()
    return sd.stop, started


def _load_for_playback(path: Path) -> tuple[np.ndarray, int]:
    """Decode for the sounddevice fallback. On macOS we use afplay directly
    on the file so this result is unused there."""
    if path.suffix.lower() in _NATIVE_EXTS:
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        return audio, sr
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError(
            f"ffmpeg/ffprobe not found for {path.suffix}. "
            "Install with: brew install ffmpeg"
        )
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, check=True, text=True,
    )
    sr = int(probe.stdout.strip() or 44100)
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
         "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(sr),
         "pipe:1"],
        capture_output=True, check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy().reshape(-1, 2), sr


def run(config_path: Path, audio_override: Path | None = None,
        state_override: Path | None = None,
        screenshot_at: float | None = None,
        screenshot_path: Path | None = None) -> None:
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    cfg_dir = config_path.parent

    audio_path = audio_override or (cfg_dir / cfg["audio"]["input_file"])
    state_path = state_override or state_path_for(cfg, cfg_dir, audio_path)

    print(f"Loading audio {audio_path}")
    # Only decode PCM for the sounddevice fallback path; afplay reads the file
    # directly on macOS.
    if platform.system() == "Darwin" and shutil.which("afplay") is not None:
        audio_arr, sr = np.empty(0, dtype=np.float32), 0
    else:
        audio_arr, sr = _load_for_playback(audio_path)

    print(f"Loading state {state_path}")
    table = pq.read_table(state_path)
    rt, ramp, rpha, rph, rres, rf = _extract_layer(table, "rhythm")
    pt, pamp, ppha, pph, pres, pf = _extract_layer(table, "pitch")
    # Optional motor layer from the two-layer pulse network.
    layers_in_file = set(table.column("layer").to_pylist())
    has_motor = "motor" in layers_in_file
    if has_motor:
        mt, mamp, mpha, mph, mres, mf = _extract_layer(table, "motor")

    # Optional learned-weights file (alongside the parquet). If present and
    # contains a W history, the viewer will animate the matrix evolving
    # in sync with the playhead. If only a final W is saved we'll show
    # that as a static panel.
    weights_path = state_path.with_suffix(".weights.npz")
    weights_npz = np.load(weights_path) if weights_path.exists() else None
    pW_hist = (weights_npz["pitch_W_history"]
               if weights_npz is not None and "pitch_W_history" in weights_npz.files
               else None)
    pW_times = (weights_npz["pitch_W_times"]
                if weights_npz is not None and "pitch_W_times" in weights_npz.files
                else None)
    rW_hist = (weights_npz["rhythm_W_history"]
               if weights_npz is not None and "rhythm_W_history" in weights_npz.files
               else None)
    rW_times = (weights_npz["rhythm_W_times"]
                if weights_npz is not None and "rhythm_W_times" in weights_npz.files
                else None)
    pW_final = (weights_npz["pitch_W"]
                if weights_npz is not None and "pitch_W" in weights_npz.files
                else None)
    rW_final = (weights_npz["rhythm_W"]
                if weights_npz is not None and "rhythm_W" in weights_npz.files
                else None)
    has_pW = pW_hist is not None or pW_final is not None
    has_rW = rW_hist is not None or rW_final is not None

    snap_hz = int(cfg["state_log"]["snapshot_hz"])
    half_samples = int(snap_hz * WINDOW_S / 2)
    window_samples = 2 * half_samples

    # Zero-pad so window slices never go out of bounds.
    pamp_pad = _pad_for_window(pamp, half_samples)   # (T + 2*half, n_pitch)
    ramp_pad = _pad_for_window(ramp, half_samples)
    pph_pad = _pad_for_window(pph.astype(np.float32), half_samples)
    rph_pad = _pad_for_window(rph.astype(np.float32), half_samples)
    mamp_pad = (_pad_for_window(mamp, half_samples)
                if has_motor else None)

    # Figure layout — heatmaps full width on top, pitch+rhythm waterfalls
    # split the bottom row. Waterfall widths match their oscillator counts
    # roughly (pitch has 2× as many, so it gets 2× the horizontal space).
    # Tall figure intended to be viewed in a scrollable container. Adds
    # optional rows for motor (two-layer pulse) and learned weights.
    show_motor_row = has_motor
    show_w_row = has_pW or has_rW
    # First row is a dedicated banner for the title + perceptual features
    # overlay. Using its own axes (rather than fig.text or ax_p.text at
    # y>1) keeps blit well-behaved — the text updates cleanly per frame.
    base_rows = [1.0, 3, 3, 3, 2, 1]  # banner, pitch heat, rhythm heat, scopes, phase, phantom
    extra = []
    if show_motor_row:
        extra.append(("motor", 2))    # compact heatmap
    if show_w_row:
        extra.append(("w", 3))
    row_names = (["banner", "pitch", "rhythm", "scope", "phase", "phantom"]
                 + [e[0] for e in extra])
    height_ratios = base_rows + [e[1] for e in extra]
    n_rows = len(height_ratios)
    fig_h = 14 + (2 if show_motor_row else 0) + (3 if show_w_row else 0)
    fig = plt.figure(figsize=(14, fig_h), constrained_layout=True,
                     facecolor=_PAPER)
    gs = GridSpec(
        n_rows, 2, figure=fig,
        height_ratios=height_ratios,
        width_ratios=[len(pf), len(rf)],
    )
    ax_banner = fig.add_subplot(gs[0, :])
    ax_banner.set_axis_off()
    ax_p = fig.add_subplot(gs[1, :])
    ax_r = fig.add_subplot(gs[2, :])
    ax_wp = fig.add_subplot(gs[3, 0])
    ax_wr = fig.add_subplot(gs[3, 1])
    ax_psp = fig.add_subplot(gs[4, 0])   # pitch phase coherence
    ax_prs = fig.add_subplot(gs[4, 1])   # rhythm phase coherence
    ax_php = fig.add_subplot(gs[5, 0])   # pitch phantom+residual
    ax_phr = fig.add_subplot(gs[5, 1])   # rhythm phantom+residual
    # Optional rows beyond row 5.
    extra_row_idx = 6
    ax_motor = None
    if show_motor_row:
        ax_motor = fig.add_subplot(gs[extra_row_idx, :])
        extra_row_idx += 1
    ax_Wp = fig.add_subplot(gs[extra_row_idx, 0]) if show_w_row else None
    ax_Wr = fig.add_subplot(gs[extra_row_idx, 1]) if show_w_row else None
    # Title sits statically at top of the banner axis.
    ax_banner.text(
        0.5, 0.85,
        f"NEURAL RESONANCE  ·  {audio_path.name}".upper(),
        transform=ax_banner.transAxes,
        ha="center", va="top", color=_INK, fontsize=12,
    )
    # Features banner text lives inside ax_banner. Returning it in the
    # blit artist list updates reliably because the text is inside an
    # axes with proper clip bounds.
    features_text = ax_banner.text(
        0.5, 0.25, "",
        transform=ax_banner.transAxes,
        ha="center", va="center", color=_INK, fontsize=14,
        animated=True,
    )

    # Percentile-based vmax + aggressive gamma<1 (PowerNorm) to boost
    # low-amplitude detail. Pitch especially benefits from strong gamma
    # because music has dozens of weakly-active oscillators at any time.
    from matplotlib.colors import PowerNorm
    p_vmax = float(max(np.percentile(pamp, 80), 1e-3))
    r_vmax = float(max(np.percentile(ramp, 90), 1e-3))

    init_p = np.zeros((len(pf), window_samples), dtype=np.float32)
    im_p = ax_p.imshow(
        init_p, aspect="auto", origin="lower",
        extent=(-WINDOW_S / 2, WINDOW_S / 2, 0, len(pf)),
        cmap=_CMAP_PITCH, norm=PowerNorm(gamma=0.35, vmin=0, vmax=p_vmax),
        interpolation="nearest",
    )
    ax_p.set_title(f"PITCH  ·  {len(pf)} oscillators  ·  "
                   f"{pf[0]:.0f}–{pf[-1]:.0f} Hz")
    ax_p.set_ylabel("freq")
    p_ticks = np.linspace(0, len(pf) - 1, 6).astype(int)
    ax_p.set_yticks(p_ticks)
    ax_p.set_yticklabels([f"{pf[i]:.0f}" for i in p_ticks])

    init_r = np.zeros((len(rf), window_samples), dtype=np.float32)
    im_r = ax_r.imshow(
        init_r, aspect="auto", origin="lower",
        extent=(-WINDOW_S / 2, WINDOW_S / 2, 0, len(rf)),
        cmap=_CMAP_RHYTHM, norm=PowerNorm(gamma=0.6, vmin=0, vmax=r_vmax),
        interpolation="nearest",
    )
    ax_r.set_title(f"RHYTHM  ·  {len(rf)} oscillators  ·  "
                   f"{rf[0]:.2f}–{rf[-1]:.2f} Hz")
    ax_r.set_ylabel("freq")
    r_ticks = np.linspace(0, len(rf) - 1, 6).astype(int)
    ax_r.set_yticks(r_ticks)
    ax_r.set_yticklabels([f"{rf[i]:.2f}" for i in r_ticks])
    ax_r.set_xlabel(f"time offset (s)  ·  ±{WINDOW_S/2:.0f}s around playhead")

    # Motor heatmap (two-layer pulse network's motor-cortex layer).
    im_motor = None
    m_line = None
    if show_motor_row:
        # Motor oscillators on the canonical Hopf limit cycle cluster
        # tightly near their saturation amplitude (e.g. 0.45–0.53 on
        # Flights). Stretch the narrow p5→p99 band across the full cmap
        # with linear norm so the dynamics are legible.
        m_vmin = float(np.percentile(mamp, 5))
        m_vmax = float(max(np.percentile(mamp, 99), m_vmin + 1e-4))
        init_m = np.zeros((len(mf), window_samples), dtype=np.float32)
        # Reuse the rhythm cmap but shift palette position to distinguish
        # the motor layer visually from sensory rhythm.
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        cmap_motor = LinearSegmentedColormap.from_list(
            "nd_motor_ink",
            [_PAPER, "#D6C4B1", "#8E7A58", "#5C3F1C", "#2C1B08"],
        )
        im_motor = ax_motor.imshow(
            init_m, aspect="auto", origin="lower",
            extent=(-WINDOW_S / 2, WINDOW_S / 2, 0, len(mf)),
            cmap=cmap_motor,
            norm=Normalize(vmin=m_vmin, vmax=m_vmax),
            interpolation="nearest",
        )
        ax_motor.set_title(
            f"MOTOR  ·  two-layer pulse  ·  {len(mf)} oscillators  ·  "
            f"{mf[0]:.2f}–{mf[-1]:.2f} Hz"
        )
        ax_motor.set_ylabel("freq")
        m_ticks = np.linspace(0, len(mf) - 1, 6).astype(int)
        ax_motor.set_yticks(m_ticks)
        ax_motor.set_yticklabels([f"{mf[i]:.2f}" for i in m_ticks])
        # Primary playhead on motor too.
        m_line = ax_motor.axvline(0, color=_INK, linewidth=2.0)
        m_line.set_path_effects([
            Stroke(linewidth=4.5, foreground=_PAPER, alpha=0.95),
            Normal(),
        ])

    # Primary playhead: thick ink line at x=0 with paper halo so it reads
    # against either the light or the deep-ink regions of the heatmaps.
    p_line = ax_p.axvline(0, color=_INK, linewidth=2.0)
    r_line = ax_r.axvline(0, color=_INK, linewidth=2.0)
    for line in (p_line, r_line):
        line.set_path_effects([
            Stroke(linewidth=4.5, foreground=_PAPER, alpha=0.95),
            Normal(),
        ])

    # Mode-lock constellation overlay collections — one LineCollection per
    # heatmap. We update the segment list periodically (not per frame) so
    # the O(n²) PLV scan doesn't tank framerate. Legend swatches in the
    # title strip explain the ratio → ink mapping.
    from matplotlib.collections import LineCollection
    lock_lines_p = LineCollection([], colors=[], linewidths=1.3,
                                   alpha=1.0, zorder=4)
    lock_lines_r = LineCollection([], colors=[], linewidths=1.3,
                                   alpha=1.0, zorder=4)
    ax_p.add_collection(lock_lines_p)
    ax_r.add_collection(lock_lines_r)

    # Voice-identity brackets — vertical segments drawn just past the
    # right edge of the time window, spanning each active voice's
    # oscillator range. Color cycles by voice id mod palette length
    # so distinct voices are visually distinct; persistent IDs keep
    # colors stable across frames.
    # Horizontal ticks at the right edge of the pitch heatmap, one
    # per oscillator member of an active voice. Voice color cycles by
    # id so identity is visually consistent across frames.
    voice_lines_p = LineCollection([], colors=[], linewidths=4.5,
                                    alpha=0.95, zorder=6,
                                    capstyle="butt")
    ax_p.add_collection(voice_lines_p)

    # Compact legend in the figure header explaining the ratio palette.
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=color, edgecolor="none", label=f"{p}:{q}")
        for (p, q), color in _LOCK_COLORS.items()
    ]
    fig.legend(
        handles=legend_patches, loc="upper right",
        ncol=len(_LOCK_COLORS), frameon=False, fontsize=7,
        title="MODE-LOCK RATIOS", title_fontsize=7,
        bbox_to_anchor=(0.99, 0.985),
    )

    # Anticipation ghost playhead — second translucent line at x = +tau on
    # any layer where delay coupling is enabled. The network's response is
    # phase-leading the driver by ~tau seconds, so its "current state"
    # is what the audio WILL look like at +tau. Drawn dashed cyan so it's
    # unambiguously distinct from the primary playhead.
    p_delay_tau = float(cfg.get("pitch_grfnn", {})
                           .get("delay", {}).get("tau", 0.0))
    r_delay_tau = float(cfg.get("rhythm_grfnn", {})
                           .get("delay", {}).get("tau", 0.0))
    p_delay_gain = float(cfg.get("pitch_grfnn", {})
                            .get("delay", {}).get("gain", 0.0))
    r_delay_gain = float(cfg.get("rhythm_grfnn", {})
                            .get("delay", {}).get("gain", 0.0))
    # Delay coupling is enabled in the engine but not visualized in this
    # round — past-shadow / forward-projection visualizations weren't
    # carrying useful meaning. The dynamics still get the phase-leading
    # effect; we just don't draw it.
    _ = (p_delay_tau, p_delay_gain, r_delay_tau, r_delay_gain)

    # Waterfalls — stacked ridges showing last WATERFALL_S seconds of
    # per-oscillator amplitude. Newer slices sit lower (in front) and brighter;
    # older slices recede upward and fade. This gives a "3D contour" view of
    # which oscillators cluster together moment-to-moment.
    pitch_amp_max = max(float(np.percentile(pamp, 95)), 1e-3)
    rhythm_amp_max = max(float(np.percentile(ramp, 95)), 1e-3)
    scope_p = _make_scope_trace(
        ax_wp, n_osc=len(pf), n_history=WATERFALL_N,
        max_amp=pitch_amp_max, ink_color=_PITCH_INK,
    )
    scope_r = _make_scope_trace(
        ax_wr, n_osc=len(rf), n_history=WATERFALL_N,
        max_amp=rhythm_amp_max, ink_color=_RHYTHM_INK,
    )
    history_s = WATERFALL_N / snap_hz
    ax_wp.set_title(f"PITCH PROFILE  ·  current + last {history_s:.2f}s")
    ax_wr.set_title(f"RHYTHM PROFILE  ·  current + last {history_s:.2f}s")
    for ax, freqs in ((ax_wp, pf), (ax_wr, rf)):
        ticks = np.linspace(0, len(freqs) - 1, 5).astype(int)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{freqs[i]:.1f}" for i in ticks], fontsize=7)

    # Phase coherence strips — per oscillator, |⟨exp(iφ)⟩| over a rolling
    # 12-sample (~200 ms at 60 Hz) window. 1 = phase has been stable; 0 =
    # tumbling. Reads like a monochrome heatmap: dark ink = locked, pale
    # paper = drift. Precomputed for the whole run so the viewer just
    # slices the current window each frame.
    COH_WIN = 12
    pcoh = _rolling_coherence(ppha, window=COH_WIN)
    rcoh = _rolling_coherence(rpha, window=COH_WIN)
    init_psp = np.zeros((WATERFALL_N, len(pf)), dtype=np.float32)
    init_prs = np.zeros((WATERFALL_N, len(rf)), dtype=np.float32)
    im_psp = ax_psp.imshow(
        init_psp, aspect="auto", origin="upper",
        cmap=_CMAP_COHERENCE, vmin=0, vmax=1, interpolation="nearest",
    )
    im_prs = ax_prs.imshow(
        init_prs, aspect="auto", origin="upper",
        cmap=_CMAP_COHERENCE, vmin=0, vmax=1, interpolation="nearest",
    )
    ax_psp.set_title(
        f"PITCH COHERENCE  ·  phase stability over {COH_WIN/snap_hz:.2f}s"
    )
    ax_prs.set_title(f"RHYTHM COHERENCE  ·  phase stability over {COH_WIN/snap_hz:.2f}s")
    for ax, freqs in ((ax_psp, pf), (ax_prs, rf)):
        ax.set_yticks([])
        ticks = np.linspace(0, len(freqs) - 1, 5).astype(int)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{freqs[i]:.1f}" for i in ticks], fontsize=7)

    # Phantom + residual composite strips (monochrome red luminance).
    # Phantom = dim rose tint (ringing without drive).
    # Positive residual = bright red flash (onset / surprise).
    pres_vmax = float(max(np.clip(pres, 0, None).max(), 1e-3))
    rres_vmax = float(max(np.clip(rres, 0, None).max(), 1e-3))
    init_php_rgb = np.zeros((WATERFALL_N, len(pf), 3), dtype=np.float32)
    init_phr_rgb = np.zeros((WATERFALL_N, len(rf), 3), dtype=np.float32)
    im_php = ax_php.imshow(
        init_php_rgb, aspect="auto", origin="upper", interpolation="nearest",
    )
    im_phr = ax_phr.imshow(
        init_phr_rgb, aspect="auto", origin="upper", interpolation="nearest",
    )
    ax_php.set_title(
        "PITCH  ·  rose = phantom (inner voice)  ·  red flash = surprise"
    )
    ax_phr.set_title("RHYTHM  ·  same encoding")
    for ax, freqs in ((ax_php, pf), (ax_phr, rf)):
        ax.set_yticks([])
        ticks = np.linspace(0, len(freqs) - 1, 5).astype(int)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{freqs[i]:.1f}" for i in ticks], fontsize=7)

    # Optional learned-weight matrices. |W_ij| as a square heatmap with
    # frequency-labeled axes and integer-ratio guide lines so you can read
    # off octaves (2:1), fifths (3:2), etc. — the structure NRT predicts
    # the network should be learning.
    def _draw_w_panel(ax, hist, final, freqs, cmap, label):
        if hist is not None:
            data = np.abs(hist)
            initial = data[0]
            # Percentile + power-norm so subtle off-diagonal structure is
            # visible without being crushed by the diagonal-adjacent peaks.
            vmax = max(float(np.percentile(data, 98)), 1e-6)
        else:
            initial = np.abs(final)
            vmax = max(float(np.percentile(initial, 98)), 1e-6)
        from matplotlib.colors import PowerNorm
        im = ax.imshow(
            initial, cmap=cmap,
            norm=PowerNorm(gamma=0.55, vmin=0, vmax=vmax),
            interpolation="nearest", aspect="equal", origin="lower",
        )
        n = len(freqs)
        ticks = np.linspace(0, n - 1, 5).astype(int)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        # Compact labels (Hz, no unit on the cell to save space).
        labels = [f"{freqs[i]:.1f}" for i in ticks]
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("freq j (Hz)", fontsize=7)
        ax.set_ylabel("freq i (Hz)", fontsize=7)
        ax.set_title(
            f"{label}  ·  bright = strong learned connection  ·  "
            f"diagonals = octave (2:1) + fifth (3:2)"
        )

        # Integer-ratio guide lines: for each j, find the i where freqs[i]
        # is closest to freqs[j] * ratio. Draw a faint line through those
        # points so you can scan along the ratio.
        def _ratio_line(num, den, color, alpha):
            ratio = num / den
            xs, ys = [], []
            for j, fj in enumerate(freqs):
                target = fj * ratio
                if target > freqs[-1] or target < freqs[0]:
                    continue
                i = int(np.argmin(np.abs(freqs - target)))
                xs.append(j)
                ys.append(i)
            if xs:
                ax.plot(xs, ys, color=color, linewidth=0.6, alpha=alpha,
                        linestyle=(0, (3, 3)))
        # Octave (most stable) and fifth (most musically prominent).
        _ratio_line(2, 1, _INK, 0.35)
        _ratio_line(3, 2, _INK_SOFT, 0.25)
        return im

    im_Wp = im_Wr = None
    if has_pW:
        im_Wp = _draw_w_panel(ax_Wp, pW_hist, pW_final, pf,
                              _CMAP_PITCH, "PITCH W")
    if has_rW:
        im_Wr = _draw_w_panel(ax_Wr, rW_hist, rW_final, rf,
                              _CMAP_RHYTHM, "RHYTHM W")

    # Pad amplitude + phantom arrays for waterfall/phantom-strip lookups.
    # History length WATERFALL_N samples at snap_hz covers WATERFALL_N/snap_hz sec.
    pamp_for_wf = _pad_for_window(pamp, WATERFALL_N)
    ramp_for_wf = _pad_for_window(ramp, WATERFALL_N)
    pph_for_wf = _pad_for_window(pph.astype(np.float32), WATERFALL_N)
    rph_for_wf = _pad_for_window(rph.astype(np.float32), WATERFALL_N)
    pres_for_wf = _pad_for_window(pres.astype(np.float32), WATERFALL_N)
    rres_for_wf = _pad_for_window(rres.astype(np.float32), WATERFALL_N)
    pcoh_for_wf = _pad_for_window(pcoh, WATERFALL_N)
    rcoh_for_wf = _pad_for_window(rcoh, WATERFALL_N)

    def slice_window(padded: np.ndarray, now: float) -> np.ndarray:
        center = half_samples + int(round(now * snap_hz))
        lo = center - half_samples
        hi = center + half_samples
        lo = max(0, min(lo, padded.shape[0] - window_samples))
        hi = lo + window_samples
        return padded[lo:hi].T

    def slice_history(padded: np.ndarray, now: float) -> np.ndarray:
        """Last WATERFALL_N snapshots up to 'now', oldest-first."""
        # padded[WATERFALL_N] corresponds to source index 0.
        end = WATERFALL_N + int(round(now * snap_hz)) + 1
        start = end - WATERFALL_N
        start = max(0, min(start, padded.shape[0] - WATERFALL_N))
        end = start + WATERFALL_N
        return padded[start:end]

    # Mutable clock state populated once playback starts, before mainloop.
    # Deferring playback start until after the figure is verified rendering
    # means an animation setup failure won't leave afplay orphaned.
    clock = {"t0": None, "stop_fn": lambda: None}

    # Mode-lock recompute cadence: every LOCK_REFRESH frames we run a
    # PLV scan on the current phase + amplitude window and rebuild the
    # constellation overlay.
    # Mode-lock locks are PRECOMPUTED across the whole run at startup, so
    # the update loop just does a list lookup by playhead time. This avoids
    # any stalls from doing a PLV scan inside a frame budget.
    LOCK_PRECOMPUTE_HZ = 2.0           # snapshots-per-second of locks
    LOCK_PLV_THRESHOLD = 0.7           # easier to trip on real audio
    LOCK_AMP_THRESHOLD_R = max(float(np.percentile(ramp, 60)), 0.05)
    LOCK_AMP_THRESHOLD_P = max(float(np.percentile(pamp, 70)), 0.01)

    def _precompute_locks(amp_arr, phase_arr, amp_thresh):
        n_steps = amp_arr.shape[0]
        stride = max(1, int(round(snap_hz / LOCK_PRECOMPUTE_HZ)))
        win = WATERFALL_N
        out_at_step = {}  # snapshot_index → list of locks
        for i in range(0, n_steps, stride):
            lo = max(0, i - win + 1)
            window_phase = phase_arr[lo:i + 1]
            window_amp = amp_arr[lo:i + 1]
            if window_phase.shape[0] < 4:
                out_at_step[i] = []
                continue
            out_at_step[i] = _detect_locks_filtered(
                window_phase, window_amp,
                threshold=LOCK_PLV_THRESHOLD,
                amp_threshold=amp_thresh,
            )
        # Convert sparse mapping into a per-step lookup that holds the most
        # recent computed locks.
        result = [[] for _ in range(n_steps)]
        last = []
        for i in range(n_steps):
            if i in out_at_step:
                last = out_at_step[i]
            result[i] = last
        return result

    print("Precomputing mode-locks…")
    t_lock_start = time.monotonic()
    locks_per_step_p = _precompute_locks(pamp, ppha, LOCK_AMP_THRESHOLD_P)
    locks_per_step_r = _precompute_locks(ramp, rpha, LOCK_AMP_THRESHOLD_R)
    print(f"  precomputed in {time.monotonic() - t_lock_start:.1f}s "
          f"(p: {sum(1 for l in locks_per_step_p if l)} active steps, "
          f"r: {sum(1 for l in locks_per_step_r if l)} active steps)")

    # Perceptual feature precompute. Features change slowly (tempo, key,
    # consonance) so we compute at FEATURE_HZ and nearest-neighbor-lookup
    # per render frame. Windowed: ~2.5 s half-window (5 s total) gives
    # the rhythm GrFNN enough context to pick a stable peak. Key
    # detection uses the final Hebbian W if available. Rhythm structure
    # carries peak-index persistence across frames so BPM doesn't flip
    # between adjacent log-spaced oscillators.
    FEATURE_HZ = 5.0
    print("Precomputing perceptual features…")
    t_feat_start = time.monotonic()
    feat_stride = max(1, int(round(snap_hz / FEATURE_HZ)))
    # Feature window kept tight (1s half → 2s total) because rhythm
    # peak picking destabilizes with a wider window — the mean amp
    # profile over too much history shifts the peak oscillator.
    feat_half = max(1, int(snap_hz * 1.0))
    # Voices want more history for envelope correlation to be
    # meaningful — 2.5s half-window → 5s total.
    voice_half = max(feat_half, int(snap_hz * 2.5))
    n_snap = pamp.shape[0]
    # We need complex state for rhythm structure (phase matters), so
    # rebuild rhythm_z from amp·exp(i·phase). Pitch state still uses
    # amplitudes only (extract_key reads W diag or |pitch_z|).
    rz_complex = (ramp.astype(np.complex128)
                  * np.exp(1j * rpha.astype(np.complex128)))
    feat_indices = np.arange(0, n_snap, feat_stride)
    features_per_step: list[dict] = []
    prev_peak_idx: int | None = None
    for i in feat_indices:
        lo = max(0, i - feat_half)
        hi = min(n_snap, i + feat_half + 1)
        sw = StateWindow(
            pitch_z=pamp[lo:hi].astype(np.complex128),
            pitch_freqs=pf,
            rhythm_z=rz_complex[lo:hi],
            rhythm_freqs=rf,
            frame_hz=float(snap_hz),
            w_pitch=pW_final,
        )
        rhythm = extract_rhythm_structure(sw, prev_peak_idx=prev_peak_idx)
        prev_peak_idx = rhythm["peak"]["idx"]
        key = extract_key(sw)
        chord = extract_chord(sw)
        conson = extract_consonance(sw)
        features_per_step.append({
            "tempo": rhythm["peak"]["bpm"],
            "peak_phase": rhythm["peak"]["phase"],
            "peak_freq": rhythm["peak"]["freq"],
            "n_companions": len(rhythm["companions"]),
            "tonic": key["tonic"],
            "mode": key["mode"],
            "key_conf": key["confidence"],
            "chord": chord["name"],
            "chord_conf": chord["confidence"],
            "consonance": conson,
        })
    features_hz_effective = snap_hz / feat_stride
    print(f"  precomputed in {time.monotonic() - t_feat_start:.1f}s "
          f"({len(feat_indices)} frames at {features_hz_effective:.1f} Hz)")

    # Voice identity precompute. The pitch amplitude envelopes across
    # the window feed into phase-coherence clustering; tracked voice
    # IDs persist across frames via Hungarian matching threaded
    # through prev_state.
    print("Precomputing voice identities…")
    t_vox_start = time.monotonic()
    voice_cfg = VoiceClusteringConfig()
    voice_state = VoiceState()
    voices_per_step: list[list[dict]] = []
    for i in feat_indices:
        lo = max(0, i - voice_half)
        hi = min(n_snap, i + voice_half + 1)
        pitch_z_vox = (pamp[lo:hi].astype(np.complex128)
                       * np.exp(1j * pph[lo:hi].astype(np.complex128)))
        sw = StateWindow(
            pitch_z=pitch_z_vox,
            pitch_freqs=pf,
            rhythm_z=rz_complex[lo:hi],
            rhythm_freqs=rf,
            frame_hz=float(snap_hz),
            w_pitch=pW_final,
        )
        voice_state = extract_voices(sw, prev_state=voice_state,
                                      config=voice_cfg)
        # Phase 2 — per-voice rhythm association. Envelope DFT
        # against the rhythm oscillator bank; each voice gets its own
        # tempo, phase, and clock-rate oscillator index.
        voice_state = extract_voice_rhythms(sw, voice_state)
        # Store lightweight per-frame snapshot for rendering/CSV.
        frame_voices = []
        for v in voice_state.active_voices:
            osc_arr = np.array(v.oscillator_indices, dtype=np.int32)
            frame_voices.append({
                "id": int(v.id),
                "osc_lo": int(osc_arr.min()),
                "osc_hi": int(osc_arr.max()),
                "osc_indices": v.oscillator_indices,
                "center_freq": float(v.center_freq),
                "amp": float(v.amp),
                "confidence": float(v.confidence),
                "age_frames": int(v.age_frames),
                "rhythm_bpm": (float(v.rhythm.bpm)
                               if v.rhythm is not None else None),
                "rhythm_phase": (float(v.rhythm.phase)
                                 if v.rhythm is not None else None),
                "rhythm_freq": (float(v.rhythm.freq)
                                if v.rhythm is not None else None),
                "rhythm_confidence": (float(v.rhythm.confidence)
                                       if v.rhythm is not None else None),
            })
        voices_per_step.append(frame_voices)
    print(f"  precomputed in {time.monotonic() - t_vox_start:.1f}s "
          f"(max {max((len(f) for f in voices_per_step), default=0)} "
          f"simultaneous voices)")

    # Temporal median smoothing on the continuous-valued features so the
    # displayed BPM and consonance don't flicker when the extractor peak
    # oscillator wobbles. 5-frame median at 5 Hz ≈ 1 s of smoothing —
    # short enough to stay responsive, wide enough to kill flicker.
    def _median_smooth(values: list[float], k: int = 5) -> list[float]:
        arr = np.array(values, dtype=np.float64)
        half = k // 2
        smoothed = np.empty_like(arr)
        for i in range(len(arr)):
            lo, hi = max(0, i - half), min(len(arr), i + half + 1)
            smoothed[i] = float(np.median(arr[lo:hi]))
        return smoothed.tolist()

    if features_per_step:
        tempos = _median_smooth([f["tempo"] for f in features_per_step])
        conson = _median_smooth([f["consonance"] for f in features_per_step])
        for i, f in enumerate(features_per_step):
            f["tempo"] = tempos[i]
            f["consonance"] = conson[i]

    def _build_lock_segments(locks: list[dict], n_osc: int):
        """Build a flat list of (vertical line + 2 endpoint dots) segments
        and matching colors for the constellation. Lanes spread by ratio so
        overlapping locks don't pile up on a single x-offset."""
        segments, colors = [], []
        for lock in locks:
            ratio = (int(lock["p"]), int(lock["q"]))
            color = _LOCK_COLORS.get(ratio, _INK_SOFT)
            lane_idx = _LOCK_RATIOS.index(ratio) if ratio in _LOCK_RATIOS else 0
            x = 0.25 + 0.45 * lane_idx
            y_lo = min(lock["i"], lock["j"]) + 0.5
            y_hi = max(lock["i"], lock["j"]) + 0.5
            segments.append([(x, y_lo), (x, y_hi)])
            colors.append(color)
            for y in (y_lo, y_hi):
                segments.append([(x - 0.10, y), (x + 0.10, y)])
                colors.append(color)
        return segments, colors

    def update(_frame):
        now = (max(0.0, time.monotonic() - clock["t0"])
               if clock["t0"] is not None else 0.0)

        im_p.set_data(slice_window(pamp_pad, now))
        im_r.set_data(slice_window(ramp_pad, now))
        if im_motor is not None and mamp_pad is not None:
            im_motor.set_data(slice_window(mamp_pad, now))

        # Lookup mode-locks for the current snapshot index. Cheap.
        step_idx_p = min(int(round(now * snap_hz)), len(locks_per_step_p) - 1)
        step_idx_r = min(int(round(now * snap_hz)), len(locks_per_step_r) - 1)
        seg_p, col_p = _build_lock_segments(locks_per_step_p[step_idx_p], len(pf))
        seg_r, col_r = _build_lock_segments(locks_per_step_r[step_idx_r], len(rf))
        lock_lines_p.set_segments(seg_p)
        lock_lines_p.set_color(col_p)
        lock_lines_r.set_segments(seg_r)
        lock_lines_r.set_color(col_r)

        _update_scope_trace(scope_p, slice_history(pamp_for_wf, now))
        _update_scope_trace(scope_r, slice_history(ramp_for_wf, now))

        # Phase coherence strips (per-oscillator rolling-window stability).
        im_psp.set_data(slice_history(pcoh_for_wf, now))
        im_prs.set_data(slice_history(rcoh_for_wf, now))

        # Phantom+residual composite strips.
        im_php.set_data(_composite(
            slice_history(pph_for_wf, now),
            slice_history(pres_for_wf, now),
            pres_vmax,
        ))
        im_phr.set_data(_composite(
            slice_history(rph_for_wf, now),
            slice_history(rres_for_wf, now),
            rres_vmax,
        ))

        # W matrix update — find the snapshot index closest to the current
        # playhead time and set_data. Static if no history is available.
        if im_Wp is not None and pW_hist is not None and pW_times is not None:
            idx = min(int(np.searchsorted(pW_times, now)), len(pW_times) - 1)
            im_Wp.set_data(np.abs(pW_hist[idx]))
        if im_Wr is not None and rW_hist is not None and rW_times is not None:
            idx = min(int(np.searchsorted(rW_times, now)), len(rW_times) - 1)
            im_Wr.set_data(np.abs(rW_hist[idx]))

        # Perceptual features banner + voice visualization.
        feat_idx = min(int(round(now * features_hz_effective)),
                       len(features_per_step) - 1)
        feat = features_per_step[feat_idx]
        # Beat pulse indicator: filled circle when peak_phase is within
        # ± π/4 of zero (near the beat), hollow otherwise.
        pulse = "●" if abs(feat["peak_phase"]) < (np.pi / 4) else "○"
        companions = (f"+{feat['n_companions']}L"
                      if feat["n_companions"] else "")
        active_voices = voices_per_step[feat_idx]
        voice_count = len(active_voices)
        features_text.set_text(
            f"{feat['tonic']} {feat['mode'].upper()} · "
            f"{feat['chord']} · "
            f"{pulse} {feat['tempo']:.0f} BPM {companions} · "
            f"CONSONANCE {feat['consonance']:.2f} · "
            f"VOICES {voice_count}"
        )

        # Voice ticks — horizontal colored segments at each oscillator
        # that belongs to an active voice. Rendered in the final 0.5 s
        # of the time window so they act like a visual "voice legend"
        # at the playhead-right. Each voice's ticks share a color
        # indexed by voice id mod palette length.
        voice_segments: list = []
        voice_colors: list = []
        tick_x_lo = WINDOW_S / 2 - 0.55
        tick_x_hi = WINDOW_S / 2 - 0.10
        for vox in active_voices:
            color = _VOICE_PALETTE[vox["id"] % len(_VOICE_PALETTE)]
            for osc in vox["osc_indices"]:
                y = float(osc) + 0.5
                voice_segments.append([(tick_x_lo, y), (tick_x_hi, y)])
                voice_colors.append(color)
        voice_lines_p.set_segments(voice_segments)
        voice_lines_p.set_color(voice_colors)

        artists = [im_p, im_r, p_line, r_line,
                   lock_lines_p, lock_lines_r, voice_lines_p,
                   scope_p["current"], *scope_p["history"], scope_p["fill"],
                   scope_r["current"], *scope_r["history"], scope_r["fill"],
                   im_psp, im_prs, im_php, im_phr, features_text]
        if im_motor is not None:
            artists.append(im_motor)
        if m_line is not None:
            artists.append(m_line)
        if im_Wp is not None:
            artists.append(im_Wp)
        if im_Wr is not None:
            artists.append(im_Wr)
        return tuple(artists)

    # Headless screenshot mode — render one frame at the requested time
    # and exit before touching Tk.
    if screenshot_at is not None:
        clock["t0"] = time.monotonic() - float(screenshot_at)
        update(0)
        fig.canvas.draw()
        out = screenshot_path or Path("snapshot.png")
        fig.savefig(out, dpi=120, facecolor=fig.get_facecolor())
        print(f"Saved snapshot to {out}")
        # Also dump the feature time series to CSV alongside the PNG
        # so we can audit things like BPM stability across the track.
        csv_path = out.with_suffix(".features.csv")
        with open(csv_path, "w") as f:
            f.write("t,tonic,mode,chord,bpm,peak_phase,peak_freq,"
                    "n_companions,consonance,n_voices,voice_ids,"
                    "voice_bpms\n")
            for i, feat in enumerate(features_per_step):
                t = i / features_hz_effective
                voices = voices_per_step[i]
                ids = "+".join(str(v["id"]) for v in voices) or "-"
                voice_bpms = "+".join(
                    f"{v['rhythm_bpm']:.0f}" if v.get("rhythm_bpm") else "?"
                    for v in voices
                ) or "-"
                f.write(
                    f"{t:.3f},{feat['tonic']},{feat['mode']},"
                    f"{feat['chord']},"
                    f"{feat['tempo']:.3f},{feat['peak_phase']:.4f},"
                    f"{feat['peak_freq']:.4f},{feat['n_companions']},"
                    f"{feat['consonance']:.4f},"
                    f"{len(voices)},{ids},{voice_bpms}\n"
                )
        print(f"Dumped features to {csv_path}")
        return

    # Scrollable Tk window. Background matches the paper figure so the
    # scrollbar gutter doesn't break the ink-on-paper illusion.
    root = tk.Tk()
    root.title(f"NRT — {audio_path.name}")
    root.geometry("1400x900")
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
        # Stretch the inner frame to the scroll canvas width so figure
        # resizes with the window horizontally.
        scroll_canvas.itemconfigure(inner_id, width=event.width)

    inner.bind("<Configure>", on_inner_configure)
    scroll_canvas.bind("<Configure>", on_scroll_configure)

    # Mouse wheel / trackpad scrolling on macOS + Linux + Windows.
    def on_mousewheel(event):
        # macOS: event.delta is small (±1..±3), scroll directly.
        # Windows/Linux: event.delta comes in multiples of 120.
        if abs(event.delta) >= 120:
            step = -int(event.delta / 120)
        else:
            step = -int(event.delta) if event.delta else 0
        scroll_canvas.yview_scroll(step, "units")

    scroll_canvas.bind_all("<MouseWheel>", on_mousewheel)
    scroll_canvas.bind_all("<Button-4>",  # Linux scroll-up
                           lambda _e: scroll_canvas.yview_scroll(-3, "units"))
    scroll_canvas.bind_all("<Button-5>",  # Linux scroll-down
                           lambda _e: scroll_canvas.yview_scroll(3, "units"))

    anim = FuncAnimation(fig, update, interval=33, blit=True,
                         cache_frame_data=False)

    def on_close():
        try:
            clock["stop_fn"]()
        except Exception:
            pass
        root.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    # NOW start playback — after all figure/canvas/animation wiring is in
    # place. If any of the above raised, afplay would never have started.
    print("Starting playback…")
    clock["stop_fn"], clock["t0"] = _playback_start(audio_path, audio_arr, sr)

    try:
        root.mainloop()
    finally:
        # Belt-and-suspenders: ensure playback is stopped if mainloop exits
        # via any path that skipped on_close.
        try:
            clock["stop_fn"]()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("config.toml"))
    ap.add_argument("--audio", type=Path, default=None,
                    help="Audio file for playback (overrides config)")
    ap.add_argument("--state", type=Path, default=None,
                    help="State parquet path (default: output/<audio_stem>.parquet)")
    ap.add_argument("--screenshot-at", type=float, default=None,
                    help="Render a single frame at this time (sec) and exit.")
    ap.add_argument("--screenshot-path", type=Path, default=None,
                    help="Where to save the screenshot PNG.")
    args = ap.parse_args()
    run(args.config, audio_override=args.audio, state_override=args.state,
        screenshot_at=args.screenshot_at,
        screenshot_path=args.screenshot_path)


if __name__ == "__main__":
    main()
