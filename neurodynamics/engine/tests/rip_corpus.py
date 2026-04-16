"""Rip the test corpus from YouTube and run the engine on each track.

Reads ``test_audio/tracklist.toml``, downloads each track via yt-dlp,
trims to the specified start/duration window, resamples to 16 kHz
mono WAV, and runs ``nd-run`` to produce ``state/<slug>.parquet``.

The audio and parquet files live in ``test_audio/`` which is
gitignored — regenerable, plus copyright. The tracklist itself is
committed so the corpus is reproducible.

Usage:
    uv run python -m tests.rip_corpus           # rip all
    uv run python -m tests.rip_corpus <slug>    # rip one
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


ENGINE_DIR = Path(__file__).resolve().parent.parent
TEST_AUDIO = ENGINE_DIR / "test_audio"
STATE_DIR = TEST_AUDIO / "state"
TRACKLIST = TEST_AUDIO / "tracklist.toml"


def load_tracklist() -> list[dict]:
    with open(TRACKLIST, "rb") as f:
        return tomllib.load(f)["track"]


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def rip_track(track: dict, *, force: bool = False) -> Path:
    """Download + trim + resample a single track. Returns the WAV path."""
    slug = track["slug"]
    wav_path = TEST_AUDIO / f"{slug}.wav"
    if wav_path.exists() and not force:
        print(f"[{slug}] already exists, skip rip")
        return wav_path
    if not _have("yt-dlp"):
        raise RuntimeError("yt-dlp not installed; brew install yt-dlp")
    if not _have("ffmpeg"):
        raise RuntimeError("ffmpeg not installed; brew install ffmpeg")

    # Prefer a YouTube search (resilient to dead links) over hard-coded URLs.
    # ``query`` falls back to artist+title if not specified.
    query = track.get("query") or f"{track['artist']} {track['title']}"
    target = track.get("url") or f"ytsearch1:{query}"
    print(f"[{slug}] downloading ({target})")
    tmp = TEST_AUDIO / f".tmp_{slug}.%(ext)s"
    # yt-dlp's best audio → M4A (or opus); we re-encode to WAV below.
    subprocess.run([
        "yt-dlp", "-f", "bestaudio",
        "-o", str(tmp),
        "--no-progress", "--quiet",
        target,
    ], check=True)

    # Find the downloaded file (extension varies).
    downloaded_candidates = list(TEST_AUDIO.glob(f".tmp_{slug}.*"))
    if not downloaded_candidates:
        raise RuntimeError(f"[{slug}] yt-dlp produced no file")
    src = downloaded_candidates[0]

    start = int(track["start_sec"])
    duration = int(track["duration_sec"])

    print(f"[{slug}] trimming {start}-{start+duration}s, resampling to 16 kHz mono WAV")
    subprocess.run([
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-ss", str(start),
        "-t", str(duration),
        "-i", str(src),
        "-ac", "1",         # mono
        "-ar", "16000",     # 16 kHz
        str(wav_path),
    ], check=True)

    src.unlink()
    return wav_path


def run_engine(wav_path: Path, slug: str, *, force: bool = False) -> Path:
    parquet_path = STATE_DIR / f"{slug}.parquet"
    if parquet_path.exists() and not force:
        print(f"[{slug}] parquet exists, skip engine run")
        return parquet_path
    print(f"[{slug}] running engine on {wav_path.name}")
    subprocess.run([
        "uv", "run", "nd-run",
        "--audio", str(wav_path),
        "--output", str(parquet_path),
    ], cwd=ENGINE_DIR, check=True)
    return parquet_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug", nargs="?", default=None,
                    help="Rip only this slug (default: all tracks)")
    ap.add_argument("--force", action="store_true",
                    help="Redo rip + engine run even if outputs exist")
    ap.add_argument("--skip-engine", action="store_true",
                    help="Only rip audio, don't run the engine")
    args = ap.parse_args()

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    tracks = load_tracklist()
    if args.slug:
        tracks = [t for t in tracks if t["slug"] == args.slug]
        if not tracks:
            print(f"No track with slug {args.slug!r}", file=sys.stderr)
            sys.exit(1)

    for track in tracks:
        slug = track["slug"]
        try:
            wav_path = rip_track(track, force=args.force)
        except subprocess.CalledProcessError as e:
            print(f"[{slug}] rip failed: {e}", file=sys.stderr)
            continue
        if args.skip_engine:
            continue
        try:
            run_engine(wav_path, slug, force=args.force)
        except subprocess.CalledProcessError as e:
            print(f"[{slug}] engine run failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
