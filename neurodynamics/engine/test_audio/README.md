# Real-audio test corpus for voice extraction

Contents are gitignored. To regenerate the corpus, run:

```
uv run python -m tests.rip_corpus
```

This will:
1. yt-dlp the tracks listed in `tracklist.toml`
2. Trim to the specified excerpt range
3. Resample to 16 kHz mono WAV
4. Run `nd-run` on each to produce `state/<slug>.parquet`

The test suite reads the parquets (not the audio) for voice-extraction
validation, so committing the corpus isn't necessary — a clone can
regenerate everything.
