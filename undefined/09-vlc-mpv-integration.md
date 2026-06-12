# VLC (and mpv) Subtitle Integration Surface for Sotto

## Overview

Sotto's architecture bet — offline preprocessing emits a rich `.ass` sidecar, rendered natively by VLC via libass, with variants muxed as selectable MKV tracks — turns out to be well supported by the actual VLC integration surface, with two important corrections to common assumptions. First, the good news: VLC passes ASS to libass **completely untouched**. The libass decoder module in VLC 3.0.x explicitly disables all style overrides (`ass_set_style_overrides(library, NULL)`) and pins font scale to 1.0, and the "Subtitles text scaling factor" preference is read *only* by the freetype renderer used for SRT/plain-text — it cannot touch libass output. There is essentially **nothing for users to configure or disable**; a default VLC install renders Sotto's ASS verbatim. Second, the caveat: VLC renders ASS at the *video's storage resolution* and upscales the resulting bitmaps before blending, so text sharpness on a Retina display is capped at the video's pixel resolution — a quality ceiling mpv doesn't have, since mpv renders at display resolution.

This research was grounded in the actual binaries: VLC 3.0.23 (the current stable, built Dec 2025) is installed on the target MacBook, and its bundled libass version (0.17.3), full command-line option surface, and decoder source were verified directly. Verdict up front: **mux variants with mkvmerge, name the tracks descriptively, set one default flag, embed fonts as attachments, tell users to do nothing — and use mpv with JSON IPC as the dev/preview harness.**

## Key findings

### Which libass ships where

- **VLC 3.0.x stable bundles libass 0.17.3.** The 3.0.x contrib build spec pins `ASS_VERSION := 0.17.3` with harfbuzz and (non-Apple) fontconfig ([contrib/src/ass/rules.mak, 3.0.x branch](https://raw.githubusercontent.com/videolan/vlc/3.0.x/contrib/src/ass/rules.mak)). Verified locally: the installed VLC 3.0.23 (Dec 2025 build) contains `liblibass_plugin.dylib` with the version string `tarball: 0.17.3` (local inspection of `/Applications/VLC.app`, VLC `--version` output: "3.0.23 Vetinari").
- **VLC 4 nightlies bundle libass 0.17.4** with `-Dauto_features=disabled`, DirectWrite on Windows, and ASM optimizations ([contrib/src/ass/rules.mak, master](https://raw.githubusercontent.com/videolan/vlc/master/contrib/src/ass/rules.mak)).
- **libass 0.17.4 is the latest upstream release** as of 2026; the 0.17.x line includes the `LayoutResX`/`LayoutResY` headers introduced in 0.17.0 ([libass releases](https://github.com/libass/libass/releases)). So even VLC stable is only one patch release behind upstream — all modern ASS features (`\pos`, `\an`, `\move`, `\fad`, `\k` karaoke, inline `\fs`/`\c`, LayoutRes) are available.

### How VLC renders ASS — and the one real quirk

- VLC's libass decoder (`modules/codec/libass.c`) sets `ass_set_extract_fonts(true)` (loads fonts embedded as MKV attachments), reads one config variable `ssa-fontsdir` for extra font directories, disables style overrides via `ass_set_style_overrides(p_library, NULL)`, and sets `ass_set_font_scale(p_renderer, 1.0)` ([VLC 3.0.x libass.c](https://github.com/videolan/vlc/blob/3.0.x/modules/codec/libass.c)). On macOS it skips fontconfig and uses libass's autodetect font provider (CoreText) with fallback family "Helvetica Neue" (same file).
- **Storage-resolution rendering:** VLC calls `ass_set_frame_size()` / `ass_set_storage_size()` with the *video's* dimensions, not the window's. As libass/Aegisub developer arch1t3cht put it: "VLC always renders subtitles at the video's storage resolution and then up/downscales all bitmaps returned by libass individually before blending them," which "creates ugly ringing artifacts on text" ([HN discussion, early 2026](https://news.ycombinator.com/item?id=46472051)). Practical consequence: on the founder's M1 Max Retina display, a 1080p film yields subtitle text rasterized at 1080p and upscaled — readable but not Retina-crisp. mpv renders at display resolution and doesn't have this ceiling.
- **The "subtitle text scaling" preference does NOT affect ASS.** In VLC 3.0.x, `sub-text-scale` ("Subtitles text scaling factor", range 10–500, help text: "Changes the subtitles size *where possible*") is read only in `modules/text_renderer/freetype/freetype.c` (line 1174: `var_InheritInteger(p_filter, "sub-text-scale")`) — the renderer used for SRT/plain text — never in libass.c ([VLC 3.0.x freetype.c](https://github.com/videolan/vlc/blob/3.0.x/modules/text_renderer/freetype/freetype.c); option text verified from the local 3.0.23 `vlc -H --advanced` dump). The same is true of the Preferences > Subtitles/OSD font/size/color settings: "Font settings cannot be changed for rich text (such as ASS)" ([VLC wiki](https://wiki.videolan.org/VLC_HowTo/Adjust_subtitle_font_settings/)). **The brief's assumption that users must disable subtitle text scaling is wrong in a good way — they can't break Sotto's rendering even if they try.**
- **PlayRes / ScaledBorderAndShadow:** rendering "subtitles whose script resolution is not equal to the actual video resolution works poorly"; border/shadow/blur scale in script pixels only when `ScaledBorderAndShadow: yes`, which "should always be enabled" ([Aegisub script resolution docs](https://aegisub.org/docs/latest/script_resolution/)). Since Sotto preprocesses the exact movie file, it can simply set `PlayResX`/`PlayResY` to the video's storage resolution, sidestepping the entire scaling-bug class.

### Loading sidecars (verified against VLC 3.0.23 `-H` output)

- `--sub-file <string>` — "Load this subtitle file. To be used when autodetect cannot detect your subtitle file."
- `--sub-autodetect-file` — default **enabled**; fuzziness `--sub-autodetect-fuzzy` levels: 0 = none, 1 = any subtitle file, 2 = any containing the movie name, 3 = matching movie name with additional chars, 4 = exact match. So `Movie.sotto-placed.ass` next to `Movie.mkv` auto-loads at the default fuzziness, and *multiple* sidecar variants (`Movie.classic.ass`, `Movie.placed.ass`, …) all appear as selectable subtitle tracks — a zero-mux distribution option.
- `--sub-autodetect-path <string>` — extra directories to search.
- GUI: Subtitles > Add Subtitle File…, and drag-and-drop of a subtitle file onto the playing window also adds it (standard VLC behavior; the Lua API equivalent is `player.add_subtitle(url, autoselect)` — [VLC lua README](https://raw.githubusercontent.com/videolan/vlc/master/share/lua/README.txt)).
- `--ssa-fontsdir <string>` — "Additional fonts directory" for non-embedded custom fonts.

### Multiple tracks in one MKV

- **mkvmerge** is purpose-built: `--track-name TID:name`, `--language TID:lang` (ISO 639-2/-1), `--default-track-flag TID[:bool]`, `--forced-display-flag TID[:bool]`, `--track-order` ([mkvmerge docs](https://mkvtoolnix.download/doc/mkvmerge.html)). Options precede the file they apply to, so per-variant configuration is clean.
- **ffmpeg** works too: `-map 0 -map 1:0 … -c copy -metadata:s:s:N language=eng -metadata:s:s:N title="…" -disposition:s:s:N default`. Caveat: ffmpeg *copies disposition flags from inputs by default*, so explicitly zero the non-default tracks (`-disposition:s:s:1 0`) ([ffmpeg docs](https://ffmpeg.org/ffmpeg.html)).
- **Flag semantics:** the default flag marks a track as "eligible to be played by default, taking other user preferences such as track language into account"; the forced flag means "must be shown no matter what" and is conventionally reserved for foreign-dialogue-only tracks ([MKVToolNix wiki](https://codeberg.org/mbunkus/mkvtoolnix/wiki/Default-and-forced-flags-and-default-yes-no-in-the-GUI)). Players diverge in interpreting these ([doom9 thread](https://forum.doom9.org/showthread.php?t=167710)); VLC honors the default flag at selection time but the user's "Preferred subtitle language" preference (`--sub-language`, comma-separated codes, verified in local help dump) takes precedence, and forced-display tracks are auto-shown.
- **Switching tracks in VLC:** `V` cycles subtitle tracks, `Alt+V` cycles in reverse ([VLC hotkeys table](https://wiki.videolan.org/Hotkeys_table/)); the Subtitles > Subtitle Track menu shows each track's **name from the MKV header**, so descriptive `--track-name` values are the entire UX for variant selection.
- **Subtitle delay in VLC:** CLI `--sub-delay <integer>` is coarse — units of **1/10 second** ("100 means 10s", local help dump). Hotkeys `H`/`G` adjust delay up/down at runtime, and `Shift+H`/`Shift+J`/`Shift+K` implement the bookmark-audio/bookmark-sub/sync workflow ([hotkeys table](https://wiki.videolan.org/Hotkeys_table/)). Delay shifts whole cues uniformly, so Sotto's intra-cue word cadence survives user resync.

### VLC Lua: honest assessment — too limited, and that's fine

The Lua surface ([share/lua/README.txt](https://raw.githubusercontent.com/videolan/vlc/master/share/lua/README.txt)) offers: `player.add_subtitle()`, `player.get_spu_tracks()` / track selection, `player.get_subtitle_delay()` / `set_subtitle_delay()`, and an OSD API limited to `osd.message(string, [id], [position], [duration])` with nine fixed positions plus icons/sliders. There is **no access to the subpicture rendering pipeline, no arbitrary text positioning, no per-frame drawing, and no libass hooks**. A Lua extension cannot implement word-streaming or dynamic placement; at most it could be a convenience helper (auto-select the Sotto track, nudge delay). The `.ass` sidecar architecture is not just convenient — it is the *only* way to do this inside stock VLC.

### Performance: hundreds of short events per minute

- Word-streaming at ~160 wpm as one Dialogue event per word ≈ 160–400 events/min with accumulation lines — far below what anime karaoke typesetting throws at libass routinely. The idiomatic, cheaper encoding is **one Dialogue event per cue using `\k` karaoke timing per word** (centisecond resolution, relative to cue start), which collapses event count back to normal subtitle density.
- Known VLC complaints about ASS stutter exist but concern *heavy typesetting* (`\t` animations, `\blur`, `\clip`) and platform-specific issues — e.g., stutter on Android/Fire TV fixed by forcing OpenGL ES2 ([Stremio issue #702](https://github.com/Stremio/stremio-bugs/issues/702)), and forum reports of lag when subtitle lines appear ([VideoLAN forum](https://forum.videolan.org/viewtopic.php?t=146513)). The HN thread also relays second-hand reports of VLC "lagging or choking on complex subtitles" ([HN](https://news.ycombinator.com/item?id=46472051)). On an M1 Max with static styling and no per-word animation tags, this is a non-issue, but it argues for restraint: avoid `\blur`/`\be`/`\t` per word.

### mpv as the dev/preview target

- mpv makes libass **non-optional** ([mpv commit](https://github.com/mpv-player/mpv/commit/0b9ed9c2744a)) and typically links the newest release (0.17.4 via Homebrew). It renders at display resolution — no storage-res upscaling artifacts.
- `--sub-ass-override=<no|yes|scale|force|strip>` — default **`scale`** (applies `--sub-scale`, default 1, so effectively faithful); `no` renders "as specified by the subtitle scripts, without overrides" ([mpv manual](https://mpv.io/manual/master/#options-sub-ass-override), verified from DOCS/man/options.rst).
- `--sub-delay=<sec>` takes **float seconds** (vs VLC's 0.1 s integer CLI units); `--sub-files`/`--sub-file` add external subs, multiple files become switchable tracks; `--sid`/`--slang` select tracks; `--secondary-sid` can display two tracks at once (styling stripped on the secondary) — all from the same manual.
- `--sub-ass-use-video-data=<none|aspect-ratio|all>` (default `all`) controls whether libass sees storage resolution/aspect — relevant for testing anamorphic sources ([mpv manual](https://mpv.io/manual/master/#options-sub-ass-use-video-data)).
- Scripting: Lua scripts plus **JSON IPC** via `--input-ipc-server=/tmp/mpv-socket` ([mpv manual](https://mpv.io/manual/master/#json-ipc)) enable programmatic seek/screenshot/frame-step — exactly what an automated "render cue N, screenshot, assert placement" test harness needs. VLC has nothing comparable for rendering verification.

## State of the art (dated)

- **VLC 3.0.21** (June 2024) release notes mention "improved ASS subtitle rendering" ([free-codecs summary](https://www.free-codecs.com/news/vlc-media-player-3-0-21-super-resolution-hdr-enhancements.htm)); **VLC 3.0.23** (builds dated Dec 2025, verified locally) is current stable with libass 0.17.3.
- **VLC 4.0 is still nightly-only as of June 2026** (e.g., build 4.0.0.20260607 on [nightlies.videolan.org](https://nightlies.videolan.org/), packaged by [Chocolatey](https://community.chocolatey.org/packages/vlc-nightly)); it carries libass 0.17.4 and the new player/Lua API. Do not target VLC 4 behavior for users.
- **libass 0.17.4** (2025) is the latest release; LayoutRes headers since 0.17.0 ([releases](https://github.com/libass/libass/releases)). 2024–2026 recent-SOTA items: libass aarch64 ASM (0.17.2+, relevant on M1), mpv's `--sub-ass-use-video-data` and `--sub-scale-signs` refinements, and the early-2026 HN thread documenting VLC's storage-res rendering divergence.

## Design implications for Sotto (concrete, opinionated)

1. **Target libass 0.17.3 semantics, not newer.** Whatever 0.17.4 adds, the user's VLC has 0.17.3. Use the conservative feature set: `\pos`, `\an`, `\fad`, `\alpha`, `\k`, per-style margins. This also keeps mpv (0.17.4) and VLC (0.17.3) behavior aligned.
2. **Generate per-movie headers that make scaling a non-problem:** `PlayResX`/`PlayResY` = video storage resolution (Sotto has the movie file, so this is free), `ScaledBorderAndShadow: yes`, `WrapStyle: 2` (no automatic line wrapping — Sotto controls line breaks), and `LayoutResX/Y` matching for 0.17+ safety.
3. **Prefer `\k` karaoke timing over one-event-per-word** for the word-stream variant where the visual design allows (accumulating text within a cue): one Dialogue event per cue, per-word centisecond timing, drastically lower event count, and battle-tested in VLC. Use separate events only where words must appear *and disappear* independently.
4. **Ship via mkvmerge, one default track, never forced:** descriptive `--track-name` is the variant-selection UI. Tested-recipe shape:
   ```
   mkvmerge -o Movie.sotto.mkv Movie.mkv \
     --attach-file SottoFont.ttf --attachment-mime-type font/ttf \
     --language 0:eng --track-name 0:"English — classic"            --default-track-flag 0:1 classic.ass \
     --language 0:eng --track-name 0:"English — Sotto placed"       --default-track-flag 0:0 placed.ass \
     --language 0:eng --track-name 0:"English — Sotto word-stream"  --default-track-flag 0:0 stream.ass \
     --language 0:eng --track-name 0:"English — Sotto placed+stream" --default-track-flag 0:0 placed-stream.ass
   ```
   Default-flag the *classic* track so first-open behavior is unsurprising; users opt into Sotto tracks via `V` or the Subtitles menu.
5. **Embed the font as an MKV attachment.** VLC calls `ass_set_extract_fonts(true)`, so an attached TTF/OTF renders identically everywhere; never depend on the user having a font installed (macOS VLC has no fontconfig — fallback is Helvetica Neue, which would silently change metrics and break ORP alignment).
6. **User instructions are one line:** "Open the MKV in VLC, press V (or Subtitles > Subtitle Track) until you see 'Sotto'." No preferences to change — VLC cannot override ASS styling. Document only the negative: hardware/output-module quirks are unrelated to Sotto.
7. **Accept the storage-resolution sharpness ceiling in VLC; develop in mpv.** Build the preview/QA harness on mpv + JSON IPC (`--no-config --sub-ass-override=no --pause`, seek, `screenshot-to-file`, assert pixels). Geometry (PlayRes-space positions) is identical across both; only edge rendering differs. Spot-check final output in VLC 3.0.23 specifically.
8. **Word-stream cadence is delay-proof:** because all timing is per-cue (events or `\k` offsets relative to cue start), VLC's `H`/`G` delay adjustment and subtitle-sync workflow shift everything coherently. No special handling needed.
9. **Sidecar mode as the lightweight path:** for quick iteration (and for users unwilling to remux), emit `Movie.<variant>.ass` files next to the movie — VLC's default fuzzy autodetect (level 3, "movie name with additional chars") loads them all as switchable tracks with zero muxing.

## Open questions

- **End-to-end smoke test still owed:** every option above was verified from binaries, source, and docs, but this session did not play a muxed 4-track MKV in VLC 3.0.23 and screenshot each variant. That one-hour test (including a `\k`-heavy word-stream stress file) should be the first Sotto milestone.
- **Anamorphic sources (DVD rips):** VLC 3.0.x's libass.c grep showed no `ass_set_pixel_aspect` call in the surveyed paths — how VLC maps PlayRes onto non-square-pixel video needs an empirical check before claiming DVD support.
- **VLC update-throttle behavior:** does VLC re-blend the SPU exactly at each `\k` boundary or on its own subpicture refresh cadence? Affects perceived word-onset precision (target ≤ ~50 ms). Measure with a high-speed screen recording.
- **VLC 4 timeline:** if VLC 4 ships during Sotto's life, re-test — same libass family (0.17.4) but a rewritten video output pipeline could change the scaling/blending quirk (possibly for the better).
- **Orchestrator note:** the assigned output path arrived as `undefined/09-vlc-mpv-integration.md` (template variable unresolved); written literally relative to the working directory.

## Sources

1. VLC 3.0.x contrib libass build spec — https://raw.githubusercontent.com/videolan/vlc/3.0.x/contrib/src/ass/rules.mak
2. VLC master (4.0 nightly) contrib libass build spec — https://raw.githubusercontent.com/videolan/vlc/master/contrib/src/ass/rules.mak
3. VLC 3.0.x libass decoder module — https://github.com/videolan/vlc/blob/3.0.x/modules/codec/libass.c
4. VLC master libass decoder module — https://github.com/videolan/vlc/blob/master/modules/codec/libass.c
5. VLC 3.0.x freetype text renderer (sub-text-scale consumer) — https://github.com/videolan/vlc/blob/3.0.x/modules/text_renderer/freetype/freetype.c
6. Local verification: VLC 3.0.23 Vetinari on macOS — `VLC --version`, `VLC -H --advanced` full option dump, `strings liblibass_plugin.dylib` (libass "tarball: 0.17.3")
7. mpv manual (options: sub-ass-override, sub-delay, sub-files, sub-ass-use-video-data, JSON IPC) — https://mpv.io/manual/master/ and https://raw.githubusercontent.com/mpv-player/mpv/master/DOCS/man/options.rst
8. mpv: libass made non-optional — https://github.com/mpv-player/mpv/commit/0b9ed9c2744a
9. Hacker News: VLC libass rendering divergence (arch1t3cht comments, early 2026) — https://news.ycombinator.com/item?id=46472051
10. mkvmerge documentation — https://mkvtoolnix.download/doc/mkvmerge.html
11. MKVToolNix wiki: default and forced flags semantics — https://codeberg.org/mbunkus/mkvtoolnix/wiki/Default-and-forced-flags-and-default-yes-no-in-the-GUI
12. Doom9: player handling of default/forced Matroska flags — https://forum.doom9.org/showthread.php?t=167710
13. libass releases — https://github.com/libass/libass/releases
14. VLC Lua API README (master) — https://raw.githubusercontent.com/videolan/vlc/master/share/lua/README.txt
15. Aegisub: script resolution, PlayRes, ScaledBorderAndShadow — https://aegisub.org/docs/latest/script_resolution/ and https://aegisub.org/docs/latest/styles/
16. ffmpeg documentation (-map, -disposition, -metadata:s) — https://ffmpeg.org/ffmpeg.html
17. VLC hotkeys table — https://wiki.videolan.org/Hotkeys_table/
18. VLC wiki: subtitle font settings don't apply to ASS — https://wiki.videolan.org/VLC_HowTo/Adjust_subtitle_font_settings/
19. VideoLAN nightlies (VLC 4.0 status) — https://nightlies.videolan.org/ and https://community.chocolatey.org/packages/vlc-nightly
20. Stremio: VLC subtitle stutter and OpenGL ES2 fix (Android) — https://github.com/Stremio/stremio-bugs/issues/702
21. VideoLAN forum: subtitles cause video lag — https://forum.videolan.org/viewtopic.php?t=146513
22. VLC 3.0.21 release coverage (ASS rendering improvements, June 2024) — https://www.free-codecs.com/news/vlc-media-player-3-0-21-super-resolution-hdr-enhancements.htm
