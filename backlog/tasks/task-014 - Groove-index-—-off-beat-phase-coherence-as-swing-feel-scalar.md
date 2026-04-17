---
id: task-014
title: Groove index — off-beat phase coherence as swing/feel scalar
status: To Do
assignee: []
created_date: '2026-04-17 02:17'
labels:
  - engine
  - perceptual
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred from task-004 Tier C. A continuous scalar capturing the "groove" or "swing" feel of a passage — how much rhythmic activity happens off the downbeat.

## Approach

The beat phase (from rhythm_structure peak oscillator) gives the downbeat. Pitch-voice onsets whose phase relative to that downbeat lands near 0 are "on-beat"; onsets near π are "half-beat"; onsets at other phases are genuinely off-grid.

Groove ~ sum over voice onsets of (off-downbeat-ness × phase coherence × amplitude), normalized to [0, 1]. Research-level — the right formula probably emerges from prototyping on real music.

## Scope

- Research phase: prototype on Flights + the techno corpus (Sandwell, Function, Burial all have distinct grooves)
- Once the formula stabilizes: extract_groove in perceptual module
- Real-audio validation: groove should differ meaningfully between straight 4/4 (low) and swung / syncopated tracks (high)
- OSC /features/groove broadcast

## Acceptance Criteria

- [ ] #1 Research spike on 3+ corpus tracks: hand-eyeball the "groove" of each, compare to candidate formulas
- [ ] #2 extract_groove function producing 0-1 scalar
- [ ] #3 Synthetic tests: click track (low groove) vs. swung 16ths (higher groove)
- [ ] #4 Real-audio test: Sandwell District (pulsing 4/4) vs. Burial (heavily off-grid) produce distinguishable values
- [ ] #5 OSC /features/groove + banner display
<!-- SECTION:DESCRIPTION:END -->
