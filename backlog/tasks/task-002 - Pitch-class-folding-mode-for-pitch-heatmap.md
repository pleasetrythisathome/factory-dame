---
id: task-002
title: Pitch-class folding mode for pitch heatmap
status: To Do
assignee: []
created_date: '2026-04-16 17:52'
labels:
  - viz
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a toggle that folds the pitch heatmap to 12 octave-equivalent rows (C, C#, D, …). Reduces the 100 oscillators to pitch classes by summing amplitudes across octaves. Makes key/chord structure far more visible than the current log-frequency view. Keep current view as default; alternate via a config flag or keybinding.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Config toggle (or viewer keybinding) switches view
- [ ] #2 Aggregation is sum-over-octaves of |z|
- [ ] #3 Y-axis labels as pitch-class names, not Hz
- [ ] #4 Scope trace follows the same view state
<!-- AC:END -->
