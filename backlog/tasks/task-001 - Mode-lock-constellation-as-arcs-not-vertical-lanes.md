---
id: task-001
title: 'Mode-lock constellation as arcs, not vertical lanes'
status: To Do
assignee: []
created_date: '2026-04-16 17:52'
labels:
  - viz
  - polish
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the current vertical-lane constellation (one vertical segment per locked pair at a per-ratio x-offset) with curved arcs that read more like a musical Feltron diagram. Arcs would start at the locked oscillator's y-position, curve to the partner's y-position, colored by ratio. Visually strikingly distinct from the heatmap cells underneath.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Arcs render as smooth Beziers or circle-arc segments
- [ ] #2 Color mapping to ratios unchanged (green 1:1, navy 2:1, terracotta 3:2, etc.)
- [ ] #3 No regression in performance (precompute stays at startup)
- [ ] #4 Legend still in header
<!-- AC:END -->
