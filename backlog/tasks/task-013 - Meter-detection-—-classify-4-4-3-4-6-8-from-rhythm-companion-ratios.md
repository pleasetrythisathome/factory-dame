---
id: task-013
title: 'Meter detection — classify 4/4, 3/4, 6/8 from rhythm companion ratios'
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
Deferred from task-004's original scope. Classify the dominant rhythmic grouping structure from the companion-ratio histogram that extract_rhythm_structure already produces.

## Approach

The rhythm GrFNN mode-locks oscillators at small-integer ratios against the main beat. The pattern of which ratios appear (and with what PLV) encodes the meter:

- Strong 2:1 + 4:1 + 8:1 companions → duple (4/4, 2/4)
- Strong 3:1 + 3:2 companions → triple (3/4, 6/8 depending on subdivision)
- Strong 3:2 without strong 2:1 → compound (6/8)
- Strong 5:1 or 7:1 → odd meter (5/4, 7/8)

## Scope

- extract_meter(window, rhythm_structure) returns {meter: str, confidence: float}
- Decision surface: a small classifier that maps companion-ratio presence patterns to meter labels
- Tests with synthetic rhythm state at known meters
- Real-audio validation across corpus: 4/4 tracks should classify as 4/4, 3/4 tracks as 3/4 (corpus is mostly 4/4 electronic so a new diverse test set may be needed for 3/4 and 6/8 coverage)

## Acceptance Criteria

- [ ] #1 extract_meter function in perceptual module
- [ ] #2 Synthetic tests for 4/4, 3/4, 6/8
- [ ] #3 OSC /features/meter broadcast
- [ ] #4 Viewer banner shows meter when confidence > 0.5
- [ ] #5 Real-audio test verifies 4/4 tracks in corpus classify correctly
<!-- SECTION:DESCRIPTION:END -->
