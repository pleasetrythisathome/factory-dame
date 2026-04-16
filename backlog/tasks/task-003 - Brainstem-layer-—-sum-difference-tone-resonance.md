---
id: task-003
title: Brainstem layer — sum/difference tone resonance
status: To Do
assignee: []
created_date: '2026-04-16 17:52'
labels:
  - engine
  - nrt-core
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a third GrFNN modeling frequency-following response at cochlear-nucleus resolution (see Harding/Kim/Large 2025 Fig. 5). This layer produces summation and difference tones from active pitches (e.g. 400+600Hz pitches → 200Hz and 1000Hz in brainstem) that are NOT in the stimulus. This is the psychophysical basis for consonance perception. Auto-harmonization use case: synthesize at the implied difference tones.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 New [brainstem_grfnn] config section, off by default
- [ ] #2 Driven by pitch layer state (not cochlea directly) since sum/diff tones are neural artifacts
- [ ] #3 Tests: driven with two sinusoids, verifies resonance at sum and difference frequencies
- [ ] #4 State log + OSC broadcast for brainstem layer
- [ ] #5 Optional viewer row
<!-- AC:END -->
