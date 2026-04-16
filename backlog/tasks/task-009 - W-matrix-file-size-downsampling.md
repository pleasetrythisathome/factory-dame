---
id: task-009
title: W matrix file-size downsampling
status: To Do
assignee: []
created_date: '2026-04-16 17:53'
labels:
  - infra
  - performance
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Current W snapshot storage: ~138MB for a 6-minute track. Complex128 100x100 pitch matrices at 2Hz. This won't scale to a playlist or live session.

Options (pick one or combine):
1. Store only the top-K values per row (sparse) — attractor structure is sparse in practice
2. Quantize to float16 or bf16 at write time
3. Store delta-encoded (snapshot minus previous)
4. Lower cadence when dynamics are quiescent (adaptive rate)

Needs thought about whether the viewer animation still works under each scheme.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 File size < 20MB for a 6-minute track
- [ ] #2 Viewer animation quality unchanged subjectively
- [ ] #3 Existing regression tests still pass (baselines may need regen)
<!-- AC:END -->
