---
id: task-010
title: OSC message schema documentation
status: To Do
assignee: []
created_date: '2026-04-16 17:53'
labels:
  - docs
  - consumer
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Canonical reference doc for downstream consumers. Today the schema is 'read osc_out.py' — needs to be a first-class doc.

Include:
- Message path (/rhythm/amp, /pitch/phase, etc.)
- Argument shape (list of floats? ints? length?)
- Rate (60 Hz currently)
- Meaning (what the values represent, units, ranges)
- Ordering guarantees (per-step, multi-message frame boundaries)
- Example SuperCollider / TouchDesigner / Overtone subscribers (small snippets)

Lives at neurodynamics/engine/docs/osc-schema.md, linked from README.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Covers all existing paths
- [ ] #2 Includes at least one subscriber snippet
- [ ] #3 Linked from engine README
<!-- AC:END -->
