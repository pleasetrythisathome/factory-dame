---
id: task-007
title: Clojure / Overtone OSC subscriber
status: To Do
assignee: []
created_date: '2026-04-16 17:53'
labels:
  - consumer
  - audio
  - clojure
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A Clojure consumer that subscribes to the engine OSC stream and maps the signals to Overtone synths. Validates the 'plays well with Clojure in the future' design goal. Natural fit with Scarlet's thi.ng / functional-generative lineage.

Demo mappings:
- /rhythm/phantom → missing-pulse drum synth (fires on phantom activation)
- /pitch/phantom → implied-harmony pad
- /*/residual → filter cutoff / distortion depth for tension-release
- Hebbian W (post-run, read from npz) → noise-seeded improvisation in the song's key

Lives at neurodynamics/consumers/overtone/ with its own project.clj / deps.edn. aero for config.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Overtone subscribes and synthesizes audio
- [ ] #2 At least two of the mappings above work
- [ ] #3 aero config for mapping parameters
- [ ] #4 Short README with deps and how-to-run
<!-- AC:END -->
