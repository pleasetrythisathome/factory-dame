---
id: task-006
title: TouchDesigner OSC particle demo
status: To Do
assignee: []
created_date: '2026-04-16 17:53'
labels:
  - consumer
  - visual
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A TouchDesigner project that subscribes to the engine's OSC stream and drives a particle system. First consumer — validates the OSC schema is actually usable.

Mappings to implement:
- /pitch/amp → per-band emission rate / color hue
- /rhythm/amp + /rhythm/phase → emission timing / orbital motion
- /rhythm/phantom → ghost emitters (firing in silence)
- /rhythm/residual → turbulence (positive = scatter, negative = coherent flow)
- /motor/amp → secondary slower-moving emitter layer

Keep visuals minimal at first — prove the data flow before going hard on aesthetics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Subscribes on 127.0.0.1:57120
- [ ] #2 Renders at 60fps on user's hardware
- [ ] #3 .toe file committed to neurodynamics/consumers/touchdesigner/
- [ ] #4 Short README with how-to-run
<!-- AC:END -->
