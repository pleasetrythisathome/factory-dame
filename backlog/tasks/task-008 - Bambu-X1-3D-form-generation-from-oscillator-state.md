---
id: task-008
title: Bambu X1 3D form generation from oscillator state
status: To Do
assignee: []
created_date: '2026-04-16 17:53'
labels:
  - consumer
  - 3d
  - exploration
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Generative geometry driven by oscillator state for 3D-printable forms on Scarlet's Bambu X1 Carbon. Potential Taylor-creative-intersection project.

Architecture idea: the engine processes a piece of music, produces the state log + W matrix. A separate Clojure or Python generator consumes the state and outputs an STL/3MF whose geometry encodes the song's structure — e.g. a voronoi lattice where cell positions are pitch oscillators, densities are amplitudes, and connection topology follows the learned W matrix.

This is exploratory. Scope down to one specific form type first (e.g. a radial column where height at angle θ = rhythm oscillator amplitude, accumulated over time).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Takes state.parquet (+ optional weights.npz) → STL/3MF
- [ ] #2 Form captures at least two oscillator signals structurally
- [ ] #3 Prints cleanly on X1 Carbon (not a blob of overhangs)
- [ ] #4 Committed to neurodynamics/consumers/3d/
<!-- AC:END -->
