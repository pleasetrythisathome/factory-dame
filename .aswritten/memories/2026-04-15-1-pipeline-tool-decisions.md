---
reviewers:
  - scarlet@aswritten.ai
---

# Pipeline Tool Decisions — What's Settled and What's Open

The tool choices are pragmatic, not architectural statements:

- **Clojure + thi.ng/geom** — Scarlet's expert language. The generative/procedural core. thi.ng/geom worked when Scarlet last tried it (~2025). The briefing's concern about staleness and needing a fork is likely overstated — verify on current JVM when work starts, but not a presumed blocker.
- **CadQuery** — appropriate tool for mechanical/precision parts that interface with generative forms. Python, programmatic, shares parameters with Clojure via EDN.
- **Rhino** — Scarlet's most familiar 3D modeling environment. Integration workspace for mating NURBS surfaces, boolean operations between different pipeline outputs. Not daily modeling.
- **Fusion 360** — Scarlet is considering learning it. Taylor has started learning Fusion.

## What's open

- LED / LX Studio history — nothing assumed to carry forward. Fresh exploration; prior work resurfaces if relevant.
- AI workflow experimentation — no specific plans. The project has no constraints, so experimentation happens organically.