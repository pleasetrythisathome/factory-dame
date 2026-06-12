# Hoppensteadt & Izhikevich (1997) — Weakly Connected Neural Networks

**Reference:** Springer, Applied Mathematical Sciences vol 126
**DOI:** 10.1007/978-1-4612-1828-9
**ISBN:** 978-0-387-94948-2 (also published as ISBN 978-1-4612-7302-8)

---

## Access status

**Springer textbook** (~$110 hardcover, available via institutional
subscriptions or library). Some chapters available as PDFs on
Izhikevich's website (https://www.izhikevich.org/publications/).

## Why this matters for factory-dame

The **textbook treatment** of the weakly-connected-neural-network
theory. Hoppensteadt & Izhikevich's 1996 BC papers (I and II) are
condensed treatments of material that gets full development here.

For factory-dame: read selectively. The most relevant chapters:

- **Chapters on Andronov-Hopf canonical models** — the math behind
  Large 2010's canonical equation. Helps when reading-up on *why*
  the equation has the form it does.
- **Chapters on synaptic organizations and Dale's principle** — the
  full development of the Fig. 1 analysis in HI 1996b. Matters for
  Phase 5 architecture choices.
- **Chapters on multiple-attractor neural networks** — relevant if
  Phase 5 ends up storing multiple voice patterns simultaneously.
- **Chapters on weak coupling** — the small-ε limit and when the
  canonical-form analysis applies vs breaks down.

## Status as a "load-bearing" reference

This is **deep background**, not implementation-critical. Read
only if:
- Wanting to extend or modify the canonical model (Phase 5+)
- Debugging an instability and suspecting the model is operating
  outside the weak-coupling regime
- Writing a paper/article and needing the rigorous citation

## TODO

- [ ] Acquire copy (library, institutional, or purchase)
- [ ] Read selectively per the chapter list above
- [ ] Cross-reference any equations in our code against textbook
      derivations for assurance
