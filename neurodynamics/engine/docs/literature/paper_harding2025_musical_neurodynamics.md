# Harding, Kim, Demos, Roman, Tichko, Palmer, Large (2025) — Musical neurodynamics

**Reference:** Nature Reviews Neuroscience 26, 293–307 (May 2025)
**DOI:** 10.1038/s41583-025-00915-4

---

## Why this matters for factory-dame

**The current canonical statement of Neural Resonance Theory** —
a Nature Reviews Perspective spanning pitch, harmony, melody,
tonality, rhythm, metre, groove, and affect. Cite as the external
reference for "factory-dame is grounded in NRT."

Also the cleanest published **glossary** of NRT terms — useful for
external docs and onboarding new contributors.

## NRT principles (Section: NRT)

Five core principles, all of which apply to factory-dame:

1. **Neural resonance** — entrainment of neural oscillations to
   external stimuli or to one another.
2. **Nonlinear resonance** — generates frequencies not present in
   input (harmonics, subharmonics, combination frequencies,
   mode-locking at n:m ratios). The missing-fundamental and
   missing-pulse phenomena.
3. **Stability and attraction** — stable resonances pull less-stable
   states toward themselves. Mode-locking stability orders
   consonance.
4. **Attunement** — adaptation of network parameters via Hebbian
   plasticity *and* natural-frequency adaptation. Long-term
   learning of cultural musical structures.
5. **Strong anticipation** — anticipation emerges from delay-coupled
   dynamics without requiring explicit predictive models.

## The central claim

> "People anticipate musical events not through predictive neural
> models, but because brain–body dynamics physically embody musical
> structure."

This is **NRT's most distinctive empirical commitment**. The
alternative framework (PCM, predictive coding of music) holds that
expectation comes from a Bayesian internal model. NRT holds that
the brain *physically resonates* with music and that's where
anticipation comes from — no internal model needed.

> "Statistically universal structures may have arisen in music
> because they correspond to stable states of complex,
> pattern-forming dynamical systems."

I.e., the cross-cultural prevalence of 2:1, 3:2 ratios isn't
because all cultures happened to invent the same conventions —
it's because human nervous systems are coupled oscillators that
naturally resonate at these ratios.

## Two timescales (Fig. 1)

Music operates on **two distinct timescales** handled by NRT:

- **Rhythm/metre/beat:** delta (~0.5–4 Hz), theta (~4–8 Hz),
  alpha (~8–12 Hz). Cortex-level oscillations.
- **Pitch/harmony/tonality:** much faster. Cochlea ~30 Hz to
  20 kHz; auditory nerve and brainstem phase- and mode-lock up
  to ~4 kHz.
- **Roughness:** the gap between rhythm and pitch (~20–40 Hz) —
  perceived as buzzing.

These two timescales use the **same canonical equations** with
different parameter regimes — this is the architectural
justification for factory-dame's unified-engine-different-config
approach.

## Specific parameters in the figures

### Fig. 2 — oscillator regime demonstration

| Regime | α | β₁ | β₂ | ε |
|---|---|---|---|---|
| Linear | -0.1 | 0 | 0 | 1 |
| Critical | 0 | -0.1 | -0.1 | 1 |
| Limit cycle | 0.01 | -0.02 | -0.02 | 1 |
| Bistable | -0.1 | 0.3 | -0.1 | 1 |

### Fig. 3 — multi-frequency mode-locking stability

```
α = 0.05, β_1 = -0.05, β_2 = -0.05, ε = 1
```

Stability ratios visualized as colored Arnold-tongue regions across
the full ratio space. This is **a comprehensive visual map** of
which ratios are robustly mode-locked: 1:1, 2:1, 3:2, 4:3, 5:4,
9:8, ... down to 16:15, 45:32 (very narrow), 7:4 (notably narrow).

The figure spans **both rhythm timescale (top, polyrhythms like
1:1, 3:2, 7:4) and pitch timescale (bottom, unison/perfect fifth/
minor seventh)** — same diagram, same parameters, just relabeled.
**Most direct visual proof that rhythm and pitch share machinery.**

## Architectural map (Fig. 5)

The central auditory pathway in NRT terms:

```
Cochlea → Cochlear nerve → DCN/VCN → Superior olive
       → Inferior colliculus → Medial geniculate → Auditory cortex
```

Each stage modeled as a GrFNN layer. The Lerud 2014 multi-layer
architecture (cochlea+CN+IC) extended to cortex (Large 2016
tonality + this paper).

**Frequency-following response (FFR)** reproduced with extreme
accuracy by the NRT multi-layer model — at fundamental + all
harmonics + difference tones + combination tones.

## Three-frequency resonance for pitch (NEW emphasis)

> "The pitch shift matches closely with the frequency of a nonlinear
> oscillator that resonates to the two lowest-frequency components
> of the stimulus (a so-called three-frequency resonance)." (ref 113)

This is an explicit citation of **three-oscillator resonance** as
the mechanism behind missing-fundamental and pitch-shift effects.
Maps directly to Large 2016's `D_ijk` three-frequency Hebbian
connections.

**Implication for factory-dame:** if we implement two-frequency
Hebbian (`C_ij`) only, we'll miss the *pitch shift* phenomenon
that distinguishes 600+800+1000 Hz (perceived ~200 Hz, exact
fundamental) from 600+850+1100 Hz (perceived shifted pitch). Adding
`D_ijk` three-freq learning captures this.

## Attunement details

Attunement = Hebbian learning + natural-frequency adaptation.
Operating mechanisms:

1. **Hebbian synaptic plasticity** (Fig. 2c, Eq. of HI 1996b /
   Large 2016 / Kim & Large 2021)
2. **Natural-frequency adaptation** — the Roman et al. 2023 ASHLE
   mechanism. Per this paper:

> "In this NRT model, the natural frequency is an attractor, but
> the frequency of another oscillator to which it is coupled is
> also an attractor. Frequency undergoes short-term adaptation,
> allowing it to adapt to the tempo of other performers, but
> synchronization is better when oscillators operate closer to
> their natural frequencies."

This is the basis of Roman et al. 2023's elasticity mechanism.

## Enculturation (Fig. 4e)

> "Neural resonance theory models predict that infants learn simple
> rhythmic relationships (2:1) more quickly than complex ratios
> (3:1), reflected in the coupling strength of oscillator
> connections."

Confirmed by infant studies. **Cross-cultural validation** of NRT:
Western listeners and Indian/Bolivian listeners rate consonance
the same way at first hearing, but enculturation reshapes detailed
tonal preferences over years.

## Performance/sync research (Fig. 4f, g)

ASHLE (Roman 2023) results presented:
- Solo: musicians' spontaneous production tempo reflects each
  performer's intrinsic dynamics (preferred tempo + variability
  around it).
- Trio: ensemble synchronization shows coupling between performers
  — they pull toward each other's tempos.

This is the architectural motivation for an *interactive*
factory-dame mode: extracted voices could be coupled to ensemble
"performers" with individual preferred tempos.

## NRT vs PCM (Predictive Coding of Music)

The paper explicitly contrasts NRT with predictive coding:

> "Theories of predictive processing such as PCM propose that
> musical expectancy is a recursive Bayesian process. ... NRT
> relies on intrinsic dynamics of physiological mechanisms to
> explain structure and expectancy, whereas predictive coding is a
> very different account based fully on prior learning."

> "Theories that rely on statistical learning, including PCM, posit
> that the brain builds internal models based on statistical
> regularities in the environment to make predictions. NRT predicts
> that musical statistics and regularities emerge from the stable
> dynamics and patterns intrinsic to neural circuits. As a result,
> NRT predicts which patterns are easier to learn than others and
> why some patterns are more commonly found in music."

**Implications for factory-dame:**
- We're implementing the NRT view by default (oscillator dynamics)
- A predictive-coding wrapper *could* be added on top (Phase 6+)
- For now, NRT alone is the simpler, more biologically motivated
  starting point

## Open research directions identified

The paper lists what's missing from NRT (i.e., research gaps we
*shouldn't* expect existing papers to fill):

1. **Direct cross-timescale modeling** — rhythm and pitch
   integrated in one network rarely studied. Factory-dame's
   unified-canonical-engine across pitch + rhythm IS this research
   direction in implementation form.
2. **Individual variation and developmental trajectories** — how
   coupling strengths and natural frequencies vary per person.
3. **Bridging dynamics with statistical learning** — hybrid
   NRT+PCM models. Recent (Kaplan 2022 PIPPET, etc.) are early
   examples.
4. **Cross-cultural rhythm corpora** — empirical study of "stable
   musical structures will surface more often than less stable
   ones" claim across cultures.

These are *all* areas where factory-dame could contribute back to
the literature if we develop interesting empirical findings.

## Glossary highlights (Box, page 303)

Key definitions to use consistently in our docs:

- **Anticipation:** "The process by which a system responds to an
  expected event before the event occurs."
- **Attraction:** "The evolving state of a dynamical system towards
  a more stable state, such as an orbit."
- **Attunement:** "The adaptation of neural circuits to the
  environment, enhancing response stability and flexibility."
- **Critical oscillation:** "An oscillation poised at a bifurcation
  point, the transition between damped and self-sustained
  oscillation."
- **Mode-locking:** "Synchronization in an integer (non-1:1) ratio."
- **Natural frequency:** "The frequency at which a system oscillates
  when not subjected to external forces; determined by its physical
  characteristics."
- **Phase-locking:** "Synchronization in 1:1 frequency ratio where
  the phase difference between two systems (or a system and an
  external stimulus) is maintained constant."
- **Strong anticipation:** "Anticipatory behaviour in interactions
  with the environment that emerges owing to transmission delays
  (delay coupling)."
- **Tonality:** "The perception of stability and attraction
  relationships among the pitches in musical work."

## Patent disclosure (relevant!)

> "Competing interests: E.W.L. is founder of, and owns stock in,
> Oscilloscape, Inc. (dba Oscillo Biosciences). J.C.K. is currently
> a paid employee of, and owns stock in, Oscilloscape, Inc.
> E.W.L. and J.C.K. are authors of patents owned by Oscilloscape,
> Inc. The subject matter of the current paper is not directly
> related to the business interests of Oscilloscape, and no
> products of Oscilloscape are discussed in this paper."

This is a useful clarifying statement: **the academic paper is not
itself a patent claim**, and the authors explicitly note the paper
doesn't discuss Oscilloscape products. For commercial purposes,
patent claims (US 8,930,292, US 8,583,442) still govern — not the
academic publication.

## Implications for factory-dame

### Use as the canonical NRT citation

When external docs need to cite "the theoretical foundation," cite
this paper. It's the most current, most comprehensive, peer-
reviewed in the highest-impact relevant venue (Nature Reviews
Neuroscience).

### Update factory-dame's vocabulary to match the glossary

Adopt the paper's glossary definitions as canonical. This reduces
ambiguity in our own docs and aligns with the NRT literature.

### Validate that we cover the principles

Cross-check that our implementation actually instantiates the
five NRT principles:

| Principle | Status |
|---|---|
| Neural resonance | ✓ (phase-locking in GrFNN) |
| Nonlinear resonance | Partial (missing-fundamental needs Phase 1.x canonical-input refactor) |
| Stability and attraction | ✓ (Voice clustering by mode-lock stability) |
| Attunement | Not yet (Phase 5 Hebbian + Roman 2023 elasticity) |
| Strong anticipation | Not yet (Phase 3 motor layer + delay coupling) |

This gives us a checklist for "Is factory-dame a real NRT
implementation?" The answer right now: mostly resonance + stability.
Phases 3/5 are needed for full coverage.

### Three-frequency resonance is now explicitly highlighted

Per "three-frequency resonance" callout above, three-oscillator
resonance is an *explicit* NRT prediction in this current
comprehensive statement. Implementing `D_ijk` (three-frequency
Hebbian) isn't an exotic extension — it's part of canonical NRT.

### Patent-IP situation is unchanged

The "competing interests" disclosure confirms that **academic
publications by Large et al. don't supersede patent claims.**
Implementing the literal equations from this Perspective for
commercial use would still require a license to US 8,930,292 etc.
For research / personal use: unaffected.
