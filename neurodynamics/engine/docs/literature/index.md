# NRT Literature Index

Complete index of patents and papers grounding the factory-dame engine.
Updated 2026-05-10 (literature pull session).

PDFs of pulled papers are mirrored locally in `_pdfs/` for offline
reference. Summary `.md` files in this directory capture the
technical content; the master summary across all of them is in this
index's "Reading order for new contributors" section at the bottom.

Sources: Edward W. Large's CV (May 2 2023, retrieved from
musicdynamicslab.uconn.edu) plus follow-on papers and patents through
2025. The CV lists 98 publications and 16 patents — this index covers
those with direct relevance to GrFNN implementation, voice extraction,
rhythm processing, and the architectural choices factory-dame depends
on. Less-directly-relevant items (psychophysics studies, tangential
applications, pre-canonical-model work) are listed at the bottom for
completeness without summaries.

## How to use this index

- Items marked **[pulled]** have a summary file in this directory.
- **[needs-pull]** items are queued for retrieval and indexing.
- The "Phase X" tag indicates which phase of our depth-pass plan the
  item directly informs. Multiple tags = relevant to multiple phases.
- For each pulled item: equations, parameter values, and architectural
  insights are quoted verbatim from the source where possible.

---

## Patents

### Active (factor into commercial-use decisions)

#### US 8,930,292 B2 — Learning and ASA in GrFNN
- Filed 2011-01-28, issued 2015-01-06, expires 2032-10-13
- Inventors: Large
- Assignee: Oscilloscape (originally Circular Logic)
- **[pulled]** [summary](patent_us_8930292.md)
- Phase 5 (multi-frequency Hebbian), Phase 4 (W-based ASA)
- Key claim: connection learning via multi-frequency phase coherency

#### US 8,583,442 B2 — Rhythm processing + frequency tracking in GrFNN
- Filed 2011-01-28, issued 2013-11-12, expires 2031-04-02
- **[pulled]** [summary](patent_us_8583442.md)
- Phase 2 (network-wide tempo estimation)
- Key claim: per-oscillator frequency-correction signal aggregated network-wide

#### US 11,508,393 — Controller for real-time visual display of music
- Filed 2019, issued 2022-11-22
- Phase 6+ (visualization), Router (output formatting)
- **[needs-pull]**

#### EP 2529369 B1 — Learning and ASA (European)
- Issued 2019-12-04
- European counterpart to US 8,930,292
- **[needs-pull]**

#### EP 1774514 B1 — Nonlinear frequency analysis (European)
- Issued 2017-01-25
- European counterpart to (now-expired) US 7,376,562
- **[needs-pull]**

#### JP 5864441 B2 — Learning and ASA (Japanese)
- Issued 2016-02-17
- Japanese counterpart to US 8,930,292
- **[needs-pull]** (lower priority — Japanese only matters if commercializing in Japan)

### Expired / Public Domain (free to implement)

#### US 7,376,562 B2 — Nonlinear frequency analysis ★ EXPIRED
- Filed 2004-06-22, issued 2008-05-20, **EXPIRED 2024-11-17**
- **[pulled]** [summary](patent_us_7376562.md)
- Phase 1.1 (coupling kernel), Phase 1.2 (per-osc τ), Phase 3 (multi-layer)
- All architectural ideas in this patent are now public domain.

#### US 5,751,899 — Method of analysis (1997)
- **LAPSED** (per Large's CV)
- Earliest NRT-related patent. Methods now public domain.
- Probably superseded by 7,376,562's claims, of less use for implementation.
- **[needs-pull]** if curious about historical roots; not load-bearing.

#### CN 2015032700059050 — Chinese ASA — LAPSED
- China-only, lapsed. Doesn't affect us.

### Pending Applications (not yet enforceable, but signal future direction)

#### Counter-phase dichotic stimulation (US App. 18/132,852, 2023)
- Medical (binaural therapy)
- Not relevant to factory-dame

#### Music recommendations for audio + neural stimulation (US 63/434,603, 2022)
- Mixed medical / recommendation system
- Phase 6+ if we build a DJ/recommendation feature

#### Optimizing neural stimulation based on neurological signals (US 63/433,547)
- Medical (closed-loop therapy)
- Not relevant

#### Audio recommendations for neural stimulation (US 63/433,535)
- Recommendation system
- Phase 6+ relevant if we build matching/DJ feature

#### Neural stimulation via music + rhythmic visual stim (multiple jurisdictions, 2021-2023)
- Medical (Alzheimer's intervention)
- Not relevant

#### Feedback-based audio/visual neural stimulation (US 63/434,591)
- Medical
- Not relevant

#### Real-time visual display of music — continuation (US 17/969,253)
- Visualization continuation of issued 11,508,393
- Phase 6+ visualization

---

## Papers

Sorted by relevance to current/upcoming phases of factory-dame.

### Tier 1 — directly grounds engine implementation

#### Large, Almonte, Velasco (2010) "A canonical model for gradient frequency neural networks" — Physica D 239, 905-911
- **[pulled]** [summary](paper_large2010_canonical.md)
- THE foundational paper. Gives the canonical equation (Eq. 15)
  including the nonlinear input transformation and internal coupling
  (which we are NOT yet implementing per-spec — see Phase 1.x
  refactor needs). Demo simulation parameters: α=0, β1=-10, δ1=-9,
  ε=0.4, 360 osc / 6 octaves @ 60 osc/oct.
- Phase 1.1, Phase 1.2, Phase 3, ALL phases

#### Kim, J.C. & Large, E.W. (2015) "Signal Processing in Periodically Forced GFNNs" — Frontiers Comp Neuroscience 9:152
- **[pulled]** [summary](paper_kimlarge2015_signal_processing.md)
- Categorizes 4 parameter regimes (Critical Hopf, Supercritical Hopf,
  Supercritical DLC, Subcritical DLC) with bifurcation analysis.
  Critical regime: "always phase-locks to any frequency/amplitude" —
  ideal for audio analysis. Our config (α=-0.05, β1=-1) is closest
  to Subcritical DLC.
- Phase 3 (regime selection per layer)

#### Kim, J.C. & Large, E.W. (2021) "Multifrequency Hebbian plasticity in coupled neural oscillators" — Biological Cybernetics 115(1), 43-57
- **[pulled]** [summary](paper_kimlarge2021_hebbian.md) ★ Phase 5 blueprint
- Full mathematical analysis of multi-frequency Hebbian. Key
  finding: **GrFNN Hebbian plasticity is locally dominated by the
  lowest-order resonant monomial for each k:m ratio.** Provides
  Eq. 26 (frequency-scaled GrFNN + plastic connections) — direct
  Phase 5 implementation form. Concrete sim params from Fig 11.

#### Kim, J.C. & Large, E.W. (2019) "Mode locking in periodically forced gradient frequency neural networks" — Phys Rev E 99(2), 022421
- **[pulled]** [summary](paper_kimlarge2019_modelocking.md)
- Complete Arnold-tongue analysis across all four parameter
  regimes. Eq. 14 closed-form tongue widths. Eq. 6 stability
  condition for k:m mode-locking. Crucial for Phase 4 voice-cluster
  confidence weighting and `modelock.py` validation.

#### Lerud, K.D., Kim, J.C., Almonte, F.V., Carney, L.H., Large, E.W. (2019) "A canonical oscillator model of cochlear dynamics" — Hearing Research 380, 100-107
- **[pulled]** [summary](paper_lerud2019_cochlea.md)
- Dedicated cochlear-stage GrFNN. Phase 3's cochlear layer target.
- Bidirectional BM↔OC coupling (Eq. 9); 496 macaque tuning-curve fits.
- Code: https://github.com/MusicDynamicsLab/GrFNNCochlea

#### Large, E.W., Roman, I., Kim, J.C., Cannon, J., Pazdera, J.K., Trainor, L.J., Rinzel, J., Bose, A. (2023) "Dynamic models for musical rhythm perception and coordination" — Frontiers Comp Neuroscience 17:1151895
- **[pulled]** [summary](paper_large2023_rhythm_models.md)
- Comprehensive review of rhythm-perception model families.
  Synthesizes phase oscillator, Hopf normal form, Bayesian PIPPET,
  and Bose biophysical beat-generator. Phase 2 family-choice guide.

#### Large, E.W., Herrera, J.A., Velasco, M.J. (2015) "Neural networks for beat perception in musical rhythm" — Frontiers Sys Neuroscience 9:159
- **[pulled]** [summary](paper_large2015_beat_perception.md)
- Beat tracking via two coupled GrFNNs (sensory + motor). The
  sensory-motor architecture that grounds Phase 2's motor-layer.
  Eq. 2 is the fully-coupled-network canonical form.

#### Large, E.W., Kim, J.C., Flaig, N., Bharucha, J.J., Krumhansl, C.L. (2016) "A neurodynamic account of musical tonality" — Music Perception 33(3), 319-331
- **[pulled]** [summary](paper_large2016_tonality.md) ★ Hebbian rules
- **The most concrete Hebbian-learning paper.** Gives:
  - Eq. 2 closed-form stability: `ε^((k+m-2)/2)` for ratio k:m
  - Eq. A1 full network equation with two- and three-frequency monomials
  - Eq. A2 Hebbian rules for both C_ij and D_ijk connections
  - Concrete supercritical Hopf parameters: α=0.002, β₁=-2, β₂=-4
  - 35 training cycles for convergence
- **Phase 5 implementation reference.**

#### Tal, I., Large, E.W., Rabinovitch, E., Wei, Y., Schroeder, C.E., Poeppel, D., Zion Golumbic, E. (2017) "Neural Entrainment to the Beat: The 'Missing Pulse' Phenomenon" — J Neuroscience 37(26), 6331-6341
- **[pulled]** [summary](paper_tal2017_missing_pulse.md)
- MEG experimental validation that the brain generates oscillation
  at frequencies physically absent from stimulus envelope. The
  empirical basis for missing-pulse / phantom-fundamental output.
- Phase 2 test criterion ("does our rhythm bank pass the MP1/MP2
  test?").

#### Lerud, K.D., Almonte, F.V., Kim, J.C., Large, E.W. (2014) "Mode-locking neurodynamics predict human auditory brainstem responses to musical intervals" — Hearing Research 308, 41-49
- **[pulled]** [summary](paper_lerud2014_brainstem.md)
- First multi-layer GrFNN paper: cochlea → CN → IC/LL pathway.
  68% variance explained in FFR data. Concrete parameters for each
  layer (cochlea OC: β₁=-10000, ε=0.0025; brainstem: α=0, β₁=0,
  β₂=-1). **Phase 3 layered architecture reference.**

#### Harding, E.E., Kim, J.C., Demos, A.P., Roman, I.R., Tichko, P., Palmer, C., Large, E.W. (2025) "Musical neurodynamics" — Nature Reviews Neuroscience 26, 293-307
- **[pulled]** [summary](paper_harding2025_musical_neurodynamics.md) ★ canonical citation
- The current comprehensive NRT statement. Five core principles:
  resonance, nonlinear resonance, stability/attraction, attunement,
  strong anticipation. Glossary defines NRT vocabulary canonically.
- **Explicit highlight: three-frequency resonance** for pitch
  perception (motivates `D_ijk` in Phase 5).
- Includes the NRT-vs-PCM (predictive coding) contrast — useful
  for positioning factory-dame externally.
- **External citation for "factory-dame is grounded in NRT".**

### Tier 2 — informs architecture, less directly load-bearing

#### Large, E.W., Kim, J.C., Flaig, N., Bharucha, J.J., Krumhansl, C.L. (2016) "A neurodynamic account of musical tonality" — Music Perception 33(3), 319-331
- **[pulled]** [summary](paper_large2016_tonality.md) — **see Tier 1 detailed entry**
- ★ The actually-most-useful Hebbian paper — moved up the priority
  list. Contains the full Eq. A1/A2 Hebbian rules with concrete
  supercritical Hopf parameters.

#### Lerud, K.D., Almonte, F.V., Kim, J.C. & Large, E.W. (2014) "Mode-locking neurodynamics predict human auditory brainstem responses to musical intervals" — Hearing Research 308, 41-49
- **[pulled]** see [Tier 1 entry above](#tier-1).

#### Tichko, P., Kim, J., Large, E., & Loui, P. (2020) "Integrating music-based interventions with Gamma-frequency stimulation: Implications for healthy aging" — European J Neuroscience
- Lower priority (medical application focus). Skip.

#### Almonte, F., Velasco, M., Large, E.W. (2009/2010) "A canonical model for gradient frequency neural networks" — earlier conference version of Physica D paper
- Skip (superseded by Physica D version, already pulled)

#### Roman, I.R., Roman, A.S., Kim, J.C., Large, E.W. (2023) "Hebbian learning with elasticity explains how the spontaneous motor tempo affects music performance synchronization (ASHLE)" — PLoS Computational Biology 19(6):e1011154
- **[pulled]** [summary](paper_roman2023_ashle.md) ★ Open access + open code
- Sensory + motor coupled oscillators with adaptive natural
  frequencies + elasticity. Concrete parameters (α=1, β=-1, λ_1=4,
  λ_2=2, γ=0.02). Validated against 3 behavioural datasets.
- Phase 2 (tempo-bank with adaptive natural frequencies), Phase 3
  (motor layer with preferred rate). Low-risk extension.
- Code: https://github.com/iranroman/ASHLE

#### Tichko, P., Kim, J.C. & Large, E.W. (2021) "Bouncing the network" — Developmental Science 24(5)
- Auditory-vestibular interactions. Niche.

#### Roman, I.R., Washburn, A., Large, E.W., Chafe, C., Fujioka, T. (2019) "Delayed feedback embedded in perception-action coordination cycles" — PLOS Comp Biology 15(10)
- Sensorimotor delays. Niche — could matter for live-modular timing.

#### Farokhniaee, A., Almonte, F.V., Yelin, S. & Large, E.W. (2020) "Entrainment of weakly coupled canonical oscillators with applications to GrFNN using approximating analytical methods" — Mathematics 8, 1312
- **[needs-pull]** for analytical insight on multi-osc coupling.

#### Tal, I., Large, E.W., Rabinovitch, E., Wei, Y., Schroeder, C.E., Poeppel, D., Zion Golumbic, E. (2017) "Neural Entrainment to the Beat: The 'Missing Pulse' Phenomenon" — J Neuroscience 37(26)
- **[pulled]** see [Tier 1 entry above](#tier-1).

### Tier 3 — overview / context (cite only)

- **Harding et al. 2025 "Musical neurodynamics"** — Nature Reviews
  Neuroscience. **[stubbed]** see [Tier 1 entry above](#tier-1).
  Recent comprehensive NRT review; the place to cite externally.
- **Large 2010 "Neurodynamics of Music"** — Springer Handbook of
  Auditory Research vol 36 chapter
- **Large 2014 "Rhythm Perception: Pulse and Meter"** — Encyclopedia
  of Computational Neuroscience entry

### Tier 2.5 — foundational math (read for *why*)

These are the **mathematical foundations** Large 2010 builds on,
plus the alternative beat-perception framework (Cannon-Patel) and
its concrete implementation (Bose-Byrne-Rinzel). Pull these to
understand the **theory underneath** the implementation papers.

#### Hoppensteadt, F.C. & Izhikevich, E.M. (1996a) "Synaptic organizations and dynamical properties of weakly connected neural oscillators. I. Analysis of a canonical model" — Biological Cybernetics 75:117-127
- **[pulled]** [summary](paper_hoppensteadt_izhikevich_1996a_canonical.md) ★ foundational math
- Derives the canonical form `z' = bz + dz|z|² + Σ c_ij z_j` (Eq. 8)
  that Large 2010 extends. Proves it's *the* unique canonical form
  for weakly-coupled neural oscillators near Andronov-Hopf
  bifurcation.
- Key results: frequency-pool segregation (oscillators only couple
  if natural frequencies match), oscillator death + self-ignition
  regimes, Type A vs B oscillator distinction.
- PDF: https://www.izhikevich.org/publications/bc1.pdf (open)

#### Hoppensteadt, F.C. & Izhikevich, E.M. (1996b) "Synaptic organizations and dynamical properties of weakly connected neural oscillators. II. Learning phase information" — Biological Cybernetics 75:129-135
- **[pulled]** [summary](paper_hoppensteadt_izhikevich_1996b_learning.md) ★ Hebbian foundation
- The complex-Hebbian rule `c_ij' = -γ c_ij + k_ij z_i z̄_j`
  (Eq. 14) plus memory theorems. **All later Large-lab Hebbian
  results extend this.**
- Plus synaptic-organization analysis showing which connectivity
  patterns can/cannot memorize phase differences (Fig. 1a-d).
- Cohen-Grossberg convergence theorem (Eq. 17): self-adjoint W
  guarantees convergence to limit cycle.
- PDF: https://www.izhikevich.org/publications/bc2.pdf (open)

#### Hoppensteadt, F.C. & Izhikevich, E.M. (1997) "Weakly Connected Neural Networks" — Springer book (AMS vol 126)
- **[stub]** [summary](textbook_hoppensteadt_izhikevich_1997_wcnn.md)
- Textbook treatment of all the above. Deep background. Read
  selectively when extending the model or debugging instability.

#### Shapira Lots, I. & Stone, L. (2008) "Perception of musical consonance and dissonance: an outcome of neural synchronization" — J Royal Soc Interface 5:1429-1434
- **[pulled]** [summary](paper_shapira_lots_stone_2008_consonance.md)
- **Independent (non-Large-lab) validation** that mode-lock stability
  predicts consonance ordering. Uses IF neurons, not canonical Hopf.
- Cite this externally as "voice clustering by mode-lock isn't a
  Large-lab idiosyncrasy — it's intrinsic to coupled oscillators."

#### Cannon, J.J. & Patel, A.D. (2021) "How Beat Perception Co-opts Motor Neurophysiology" — Trends Cog Sci 25(2):137-150
- **[pulled]** [summary](paper_cannon_patel_2021_motor_neurophysiology.md)
- **Alternative framework** to GrFNN for beat perception. Proposes
  SMA + dorsal striatum mechanism. Handles non-isochronous meters
  (Balkan 2/2/3 etc.) better than pure oscillators.
- Useful for: understanding the cog-sci-mainstream alternative;
  designing a hybrid sensory(GrFNN)+motor(striatum) Phase 3.

#### Bose, A., Byrne, Á., Rinzel, J. (2019) "A neuromechanistic model for rhythmic beat generation" — PLoS Comp Biology 15(5):e1006450
- **[pulled]** [summary](paper_bose_byrne_rinzel_2019_beat_generator.md)
- The **concrete computational model** behind Cannon-Patel's
  proposal. Specific equations: LIF beat generator + gamma-counter +
  period/phase learning rules. Parameters fully specified.
- Patent-IP safe (independent of Large lab). Could replace or
  augment our motor layer if non-isochronous meter tracking matters.
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC6508617/

### Tier 2.6 — physiological-fidelity front-ends

For Phase 3's cochlear layer, alternatives to building a GrFNN
cochlea from scratch.

#### Zilany, M.S.A., Bruce, I.C., Carney, L.H. (2014) "Updated parameters and expanded simulation options for a model of the auditory periphery" — JASA 135(1):283-286
- **[pulled]** [summary](paper_zilany_bruce_carney_2014_auditory_periphery.md)
- Standard auditory-nerve model used throughout computational
  hearing research. Multi-stage pipeline: middle-ear → BM →
  IHC → IHC-AN synapse (power-law adaptation) → spike generator.
- Open MATLAB/C code at urmc.rochester.edu/labs/Carney-Lab.
  Python wrapper: https://github.com/tbekolay/auditory_nerve
- **Alternative to GrFNN cochlea (Lerud 2019)** — more
  physiologically accurate but heavier preprocessing stage.

### Tier 4 — historical / foundational (no pull, listed for completeness)

Pre-canonical model papers (1992-2009). Foundational thinking
superseded by the canonical formulation. Bibliography only:

- Large 1992 "A neural network model of recoding for musical stimuli"
- Large 1995 "Beat tracking with a nonlinear oscillator"
- Large & Kolen 1994/1999 "Resonance and the perception of musical meter"
- Large 2001 "Periodicity, pattern formation, and metric structure"
- Large & Snyder 2009 "Pulse and meter as neural resonance"
- Large 2008 "Resonating to Musical Rhythm: Theory and Experiment"
- Large & Tretakis 2005 "Tonality and Nonlinear Resonance"
- Large 2006 "A generic nonlinear model for auditory perception"
- Large 2011 "Musical tonality, neural resonance and Hebbian learning"
  (chapter, partly captured in Kim & Large 2021)

These pre-2010 papers establish ideas that the canonical model and
its refinements (2015, 2019, 2021) make precise. For implementation,
the post-2010 papers are sufficient grounding.

### Tier 5 — open-source code references

Public toolboxes from the Music Dynamics Lab. Implementation
references; check LICENSE on each before using directly.

- **GrFNN Toolbox 1.2.1** (MATLAB) — Large, Kim, Lerud, Harrell 2016.
  https://github.com/MusicDynamicsLab/GrFNNToolbox. Reference
  implementation of the canonical model.
- **GrFNN Cochlea** — Lerud, Kim, Large 2016. Cochlear-stage GrFNN.
  https://github.com/MusicDynamicsLab/GrFNNCochlea
- **GrFNN Brainstem** — Lerud, Kim, Large 2016. Brainstem GrFNN.
  https://github.com/MusicDynamicsLab/GrFNNBrainstem
- **GrFNN Rhythm** — Large, Herrera, Velasco 2016. Rhythm-perception
  GrFNN. https://github.com/MusicDynamicsLab/GrFNNRhythm

### Tier 6 — recent applications papers (cite if context demands)

- Tichko, Page, Kim, Large, Loui (2022) "Neural entrainment to musical
  pulse in naturalistic music is preserved in aging" — Brain Sciences 12(12)
- Dotov, Delasanta, Cameron, Large, Trainor (2022) "Collective dynamics
  support group drumming" — eLife 11
- Wei, Hancock, Mozeiko, Large (2022) "Entrainment dynamics and reading
  fluency" — Experimental Brain Research 240(6)
- Tichko, Kim, Large (2022) "A dynamical, radically embodied, and
  ecological theory of rhythm development" — Frontiers Psychology 13
- Chew et al (2021) "Music should be part of every physician's toolkit"
  — Scientific American
- Tichko, Kim, Large (2021) "Bouncing the network" — Developmental Science
- Heo, Soleymanpour, Lam, Goldberg, Large, Park, Kim (2021)
  "Wide-range Motion Recognition through Insole Sensor"
  IEEE J Biomed Health Inform 13. Sensor application — tangential.
- Tichko & Large (2019) "Modeling infants' perceptual narrowing to
  musical rhythms" — Annals NYAS
- Aydogan, Flaig, Ravi, Large, McClure, Margulis (2018) "Overcoming
  bias: Cognitive control reduces susceptibility to framing effects in
  evaluating musical performance" — Scientific Reports
- Kim, Large, Gwon, Ashley (2018) "Online processing of implied
  harmony" — Music Perception
- Harding, Sammler, Henry, Large, Kotz (2019) "Cortical tracking of
  nested beat structure in music and speech" — NeuroImage 185

These are CONFIRMATION studies (NRT predictions tested against EEG /
fMRI data, applied in clinical / behavioral / developmental contexts).
Useful for "this isn't speculative, the model is grounded in measured
human data" but don't add to implementation specs.

---

## Reading order for new contributors

Tackle in this order:

1. **patent_us_7376562.md** — public-domain foundation, the GrFNN
   architecture as established
2. **paper_large2010_canonical.md** — the canonical equation
3. **paper_kimlarge2015_signal_processing.md** — four-regime
   bifurcation analysis (when to use which α/β)
4. **paper_lerud2014_brainstem.md** — first multi-layer GrFNN; the
   blueprint for Phase 3
5. **paper_large2015_beat_perception.md** — sensory+motor architecture
6. **paper_large2016_tonality.md** — ★ Hebbian rules + the
   `ε^((k+m-2)/2)` stability formula
7. **patent_us_8930292.md** — what's claimed in active IP
8. **paper_tal2017_missing_pulse.md** — experimental validation of
   phantom-pulse/-fundamental generation
9. **paper_lerud2019_cochlea.md** — Phase 3 cochlear-layer reference
10. **paper_kimlarge2019_modelocking.md** + **paper_kimlarge2021_hebbian.md**
    — full Arnold-tongue + Hebbian theory (Phase 4/5)
11. **paper_large2023_rhythm_models.md** — comprehensive rhythm-model
    review (Phase 2 family-choice guide)
12. **paper_harding2025_musical_neurodynamics.md** — current
    comprehensive NRT statement, for external citation

### For mathematical foundations (read for why)

A. **paper_hoppensteadt_izhikevich_1996a_canonical.md** — derivation
   of the canonical form Large 2010 builds on
B. **paper_hoppensteadt_izhikevich_1996b_learning.md** — complex
   Hebbian rule + Cohen-Grossberg convergence
C. **textbook_hoppensteadt_izhikevich_1997_wcnn.md** — textbook
   treatment for deep dives

### For alternative frameworks (read for breadth)

D. **paper_shapira_lots_stone_2008_consonance.md** — independent
   confirmation that mode-lock stability orders consonance
E. **paper_cannon_patel_2021_motor_neurophysiology.md** — SMA +
   dorsal striatum alternative to oscillator models
F. **paper_bose_byrne_rinzel_2019_beat_generator.md** — concrete
   beat-generator equations (Cannon-Patel-style)
G. **paper_roman2023_ashle.md** — Hebbian + elasticity for adaptive
   tempo learning (open-source, low-risk Phase 2 extension)

### For physiological front-end (Phase 3 fidelity)

H. **paper_zilany_bruce_carney_2014_auditory_periphery.md** —
   standard auditory-nerve model (open MATLAB/Python)

---

## Phase-to-paper decision matrix

When picking an approach, here's which papers ground each option:

### Phase 1: foundation upgrades
- **Coupling kernel D**: us_7376562 + large2010_canonical
- **Per-osc τ**: us_7376562 + large2010_canonical (Eq. 19/20)
- **Gaussian noise**: us_7376562 + zilany2014 (fixed-seed fGn)
- **Canonical input form**: large2010_canonical + large2015_beat
  (Eqs. 3-4 P(ε,z) and A(ε,z̄))

### Phase 2: rhythm bank + tempo
- **Bank-based tempo** (current): large2015_beat + large2023_rhythm
- **Adaptive natural freq + elasticity**: roman2023_ashle ★ low-risk
- **Bose-style discrete beat generator** (alternative):
  bose_byrne_rinzel_2019
- **Patent-IP-safe aggregation**: avoid us_8583442 specific equation

### Phase 3: multi-layer architecture
- **Cochlear layer (GrFNN, native)**: lerud2019_cochlea
- **Cochlear layer (alternative front-end)**: zilany_bruce_carney_2014
- **Brainstem layers**: lerud2014_brainstem (concrete params)
- **Cortical layer (supercritical, memory)**:
  large2016_tonality (α=0.002, β₁=-2, β₂=-4) + us_8930292
- **Motor layer**: large2015_beat_perception (sensory+motor),
  + cannon_patel_2021 / bose_byrne_rinzel_2019 (alternative)

### Phase 4: voice extraction via W
- **Community detection on W**: us_8930292 (concept; not specific
  claim) + large2016_tonality (showing what stable W looks like)
- **Mode-lock stability hierarchy** (which connections matter):
  large2016_tonality Eq. 2 + shapira_lots_stone_2008

### Phase 5: multi-frequency Hebbian
- **The equations**: large2016_tonality Eq. A2 (most concrete)
- **Theoretical basis**: hi_1996b + kimlarge2021_hebbian (stub)
- **Patent-safe path**: PLV-gated update, not the literal expansion
- **Three-frequency D_ijk**: large2016_tonality (possibly not
  claimed in us_8930292 — research opportunity)
- **Convergence guarantees**: hi_1996b Thm 2 (self-adjoint W →
  Cohen-Grossberg convergence)

### Phase 6+: extensions / commercial
- **External NRT citation**: harding2025_musical_neurodynamics
- **Non-Western rhythms**: cannon_patel_2021 (Balkan meters)
- **Cross-cultural tonality**: large2016_tonality (rāgas)
- **Performance synchronization**: roman2023_ashle (SMT effects)
