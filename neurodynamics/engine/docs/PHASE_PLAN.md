# factory-dame engine: revised phase plan

**Updated 2026-05-13 (Phase 4.3 W-community extraction — over-clustering resolved).**

## Status — 299 tests pass (+1 documented xfail); voice extraction functional with the cascade on

The over-clustering and voice-jitter that made dense music unusable
are fixed by W-community extraction (see "Real-audio corpus state"
below). The cascade (Lerud 2014) stays on; its harmonics now fold
into their fundamentals via the learned Hebbian W. Voice counts are
musical (3–5 mean), IDs are stable, and the viewer shows honest
fundamental trails.

### Earlier — Phase 4.1/4.2 depth pass (2026-05-11)

Additions in this depth pass:
- **Phase 4.1 PLV gate** (Kim & Large 2021 Eq. 14): primary same-source
  evidence for harmonic merging. Test:
  `tests/test_voice_plv_gate.py` (locked vs independent at 1:2).
- **Vectorized harmonic pairwise check**: `_pair_is_harmonic` was 58 %
  of total runtime per cProfile (2.5 M calls / 5 s audio in a Python
  loop). Replaced with `_pairwise_harmonic_matrix` (numpy broadcast).
  ~3× speedup overall.
- **CSR sparse coupling kernel** in JIT: integer-ratio coupling matrix
  is ~10 % non-zero. Dense `Σ_j C[i,j] · P[j]` replaced with CSR
  sparse matvec. Per-step ~9× speedup at the coupling stage.
- **`--no-osc` flag for `nd-run`**: batch corpus runs skip pythonosc
  serialization (~30 % CPU saving when no live listener).
- **Chunked `step_many` integration loop in `run.py`**: previously
  per-sample step() calls, now snap-interval-sized chunks. Combined
  with vectorized harmonic + sparse coupling, offline mode runs at
  reasonable speed.
- **Phase 4.3 peak detection on active set**: cascade produces wall-
  to-wall pitch-bank activity (~270/278 bins above 0.005 noise floor
  on dense music). Local-maxima detection (±1 semitone window,
  90 % shoulder) collapses bandwidth tails to their centers,
  recovering a sparse set of meaningful voice candidates.
- **Phase 4.4 envelope-veto on PLV merge**: PLV at exact integer
  ratios = 1 even for "independent" synthetic sinusoids (carrier
  phases mathematically locked). Envelope correlation < 0 vetoes
  merge regardless of PLV; corr < 0.5 vetoes when PLV is the only
  signal. Disambiguates synthetic vs same-source cases.
- **Phase 4.5 harmonic-child collapse**: post-merge pass where a
  cluster at integer ratio of another with strictly higher amp
  (≥ 1.3×) AND positive envelope correlation (≥ 0.2) merges into
  the louder cluster — captures natural 1/n harmonic amplitude
  falloff. Two equally-loud voices at integer ratios (chord) stay
  separate.

## Real-audio corpus state — RESOLVED via W-community extraction (2026-05-13)

Phase 4.3 (W-community voice extraction) closed the over-clustering
gap. Clustering now keys on the LEARNED Hebbian W (Kim & Large 2021
Eq. 26): two oscillators merge as a harmonic family when one is an
integer overtone of the other AND the network learned a strong bond
between them. W is the time-integrated same-source evidence that
survives the per-window envelope decoupling of cascade-generated
harmonics — the exact failure mode that defeated the older
envelope-correlation merge. The cascade stays ON; its harmonics now
fold back into their fundamental.

| Track | mean (before→after) | max (before→after) |
|---|---:|---:|
| four_tet_angel_echoes | 11.2 → 3.7 | 19 → 8 |
| fred_again_marea | 12.8 → 3.6 | 30 → 9 |
| chemical_brothers_hey_boy_hey_girl | 8.5 → 3.5 | 23 → 8 |
| underworld_born_slippy_nuxx | 4.4 → 3.2 | 19 → 8 |
| function_voiceprint | 8.8 → 3.1 | 18 → 7 |
| sandwell_district_immolare | 12.1 → 5.6 | 19 → 11 |
| burial_archangel | 10.6 → 2.8 | 23 → 7 |
| disclosure_latch | 12.0 → 4.0 | 28 → 9 |
| kpop_demon_hunters_golden | 6.3 → 3.3 | 21 → 8 |
| daft_punk_da_funk | 10.2 → 4.2 | 22 → 10 |

All tracks now land in the musically-plausible 3–5 mean range
(matching the pre-cascade baseline) with stable voice IDs — center-
frequency jitter dropped from median ~1.2 st (p90 ~2.7) to ≤1.1 st
(p90 ≤1.9). Three additional fixes landed with it: **dominant-peak
fundamentals** (a voice's pitch is its loudest partial, not the
lowest member, so phantom subharmonics no longer drag voices an
octave low), **log-frequency smoothing** on matched voices (kills
frame-to-frame jitter), and **frequency-local co-modulation** (a
chord struck on one envelope resolves into its notes instead of
collapsing).

Validation: `tests/test_voices_ground_truth.py` (known-answer
synthetic fixtures — single tone → 1 voice, harmonic stack → 1, two
independent → 2, chord → 2-4) and `tests/test_voices_real_audio.py`
(honest musical bounds + a center-frequency stability gate). The
viewer (`nd-view`) draws each voice as a trail at its fundamental
across the time window, coloured by persistent ID — the honest voice
layer over the raw pitch heatmap.

One documented limitation (xfail): a fast exponential glissando
(octave-per-window sweep) gets pulled to the cascade's phantom
subharmonic; the voice count stays correct (1) but the tracked pitch
runs ~an octave low. Real music rarely sweeps that fast inside the
2.5 s window.

## Original status — Lerud 2014 architecture in place

Final session adds:
- **Cascade architecture wired in run.py** (offline mode) — previously
  only in live.py, which caused Tier A audit to not actually exercise
  the cascade. Now both paths use the same architecture.
- **Phase 0.3 missing-pulse test** (Tal 2017) — passes at 1.35× pulse-to-
  background ratio. Required enabling canonical input + coupling on
  the rhythm bank (not just pitch/brainstem).
- **Phase 0.4 FFR signature test** (Lerud 2014 Fig. 4) — passes with
  complete signature: f1, f2, 2f1, 2f2, difference tone (f2-f1),
  summation tone (f1+f2), and cubic distortion product (2f1-f2) all
  emerging from the cascade.
- **Brainstem panel in viewer** — teal-green panel below pitch shows
  the nonlinear layer's state. Harmonics visible as horizontal stripes
  at multiples of any tonal input. Distinct color palette so the
  cascade architecture is read top-down at a glance.
- **Voice extraction bandwidth-span gate** — fixed a bug where bandwidth
  spread of a single peak was misread as a harmonic stack, causing
  pure tones to report at f/2.
- **Extended harmonic ratio set** — added 1:1 (bandwidth-spread same-
  source), 1:6, 1:7, 1:8, and other ratios up to k+m ~ 11 for the
  cascade regime.

### Quality metrics

| Metric | Before depth pass | After full architecture |
|---|---|---|
| Tests passing | 185 | 192 |
| FFR signature (Lerud 2014) | n/a | reproduced (7/8 expected peaks) |
| Missing-pulse (Tal 2017) | not tested | reproduces (35% above background) |
| Tier A single_tone (440 Hz) | 7-15 voices wrong freq | V0 at 442 Hz ✓ |
| Tier A two_independent (330 + 880) | 26 IDs scattered | 5 IDs near correct freqs |
| Tier A pulsed_hihat (2 kHz) | spurious | V0 at 1990 Hz ✓ |
| Live throughput (step_many) | 1.12× | 11× realtime |
| Offline throughput (per-sample) | 0.56× | 0.91× realtime |



The complete Lerud 2014 / Kim & Large 2021 / Roman 2023 architecture
is now in place. 190 tests pass. End-to-end real audio works.

Architecture:

```
audio
  └→ Gammatone filterbank (cochlea, linear bandpass)
       └→ Brainstem GrFNN (Lerud 2014 critical Hopf)
            • α=0, β₁=0, β₂=-1, ε=0.5
            • Canonical nonlinear input P(ε,x)·A(ε,z̄)
            • Integer-ratio coupling kernel D enabled at gain=0.5
            • Generates harmonics + combination tones natively
                 └→ Pitch GrFNN (cortex, subcritical DLC)
                      • Multi-frequency Hebbian (K&L 2021 Eq. 26)
                      • Mode-locks to brainstem feedforward
                      • W matrix learns harmonic-stack communities
                           └→ Voice extraction (Phase 4)
                                • Harmonic-stack merging w/
                                  envelope-correlation gate
                                • Phase 4 + Phase 5 produce
                                  clean per-voice output
                                     └→ OSC/CV/MIDI router

audio
  └→ Onset envelope
       └→ Rhythm GrFNN (sensory)
            └→ Motor GrFNN (ASHLE: adaptive natural frequencies +
                            elasticity per Roman 2023)
                 • Adaptive f tracks tempo via Hebbian frequency rule
                 • Elasticity pulls f back to initial log-spaced value
                 • Sensory↔motor bidirectional coupling per Large 2015
```

## What each phase delivered

### Phase 1 — foundation upgrades

- 1.1 Internal coupling kernel D (now active in brainstem layer)
- 1.2 Per-oscillator τ (enabled bank-wide)
- 1.3 Minimal Gaussian noise 1e-5 (symmetry-breaking, far below
  voice-extraction noise floor)
- 1.4 ★ Canonical nonlinear input form `P(ε,x)·A(ε,z̄)` — Large
  2010 Eq. 15 / Lerud 2014 Eq. 2 / Kim & Large 2019 Eq. 15. Enabled
  in brainstem layer; ready for any layer where amplitudes warrant.

### Phase 0 — NRT-prediction validation tests

- 0.1 Arnold tongue width — `tests/test_canonical_input.py` confirms
  1:2 mode-locking works in test bank
- 0.2 Mode-lock stability hierarchy — same file confirms 1:1 > 2:1
  ≈ 1:2 > 3:2 ordering
- Phase 5 community structure test — `tests/test_multifreq_hebbian.py`
  shows W learns integer-ratio communities, distractor stays decoupled
- Phase 3 cascade test — `tests/test_brainstem_layer.py` confirms
  brainstem's 1:2 harmonic propagates to pitch (36× amplification
  over linear bypass mode)
- Phase 2 ASHLE adaptation test — `tests/test_adaptive_frequency.py`
  confirms motor freq tracks drive when phase-locked, relaxes back
  to f_0 in silence

### Phase 5 — Multi-frequency Hebbian (Kim & Large 2021 Eq. 26)

Multi-freq form replaces classical 1:1 Hebbian throughout. The
P-transform `z/(1-√ε z)` expands z into all integer-frequency
multiples; the outer product P(z_i)·P(z_j)* becomes stationary at
integer ratios and oscillating (zero-averaging) at irrational
ratios. Result: W matrix automatically encodes harmonic
relationships.

Bug fixed: Hebbian update used per-sample dt regardless of chunk
size, so step_many learned `chunk_size`× slower than step. Fixed
by passing n_steps through to `_hebbian_update`.

### Phase 4 — Voice cluster harmonic-stack merging

Post-clustering pass merges components at integer-ratio frequencies
WHEN their aggregated envelopes correlate. Envelope-correlation
gate prevents accidental merges of independent sources at integer
ratios (e.g., a 220 Hz voice + 880 Hz hi-hat). Union-find for
transitive harmonic closure.

### Phase 3 — Cascaded multi-layer architecture (Lerud 2014)

Brainstem GrFNN inserted between gammatone and pitch. Brainstem in
critical Hopf regime (α=0) with canonical input + coupling enabled
— generates harmonic enrichment via mode-locking. Pitch sees
brainstem's per-sample z trajectory via tonotopic 1-to-1
projection. Required adding `step_many_record` method on GrFNN
that captures z history per sample for downstream layers.

**Key validation**: drive engine with pure 220 Hz, observe pitch
oscillator at 440 Hz:
- Bypass mode (gammatone → pitch direct): |z_440| = 0.008
- Cascade mode (gammatone → brainstem → pitch): |z_440| = 0.302
- **36× amplification** of the propagated harmonic.

### Phase 2 — ASHLE adaptive natural frequencies (Roman 2023)

Each motor oscillator's natural frequency adapts toward drive phase
rate when driven, with elasticity pulling back to its log-spaced
initial value:

```
ḟ = f · (λ₁ |x| sin(φ_x − φ_z) − λ_e (exp((f − f_0)/f_0) − 1))
```

Updates applied per chunk in Python (not JIT) — frequency dynamics
are slow enough (~seconds) that per-chunk integration suffices.
Enabled on motor layer; sensory rhythm bank kept at static
frequencies (its role is spectral analysis, not single-tempo
tracking).

## What's NOT done (honest list)

- **Tier A full audit**: haven't run the synthetic ground-truth
  harness end-to-end with all features enabled. The downstream
  voice-extraction thresholds (`voices.py` config) were calibrated
  for the pre-cascade pitch bank state. Cascade pipes much richer
  state through pitch; thresholds may want re-tuning. **First
  follow-up after this checkpoint.**
- **Missing-pulse test (Tal 2017)**: not implemented as a Tier 0
  test. The rhythm bank is structurally capable (with ASHLE motor)
  but the test fixture isn't built.
- **FFR shape test (Lerud 2014)**: not implemented. Would require
  building Tone+Interval fixtures matching Lerud 2014 Fig. 4.
- **Pre-cochlear middle-ear filter (Zilany-Bruce 2006)**: not added.
  Gammatone front-end is sufficient for the demo; MEF is fidelity
  polish for FFR validation.
- **Three-frequency Hebbian `D_ijk`** (Large 2016 Eq. A2): not
  implemented. Only the two-frequency form is active. Three-freq
  captures combination-tone learning explicitly (better
  missing-fundamental for non-harmonic spectra). Likely
  patent-safer than two-freq.
- **Coupling kernel performance**: the brainstem coupling kernel D
  has O(n²) cost per step. At n=278 with full RK4, this is
  ~10× the bare GrFNN cost. Live throughput may regress; haven't
  measured yet.

## Status



Phase 1 is fully done — including the new 1.4 (canonical input) and
1.3 (noise). What's now wired:

- ✓ 1.1 Coupling kernel D infrastructure (disabled in config; works
  in probe at gain=1.0 + canonical input — see
  `tests/probe_harmonic_emergence.py`)
- ✓ 1.2 Per-osc τ (enabled)
- ✓ 1.3 Gaussian noise at 1e-5 (re-enabled)
- ✓ 1.4 Canonical nonlinear input form (enabled in config; identical
  to linear at our operating amplitudes, ready for Phase 3 regimes)
- ✓ Phase 0.1, 0.2 validation tests folded into
  `tests/test_canonical_input.py`. All pass.

185/185 tests pass.

## Critical finding from Phase 1 experiments

`probe_harmonic_emergence.py` shows the canonical input + coupling
kernel D produces **8× more 2:1 harmonic energy** than baseline at
realistic audio amplitudes — the missing-fundamental mechanism is
working mathematically.

**But:** enabling this on the existing single-layer pitch bank
produces voice-extraction explosion (7–15 simultaneous spurious
voices in Tier A). The bank correctly generates harmonics; voice
extraction interprets each as a separate voice.

**This is the gating issue for the demo.** The fix has two parts:

1. **Phase 3 multi-layer architecture** — separate the "harmonic
   generation" layer (brainstem with canonical+coupling) from the
   "pitch detection" layer (cortex, simpler dynamics). Brainstem
   has rich harmonic content; cortex tracks fundamentals via mode-
   locking against brainstem.
2. **Phase 4 voice extraction refinement** — recognize harmonic
   stacks and group them as one voice. The current
   `_cluster_stats` harmonic-stack centroid helps but doesn't merge
   separate clusters at integer ratios.

Both are substantial. Neither fits in a single focused session.

## Recommended next direction (pick one)

### Option A: Phase 5 first (Hebbian learning) — most demo-able

The existing engine has a simple 1:1 Hebbian rule
(`grfnn.py::_hebbian_update`). Per Kim & Large 2021 Eq. 26 + Large
2016 Eq. A2, the canonical multi-frequency form is:

```
(1/f_i) ċ_ij = -γ_ij c_ij + κ_ij · z_i/(1-z_i) · z̄_j/(1-z̄_j)
```

This learns connections at **all integer ratios** automatically.
With parameters from Kim & Large 2021 Fig. 11 (α=2, β₁=β₂=-1,
γ=0.5, κ=0.05/(n-1)) and ~35 training cycles per Large 2016, the W
matrix develops community structure where each community = a
harmonic stack of one voice.

**Demo:** train on a Bach chorale → watch W learn tonal hierarchy
→ play single notes → see other notes in the key "ring" via
learned connections. The viewer already supports W visualization.

**Effort:** ~1-2 days. Bounded. Replaces existing 1:1 Hebbian with
multi-freq form. Tests it on synthetic harmonic stacks.

### Option B: Phase 3 multi-layer architecture — most physiological

Full Lerud 2014 cochlea→brainstem→cortex pipeline. Each layer is a
GrFNN with literature-prescribed parameters. Requires
per-sample inter-layer coupling (modifying step_many for z history
or using small step chunks). Substantial.

**Demo:** drive engine with a missing-fundamental complex (200+400+600 Hz)
→ watch 100 Hz emerge in brainstem layer → cortex tracks 100 Hz as
the pitch. The viewer would need to show multi-layer state.

**Effort:** ~3-5 days. Includes architectural changes to live.py,
new layer classes, viewer updates, integration with voice
extraction.

### Option C: Phase 4 voice refinement first — minimal scope

Fix the harmonic-stack-merging in `_cluster_stats` to detect when
two clusters are at integer ratios and merge them. This unblocks
Option A and Option B without further architectural work.

**Demo:** none on its own — it's enabling work.

**Effort:** ~0.5-1 day. Tightest scope.

### My recommendation

**Option A (Phase 5 Hebbian) → Option C (Phase 4 cleanup) → Option B
(Phase 3 multi-layer).**

Reasoning:
- Phase 5 Hebbian alone produces the most striking single demo
  (watching W learn tonal structure in real time)
- It's bounded scope, builds toward Phase 4 (W community → voices)
- Phase 3 multi-layer is the big upgrade but takes time and isn't
  the most demonstrable on its own



This document supersedes the implicit phase plan held in conversation
and TaskCreate. Reading it should give a complete picture of where
we've been, what the literature showed, and what changes about the
path forward.

References to `paper_X.md` are in `docs/literature/`. PDFs in
`docs/literature/_pdfs/`.

---

## TL;DR — three biggest plan changes

1. **NEW Phase 1.4 — canonical nonlinear input form** is the single
   biggest gap in our current engine. We currently add `+ x`
   linearly to the oscillator equation. Per Large 2010 / Large 2015
   / Kim & Large 2021, this should be
   `+ x_total/(1-√ε x_total) · 1/(1-√ε z̄_i)`. **Without this, the
   missing-fundamental, phantom-pulse, and three-frequency-resonance
   phenomena that motivate the entire NRT framework cannot emerge.**
   This is more important than Phase 1.3 (noise).

2. **Phase 1.1 (internal coupling kernel) was correctly implemented
   but in the wrong operating regime.** Kim & Large 2019 Eq. 14
   explains why: at our typical |z| ~ 0.04 and ε = 1, the resonant-
   monomial prefactor `ε^((k+m-2)/2) · |z|^(m-1)` is essentially
   zero for any k+m ≥ 3. **The kernel will start working once
   Phase 3 puts oscillators in the high-amplitude regime** (cochlea
   or supercritical cortex). We don't need to fix the kernel; we
   need to give it the conditions it needs.

3. **Single-layer architecture has a hard performance ceiling.** Our
   β₁ = -1 single layer sits in the Subcritical DLC regime (Kim &
   Large 2015 four-regime analysis). Lerud 2014's multi-layer
   architecture is the only published configuration that hits 68%
   variance-explained on real FFR data. **Phase 3 is not optional
   for "best possible accuracy" — it's the rate-limiting step.**

---

## What the research changed about our understanding

### The core mathematical claim is bigger than we thought

The canonical NRT equation isn't just "Hopf oscillator with cubic
nonlinearity." The **resonant-input transformation**
`P(ε,x) · A(ε,z̄) = x/(1-√εx) · 1/(1-√εz̄)` is the load-bearing
piece. Without it:

- No harmonics generated from a pure tone
- No subharmonics or missing fundamentals
- No combination frequencies (sum/difference tones)
- No three-frequency resonance (the pitch-shift mechanism)

These are *not* emergent from the cubic nonlinearity alone — they
come from the geometric-series expansion that the nonlinear input
form provides. Hoppensteadt-Izhikevich 1996a Corollary 1 makes this
explicit: in the *plain* canonical model, oscillators at different
frequencies don't interact. The resonant-monomial expansion is what
breaks the frequency-pool segregation.

**Practical:** our pitch bank currently has each oscillator driven
linearly by the stimulus. The 220 Hz oscillator is excited only by
the 220 Hz component of the input. There's no harmonic enrichment
or phantom-fundamental generation happening. Voice extraction
heuristics (like our harmonic-stack centroid) are *approximating*
what the canonical form would produce natively.

### Hebbian plasticity has a concrete recipe now

Kim & Large 2021 Eq. 26 + Large 2016 Eq. A2 give the full Hebbian
learning rule with concrete parameters and a known convergence
benchmark (35 training cycles on a IV–V₇–I cadence in Large 2016).

The key insight: **Hebbian plasticity in GrFNNs is locally
dominated by the lowest-order resonant monomial for each k:m ratio**
(Kim & Large 2021 main theorem). So we don't need to compute the
infinite series; we can analyze and implement ratio-by-ratio with
single-monomial coupling per pair.

### Voice extraction has a stability hierarchy from physics

Kim & Large 2019 Eq. 14 gives the Arnold tongue width:
`width ≈ 2 · (√ε F)^k · (√ε r_s)^(m-2)`. This means:

- k+m = 2 (just 1:1): widest, robust under any conditions
- k+m = 3 (2:1, 1:2): wide if F or r_s is ~1
- k+m = 4 (3:1, 1:3, etc.): narrow unless amplitudes are strong
- k+m ≥ 7: essentially noise at our parameters

Our voice clustering should *bake in* this hierarchy. Currently
`voices._cluster_stats` accepts harmonic relationships up to roughly
8:1 — half of those are below detection threshold per the Arnold-
tongue math.

### Single layer is the wrong architecture for the goal

Lerud 2014 explicitly shows 68% variance-explained on FFR data
*only with* the three-layer cochlea→CN→IC/LL pathway. Single-layer
GrFNNs can do mode-locking and 1:1 phase locking, but they cannot
reproduce the full nonlinear FFR signature (harmonics + difference
tones + combination tones) that drives "voice identity" perception.

Multi-layer isn't a polish step. It's *the* step that takes us from
"phase-locking detector" to "full auditory-pathway model."

### Internal coupling kernel D — vindicated and de-prioritized

The Phase 1.1 implementation was structurally correct, but the
operating regime made it a no-op. This isn't a bug — Hoppensteadt-
Izhikevich 1996a Corollary 1 + Kim & Large 2019 Eq. 14 both say
the same thing: cross-frequency coupling at low amplitudes is
mathematically negligible.

The kernel will start to matter at amplitudes typical of:
- Cochlear layer (Lerud 2014: high amps, β₁=-10000 active tuning)
- Cortical / memory layer (supercritical Hopf, spontaneous |z|≈1)

So: **don't fix Phase 1.1, just defer using it until Phase 3
gives us layers where amplitudes warrant it.**

### Three-frequency resonance is part of canonical NRT

Harding 2025 explicitly highlights "three-frequency resonance" for
pitch perception. Large 2016 Eq. A1/A2 includes three-frequency
`D_ijk` monomials and learning rules. **Two-frequency Hebbian alone
isn't enough** to capture the full pitch-shift phenomenon (where
600+800+1000 Hz vs 600+850+1100 Hz produce subtly different
perceived pitches).

This bumps up the priority of three-frequency learning in Phase 5
and is a **likely patent-safer path** (US 8,930,292 focuses on
two-oscillator coherency).

### Alternative frameworks exist but don't supplant GrFNN for our domain

Cannon-Patel 2021 (SMA + dorsal striatum) and Bose-Byrne-Rinzel 2019
(LIF beat generator) are credible alternatives for beat perception,
particularly for non-isochronous meters. For factory-dame's primary
domain (Western tonal music, isochronous or simple-meter), GrFNN
remains the right call. **Keep these papers indexed for Phase 6+
extensions.**

---

## What the research changed about earlier work

### Phase 1.0 (baselines) — unchanged

Captured. Tier A 30.6% baseline + Tier B/C measurements are the
starting point.

### Phase 1.1 (internal coupling kernel D) — vindicated

The literature confirms:
- The kernel form is correct (matches Eq. 14 of Kim & Large 2021,
  Eq. 4 of Kim & Large 2019)
- The lack of effect at our amplitudes is **expected**, not a bug
- Re-enabling is correct once amplitude regime is appropriate

**Action:** keep code as-is, default disabled. Re-enable per-layer
in Phase 3 (cochlear and cortical layers).

### Phase 1.2 (per-oscillator τ) — confirmed canonical

Per Large 2010 Eq. 19/20 and Kim & Large 2021 Eq. 26, the `1/f_i`
scaling is canonical. The frequency-scaled form gives constant-Q
Arnold tongue widths in log-frequency space — exactly what we want
for tonotopic networks.

**Action:** finalize Phase 1.2, declare complete, and ensure both
JIT paths (`_step_many_jit`, `_rk4_step_jit`) use it.

### Voice clustering harmonic-stack heuristic — keep but refine

Our current 0.3×peak / 60% alignment heuristic in
`voices._cluster_stats` works but is ad hoc. The literature
suggests two refinements:

1. **Restrict to k+m ≤ 7** per Kim & Large 2019 Eq. 14
2. **Weight by Arnold-tongue width** — a 1:1 match is strong
   evidence; a 5:4 match is weak evidence even with high amplitude

**Action:** in Phase 4 refactor, replace ad hoc heuristic with
Arnold-tongue-weighted scoring.

### Test harness (Tier A/B/C) — augment with NRT-specific validation

The existing Tier A/B/C harness measures voice-extraction accuracy.
The literature adds two principled validation criteria:

1. **Missing-pulse test** (Tal 2017): rhythm bank must produce
   strong output at a pulse frequency that has zero energy in the
   stimulus envelope. This is the *correctness* test for nonlinear
   resonance in the rhythm domain.
2. **FFR shape test** (Lerud 2014): multi-layer output for the
   G2 + E3 consonant interval (99 Hz + 166 Hz) should produce peaks
   at f₁, f₂, 2f₁, 2f₂, 3f₁, 3f₂, f₁+f₂, 2f₁-f₂, etc., with
   relative amplitudes correlated to real FFR data.

**Action:** add Tier 0 (NRT-prediction validation) before Tier A.
Failing Tier 0 means the engine isn't doing NRT, regardless of
what Tier A says.

---

## Revised plan

### Phase 0 — NRT-prediction validation (NEW)

**Purpose:** verify the engine actually does what NRT claims.
These are *correctness* tests for the model, not accuracy tests
for a downstream feature.

- **0.1** Arnold tongue width measurement
  - Drive a single oscillator at varying offset frequencies
  - Measure locking range; compare to Kim & Large 2019 Eq. 14
  - Pass: measured width within 20% of predicted for k+m ∈ {2,3,4}
- **0.2** Mode-lock stability hierarchy
  - Drive at exact 1:1, 2:1, 3:2, 5:4 etc.
  - Measure PLV at each ratio
  - Pass: PLV decreases monotonically with k+m, matching
    Large 2016 Eq. 2 / Shapira Lots & Stone 2008 ordering
- **0.3** Missing-pulse generation (Tal 2017)
  - Synthesize MP1/MP2 stimuli per Tal 2017
  - Feed to rhythm bank
  - Pass: bank produces oscillation at the perceived pulse
    frequency even though stimulus has no energy there
- **0.4** FFR-shape validation (Lerud 2014)
  - Synthesize G2+E3 consonant interval
  - Run through multi-layer pipeline (deferred until Phase 3)
  - Pass: output PSD matches Lerud 2014 Fig. 4 nonmusician
    response at R² ≥ 0.5

Phase 0 partially executable now (0.1, 0.2). Phase 0.3, 0.4
require Phases 2, 3 respectively.

### Phase 1 — Foundation upgrades

- **1.0** ✓ Baselines captured
- **1.1** ✓ Internal coupling kernel D (correct, disabled at low
  amps; will be re-enabled at high-amp layers in Phase 3)
- **1.2** ◐ Per-oscillator τ (in progress; finalize)
- **1.3** Restore minimal noise (fGn fixed-seed per ZBC 2014,
  amplitude ~ 1e-5)
- **1.4** ★ **NEW HIGHEST PRIORITY: Canonical nonlinear input
  form**
  - Replace `+ x` in `_deriv_jit` with
    `+ P(ε, x_total) · A(ε, z̄)` where:
    - `P(ε, x) = x / (1 - √ε x)`
    - `A(ε, z̄) = 1 / (1 - √ε z̄)`
    - `x_total = stimulus + Σ_j c_ij z_j` (internal coupling
      enters x_total, not as separate additive term)
  - Per Large 2010 Eq. 15, Large 2015 Eq. 2, Kim & Large 2021
    Eq. 26
  - **This unlocks all multi-frequency phenomena.** Expect Tier A
    quarter-notes, phantom-fundamental, sustained-chord to improve
    significantly.
  - **Run Phase 0.1, 0.2 after this change** — the canonical input
    is what makes Arnold tongues at integer ratios actually form
- **1.5** Three-frequency resonant `D_ijk` coupling (defer until
  Phase 5 — see there)

### Phase 2 — Rhythm bank & tempo

- **2.1** Implement ASHLE-style adaptive natural frequencies +
  elasticity (Roman 2023)
  - Sensory rhythm oscillators with `ḟ_s = f_s(λ₁ sin(φ_x - φ_s) -
    γ(exp((f_s - f_m)/f_m) - 1))`
  - Replace the KDE-peak tempo aggregation with bank-native
    consensus (peaks emerge from adaptive frequencies)
- **2.2** Sensory + motor coupled layers (Large 2015)
  - Sensory: subcritical or critical Hopf
  - Motor: supercritical Hopf (spontaneous oscillation +
    entrainment)
  - Reciprocal connections
- **2.3** **Phase 0.3 missing-pulse validation here.** Must pass
  before declaring Phase 2 done.

### Phase 3 — Multi-layer architecture (cochlea + brainstem + cortex)

- **3.1** Cochlear layer (Lerud 2019 bidirectional BM↔OC)
  - Per-place: two complex states (z_bm, z_oc)
  - α_bm = -1 (passive, damped BM)
  - α_oc = 0 (critical, sharp OC tuning)
  - β₁ ≈ -10000, β₂ = -1, ε ≈ 0.0025 (per Lerud 2014 macaque fits)
  - Pre-cochlear middle-ear filter (Zilany-Bruce 2006)
- **3.2** Brainstem layer (Lerud 2014)
  - Single GrFNN, 99/octave, 4 octaves
  - α = 0 (critical Hopf), β₁ = 0, β₂ = -1
  - ε varied per fit (0.07 to 0.48)
- **3.3** Cortical / memory layer (US 8,930,292 + Large 2016)
  - Supercritical Hopf: α > 0 (try α = 0.002 per Large 2016,
    or α = 2 per Kim & Large 2021)
  - β₁ = -1 or -2 (mild)
  - This is the layer that hosts Phase 5 plastic connections
- **3.4** Inter-layer coupling
  - Feedforward (cochlea → brainstem → cortex): strong
  - Feedback (cortex → brainstem → cochlea): weak (per Lerud 2014
    architecture)
- **3.5** Output weighting per Lerud 2014: cochlea 25% +
  brainstem 50% + cortex 25% for FFR-style aggregation
- **3.6** **Phase 0.4 FFR validation here.** Pass criterion: R² ≥
  0.5 against Lerud 2014 Fig. 4 nonmusician data.

### Phase 4 — Voice extraction (refined)

- **4.1** Restrict integer-ratio detection to k+m ≤ 7
- **4.2** Weight cluster-membership evidence by Arnold-tongue
  width per Kim & Large 2019 Eq. 14
  - `confidence(k:m) ∝ (√ε F)^k · (√ε r_s)^(m-2)`
- **4.3** When Phase 5 is operational, **read learned W matrix
  community structure as voice clustering** (replacing or
  augmenting harmonic-stack heuristic)

### Phase 5 — Hebbian learning (multi-frequency)

- **5.1** Implement Kim & Large 2021 Eq. 26 (frequency-scaled
  GrFNN + plastic connections)
  - Per-pair connection update:
    `(1/f_ij) ċ_ij = -γ_ij c_ij + κ_ij · z_i/(1-z_i) · z̄_j/(1-z̄_j)`
  - Initial parameters from Fig. 11 sim:
    `α = 2, β₁ = β₂ = -1, γ = 0.5, κ = 0.05/(n-1)`
- **5.2** Training convergence test: feed C IV-V₇-I cadence,
  verify W stabilizes within ~35 cycles (Large 2016 Methods)
- **5.3** Three-frequency `D_ijk` connections (Large 2016 Eq. A2)
  - Captures combination frequencies (sum/difference of pairs)
  - Likely patent-safer than two-frequency rule
- **5.4** Patent-safe variant for commercial path
  - Cubic damping on c_ij (Kim & Large 2021 Eq. 9 Strategy 2)
    instead of `-γ c_ij`
  - Mathematically distinct from US 8,930,292's claim
  - Needs legal review before commercial deployment

### Phase 6 — Performance & deployment

- **6.1** Profile + optimize hot paths (current performance
  dropped from 1.12× → 0.56× realtime after recent changes)
- **6.2** Live-mode tuning for ≥ 1× realtime on target hardware
- **6.3** Output formatting (CV / MIDI / OSC for VCV bridge)

### Phase 7+ — Extensions

- **7.1** Cannon-Patel-style motor layer for non-isochronous
  meters (Balkan 2/2/3 etc.) — alternative to GrFNN motor
- **7.2** Predictive-coding (PCM) wrapper for hybrid NRT+PCM
  expectation tracking (Harding 2025 lists this as open
  research direction)
- **7.3** Cross-cultural tonality validation (Large 2016 rāgas)
- **7.4** Performance synchronization (Roman 2023 SMT effects)

---

## Sequencing & dependencies

```
1.0 (done) ─┬─→ 1.1 (done, disabled)
            ├─→ 1.2 (in progress) ──┐
            └─→ 1.3 (pending)       │
                                    ▼
                                ┌── 1.4 ★ canonical input ──→ 0.1, 0.2 validation
                                │
                                ▼
                                2.1, 2.2 (rhythm) ──→ 0.3 missing-pulse
                                │
                                ▼
                                3.1, 3.2, 3.3 (multi-layer) ──→ 0.4 FFR validation
                                │
                                ▼
                                4.1, 4.2 (voice refinements)
                                │
                                ▼
                                5.1, 5.2, 5.3 (Hebbian)
                                │
                                ▼
                                4.3 (W-based voice extraction)
                                │
                                ▼
                                6.x (performance)
```

**Critical path:** 1.2 → 1.4 → 2 → 3 → 5 → 4 (W-based)

**Phase 1.3 can be done in parallel** with 1.4 (independent
concern: numerical noise floor).

**Phase 1.5 (three-freq D_ijk)** is folded into Phase 5.3.

---

## Patent-IP map (where to be careful)

| What we implement | Patent claim risk |
|---|---|
| Plain canonical equation (Hopf + cubic) | Public domain (HI 1996a) |
| Per-osc τ scaling | Public domain (US 7,376,562 expired) |
| Internal coupling kernel D (integer-ratio resonant monomials) | Public domain (US 7,376,562 expired) |
| Canonical nonlinear input form `P(ε,x)·A(ε,z̄)` | Likely public domain (Large 2010 published; predates patent claims about *learning*) |
| Two-freq Hebbian `ċ = -γc + κ z z̄/...` | **US 8,930,292 claim 1, active to 2032** |
| Three-freq Hebbian `D_ijk` | Likely not claimed (US 8,930,292 focuses on two-osc coherency) — research opportunity |
| Cubic-damping stabilization for c_ij (Strategy 2) | Different math from US 8,930,292 — likely patent-safe |
| Network-wide tempo aggregation (specific eq.) | **US 8,583,442 claim 1, active to 2031** |
| KDE-peak tempo aggregation (alternative) | Patent-safe (different math) |
| ASHLE elasticity mechanism | PLoS open access; not patented |
| Multi-layer cochlea→brainstem→cortex | Architecture not patent-claimed (specific learning rules in combination are) |

For research / personal use: all of the above is fine.
For commercial path (~2027–2029 question per Phase 6 timing):
need legal review of US 8,930,292, US 8,583,442 against final
implementation. Patent-safe variants documented above.

---

## Decision points called out

1. **Phase 1.4 vs Phase 1.3 priority.** Recommendation: 1.4 first
   (canonical input). 1.3 is incremental polish; 1.4 unlocks
   capabilities.
2. **Phase 3 layer parameters.** Two options:
   - Lerud 2014 / Lerud 2019 (cochlea β₁=-10000)
   - US 8,930,292 / Kim & Large 2015 (cochlea β₁=-100)
   Recommendation: start with -100 (less stiff numerically),
   compare against -10000 once integrator is stable enough.
3. **Phase 5 patent path.** For research: literal equations are
   fine. For commercial: cubic-damping variant + three-freq D_ijk
   may suffice. Defer decision until commercial path is concrete.
4. **Phase 2 motor layer.** GrFNN motor (Large 2015) vs Cannon-
   Patel-inspired or Bose 2019 LIF. Recommendation: GrFNN motor
   first (matches our framework); Cannon-Patel variant available
   in Phase 7.1 if non-isochronous meter support is needed.

---

## What hasn't changed from earlier thinking

- **"Best possible accuracy first"** still drives sequencing
- **Tier A / Tier B / Tier C test harness** is still the
  performance measurement framework (Phase 0 augments it, doesn't
  replace it)
- **Patent-aware-not-patent-avoidant** posture for personal use;
  patent-safe variants documented for future commercial decisions
- **Voice extraction via W community structure** is still the
  target architecture (Phase 4 + Phase 5)
- **Subagents and parallelism** for independent literature pulls,
  testing, etc.
- **Numba JIT** is the right compilation strategy

---

## Open questions

1. **Should Phase 1.4 swap the entire input form or be a parallel
   path** (selectable via config)? Probably the latter — keep
   linear input as a fallback for debugging while validating the
   nonlinear form against Phase 0 tests.
2. **Does the canonical input form change our integration
   stability?** The `1/(1-√εx)` term has a singularity at
   `x = 1/√ε`. We currently constrain `|z| < 1/√ε`. Need to also
   constrain `|x| < 1/√ε` — this becomes a hard input-amplitude
   clip.
3. **Phase 3 layer sizes — keep 36/osc/octave or go to Lerud
   2014's 99/octave?** Compute-cost vs fidelity tradeoff.
   Recommendation: 36/octave for cochlea+brainstem; smaller bank
   for cortical/memory layer (the Large 2016 simulation used only
   25 oscillators).
4. **When does external positioning matter?** If we need a
   one-sheet / blog post / demo for fundraising or external
   collaboration, Harding 2025 is the right canonical NRT citation.
   We don't have to produce that material now, but it's available
   when needed.
