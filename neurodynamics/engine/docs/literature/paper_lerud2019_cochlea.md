# Lerud, Kim, Almonte, Carney, Large (2019) — A canonical oscillator model of cochlear dynamics

**Reference:** Hearing Research 380, 100–107
**DOI:** 10.1016/j.heares.2019.06.001
**PMC:** [PMC6669083](https://pmc.ncbi.nlm.nih.gov/articles/PMC6669083/)

---

## Why this matters for factory-dame

This is the **dedicated cochlear-stage GrFNN paper** — replacing
the Appendix B of Lerud 2014 with a more careful treatment. Phase
3's cochlear layer should target this paper's formulation.

The model adds **bidirectional coupling** between the basilar
membrane and organ of Corti (vs Lerud 2014's unidirectional BM→OC).
This better matches biophysics: the OC's electromotile cochlear
amplifier feeds back onto the BM.

## Unidirectional coupling (Eq. 4)

```
ż_bm = z_bm (α_bm + i·2πf) + F·e^(i·2π f_0 t)

ż_oc = z_oc (α_oc + i·2πf + (β + iδ)|z_oc|²) + c_21 · z_bm
```

- `z_bm` — basilar membrane state (linear)
- `z_oc` — organ-of-Corti state (active nonlinear)
- `α_bm < 0` — BM is passive, damped
- `α_oc = 0` — OC critical (sharp tuning)
- `c_21` — BM-to-OC coupling

> "We assumed a linear BM, thus there was no nonlinear damping
> parameter β for the BM."
>
> "For the OC we assumed critical nonlinear oscillation, i.e.,
> α_oc = 0."

## Bidirectional coupling (Eq. 9)

```
ż_bm = z_bm (α_bm + i·2πf) + F·e^(i·2π f_0 t) + c_12 · z_oc

ż_oc = z_oc (α_oc + i·2πf + (β + iδ)|z_oc|²) + c_21 · z_bm
```

Adds `c_12 · z_oc` to the BM equation — the OC's amplification
feeds back. This is the **canonical cochlea model.**

## Parameter fitting methodology

> "Parameters were determined by fitting 496 macaque auditory-nerve
> tuning curves individually. The resulting parameter values varied
> widely, even for curves that had nearby CFs."

The fitting compares three tuning-curve properties:

1. **Quality factor Q_ERB**: `CF / ERB`
2. **Tip-to-tail difference**: dB drop from peak to tail-region
3. **Tip height**: threshold dB SPL at CF

A closed-form expression for threshold-forcing is given as
**Eq. 7** (unidirectional) and **Eq. 10** (bidirectional).

The varying-per-CF parameters were "smoothed and interpolated" to
produce parameters for the computational model.

## Limitations stated

> "This model does not include longitudinal coupling between
> oscillator systems and so does not capture the BM traveling wave."

For factory-dame purposes: we don't need traveling-wave physics.
Our cochlea layer just needs frequency-selective filtering with
sharp tuning. Bidirectional BM↔OC at each CF gives that without
needing longitudinal wave propagation.

## Implementation note

> "The resulting model is implemented in MATLAB, and is publicly
> available on GitHub (Lerud et al., 2018)."

Repository: https://github.com/MusicDynamicsLab/GrFNNCochlea —
reference implementation.

## Access status

PMC version pulled successfully. Specific numerical parameters
(values for α_bm, β, δ, c_12, c_21, F thresholds) require deeper
PDF extraction; this summary captures the structural equations
and the methodology.

## Implications for factory-dame

### Phase 3 cochlear layer

**Use the bidirectional form (Eq. 9), not Lerud 2014's
unidirectional Appendix-B form.** Two state variables per cochlear
place: BM (passive, linear) and OC (active, critical-Hopf).

Per-CF parameters can be:
- α_bm = -1 (heavy damping, from Lerud 2014)
- α_oc = 0 (critical, from this paper)
- β₁_oc strong negative (matched to tuning data; values vary per
  CF — Lerud 2014 reports β₁ = -10000 for the macaque-fitted
  composite)
- β₂_oc = -1
- ε small (Lerud 2014 reports ε = 0.0025)
- c_12, c_21 are tuning parameters — start with both ~ 0.5–1.0 and
  fit against synthetic tuning curves

### Multi-state oscillator vs single-state

Our current GrFNN engine treats each oscillator as a single
complex z. The cochlear stage needs **two coupled complex z's per
CF** (BM and OC). This is a structural difference — Phase 3 should
either:
- (a) Treat cochlea layer as 2N oscillators with internal pairing
- (b) Add a dedicated cochlear-cell data structure with (z_bm, z_oc)
  pair

(b) is cleaner; (a) reuses existing code paths.

### Pre-cochlea preprocessing

Lerud 2014 included a middle-ear filter (MEF, Zilany & Bruce 2006)
+ Hilbert + half-rectification before driving the cochlea. We
should add this when implementing the cochlear layer — without it,
the cochlear model receives unfiltered raw audio, which doesn't
match physiological calibration.

### Open-source MATLAB reference

The GrFNNCochlea repo on GitHub is a reference implementation.
Worth pulling and porting parameter values during Phase 3 work.

## TODO

- [ ] Extract specific α_bm, β, δ, c_12, c_21 values from full PDF
      (Tables/Figures of the paper)
- [ ] Pull GrFNNCochlea MATLAB code and port parameter tables
- [ ] Compare Eq. 9 bidirectional behaviour against unidirectional
      Eq. 4 — see if bidirectional is necessary for our pitch
      tracking goals or if unidirectional suffices
