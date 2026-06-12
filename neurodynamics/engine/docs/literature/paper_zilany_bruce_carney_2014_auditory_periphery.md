# Zilany, Bruce, Carney (2014) — Updated parameters and expanded simulation options for a model of the auditory periphery

**Reference:** Journal of the Acoustical Society of America 135(1), 283–286
**DOI:** 10.1121/1.4837815

---

## Why this matters for factory-dame

This is **the standard auditory-nerve front-end** used throughout
computational hearing research. Lerud 2014 references the
Zilany-Bruce 2006 middle-ear filter (predecessor of this 2014
version). For Phase 3's cochlear layer, we have two choices:

1. **Use the Zilany-Bruce-Carney model as the cochlear front-end**,
   then drive a downstream GrFNN brainstem layer with its output.
   Advantage: physiologically accurate, well-validated, MATLAB code
   public. Disadvantage: a heavy preprocessing stage that's not
   part of our oscillator framework.
2. **Use a GrFNN cochlear layer** (per Lerud 2019, bidirectional
   BM↔OC). Advantage: stays in our oscillator framework. Disadvantage:
   less physiologically validated than ZBC.

Worth pulling because option (1) might be the right call for any
fidelity-critical work.

## What the model computes

The ZBC model is a **multi-stage phenomenological model of the cat
auditory periphery** (with humanized parameter options):

```
acoustic input
    ↓
[middle ear filter]
    ↓
[basilar membrane: nonlinear C1 + linear C2 filters, level-dependent
                    compression, suppression]
    ↓
[inner hair cell (IHC) transduction]
    ↓
[IHC-AN synapse: three-store diffusion + power-law adaptation]
    ↓
[spike generator: refractoriness + variance]
    ↓
auditory-nerve spike trains
```

## Key equations (this 2014 update)

**Power-law adaptation** at the IHC-AN synapse (Eq. 1):

```
I(t) = α ∫₀ᵗ r(t') / (t - t' + β) dt'
     = α · r(t) ∗ f(t)
     where f(t) = 1 / (t + β)
```

Models long-term adaptation with power-law memory (not exponential).
`α` controls adaptation amount; `β` is time-scale.

In this 2014 version, the power-law `α` was readjusted:
- Slow path α: 5×10⁻⁶ → 2.5×10⁻⁶
- `A_ss = 800 × (1 + CF/10³)` (CF-dependent saturation rate)

**Mean discharge rate** (Eq. 2 — accounts for refractoriness):

```
R(t) = S_out(t) / (1 + τ · S_out(t))
```

**Variance of discharge rate** (Eq. 3):

```
σ²(t) = S_out(t) / (1 + τ · S_out(t))³
```

With **absolute refractory period τ = 0.75 ms**.

## What this 2014 update fixes vs 2009

- High-CF fiber saturation rates better match Liberman 1978 data
- Low-frequency-tone responses of high-CF fibers no longer
  erroneously inflated
- Refractoriness explicitly modeled (analytical mean + variance)
- Optional fixed-seed fractional Gaussian noise for reproducibility

## Code availability

Source: http://www.urmc.rochester.edu/labs/Carney-Lab/publications

The model is implemented in MATLAB and C. Python ports exist:
https://github.com/tbekolay/auditory_nerve (Python wrapper).

This is **widely used** — appears in many computational-hearing
papers as the "standard front-end." Switching costs to a
GrFNN-only cochlear model are real if we're aiming for
physiological fidelity.

## Implications for factory-dame

### Decision: ZBC front-end vs GrFNN cochlear layer

Trade-offs:

| Criterion | ZBC front-end | GrFNN cochlea (Lerud 2019) |
|---|---|---|
| Physiological accuracy | Very high | High (fits 496 macaque curves) |
| Computational cost | Moderate (multi-stage) | Lower (oscillator-only) |
| Real-time capability | Demonstrated | Demonstrated |
| Integrates with our codebase | Foreign code | Native (just more oscillators) |
| Provides spike trains | Yes | No (continuous z) |
| Provides FFR analog | Indirect (sum spikes) | Direct (oscillator amplitude) |
| Patent-IP | Open | Open (Lerud 2019 in PD-able journal) |
| Validation against psychoacoustics | Extensive | More limited |

**Recommendation:** Phase 3 prototype both:
- Phase 3a: GrFNN cochlea (Lerud 2019) — fastest path, stays in
  framework
- Phase 3b: ZBC front-end (if Phase 3a underperforms on real audio)
  — fallback for fidelity

We don't have to commit at design time; let test results drive it.

### Output type mismatch

ZBC outputs spike trains; GrFNN works on continuous z. To couple
ZBC → GrFNN brainstem layer, we'd need:
- Aggregate spikes over short windows to get firing-rate continuous
  signal, OR
- Drive GrFNN with the (continuous) synapse output before the spike
  generator stage

The latter is cleaner. The ZBC model's `S_out(t)` (synapse output)
is the natural connection point.

### Pre-cochlea: middle-ear filter

Lerud 2014's Appendix B mentions the middle-ear filter (MEF) from
Zilany-Bruce 2006. This is part of the ZBC pipeline. Even if we
use GrFNN cochlea (not full ZBC), **we should add the MEF as a
preprocessing step before our cochlear layer.** This is what gives
us the canonical "auditory filter" frequency response. Easy win
for physiological accuracy.

### Reproducibility note

ZBC 2014 introduced the **fixed-seed fractional Gaussian noise**
option. Useful for our Phase 1.3 (Gaussian noise) — adopt the same
approach (fGn rather than white noise, fixed seed for repeatable
testing).

## TODO

- [ ] Pull the C/MATLAB ZBC implementation, profile its overhead
- [ ] Compare ZBC `S_out` against GrFNN cochlea `|z_oc|²` on the
      same tone stimuli — see how different the representations are
- [ ] Add a middle-ear filter (linear, ~1–4 kHz peak) as a
      preprocessing stage before our cochlear layer regardless of
      which option we choose
