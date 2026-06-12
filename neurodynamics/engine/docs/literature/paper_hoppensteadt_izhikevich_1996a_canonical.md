# Hoppensteadt & Izhikevich (1996a) — Synaptic organizations and dynamical properties of weakly connected neural oscillators. I. Analysis of a canonical model

**Reference:** Biological Cybernetics 75, 117–127
**DOI:** 10.1007/s004220050279

---

## Why this matters for factory-dame

The **mathematical derivation** that produces the canonical model
all of NRT builds on. Paper I of the two-part series; Paper II
(1996b) builds learning on top. This is where the
`z' = bz + dz|z|² + Σ c_ij z_j` form **first appears** and is
proven to be a universal canonical form for weakly-coupled neural
oscillators near multiple Andronov-Hopf bifurcations.

For factory-dame: read this when reasoning about *why* the
canonical equation has the form it does. Especially for the
"oscillator death" and "self-ignition" results (Section 5), which
explain why our subcritical / critical / supercritical α choices
have different qualitative behaviors.

## Setup

Start from a general weakly-coupled neural network (Eq. 1):

```
ẋ_i = f_i(x_i, y_i, λ) + ε p_i(x_1, y_1, …, x_n, y_n, ε)
ẏ_i = g_i(x_i, y_i, λ) + ε q_i(x_1, y_1, …, x_n, y_n, ε)
```

- `x_i` excitatory, `y_i` inhibitory activities of *i*-th neural
  oscillator
- `λ` bifurcation parameter
- `ε` small (weak coupling — synaptic weights ~1% of action
  potential)

## Andronov-Hopf bifurcation condition

For each oscillator to contribute non-trivially, each must be near
a non-degenerate Andronov-Hopf bifurcation:

```
tr L_i = a_1 + a_4 = 0
det L_i = a_1 a_4 - a_2 a_3 > 0
```

(where `L_i` is the Jacobian of the uncoupled system.) Then the
natural frequency is:

```
Ω_i = √(a_{i1} a_{i4} - a_{i2} a_{i3})
```

## The canonical model (Theorem 1, Eq. 8)

**Main theorem:** Near multiple Andronov-Hopf bifurcation, any
weakly-coupled neural network is equivalent (via invertible change
of variables Eq. 7) to:

```
z_i' = b_i z_i + d_i z_i |z_i|² + Σ_{j≠i, Ω_i=Ω_j} c_ij z_j + O(√ε)
```

(`' = d/dτ` where `τ = εt` is "slow time")

- `b_i = ρ_i + iω_i` — bifurcation parameter + frequency detuning
- `d_i = α_i + iβ_i` — nonlinear stabilization
- `c_ij ∈ ℂ` — complex synaptic connections

**This is the equation Large 2010 extends.** Adding higher-order
terms `εβ_2|z|⁴/(1-ε|z|²)` and multi-frequency resonant monomials
produces the GrFNN canonical form.

## Synaptic-coefficient formula (Eq. 9)

The complex `c_ij` are computed from real synaptic strengths:

```
c_ij = (1/2) · (1 + i·a_{i4}/Ω, -i·a_{i2}/Ω) · S_ij · (1, (a_{j4}+iΩ)/a_{j2})ᵀ
```

with `S_ij = ∂(p_i, q_i)/∂(x_j, y_j)` the rescaled synaptic Jacobian.

**Implication:** the complex `c_ij` aren't arbitrary — they come
from specific combinations of excitatory-excitatory, excitatory-
inhibitory, etc. synapses. The *anatomy* determines the *complex
weight pattern*. This is why some architectures can encode phase
and others cannot (see HI 1996b).

## Corollary 1 — frequency pools

> "All neural oscillators can be divided into groups, or pools,
> according to their natural frequencies Ω_i. Oscillators from
> different pools have different natural frequencies and
> interactions between them are negligible."

In the canonical model, oscillators only interact when their
**natural frequencies match** (or are ε-close). Synaptic
connections between oscillators at different frequencies are
"functionally insignificant" — their c_ij becomes zero in the
canonical form.

**Implication for factory-dame:** in the *plain* canonical model,
oscillators don't talk across frequencies. To get multi-frequency
resonance (1:2, 2:3, etc.), you need **higher-order terms** —
which is exactly what Large 2010 adds via the resonant-monomial
expansion. Without those, our pitch bank is just N independent
oscillators.

## Oscillator death (Theorem 2 / Section 5)

> "Even though each oscillator is a pacemaker (α > 0), the coupled
> system may approach z = 0."

For network of identical oscillators (Eq. 14):

```
z_i' = (ρ + iω) z_i + d z_i |z_i|² + Σ_{j=1}^n c_ij z_j
```

Equilibrium z=0 is stable if `ρ < -α` (where `α = max{Re λ_i(C)}`,
largest real eigenvalue of connection matrix C).

**Two regimes:**

- **If α < 0** (network damping): equilibrium is stable for
  `0 < ρ < -α` — i.e., **passive coupling can quench active
  oscillators**. Each oscillator alone would oscillate (ρ > 0),
  but the network damps them out.
- **If α > 0** (network amplification): network exhibits
  spontaneous activity for `-α < ρ < 0` — i.e., **active coupling
  can ignite passive oscillators**. Each oscillator alone is
  silent (ρ < 0), but the network produces synchronized activity.

**Implication for factory-dame:** Phase 3's layered architecture
must consider both regimes. The cortex layer at α > 0 + strong
coupling can ignite spontaneous patterns (which is how memory
patterns appear). The brainstem layer at α ≈ 0 with damped coupling
is stably-driven.

## Type A vs Type B oscillators

Based on the sign pattern of the Jacobian (Section 6). Type A is
"smarter" — has the standard `(+ −; + −)` sign pattern (excitatory
self-excites, inhibitory self-inhibits) and can reproduce the
entire range of natural phase differences via appropriate synaptic
connections.

Type B (`(− −; + +)` pattern) is more limited — cannot achieve
ψ_ij = 0, only ψ_ij near `Arg v_1` or `Arg(−v_4)`.

**Implication:** if Phase 5's learned-memory layer requires
arbitrary phase relationships (it does — to memorize different
chord voicings, etc.), the underlying oscillator must be Type A.
Most biophysical neuron models are. Our abstract Hopf-form
oscillator is Type A by construction.

## Implications for factory-dame

### Why our canonical equation is what it is

This paper proves the canonical form is *unique* (up to change of
variables) for weakly-coupled networks near Andronov-Hopf
bifurcation. **We're not free to choose a different equation** —
the math forces this one. Large 2010 extends it to handle
multi-frequency resonance, but the cubic term and the complex
coupling matrix come from this paper.

### Frequency-pool principle limits 1:1 dynamics

If we run our pitch bank with only the basic canonical form, the
220 Hz oscillator never talks to the 440 Hz oscillator —
different natural frequencies = no functional coupling. To get
2:1 interaction (an octave), we need the higher-order resonant
monomials Large 2010 adds.

This is why our Phase 1.1 internal coupling kernel D — implemented
correctly — gave near-zero effects at typical amplitudes. The
linear-coupling term (`z_j` for 1:1) only works for same-frequency
oscillators. Cross-frequency coupling needs the `z_j^k z̄_i^(m-1)`
expansion, and those terms have ε^((k+m-2)/2) attenuation. At low
ε (= near-linear regime), they're negligible — which matches what
we observed.

### Strong-amplitude regime is where the action is

The multi-frequency resonance (and hence missing-fundamental,
phantom-pulse, voice clustering) is **inherently a strong-coupling,
high-ε phenomenon.** Our current config sits in the weak regime
(ε = 1.0 nominal but effective ε small due to amp scaling). For
voice extraction to work robustly, we need to ensure oscillators
reach amplitudes where the resonant expansion matters.

**Practical implication:** Phase 3 layered architecture must run
each layer in the right amplitude regime. Cochlear layer (strong
input) reaches high |z|; cortical layer (Hebbian-learned coupling)
needs to *amplify* this so that multi-frequency coupling has
effect. This is one motivation for the α-supercritical cortical
layer per US 8,930,292.

### Oscillator-death prevents runaway oscillation

The amplitude `|z| < 1/√ε` constraint in the canonical model
(Large 2010) is the oscillator-death mechanism in disguise.
Strongly-coupled networks of pacemakers self-quench — preventing
unbounded growth. Our code's saturation in the cubic+quintic
nonlinearity reproduces this naturally.

## Cross-references

- **Paper II (HI 1996b)**: learning extends this canonical form.
  Eq. 14 + a learning rule on c_ij = the Hebbian memory machinery.
- **Large 2010**: extends Eq. 8 with higher-order terms.
- **Large 2016 tonality**: the `ε^((k+m-2)/2)` stability formula
  is a direct consequence of the resonant-monomial expansion of
  the canonical form derived here.
