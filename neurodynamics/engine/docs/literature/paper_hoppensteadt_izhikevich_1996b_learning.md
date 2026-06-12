# Hoppensteadt & Izhikevich (1996b) — Synaptic organizations and dynamical properties of weakly connected neural oscillators. II. Learning phase information

**Reference:** Biological Cybernetics 75, 129–135
**DOI:** 10.1007/s004220050281

---

## Why this matters for factory-dame

The **mathematical foundation** Large 2010, Kim 2015/2019/2021, and
Large 2016 all build on. This is paper II of a two-part series:

- **Paper I** (BC 75:117–127, 1996a) — derives the canonical model
  for weakly-coupled oscillators near a multiple Andronov-Hopf
  bifurcation. Still needs pull.
- **Paper II** (this paper) — analyzes learning rules on that
  canonical model. Proves what synaptic architectures can memorize
  phase differences. **The Hebbian rule used by Large 2016 is a
  direct consequence of Eq. 11 here.**

## The canonical model (Eq. 3)

```
z_i' = (ρ_i + iω_i) z_i - z_i |z_i|² + Σ_{j=1}^n c_ij z_j   ,   i = 1, …, n
```

This is Eq. 3 in the rescaled form (after `z_i → z_i / √|d_i|`).
Note: `' = d/dτ` where `τ = εt` is "slow time" — learning happens
on slow time.

- `ρ_i + iω_i` — bifurcation parameter + natural frequency
- `-z_i |z_i|²` — cubic stabilizing nonlinearity (Hopf normal form)
- `c_ij ∈ ℂ` — synaptic connection (complex; encodes amplitude AND
  relative phase of the connection)

**This is the simplest possible Hopf-bifurcation canonical model.**
Everything in NRT (Large 2010 onward) extends this — adding higher-
order terms, multi-frequency resonance, etc. But the **memory
machinery is here**, in 1996.

## Hebbian learning rule (Eq. 10, polar form Eq. 14)

The "Hebbian synaptic modification rule" after rescaling:

```
s' = -γ s + θ x y + O(√ε)        (Eq. 10, scalar form)

c_ij' = -γ c_ij + k_ij z_i z̄_j  (Eq. 14, complex Hebbian)
```

- `γ` — fading rate ("in the absence of either pre- or postsynaptic
  activity, the synapse weakens (forgets)")
- `k_ij > 0` — Hebbian learning rate
- `z_i z̄_j` — coactivity product. **The complex conjugate `z̄_j`
  is what makes this lock to phase differences.**

In polar coordinates `c_ij = |c_ij| e^{iψ_ij}`, `z_i = r_i e^{iφ_i}`:

```
|c_ij|' = -γ|c_ij| + k_ij r_i r_j cos(φ_i - φ_j - ψ_ij)
ψ_ij'  = (1/|c_ij|) k_ij r_i r_j sin(φ_i - φ_j - ψ_ij)
```

The phase equation drives `ψ_ij → φ_i - φ_j (mod 2π)` — the
connection learns the *phase difference* between the oscillators.

## Memory theorem (Theorem 1)

> "A weakly connected network of neural oscillators can memorize
> phase differences if and only if the plasticity rates satisfy
> `θ_ij1 = θ_ij3, θ_ij2 = θ_ij4` AND `k_ij > 0`."

The condition `k_ij > 0` (Eq. 16) is the **load-bearing requirement**.
It says: for memory to form, the relationship between synaptic
plasticity at different synapses on the same oscillator must be
balanced.

## Lemma 1 — fully general form (Eq. 11)

The most general Hebbian-style rule consistent with the assumptions:

```
c_ij' = -γ c_ij + k_ij2 z_i z̄_j + k_ij3 z̄_i z_j
```

The two coactivity terms `z_i z̄_j` and `z̄_i z_j` differ in which
oscillator's phase they track. For memory, Theorem 1 shows we need
`k_ij2 > 0` AND `k_ij3 = 0` (Eq. 13). The asymmetry is what makes
the synapse encode "i fires after j" vs "i fires before j"
distinctly.

## Synaptic-organization analysis (Section 5, Fig. 1)

Four neural architectures classified by what they can memorize:

| Architecture | Description | Can memorize? |
|---|---|---|
| Fig. 1a | No bidirectional connections | **No** (Corollary 3) |
| Fig. 1b | Type-A oscillators only | Yes, with caveats (Corollary 4) |
| Fig. 1c | Long-axon excitatory + short-axon inhibitory | **Yes** (Corollary 5) |
| Fig. 1d | Long-axon excitatory + long-axon inhibitory | **Yes**, can also unlearn |

**Implication: not every connectivity pattern supports learning.**
The cortical-pyramidal-cell + local-inhibitory-interneuron pattern
(Fig. 1c) is what biology uses. Phase 5 architectures need to
respect this — purely all-to-all connections aren't sufficient
even when the rule is correct.

## Synchronization theorem (Theorem 2, Cohen-Grossberg)

> "If C = (c_ij) is self-adjoint, i.e., c_ij = c̄_ji, then the
> neural network dynamics converges to a limit cycle."

`U(z, z̄) = -Σ ρ_i|z_i|² + (1/2)|z|⁴ - Σ c̄_ij z̄_i z_j` is a
**global Lyapunov function** (Eq. 17 in appendix). The network is
guaranteed to reach a memorized fixed-point pattern.

**Phase 5 implication:** if our learned W is forced symmetric
(`W_ij = W̄_ji`), the network has guaranteed convergence. If not,
chaotic dynamics are possible. Worth enforcing symmetry initially.

## Implications for factory-dame

### Why the Hebbian rule has complex weights

Real synapses are scalar; why do we use complex `c_ij`? Because in
the canonical model (Eq. 3), connections appear as `c_ij z_j` where
`z_j` is complex. The phase of `c_ij` rotates `z_j` to align it
with the receiving oscillator's preferred phase. This is what makes
**phase-difference memory** work. A real-valued weight can encode
amplitude correlations but cannot encode phase relationships.

For factory-dame: Phase 5 weights must be complex. Two real numbers
per pair (amplitude + phase) — this is what voice extraction needs
anyway (which partials are at phase-coherent ratios).

### The forgetting parameter γ

`γ > 0` makes synapses decay to zero in absence of coactivity.
This is the **forgetting** parameter. Memory has finite duration
on time scale `1/γ`. For factory-dame:
- Long γ (small γ⁻¹) — synapses forget quickly. Good for fast voice
  re-formation when source identity changes.
- Short γ (large γ⁻¹) — synapses persist. Good for stable tonal
  hierarchy formation (Large 2016 used `λ > 0` instead — supercritical
  for memory persistence).

Phase 5 likely wants γ tuned per layer: cortical (slow forgetting,
persistent memory) vs sensory (fast forgetting, responsive).

### What's NOT in this paper (and why)

- Multi-frequency resonance (covered in Large 2010 / Kim 2021).
  This 1996 canonical model only does 1:1 mode-locking. To get
  integer-ratio resonance, you need the higher-order terms Large
  2010 adds.
- Specific parameter values — this is a pure mathematical analysis.
- Cochlear / auditory specifics — pure abstract theory.

### Slow time vs fast time

`τ = εt` — learning is slow compared to oscillator activity. In
factory-dame's frame:
- Oscillator z dynamics: ~Hz to kHz timescales
- Learning c_ij dynamics: ~seconds to minutes
- Forgetting: should be longer than typical "voice duration" but
  shorter than "session duration"

Concrete: at 16 kHz sample rate, oscillator updates per sample.
Hebbian updates should be downsampled — e.g., updated every 10 ms
or every 100 samples. This dramatically reduces compute and matches
the "slow time" assumption.

## TODO

- [ ] Pull paper I of the series (Hoppensteadt & Izhikevich 1996a,
      "Analysis of a canonical model" Biol Cybern 75:117-127) for
      the canonical-model derivation
- [ ] Verify the multi-frequency-resonance extension (Large 2010
      adds higher-order terms to this canonical Eq. 3 — check the
      math chain)
- [ ] Decide γ value for Phase 5 prototype (start with γ ~ 1/T_voice
      where T_voice is typical voice duration ~ few seconds)
