# Phase-fingerprint protocol: decision tree (SU(2), parity-only pilot)

**Goal.** Bound Var_ν(q), the spread of the P-odd density q across the pure phases of the mixture, using ordinary lattice Monte Carlo as a proxy. The mixture structure, a measure ν over the phase space X, comes from the 10 November 2026 assessment of the OpenAI Yang-Mills Lean construction. We have not verified it independently. Whatever comes out, the result is **guidance, not proof**.

**Signal.** Every measurement below estimates the same quantity in a different way:

1. **Two-point plateau.** As r → ∞, C(r) = ⟨q(0)q(r)⟩ tends to Var_ν. This holds if the vacuum generates the zero-energy subspace, which is an inference from the assessment.
2. **Block scaling.** The fit has the form Var(q̄_ℓ) = Var_ν + χ_ℓ/ℓ⁴ + surface terms.
3. **Peak shape.** Two peaks at ±m that sharpen with volume push the Binder cumulant U toward 2/3. A single Gaussian peak gives U → 0.

## Decision tree

1. **Check sampling first.** If any frozen-topology warning fires (Q stuck in one sector, τ_int(Q) comparable to the chain length, streams sitting in different sectors), treat every peak and plateau as suspect. Fix it with more seeds and longer chains at coarser β before you interpret anything.
2. **Look at the plateau and the scaling.** If C(r) shows no plateau, the fitted Var_ν is consistent with 0 (report the 95% upper bound relative to χ/V at the largest volume), Binder(Q/V) → 0, and ℓ⁴·Var is flat near χ, so the slope against ℓ⁴ is about 0:
   - **Outcome:** a null result. This **supports pursuing the one-phase theorem in Lean**, meaning X is a single point for this order parameter. It is consistent with the lore that parity is unbroken at θ = 0. It is still only an upper bound and cannot see the proof's subsequence.
3. **Positive plateau or slope.** Then run the frozen-topology discriminator:
   - **The peaks of Q/V shrink like 1/V across volumes** (log-log slope ≈ −1), or the plateau tracks ⟨Q²⟩/V²: the signal is frozen sectors. Treat it as a **fake mixture**, fix the sampling, and go back to step 1.
   - **The peaks stay at ±m independent of V**, sharpen with L, survive several seeds, survive a second flow radius, survive a coarser β with healthy tunnelling, and ideally survive an open-boundary cross-check: this is a **persistent plateau**. Treat it as a candidate order parameter, not a discovery. Look hardest at artifacts first.
4. **A persistent plateau tells you which symmetry to formalize first.** In SU(2) the only candidate is **parity**: define P on the proof's labels, show the mixed Schwinger functions are P-invariant, and study P acting on X. In SU(3), which plateau survives decides the order:
   - a P-odd density (q) points to P first;
   - a C-odd density (Im tr of small Wilson loops) points to C first;
   - both point to CP-type structure.
5. **Inconclusive** cases, such as slopes between −1 and 0, low effective sample size, or non-integer Q: go to larger volumes and longer flow, and leave the Lean plan unchanged.

## Standing caveats

- A finite periodic volume never breaks symmetry. Any genuine breaking shows up only as peaks that sharpen as L grows.
- Frozen topology fakes clusters. The discriminator is peaks at ±Q/V that shrink like 1/V against peaks at ±m that stay put.
- The pilot cannot target the proof's odd-ratio spacing subsequence (a_N = 1/∏b_k, b_k ≥ 3 odd), its RG-matched couplings, or its compactness subsequence.
- The flowed-field observables are not the proof's renormalized lattice polynomials. We expect, without having shown it, that the two agree on whether a vacuum expectation value vanishes, not on its value.
- Dip and BIC p-values assume independent samples. Blocks and successive configurations are correlated.
