# yangmills-fingerprint-pilot

SU(2) Wilson lattice gauge theory pilot for the Yang-Mills **phase-fingerprint** protocol: gradient-flowed topological charge density, block averages, dip test, Binder cumulant, and q-q two-point plateau search.

Pure Python 3 + numpy + scipy, CPU only. No GPU frameworks, no compiled extensions.

## Context

The 10 November 2026 assessment of the OpenAI Yang-Mills Lean construction describes the limiting theory as a mixture: a probability measure ν over a compact phase space X, with each point ξ ∈ X a pure phase that has its own unique vacuum. The assessment leaves open whether X is a single point. This repo takes that description as its premise. We have **not** verified it independently, and nothing here claims the Millennium problem is solved.

The Lean objects are non-computable existence statements, so you can't run anything on them directly. This pilot is a numerical proxy. It asks whether ordinary lattice Yang-Mills shows any sign that an odd order parameter has different values in different phases.

For an odd field such as the topological charge density q, the protocol defines

    Var_ν(q) = ∫ ⟨q⟩_ξ² dν(ξ) − (∫ ⟨q⟩_ξ dν)²

The mixed theory is P-invariant, which is an inference, so the second term vanishes. Var_ν(q) > 0 would mean phases carrying ⟨q⟩ of opposite sign that cancel in the mixture. Var_ν(q) is also the large-separation limit of ⟨q(0) q(r)⟩.

### Why SU(2), and why only parity

In SU(2) every gauge-invariant trace is real, so charge conjugation acts trivially and there is no C-odd order parameter. The pilot therefore tests **parity** only, through q(x), which is P-odd and T-odd. SU(2) at β ≈ 2.2–2.6 is used as a **rough** analogue of the SU(3) range β ≈ 5.9–6.4. The analogy is in lattice spacing only, and it is approximate.

## What it does

| Step | Implementation |
|---|---|
| Action | Wilson plaquette action `S = β Σ_p (1 − ½ Re tr U_p)` on a periodic 4D torus. Links are SU(2) stored as unit quaternions `U = a0 + i a·σ` (2×2 unitary, det 1). |
| Updates | Kennedy–Pendleton heatbath plus optional overrelaxation (`U → V† U† V†`). Updates are checkerboard per direction, so same-parity links never share a staple. |
| Smoothing | Wilson gradient flow using Lüscher's RK3 integrator. Flow time is set by the smoothing radius `√(8t)` in lattice units. Pass one value, or `beta:radius` pairs to hold a physical radius fixed across β. |
| q(x) | Clover F_μν = −(i/8)(Q_μν − Q_μν†), with `q = (1/32π²) ε_μνρσ tr(F_μν F_ρσ)`. After enough flow, Q = Σ_x q(x) lands near an integer. |
| Blocks | Averages of q over ℓ⁴ hypercubes, with ℓ dividing L. Variance is split into within-config block spread and between-config spread of Q/V. |
| Diagnostics | Histograms written as CSV plus ASCII bars in the summary. Hartigan dip statistic, a pure-Python port of AS 217 / R `diptest`, with a simulated uniform-null p-value. 1- vs 2-component Gaussian mixture by EM, scored with BIC. Binder cumulant `U = 1 − ⟨m⁴⟩/(3⟨m²⟩²)`. |
| Two-point | `C(r) = ⟨q(0) q(r)⟩` on the torus via FFT, binned radially and axially with jackknife errors. Includes a sign check at r ≥ r_min, a large-r plateau estimate, and fits of `A e^{−mr}` and `A e^{−mr} + c`. |
| Scaling | Weighted fit of `Var(q̄_v) = Var_ν + χ/v (+ c/v^{5/4})` with v = ℓ⁴, using the full volume as the last point. A config bootstrap gives a one-sided 95% upper bound on Var_ν. Output includes a table of v·Var against v. |
| Topology | τ_int(Q) with automatic windowing, sectors visited and sector changes per stream, frozen-stream warnings, an integer-ness check, and a cross-volume 1/V-vs-constant peak test. |

## Install and run

```bash
python3 -m pip install -r requirements.txt     # numpy, scipy only
python3 -m fingerprint.selftest                 # correctness checks (~15 s; --quick skips the 8^4 Q check)
bash run_smoke_test.sh                          # 4^4 and 6^4, a handful of configs, ~5 s
make test                                       # selftest --quick + smoke (if make is installed)
```

A pilot run looks like this. Pick sizes and statistics to fit your laptop. 16⁴ is roughly 16× the cost of 8⁴ per sweep.

```bash
python3 -m fingerprint \
  --sizes 8 12 16 --betas 2.3 2.4 2.5 \
  --n-therm 200 --n-configs 200 --n-sweeps 10 --n-or 3 \
  --flow-radius 2.3:R1 2.4:R2 2.5:R3 \
  --block-sizes 1 2 4 --seeds 1 2 3 4 \
  --outdir pilot_output
```

`--flow-radius` is given in lattice units. To hold a **physical** radius fixed, for example the protocol's ≈ 0.2–0.3 fm, scale it with a measured scale such as √t₀. Each ensemble writes `flow_history.csv` containing t²E(t), which you can use to find t₀. Choose your own reference value and convention. SU(2) conventions vary, so the pilot does not hard-code one.

Run `python3 -m fingerprint --help` for every option, including `--start hot|cold`, `--flow-eps`, `--dip-sims`, `--n-boot`, `--surface-term`, `--twopoint-rmin`, and `--save-q-fields`.

### Outputs

`<outdir>/summary.txt` has a cross-ensemble table, the cross-volume frozen-topology check, and all warnings. `<outdir>/run_config.json` records the run settings. Each ensemble directory `<outdir>/L<dims>_b<beta>/` contains:

- `summary.txt`: plaquette, χ, the Var_ν fit and 95% bound, Binder, dip, BIC, plateau verdict, and frozen-topology warnings
- `config_measurements.csv`: per config, the seed, plaquette, Q, round(Q), Q/V, flowed E, and t²E
- `measurements.npz`: all raw arrays, including block averages per ℓ and per-config C(r)
- `block_averages_ell<ℓ>.csv`, `hist_Q_over_V.csv`, `hist_qbar_ell<ℓ>.csv`
- `block_scaling.csv`: v, ℓ, Var, error, and v·Var
- `twopoint_radial.csv`, `twopoint_axial.csv`, `flow_history.csv`, `results.json`

Plots are optional and aren't a dependency. With matplotlib installed, plot `v_times_var` against `block_volume_v` from `block_scaling.csv`, and `C` against `r` from `twopoint_radial.csv`.

χ is reported in lattice units. The literature value χ^{1/4}/√σ ≈ 0.48 for SU(2) (continuum, Lucini & Teper) is quoted only as an approximate orientation point. This pilot does not measure σ.

## Reading the result

See [PROTOCOL.md](PROTOCOL.md) for the decision tree. In short:

- **Single phase.** ℓ⁴·Var stays flat near χ. The fitted Var_ν is consistent with 0. C(r) is negative beyond the smearing range and decays to 0. Binder(Q/V) → 0.
- **Mixture.** v·Var grows with slope Var_ν. C(r) approaches a positive plateau equal to Var_ν. Two peaks at ±m sharpen with volume, and Binder tends toward 2/3.

The sign check needs care. Reflection positivity makes the **unflowed** pseudoscalar correlator negative at nonzero separation. Flow smears q over about √(8t), so the flowed C(r) is positive at short distance. The default r_min = 2√(8t) is a heuristic cutoff, not a theorem. Small lattices may have no negative region resolvable at all.

## Caveats that kill or fake the signal

1. **Frozen topology fakes clusters.** As a → 0, the autocorrelation time of Q grows very steeply. A chain stuck at charge Q has q̄ = Q/V ≠ 0, and independent stuck streams look like separate clusters. That is a counterfeit mixture in exactly this observable.
   - **Discriminator:** frozen-sector peaks sit at about ±Q/V and **shrink like 1/V**. Genuine phase peaks sit at **±m, independent of V**. The top-level summary fits log|peak| against log V across volumes at fixed β: a slope near −1 means frozen-like, a slope near 0 means a candidate.
   - **Remedies:** run many independent seeds and use the coarser β for topology. Cross-checks that this pilot does not implement include open boundary conditions in time (Lüscher–Schaefer), which break the match with the proof's torus, and parallel tempering in boundary conditions (Hasenbusch).
2. **Finite periodic volume never breaks a symmetry.** The finite-lattice measure is exactly P-invariant, so with full sampling every odd expectation is zero and every histogram is symmetric. Real breaking would appear as two peaks that sharpen as L grows, with tunnelling suppressed exponentially in the box size. On small boxes it would look like a single fat peak, so one peak at modest L is weak evidence.
3. **Correlated data.** Blocks from one configuration are spatially correlated, and successive configurations are autocorrelated. The dip-test and BIC p-values assume i.i.d. samples, so they are optimistic. Treat them as indicators.
4. **Discretization.** The clover q is not exactly integer before flow. Any nonzero result should be checked against a second flow radius, and ideally a second discretization.

## Guidance, not proof

- Agreement or disagreement with these numerics proves nothing about the Lean objects.
- The proof uses its own spacing sequence a_N = 1/∏b_k with odd ratios b_k ≥ 3 chosen to satisfy an explicit inequality. It sets couplings with its own RG map and takes infinite volume along a compactness subsequence. **This pilot cannot target that subsequence.** Those spacings are probably far finer than anything simulable, which is an inference. Behaviour specific to that subsequence would be invisible to a standard β scan.
- The proof's observables are its renormalized lattice polynomials, not flowed fields. We expect, without having shown it, that the two agree on whether a vacuum expectation value vanishes, but not on its value.
- Physics lore says P is unbroken in pure Yang-Mills at θ = 0. The Vafa–Witten argument is strong heuristics, and its rigor has been disputed. A clean null result is the expected outcome.

## Extending to SU(3)

1. Replace the quaternion links with 3×3 complex matrices. Use Cabibbo–Marinari heatbath on SU(2) subgroups plus overrelaxation, and reunitarize by Gram–Schmidt.
2. Flow needs a matrix exponential of traceless antihermitian 3×3 matrices, for example Cayley–Hamilton or `scipy.linalg.expm` vectorized. F_μν takes the explicit traceless antihermitian projection, which is automatic only for SU(2).
3. Add a **C-odd** density. The simplest choice is **Im tr of small Wilson loops**, for example `Im tr P_μν(x)` summed over planes or rectangles in a P-even combination. Charge conjugation (U → Ū) flips its sign. A smoother alternative is the d-symbol cubic `tr(F{F,F})` with a P-even index contraction.
4. Run the same block, two-point, Binder, dip, and BIC pipeline on each odd density, plus an even control such as tr F². SU(3) then tests C and P together.

## Known limitations

- **Sub-block fit bias.** The Var(q̄_ℓ) fit assumes ℓ is much larger than both the flow smearing radius and the correlation length. On small lattices it is not. The flowed q-q correlator is positive at short range and negative further out, so χ_ℓ approaches χ slowly, and the sub-block fit can return a spurious positive Var_ν. A surface term fixes only part of this. By default, blocks with ℓ < 2√(8t) are excluded (`--fit-min-ell`). The **cross-volume fit** in the top-level summary, Var(Q/V) = Var_ν + χ/V across lattice sizes at fixed β, has no smearing or surface terms and is the cleaner estimate. Use both, and believe neither on small lattices.
- **Correlations ignored in the fits.** The scaling, cross-volume, and two-point fits ignore correlations between data points in χ²; the bootstrap and jackknife errors account for them only partly.
- **Speed.** Everything is vectorized numpy and runs on a single core. On the development box, one heatbath plus 2 overrelaxation updates on 8⁴ took about 0.04 s, and flowing an 8⁴ configuration to t = 3 took about 3.5 s. Cost scales with volume.
- **What isn't implemented.** There are no open boundary conditions or parallel tempering (remedies for frozen topology), no string-tension or t₀ scale setting (only the raw t²E(t) history is recorded), and no SU(3) yet.

## Layout

```
fingerprint/
  su2.py          SU(2) quaternion algebra (vectorized)
  lattice.py      shifts, checkerboard, staples, plaquettes, gauge transforms, reflections
  heatbath.py     Kennedy-Pendleton heatbath, overrelaxation, checkerboard sweeps
  flow.py         Wilson flow, Lüscher RK3
  observables.py  clover F, q(x), E(x)
  blocks.py       block averages, variance split, Var_ν scaling fit
  diagnostics.py  histograms, dip test, GMM/BIC, Binder, two-point function
  autocorr.py     tau_int, frozen-topology warnings, cross-volume check
  run.py          Markov stream driver
  analysis.py     per-ensemble analysis and output
  cli.py          argparse CLI (python -m fingerprint)
  selftest.py     correctness checks (python -m fingerprint.selftest)
```
