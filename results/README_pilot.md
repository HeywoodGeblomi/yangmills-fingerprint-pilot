# Multi-volume pilot at β = 2.3: cross-volume Var_ν fit

This is a lattice proxy only. It says nothing about the Lean construction or the Millennium problem. Every statement below is about this SU(2) Wilson Monte Carlo at one coarse lattice spacing.

## Parameters

| | |
|---|---|
| Action and group | SU(2) Wilson action, β = 2.3, periodic L⁴ |
| Volumes and statistics | 6⁴: 1200 configs (1 stream). 8⁴: 1000 (1 stream). 10⁴: 2 × 400. 12⁴: 4 × 150. |
| Updates | Hot start, 200 thermalization updates per stream. One update = 1 Kennedy–Pendleton heatbath sweep + 2 overrelaxation sweeps. 5 updates between measurements. |
| Flow | Wilson flow to t = 0.5 (√(8t) = 2 lattice units) on every volume, RK3 step 0.025 |
| Block fits | Block sizes up to L, with `--fit-min-ell 3` |

**Why β = 2.3?** It is the coarsest end of the requested range. Topology decorrelates fast there (τ_int(Q) ≈ 0.5–0.8 at 5 updates between measurements), and a 12⁴ box is affordable. At β = 2.4 the same physical box needs a larger L. A rough literature scale, not measured here, puts L = 12 at about 2 fm.

**Why these volumes?** Measured cost per configuration was about 0.5 s (6⁴), 0.8 s (8⁴), 1.8 s (10⁴) and 5.3 s (12⁴). The streams ran as 8 parallel single-thread processes. √(8t) = 2 is L/3 at L = 6, so 6⁴ is marginal. The fit is therefore also reported without it.

## Per-volume results

Errors on Var(Q/V) and χ come from a stream-wise moving-block bootstrap with block length 2·⌈2τ_int⌉. The naive i.i.d. bootstrap agrees within 5%, consistent with τ_int ≈ 0.5.

| L | configs | plaquette | ⟨Q⟩ | Var(Q) | χ = Var(Q)/V [a⁻⁴] | Var(Q/V) | τ_int(Q) per stream | mean \|Q − round(Q)\| | dip p (Q/V) | Binder (Q/V) | largest sub-block ℓ: dip p, Binder |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 1200 | 0.60289(18) | +0.031(13) | 0.211 | 1.63(8)e-4 | 1.255(58)e-7 | 0.52 | 0.229 | 1.00 | −0.10 ± 0.05 | ℓ=3: 1.00, −0.56 ± 0.03 |
| 8 | 1000 | 0.60235(11) | −0.022(27) | 0.749 | 1.83(8)e-4 | 4.47(21)e-8 | 0.50 | 0.245 | 1.00 | −0.04 ± 0.05 | ℓ=4: 1.00, −0.21 ± 0.03 |
| 10 | 800 | 0.60224(8) | −0.052(46) | 1.714 | 1.71(8)e-4 | 1.714(80)e-8 | 0.50, 0.50 | 0.254 | 0.99 | +0.08 ± 0.04 | ℓ=5: 0.95, −0.07 ± 0.02 |
| 12 | 600 | 0.60218(6) | −0.030(85) | 3.473 | 1.68(9)e-4 | 8.08(42)e-9 | 0.80, 0.57, 0.61, 0.50 | 0.248 | 1.00 | +0.09 ± 0.04 | ℓ=6: 1.00, −0.05 ± 0.02 |

**Topology sampling**
- No stream looks frozen. τ_int(Q) is about 0.5–0.8 measurements, and rounded Q changed sector hundreds of times per stream.
- The 12⁴ ensembles visited sectors −6 through +5.
- ⟨Q⟩ is consistent with zero on every volume. 6⁴ is the largest deviation, at 2.4σ.

**Q is not near integers.** On every volume, mean |Q − round(Q)| ≈ 0.23–0.25, which is what a uniform fractional part would give. So the "sector" bookkeeping is only nominal here.
- **Normalization ruled out.** `q_normalization_check.txt` uses an abelian constant-flux configuration, where the continuum value is Q = 2·n₁·n₂ for the σ₃ embedding. The code reproduces it to 0.3% at L = 8 and 0.02% at L = 16, with the correct sign.
- **Longer flow alone does not fix it.** `q_integer_check.txt` follows six 8⁴ configurations at β = 2.3. Flowing longer barely helps: the mean distance falls from 0.28 at t = 0.5 to 0.19 at t = 4.
- **Likely explanation (inference, not verified):** coarse spacing plus small boxes, combined with the known absence of smooth charge-1 self-dual solutions on an untwisted T⁴, which keeps the smoothed charge from locking to integers.
- **Consequence:** χ here is the susceptibility of the flowed clover charge at fixed t = 0.5. It is not a continuum topological susceptibility.

**Dip test and Binder cumulant**
- **Q/V:** the dip test never rejects a single peak (p ≥ 0.99). The Binder cumulant is near 0. The two largest volumes sit at about +0.09 ± 0.04, roughly 2σ, which is not significant here, but a larger run should watch it.
- **Sub-block averages:** these have negative Binder cumulants, meaning heavy tails. That is expected for small blocks of a local density. They are not a two-peak signal, which would push U toward +2/3.

## Cross-volume fit: Var(Q/V) = Var_ν + χ/V

The fit is weighted least squares. Errors come from 2000 block-bootstrap replicas, and the volumes are bootstrapped independently.

| fit | Var_ν | 95% upper bound | χ [a⁻⁴] | χ²/dof |
|---|---|---|---|---|
| all four volumes | −0.7 ± 5.6 × 10⁻¹⁰ (−0.1σ) | 8.7 × 10⁻¹⁰ | 1.712(62) × 10⁻⁴ | 3.37 / 2 |
| 8⁴, 10⁴, 12⁴ | −8.6 ± 7.2 × 10⁻¹⁰ (−1.2σ) | 3.6 × 10⁻¹⁰ | 1.837(94) × 10⁻⁴ | 0.35 / 1 |
| null model (Var_ν ≡ 0) | 0 | — | 1.706 × 10⁻⁴ | 3.39 / 3 |

**Var_ν is consistent with zero.** With all volumes, the 95% upper bound is about 0.1 × (χ/V) at 12⁴. It corresponds to an RMS phase value of ⟨q⟩_ξ below 3 × 10⁻⁵ a⁻⁴.

Several limits apply:
- A slope of 1/V per volume is exactly what a single phase predicts, so this test has no power against breaking that only shows up at volumes beyond 12⁴.
- One coarse β, non-integer flowed Q, and finite periodic volume, which never breaks the symmetry.
- There is no claim about continuum physics, and none about the proof's spacing subsequence.

**Plot:** `var_QV_vs_invV.svg`. (A PNG copy exists only on the box that ran the pilot; the GitHub connector available here pushes text, so the repo carries the SVG, regenerated by `cross_volume_analysis.py`.) The left panel shows Var(Q/V) against 1/V with both fits. The right panel shows χ per volume.

## Block-size fit per volume

This is the within-volume Var(q̄_ℓ) fit from each stream's `summary.txt`, with ℓ ≥ 3 plus the full volume.

| L | Var_ν (block fit) | dof | comment |
|---|---|---|---|
| 6 | +8.3(5) × 10⁻⁸ | 0 | ℓ = 3 only; biased |
| 8 | +2.4(2) × 10⁻⁸ | 0 | ℓ = 4 only; biased |
| 10 | +6.6(1.2), +8.1(1.1) × 10⁻⁹ | 0 | ℓ = 5 only; biased |
| 12 | +1.0(8), −1.0(7), +1.4(1.1), +0.7(8) × 10⁻⁹ | 1 | ℓ = 3, 4, 6 + surface term; consistent with 0 |

With only one sub-block size and no surface term (dof = 0), the block fit returns a spurious positive Var_ν. Small blocks have χ_ℓ < χ because of flow smearing and boundary effects, and the README's "Known limitations" section already flags this bias. On 12⁴, where the surface term can be fitted, every stream is consistent with zero. The cross-volume fit is the one to trust.

## Reproducing this pilot

From a fresh clone (Python 3, about 33 minutes on an 8-core machine):

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-plot.txt
.venv/bin/python -m fingerprint.selftest --quick
PYTHON=.venv/bin/python bash results/reproduce_pilot.sh
```

`results/reproduce_pilot.sh` holds the exact CLI invocations, seeds and block sizes: 8 single-thread streams (6⁴ seed 1 × 1200 configs, 8⁴ seed 1 × 1000, 10⁴ seeds 1–2 × 400, 12⁴ seeds 1–4 × 150), then `cross_volume_analysis.py`. A re-run on the same pinned numpy/scipy reproduced the raw per-stream Q, plaquette and two-point arrays and the committed `cross_volume_results.json` bit for bit. Different numpy/scipy versions may change the random streams.

## Files

| File | Contents |
|---|---|
| `reproduce_pilot.sh` | Exact commands to regenerate `runs/` and rerun the analysis |
| `cross_volume_analysis.py` | Merges streams, block bootstrap, fits, plot. Reads `runs/`, which is **not** in the repo (about 45 MB of raw per-stream output); rerun the CLI to regenerate it. |
| `cross_volume_results.json` | All numbers |
| `cross_volume_analysis_output.txt` | Script output |
| `var_QV_vs_invV.svg` | The plot (the PNG original was not pushed) |
| `q_integer_check.py` / `.txt` | Q(t) on six 8⁴ configurations |
| `q_normalization_check.py` / `.txt` | Constant-flux normalization test |
| `runs/L<L>_s<seed>/` | Per-stream CLI output (CSV, npz, histograms, `summary.txt`). About 45 MB. Box only; git-ignored. |
