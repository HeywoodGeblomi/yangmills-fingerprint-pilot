"""Command-line interface: python -m fingerprint [options]."""

import argparse
import json
import os
import sys
import time

import numpy as np

from . import __version__
from .analysis import analyse_ensemble
from .autocorr import cross_volume_check
from .blocks import valid_block_sizes, cross_volume_fit
from .run import run_stream


def parse_size(s):
    parts = [int(p) for p in s.lower().split("x")]
    if len(parts) == 1:
        parts = parts * 4
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"lattice size '{s}' must be L or L0xL1xL2xL3")
    if any(p < 2 or p % 2 for p in parts):
        raise argparse.ArgumentTypeError("every extent must be even and >= 2 (checkerboard updates)")
    return tuple(parts)


def parse_flow_radius(items, betas):
    """Either one number (same radius for every beta) or beta:radius pairs."""
    if len(items) == 1 and ":" not in items[0]:
        r = float(items[0])
        return {b: r for b in betas}
    m = {}
    for it in items:
        if ":" not in it:
            raise SystemExit("--flow-radius: use one value or beta:radius pairs for every beta")
        b, r = it.split(":")
        m[float(b)] = float(r)
    missing = [b for b in betas if b not in m]
    if missing:
        raise SystemExit(f"--flow-radius: no radius given for beta {missing}")
    return m


def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m fingerprint",
        description="SU(2) Wilson lattice pilot for the Yang-Mills phase-fingerprint protocol "
                    "(gradient-flowed q(x), block averages, dip test, GMM/BIC, Binder, q-q plateau search).")
    p.add_argument("--sizes", nargs="+", type=parse_size, default=[parse_size("8")],
                   help="lattice sizes, e.g. 8 12 16 (L^4) or 8x8x8x16 (axis 0 = time). Default: 8")
    p.add_argument("--betas", nargs="+", type=float, default=[2.4],
                   help="Wilson couplings beta = 4/g0^2 (SU(2) ~2.2-2.6 is a rough analogue of SU(3) 5.9-6.4). Default: 2.4")
    p.add_argument("--n-configs", type=int, default=50, help="measured configurations per stream (default 50)")
    p.add_argument("--n-therm", type=int, default=200, help="thermalization updates per stream (default 200)")
    p.add_argument("--n-sweeps", type=int, default=10, help="updates between measurements (default 10)")
    p.add_argument("--n-or", type=int, default=3,
                   help="overrelaxation sweeps per heatbath sweep in each update (default 3; 0 = pure heatbath)")
    p.add_argument("--flow-radius", nargs="+", default=["2.0"],
                   help="smoothing radius sqrt(8t) in lattice units: one value, or beta:radius pairs to hold a "
                        "physical radius fixed across betas (default 2.0)")
    p.add_argument("--flow-eps", type=float, default=0.02, help="RK3 flow step size (default 0.02)")
    p.add_argument("--block-sizes", nargs="+", type=int, default=None,
                   help="block sides ell (must divide every extent); default: all divisors of the lattice")
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2],
                   help="one independent Markov stream per seed (default 1 2)")
    p.add_argument("--start", choices=["hot", "cold"], default="hot", help="initial configuration (default hot)")
    p.add_argument("--outdir", default="output", help="output directory (default ./output)")
    p.add_argument("--hist-bins", type=int, default=30)
    p.add_argument("--dip-sims", type=int, default=200, help="uniform-null simulations for the dip p-value")
    p.add_argument("--n-boot", type=int, default=200, help="bootstrap samples")
    p.add_argument("--surface-term", choices=["auto", "on", "off"], default="auto",
                   help="include c/ell^5 in the scaling fit (auto: if >=4 block volumes)")
    p.add_argument("--fit-min-ell", type=int, default=None,
                   help="smallest block side used in the Var_nu scaling fit (default ceil(2*flow radius))")
    p.add_argument("--twopoint-rmin", type=float, default=None,
                   help="minimum r for the two-point sign check and fits (default 2*flow radius)")
    p.add_argument("--min-eff", type=float, default=50, help="warn if n/(2 tau_int(Q)) is below this")
    p.add_argument("--save-q-fields", action="store_true", help="also store every flowed q(x) field (float32)")
    p.add_argument("--analysis-seed", type=int, default=12345)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    log = (lambda *a: None) if args.quiet else (lambda s: print(s, flush=True))
    radii = parse_flow_radius(args.flow_radius, args.betas)
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "run_config.json"), "w") as f:
        cfg = vars(args).copy()
        cfg["sizes"] = ["x".join(map(str, s)) for s in args.sizes]
        cfg["flow_radius_by_beta"] = {str(b): r for b, r in radii.items()}
        cfg["version"] = __version__
        cfg["numpy"] = np.__version__
        json.dump(cfg, f, indent=1)
    arng = np.random.default_rng(args.analysis_seed)
    t_all = time.time()
    results = []
    for dims in args.sizes:
        ells = valid_block_sizes(dims, args.block_sizes)
        if args.block_sizes:
            dropped = sorted(set(args.block_sizes) - set(ells))
            if dropped:
                log(f"  note: block sizes {dropped} do not divide {dims}; skipped")
        for beta in args.betas:
            tag = f"L{'x'.join(map(str, dims))}_b{beta:g}"
            log(f"== ensemble {tag}: flow radius {radii[beta]}, blocks {ells}, seeds {args.seeds}")
            streams = [run_stream(dims, beta, s, args.n_configs, args.n_therm, args.n_sweeps, args.n_or,
                                  radii[beta], args.flow_eps, ells, args.start, args.save_q_fields, log)
                       for s in args.seeds]
            res = analyse_ensemble(streams, dims, beta, radii[beta], os.path.join(args.outdir, tag), args, arng)
            res["tag"] = tag
            results.append(res)
            for line in res["verdict"]:
                log("   " + line)
    write_top_summary(args, results, time.time() - t_all)
    log(f"done in {time.time() - t_all:.1f}s; summary: {os.path.join(args.outdir, 'summary.txt')}")
    return 0


def write_top_summary(args, results, elapsed):
    L = ["SU(2) phase-fingerprint pilot -- run summary",
         "Context: protocol derived from the 10 November 2026 assessment of the OpenAI Yang-Mills Lean construction",
         "(not independently verified). Monte Carlo output is guidance, not proof.", ""]
    L.append(f"{'ensemble':<22}{'plaq':>10}{'chi':>12}{'Binder':>9}{'plateau_sig':>13}{'Var_nu_ub95':>13}{'warn':>6}")
    for r in results:
        fit = r["scaling"]
        ub = fit.get("var_nu_upper95", float("nan")) if fit.get("status") == "ok" else float("nan")
        L.append(f"{r['tag']:<22}{r['plaquette'][0]:>10.5f}{r['chi_full_volume'][0]:>12.3e}"
                 f"{r['binder_QV'][0]:>9.3f}{r['twopoint'].get('plateau_sigma', float('nan')):>13.2f}"
                 f"{ub:>13.3e}{len(r['frozen_warnings']):>6d}")
    L.append("")
    L.append("Cross-volume frozen-topology discriminator (peaks at +-Q/V shrink like 1/V; genuine +-m peaks do not)")
    any_check = False
    for beta in sorted({r["beta"] for r in results}):
        rs = [r for r in results if r["beta"] == beta]
        if len(rs) < 2:
            continue
        any_check = True
        peaks = {}
        for r in rs:
            g = r["gmm_QV"]
            if g.get("status") == "ok" and g["preferred"] == 2:
                peaks[r["V"]] = float(np.mean(np.abs(g["two"]["means"])))
        chk = cross_volume_check(peaks)
        if chk["status"] == "ok":
            L.append(f"  beta={beta}: slope d log|peak| / d log V = {chk['slope']:+.2f} -> {chk['verdict']}")
        else:
            L.append(f"  beta={beta}: {chk['status']} (two-component GMM preferred in {len(peaks)} of {len(rs)} volumes)")
    if not any_check:
        L.append("  (needs >= 2 lattice sizes at the same beta)")
    L.append("")
    L.append("Cross-volume Var_nu fit at fixed beta: Var(Q/V) = Var_nu + chi/V (no smearing or surface terms)")
    any_fit = False
    for beta in sorted({r["beta"] for r in results}):
        rs = [r for r in results if r["beta"] == beta]
        if len(rs) < 2:
            continue
        any_fit = True
        cf = cross_volume_fit({r["V"]: r["_QV"] for r in rs}, args.n_boot, np.random.default_rng(args.analysis_seed))
        if cf["status"] == "ok":
            L.append(f"  beta={beta}: Var_nu = {cf['var_nu']:.3e} +- {cf['var_nu_err']:.3e}, 95% upper bound "
                     f"{cf['var_nu_upper95']:.3e}; chi = {cf['chi']:.3e} +- {cf['chi_err']:.3e} (volumes {[int(v) for v in cf['V']]})")
        else:
            L.append(f"  beta={beta}: {cf['status']}")
    if not any_fit:
        L.append("  (needs >= 2 lattice sizes at the same beta)")
    L.append("")
    L.append("Frozen-topology warnings")
    nw = 0
    for r in results:
        for w in r["frozen_warnings"]:
            L.append(f"  [{r['tag']}] {w}")
            nw += 1
    if nw == 0:
        L.append("  none triggered (does not prove adequate topology sampling)")
    L.append("")
    L.append("Decision tree: see PROTOCOL.md. Per-ensemble details: <outdir>/<ensemble>/summary.txt")
    L.append(f"wall time {elapsed:.1f}s")
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
