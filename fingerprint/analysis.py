"""Per-ensemble analysis and file output."""

import json
import os

import numpy as np

from . import diagnostics as dg
from .autocorr import warnings_for_streams
from .blocks import variance_split, scaling_fit

CHI_NOTE = ("chi is in lattice units (a^-4). A physical comparison needs a scale, e.g. "
            "sqrt(t0) from the recorded t^2 E(t) flow history, or the string tension. "
            "Literature value for SU(2), quoted as approximate: chi^(1/4)/sqrt(sigma) ~ 0.48 "
            "(continuum extrapolation, Lucini & Teper). This pilot does not measure sigma.")


def _fmt(x, p=4):
    if x is None:
        return "n/a"
    if isinstance(x, float) and not np.isfinite(x):
        return "nan"
    return f"{x:.{p}e}" if isinstance(x, float) else str(x)


def analyse_ensemble(streams, dims, beta, flow_radius, outdir, args, rng):
    os.makedirs(outdir, exist_ok=True)
    V = int(np.prod(dims))
    L_tag = "x".join(str(d) for d in dims)
    Q = np.concatenate([s["Q"] for s in streams])
    plaq = np.concatenate([s["plaq"] for s in streams])
    seeds = np.concatenate([s["seed"] for s in streams])
    E = np.concatenate([s["E_clover"] for s in streams])
    t2E = np.concatenate([s["t2E_plaq"] for s in streams])
    n = len(Q)
    QV = Q / V
    ells = sorted(streams[0]["blocks"])
    qbar = {l: np.concatenate([s["blocks"][l] for s in streams]) for l in ells}
    radial = np.concatenate([s["radial"] for s in streams])
    axial = np.concatenate([s["axial"] for s in streams])
    r_rad = streams[0]["r_radial"]
    fh = np.mean([s["flow_history"] for s in streams], axis=0)

    # ---------------- raw output
    with open(os.path.join(outdir, "config_measurements.csv"), "w") as f:
        f.write("seed,index_in_stream,plaquette,Q_flowed,Q_rounded,Q_over_V,E_clover_flowed,t2E_plaq_flowed\n")
        k = 0
        for s in streams:
            for i in range(len(s["Q"])):
                f.write(f"{int(s['seed'][i])},{i},{s['plaq'][i]:.10f},{s['Q'][i]:.8f},{int(np.rint(s['Q'][i]))},"
                        f"{s['Q'][i] / V:.10e},{s['E_clover'][i]:.10e},{s['t2E_plaq'][i]:.10e}\n")
                k += 1
    npz = {"dims": np.array(dims), "beta": beta, "flow_radius": flow_radius,
           "Q": Q, "plaq": plaq, "seed": seeds, "E_clover": E, "t2E_plaq": t2E,
           "r_radial": r_rad, "C_radial": radial, "C_axial": axial, "flow_history": fh}
    for l in ells:
        npz[f"qbar_ell{l}"] = qbar[l]
    if "q_fields" in streams[0]:
        npz["q_fields"] = np.concatenate([s["q_fields"] for s in streams])
    np.savez_compressed(os.path.join(outdir, "measurements.npz"), **npz)
    for l in ells:
        np.savetxt(os.path.join(outdir, f"block_averages_ell{l}.csv"), qbar[l], delimiter=",",
                   header=f"rows=configs, cols=blocks of side {l}", fmt="%.8e")
    with open(os.path.join(outdir, "flow_history.csv"), "w") as f:
        f.write("t,E_plaq,t2E_plaq\n")
        for row in fh:
            f.write(f"{row[0]:.6f},{row[1]:.10e},{row[2]:.10e}\n")

    # ---------------- histograms
    bins = args.hist_bins
    dg.write_histogram_csv(os.path.join(outdir, "hist_Q_over_V.csv"), QV, bins)
    for l in ells:
        dg.write_histogram_csv(os.path.join(outdir, f"hist_qbar_ell{l}.csv"), qbar[l].ravel(), bins)

    # ---------------- chi, Binder, dip, GMM
    res = {"dims": list(dims), "V": V, "beta": beta, "flow_radius": flow_radius,
           "flow_t": flow_radius ** 2 / 8.0, "n_configs": n, "n_streams": len(streams)}
    res["plaquette"] = (float(plaq.mean()), float(plaq.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"))
    chi = float(np.mean(Q ** 2) / V)
    chi_b = [np.mean(Q[rng.integers(0, n, n)] ** 2) / V for _ in range(args.n_boot)]
    res["chi_full_volume"] = (chi, float(np.std(chi_b, ddof=1)) if n > 1 else float("nan"))
    res["mean_Q"] = float(Q.mean())
    res["mean_abs_Q_minus_round"] = float(np.mean(np.abs(Q - np.rint(Q))))
    res["binder_QV"] = dg.binder_with_error(QV, args.n_boot, rng)
    res["dip_QV"] = dg.dip_test(QV, args.dip_sims, rng)
    res["gmm_QV"] = dg.gmm_bic(QV, rng=rng)
    res["per_ell"] = {}
    block_data = {}
    for l in ells:
        x = qbar[l]
        d = {"variance": variance_split(x),
             "binder": dg.binder_with_error(x.ravel(), args.n_boot, rng),
             "dip": dg.dip_test(x.ravel(), args.dip_sims, rng),
             "gmm": dg.gmm_bic(x.ravel(), rng=rng)}
        res["per_ell"][l] = d
        block_data[l ** 4] = x
    block_data[V] = QV[:, None]
    ell_min = args.fit_min_ell if args.fit_min_ell is not None else int(np.ceil(2.0 * flow_radius))
    fit = scaling_fit(block_data, V, args.n_boot, rng, args.surface_term, min_v=float(ell_min) ** 4)
    fit["ell_min"] = ell_min
    res["scaling"] = fit
    with open(os.path.join(outdir, "block_scaling.csv"), "w") as f:
        f.write("block_volume_v,ell,var_qbar,v_times_var,used_in_fit,is_full_volume\n")
        fitted = set(int(v) for v in fit.get("v", [])) if fit.get("status") == "ok" else set()
        for v, y, vy in fit.get("table", []):
            f.write(f"{int(v)},{v ** 0.25:.4g},{y:.8e},{vy:.8e},{int(int(v) in fitted)},{int(int(v) == V)}\n")

    # ---------------- two-point function
    r_min = args.twopoint_rmin if args.twopoint_rmin is not None else 2.0 * flow_radius
    tp = dg.analyse_twopoint(r_rad, radial, r_min)
    res["twopoint"] = tp
    with open(os.path.join(outdir, "twopoint_radial.csv"), "w") as f:
        f.write("r,C,dC\n")
        for r, c, e in zip(tp["r"], tp["C"], tp["dC"]):
            f.write(f"{r:.6f},{c:.8e},{e:.8e}\n")
    Ca, dCa = dg.jackknife(axial)
    with open(os.path.join(outdir, "twopoint_axial.csv"), "w") as f:
        f.write("d,C,dC\n")
        for d, (c, e) in enumerate(zip(Ca, dCa)):
            f.write(f"{d},{c:.8e},{e:.8e}\n")
    res["frozen_plateau_scale"] = float(np.mean(Q ** 2) / V ** 2)

    # ---------------- frozen topology
    warns, info = warnings_for_streams([s["Q"] for s in streams], V, args.min_eff)
    res["frozen_warnings"] = warns
    res["streams"] = info

    res["verdict"] = verdict(res)
    write_summary(os.path.join(outdir, "summary.txt"), res, ells, QV, qbar)
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    res["_QV"] = QV  # kept in memory for the cross-volume fit, not written to results.json
    return res


def verdict(res):
    lines = []
    tp = res["twopoint"]
    ps = tp.get("plateau_sigma")
    if ps is None or not np.isfinite(ps):
        lines.append("two-point plateau: not determined")
    elif ps > 3:
        lines.append(f"two-point plateau: POSITIVE at {ps:.1f} sigma (mixture-like signal; check frozen topology: "
                     f"frozen-sector scale <Q^2>/V^2 = {res['frozen_plateau_scale']:.3e})")
    else:
        lines.append(f"two-point plateau: none resolved ({ps:+.1f} sigma)")
    fit = res["scaling"]
    if fit.get("status") == "ok":
        chi = fit["chi"]
        ub = fit["var_nu_upper95"]
        lines.append(f"block scaling: Var_nu = {fit['var_nu']:.3e} +- {fit['var_nu_err']:.3e}; "
                     f"95% upper bound {ub:.3e} = {ub / (res['chi_full_volume'][0] / res['V']) if res['chi_full_volume'][0] > 0 else float('nan'):.2f} x (chi/V)")
    else:
        lines.append(f"block scaling: {fit.get('status')}")
    u, ue = res["binder_QV"]
    lines.append(f"Binder(Q/V) = {u:.3f} +- {ue:.3f} (0 = single Gaussian peak, 2/3 = two sharp peaks)")
    if res["frozen_warnings"]:
        lines.append(f"FROZEN-TOPOLOGY WARNINGS: {len(res['frozen_warnings'])} (see below); "
                     "any peak structure may be fake")
    return lines


def write_summary(path, res, ells, QV, qbar):
    L = []
    w = L.append
    w("SU(2) phase-fingerprint pilot -- ensemble summary")
    w("Guidance only: a Monte Carlo result proves nothing about the Lean construction.")
    w("")
    w(f"lattice {'x'.join(map(str, res['dims']))} (V={res['V']}), beta={res['beta']}, "
      f"flow radius sqrt(8t)={res['flow_radius']} (t={res['flow_t']:.4f}) lattice units")
    w(f"configs: {res['n_configs']} in {res['n_streams']} independent stream(s)")
    p, pe = res["plaquette"]
    w(f"plaquette (unflowed) = {p:.6f} +- {pe:.6f}")
    w(f"<Q> = {res['mean_Q']:+.4f};  <|Q - round(Q)|> = {res['mean_abs_Q_minus_round']:.3f} (integer-ness check)")
    c, ce = res["chi_full_volume"]
    w(f"chi = <Q^2>/V = {c:.4e} +- {ce:.4e}   [lattice units]")
    w("  note: " + CHI_NOTE)
    w("")
    w("VERDICT")
    for line in res["verdict"]:
        w("  " + line)
    w("")
    w("Full-volume Q/V")
    u, ue = res["binder_QV"]
    d = res["dip_QV"]
    w(f"  Binder U = {u:.4f} +- {ue:.4f}")
    w(f"  dip = {_fmt(d['dip'])}, p(uniform null, {d['n_sim']} sims) = {d['p_value']:.3f}, n = {d['n']}")
    g = res["gmm_QV"]
    if g.get("status") == "ok":
        w(f"  GMM: BIC1 = {g['one']['bic']:.2f}, BIC2 = {g['two']['bic']:.2f}, dBIC(2-1) = {g['delta_bic']:+.2f} "
          f"-> prefers {g['preferred']} component(s); 2-comp means {[f'{m:.3e}' for m in g['two']['means']]}")
    else:
        w(f"  GMM: {g.get('status')}")
    w(dg.ascii_histogram(QV, 15, 40, "Q/V"))
    w("")
    w("Block averages qbar_ell (ell^4 hypercubes)")
    w("  ell  n_blk  pooled_var     within_cfg     between_cfg    ell^4*var      Binder         dip(p)         dBIC")
    for l in ells:
        d = res["per_ell"][l]
        v = d["variance"]
        u, ue = d["binder"]
        dp = d["dip"]
        g = d["gmm"]
        db = g.get("delta_bic", float("nan"))
        w(f"  {l:3d}  {v['n_blocks']:5d}  {v['pooled']:.4e}  {v['within']:.4e}  {v['between']:.4e}  "
          f"{l ** 4 * v['pooled']:.4e}  {u:+.3f}+-{ue:.3f}  {dp['dip']:.4f}({dp['p_value']:.2f})  {db:+.2f}")
    w("")
    fit = res["scaling"]
    w("Block-size scaling: Var(qbar_v) = Var_nu + chi/v" + (" + c/v^(5/4) [sub-blocks only]" if fit.get("surface_term") else "")
      + "   (v = ell^4; last point = full volume)")
    w(f"  fitted: sub-blocks with ell >= {fit.get('ell_min')} (default ceil(2*sqrt(8t)); smaller blocks are dominated by "
      "flow smearing) plus the full volume")
    w("  v          v*Var        fitted   (flat ~ chi for one phase; grows with slope Var_nu for a mixture)")
    fitted = set(int(v) for v in fit.get("v", [])) if fit.get("status") == "ok" else set()
    for v, y, vy in fit.get("table", []):
        w(f"  {int(v):<9d}  {vy:.4e}   {'yes' if int(v) in fitted else 'no'}")
    if fit.get("status") == "ok":
        w(f"  Var_nu = {fit['var_nu']:.4e} +- {fit['var_nu_err']:.4e}; one-sided 95% bootstrap upper bound = "
          f"{fit['var_nu_upper95']:.4e}")
        w(f"  chi (fit) = {fit['chi']:.4e} +- {fit['chi_err']:.4e};  chi2/dof = {fit['chi2']:.2f}/{fit['dof']}")
        if fit.get("surface_term"):
            w(f"  surface coefficient = {fit['surface_coef']:.4e}")
        if fit["dof"] == 0:
            w("  (dof = 0: the fit interpolates; the bootstrap bound is the only meaningful output)")
    else:
        w(f"  {fit.get('status')}")
    w("")
    tp = res["twopoint"]
    w("q-q two-point function (radial, jackknife errors; see twopoint_radial.csv)")
    w(f"  r_min for sign check / fits = {tp['r_min']:.2f} (default 2*sqrt(8t): the flowed correlator is "
      "smeared positive at short distance)")
    w(f"  negative somewhere at r >= r_min: {tp['negative_beyond_rmin']}"
      + (f" (first at r={tp['first_negative_r']:.2f}, min C={tp['most_negative']:.3e})" if "first_negative_r" in tp else ""))
    if "plateau" in tp:
        w(f"  plateau (mean C over r in [{tp['plateau_r_range'][0]:.2f}, {tp['plateau_r_range'][1]:.2f}]) = "
          f"{tp['plateau']:.4e} +- {tp['plateau_err']:.4e}  ({tp['plateau_sigma']:+.2f} sigma)")
    w(f"  frozen-sector scale <Q^2>/V^2 = {res['frozen_plateau_scale']:.4e} (a chain frozen at Q gives a large-r "
      "offset of order (Q^2/V - chi)/V; inference from fixed-Q finite-volume formulas)")
    fe = tp.get("fit_exp")
    if fe:
        w("  fit A exp(-m r): " + (f"m = {fe['m']:.3f}, chi2/dof = {fe['chi2']:.1f}/{fe['dof']}" if "m" in fe else fe.get("error", "")))
    fc = tp.get("fit_exp_const")
    if fc:
        w("  fit A exp(-m r) + c: " + (f"m = {fc['m']:.3f}, c = {fc['c']:.3e} +- {fc['c_err']:.3e}, chi2/dof = {fc['chi2']:.1f}/{fc['dof']}" if "c" in fc else fc.get("error", "")))
    w("  (a single gapped phase: C<0 beyond the smearing range, decaying to 0; a mixture: C -> Var_nu > 0)")
    w("")
    w("Frozen-topology diagnostics")
    for s in res["streams"]:
        w(f"  stream {s['stream']}: n={s['n']}, tau_int(Q)={s['tau_int_Q']:.2f} (W={s['window']}), "
          f"n_eff={s['n_eff']:.1f}, sectors={s['sectors_visited']}, sector changes={s['sector_changes']}")
    if res["frozen_warnings"]:
        for x in res["frozen_warnings"]:
            w("  WARNING: " + x)
    else:
        w("  no frozen-topology warnings triggered (this does not prove adequate sampling)")
    w("")
    w("Caveats: finite periodic volume never breaks a symmetry (exact P-invariance of the finite-lattice measure);")
    w("dip/BIC p-values assume i.i.d. data but blocks are spatially correlated and configs autocorrelated;")
    w("this ensemble cannot target the Lean proof's odd-ratio spacing subsequence.")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
