"""Cross-volume Var_nu analysis of the multi-volume pilot (box-local, not part of the package).

Merges independent streams per volume from results/runs/L<L>_s<seed>/, then fits
Var(Q/V) = Var_nu + chi/V with a moving-block bootstrap per stream
(block length >= 2*tau_int(Q), so autocorrelation is accounted for).
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fingerprint.autocorr import tau_int
from fingerprint import diagnostics as dg

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
rng = np.random.default_rng(7)
NB = 2000


def load():
    vols = {}
    for d in sorted(glob.glob(os.path.join(RUNS, "L*_s*"))):
        sub = glob.glob(os.path.join(d, "L*_b*"))
        if not sub or not os.path.exists(os.path.join(sub[0], "measurements.npz")):
            continue
        z = np.load(os.path.join(sub[0], "measurements.npz"))
        res = json.load(open(os.path.join(sub[0], "results.json")))
        L = int(z["dims"][0])
        vols.setdefault(L, []).append({"z": z, "res": res, "dir": sub[0]})
    return vols


def block_boot_idx(n, b):
    nb = int(np.ceil(n / b))
    starts = rng.integers(0, n - b + 1, nb)
    return np.concatenate([np.arange(s, s + b) for s in starts])[:n]


def wls(V, y, err):
    X = np.vstack([np.ones_like(V), 1 / V]).T
    w = 1 / err
    coef, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    chi2 = float(np.sum(((y - X @ coef) / err) ** 2))
    return coef, chi2


def main():
    vols = load()
    rows, boot_y = [], []
    for L in sorted(vols):
        streams = vols[L]
        V = L ** 4
        Qs = [s["z"]["Q"] for s in streams]
        Q = np.concatenate(Qs)
        plaq = np.concatenate([s["z"]["plaq"] for s in streams])
        taus = [tau_int(q)[0] for q in Qs]
        bl = [max(1, int(np.ceil(2 * t))) * 2 for t in taus]  # generous block length
        # stream-wise block bootstrap of Var(Q/V) over the pooled sample
        bs = []
        for _ in range(NB):
            qq = np.concatenate([q[block_boot_idx(len(q), b)] for q, b in zip(Qs, bl)])
            bs.append(np.var(qq / V, ddof=1))
        bs = np.array(bs)
        # naive (iid) bootstrap for comparison
        naive = np.array([np.var(Q[rng.integers(0, len(Q), len(Q))] / V, ddof=1) for _ in range(500)])
        # pooled per-volume dip/Binder from the per-stream runs: recompute on merged data
        qv = Q / V
        dip = dg.dip_test(qv, 200, rng)
        bnd = dg.binder_with_error(qv, 500, rng)
        # largest proper sub-block: merge block arrays
        ells = sorted(int(k[8:]) for k in streams[0]["z"].files if k.startswith("qbar_ell"))
        sub_ells = [l for l in ells if l < L]
        lb = max(sub_ells) if sub_ells else None
        blk = {}
        if lb:
            xb = np.concatenate([s["z"][f"qbar_ell{lb}"] for s in streams]).ravel()
            blk = {"ell": lb, "dip": dg.dip_test(xb, 200, rng), "binder": dg.binder_with_error(xb, 300, rng)}
        sc = [s["res"]["scaling"] for s in streams]
        rows.append({
            "L": L, "V": V, "n_streams": len(streams), "n": int(len(Q)),
            "plaq": float(plaq.mean()), "plaq_err": float(plaq.std(ddof=1) / np.sqrt(len(plaq))),
            "meanQ": float(Q.mean()), "meanQ_err": float(Q.std(ddof=1) / np.sqrt(len(Q) / (2 * max(np.mean(taus), 0.5)))),
            "varQ": float(np.var(Q, ddof=1)), "chi": float(np.var(Q, ddof=1) / V),
            "chi_err": float(bs.std(ddof=1) * V),
            "varQV": float(np.var(qv, ddof=1)), "varQV_err": float(bs.std(ddof=1)),
            "varQV_err_naive": float(naive.std(ddof=1)),
            "tau_int": [round(t, 2) for t in taus],
            "sector_changes": [s["res"]["streams"][0]["sector_changes"] for s in streams],
            "sectors": sorted(set(sum([s["res"]["streams"][0]["sectors_visited"] for s in streams], []))),
            "int_dev": float(np.mean(np.abs(Q - np.rint(Q)))),
            "int_dev_med": float(np.median(np.abs(Q - np.rint(Q)))),
            "dip_p": dip["p_value"], "dip": dip["dip"], "binder": bnd,
            "block": {"ell": blk.get("ell"), "dip_p": blk["dip"]["p_value"] if blk else None,
                      "binder": blk["binder"] if blk else None},
            "block_fit": [{"var_nu": c.get("var_nu"), "var_nu_err": c.get("var_nu_err"),
                           "ub95": c.get("var_nu_upper95"), "chi": c.get("chi"),
                           "chi2": c.get("chi2"), "dof": c.get("dof"), "v": c.get("v"),
                           "status": c.get("status")} for c in sc],
            "warnings": sum([s["res"]["frozen_warnings"] for s in streams], []),
        })
        boot_y.append(bs)
    Vs = np.array([r["V"] for r in rows], float)
    y = np.array([r["varQV"] for r in rows])
    e = np.array([r["varQV_err"] for r in rows])
    boot_y = np.array(boot_y)  # (nvol, NB): volumes independent, so pair boots column-wise

    def fit_subset(mask, label):
        coef, chi2 = wls(Vs[mask], y[mask], e[mask])
        bc = np.array([wls(Vs[mask], boot_y[mask, k], e[mask])[0] for k in range(NB)])
        return {"label": label, "volumes": [int(v) for v in Vs[mask]],
                "var_nu": float(coef[0]), "var_nu_err": float(bc[:, 0].std(ddof=1)),
                "var_nu_ub95": float(np.percentile(bc[:, 0], 95)),
                "var_nu_sigma": float(coef[0] / bc[:, 0].std(ddof=1)),
                "chi": float(coef[1]), "chi_err": float(bc[:, 1].std(ddof=1)),
                "chi2": chi2, "dof": int(mask.sum() - 2)}

    fits = [fit_subset(np.ones(len(Vs), bool), "all volumes")]
    if len(Vs) >= 3:
        fits.append(fit_subset(Vs > Vs.min(), "excluding smallest volume"))
    # constrained one-parameter check: Var_nu = 0, chi const
    w = 1 / e ** 2
    chi0 = float(np.sum(w * y * Vs ** -1) / np.sum(w * Vs ** -2))
    chi2_0 = float(np.sum(((y - chi0 / Vs) / e) ** 2))
    out = {"rows": rows, "fits": fits,
           "null_fit": {"chi": chi0, "chi2": chi2_0, "dof": int(len(Vs) - 1)}}
    json.dump(out, open(os.path.join(HERE, "cross_volume_results.json"), "w"), indent=1, default=float)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    x = 1 / Vs
    ax[0].errorbar(x, y, yerr=e, fmt="o", color="k", capsize=3, label="Var(Q/V), block bootstrap")
    xx = np.linspace(0, x.max() * 1.05, 200)
    f = fits[0]
    ax[0].plot(xx, f["var_nu"] + f["chi"] * xx, "C0-",
               label=f"fit all: Var_nu={f['var_nu']:.2e}±{f['var_nu_err']:.1e}, chi={f['chi']:.2e}")
    if len(fits) > 1:
        g = fits[1]
        ax[0].plot(xx, g["var_nu"] + g["chi"] * xx, "C1--",
                   label=f"fit L>{min(r['L'] for r in rows)}: Var_nu={g['var_nu']:.2e}±{g['var_nu_err']:.1e}")
    for r in rows:
        ax[0].annotate(f"L={r['L']}", (1 / r["V"], r["varQV"]), textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax[0].axhline(0, color="gray", lw=0.5)
    ax[0].set_xlabel("1/V  (lattice units)")
    ax[0].set_ylabel("Var(Q/V)")
    ax[0].set_title("SU(2) Wilson, beta=2.3, sqrt(8t)=2: Var(Q/V) vs 1/V")
    ax[0].legend(fontsize=7)
    ax[1].errorbar(Vs, [r["chi"] for r in rows], yerr=[r["chi_err"] for r in rows], fmt="s", color="k", capsize=3)
    ax[1].axhline(f["chi"], color="C0", lw=1, label="chi from cross-volume fit")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("V")
    ax[1].set_ylabel("chi = Var(Q)/V  (a^-4)")
    ax[1].set_title("Susceptibility per volume")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "var_QV_vs_invV.png"), dpi=130)
    matplotlib.rcParams["svg.fonttype"] = "none"  # keep the SVG small and text-based
    fig.savefig(os.path.join(HERE, "var_QV_vs_invV.svg"))
    print(json.dumps({"fits": fits, "null_fit": out["null_fit"]}, indent=1, default=float))
    for r in rows:
        print({k: r[k] for k in ("L", "n", "plaq", "meanQ", "varQ", "chi", "chi_err", "varQV_err", "varQV_err_naive",
                                 "tau_int", "sector_changes", "sectors", "int_dev", "int_dev_med", "dip_p", "binder", "block")})
        for bf in r["block_fit"]:
            print("   block fit:", bf)
        print("   warnings:", r["warnings"])


if __name__ == "__main__":
    main()
