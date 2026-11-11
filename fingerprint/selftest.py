"""Correctness checks: python -m fingerprint.selftest [--quick]

Runs in well under a minute on a laptop CPU (the optional Q-integer check on
an 8^4 lattice adds roughly 15-30 s; skip it with --quick).
"""

import argparse
import sys
import time

import numpy as np
from scipy import special

from . import su2
from .lattice import (hot_start, mean_plaquette, plaquette_field, gauge_transform, random_gauge,
                      reflect_gauge_field, reflect_sites, wilson_action)
from .heatbath import Updater, kp_sample_w0
from .flow import gradient_flow
from .observables import topological_charge_density, measure_flowed
from .blocks import block_average
from . import diagnostics as dg

RESULTS = []


def check(name, ok, detail):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def info(name, detail):
    RESULTS.append((name, None, detail))
    print(f"[INFO] {name}: {detail}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m fingerprint.selftest")
    ap.add_argument("--quick", action="store_true", help="skip the 8^4 Q-near-integer check")
    args = ap.parse_args(argv)
    t0 = time.time()
    rng = np.random.default_rng(2026)

    # 1. quaternion algebra matches 2x2 matrices
    a = su2.random_su2(rng, (50,))
    b = su2.random_su2(rng, (50,))
    err = np.abs(su2.to_matrix(su2.qmul(a, b)) - su2.to_matrix(a) @ su2.to_matrix(b)).max()
    err2 = np.abs(su2.to_matrix(su2.dag(a)) - np.conj(np.swapaxes(su2.to_matrix(a), -1, -2))).max()
    check("quaternion product / adjoint == 2x2 matrix product / adjoint", max(err, err2) < 1e-13,
          f"max deviation {max(err, err2):.1e}")

    # 2. KP heatbath distribution: <w0> = I2(alpha)/I1(alpha)
    worst = 0.0
    for alpha in (0.5, 2.0, 8.0):
        w = kp_sample_w0(np.full(200000, alpha), rng)
        exact = special.iv(2, alpha) / special.iv(1, alpha)
        worst = max(worst, abs(w.mean() - exact) / (w.std() / np.sqrt(len(w))))
    check("Kennedy-Pendleton <w0> vs I2(a)/I1(a) at a = 0.5, 2, 8", worst < 4.5,
          f"max deviation {worst:.2f} sigma")

    # 3. heatbath on 6^4 at beta = 2.3: unitarity, det, plaquette
    dims = (6, 6, 6, 6)
    U = hot_start(dims, rng)
    up = Updater(dims, 2.3, rng)
    for _ in range(60):
        up.update(U, 2)
    plaqs = []
    for _ in range(60):
        up.update(U, 2)
        plaqs.append(mean_plaquette(U))
    M = su2.to_matrix(U)
    unit = np.abs(M @ np.conj(np.swapaxes(M, -1, -2)) - np.eye(2)).max()
    det = np.abs(np.linalg.det(M) - 1).max()
    check("SU(2) unitarity U U^+ = 1 after 120 HB+2xOR updates", unit < 1e-12, f"max |UU^+ - 1| = {unit:.1e}")
    check("SU(2) det U = 1 after updates", det < 1e-12, f"max |det U - 1| = {det:.1e}")
    p = float(np.mean(plaqs))
    # naive error (successive measurements are correlated, so this underestimates)
    pe = float(np.std(plaqs, ddof=1) / np.sqrt(len(plaqs)))
    check("heatbath plaquette at beta=2.3, 6^4 (expect roughly 0.60)", 0.58 < p < 0.63,
          f"<P> = {p:.4f} +- {pe:.4f} (naive error; 60 measurements after 60 thermalization updates)")

    # 4. overrelaxation is microcanonical
    s0 = wilson_action(U, 2.3)
    up.sweep(U, "or")
    s1 = wilson_action(U, 2.3)
    check("overrelaxation preserves the Wilson action", abs(s1 - s0) < 1e-8 * abs(s0),
          f"|dS|/S = {abs(s1 - s0) / abs(s0):.1e}")

    # 5. gradient flow decreases the action monotonically
    V, hist = gradient_flow(U, 1.0, 0.02)
    dE = np.diff(hist[:, 1])
    check("Wilson flow: E_plaq(t) strictly decreasing over 50 RK3 steps to t=1", np.all(dE < 0),
          f"E: {hist[0, 1]:.4f} -> {hist[-1, 1]:.4f}, max step change {dE.max():.2e}")

    # 6. gauge invariance (plaquette, q, and gauge covariance of the flow)
    g = random_gauge(dims, rng)
    Ug = gauge_transform(U, g)
    dp = max(np.abs(plaquette_field(U, m, n)[..., 0] - plaquette_field(Ug, m, n)[..., 0]).max()
             for m in range(4) for n in range(m + 1, 4))
    q = topological_charge_density(U)
    qg = topological_charge_density(Ug)
    Vg, _ = gradient_flow(Ug, 1.0, 0.02, record=False)
    qf, _ = measure_flowed(V)
    qfg, _ = measure_flowed(Vg)
    dq = np.abs(q - qg).max()
    dqf = np.abs(qf - qfg).max()
    check("gauge invariance of plaquette field", dp < 1e-12, f"max |dP| = {dp:.1e}")
    check("gauge invariance of q(x) (unflowed) and of q(x) after flow", max(dq, dqf) < 1e-11,
          f"max |dq| = {dq:.1e} (unflowed), {dqf:.1e} (flowed, |q| max {np.abs(qf).max():.1e})")

    # 7. parity: q -> -q
    for axes, name in (((1, 2, 3), "spatial parity x->-x"), ((3,), "single-axis reflection")):
        UP = reflect_gauge_field(V, axes)
        qP, _ = measure_flowed(UP)
        loc = np.abs(qP + reflect_sites(qf, axes)).max()
        even = np.abs(qP - reflect_sites(qf, axes)).max()
        check(f"q(x) is odd under {name}", loc < 1e-12 and even > 1e-6 and abs(qP.sum() + qf.sum()) < 1e-10,
              f"Q = {qf.sum():+.6f} -> {qP.sum():+.6f}; max |q'(x) + q(Px)| = {loc:.1e} "
              f"(vs max |q'(x) - q(Px)| = {even:.1e})")
    pl_par = abs(mean_plaquette(reflect_gauge_field(U)) - mean_plaquette(U))
    check("plaquette even under parity", pl_par < 1e-13, f"|dP| = {pl_par:.1e}")

    # 8. blocks and two-point bookkeeping
    bavg = [block_average(qf, l).mean() for l in (1, 2, 3, 6)]
    check("block averages are consistent with Q/V for all ell", np.ptp(bavg) < 1e-15,
          f"spread {np.ptp(bavg):.1e}")
    tp = dg.TwoPoint(dims)
    c = tp.correlate(qf)
    check("two-point sum rule sum_r C(r) = Q^2/V", abs(c.sum() - qf.sum() ** 2 / qf.size) < 1e-12,
          f"{c.sum():.6e} vs {qf.sum() ** 2 / qf.size:.6e}")

    # 9. statistical diagnostics on synthetic data
    x1 = rng.standard_normal(2000)
    x2 = np.concatenate([rng.normal(-3, 1, 1000), rng.normal(3, 1, 1000)])
    d1 = dg.dip_test(x1, 100, rng)
    d2 = dg.dip_test(x2, 100, rng)
    check("dip test: unimodal Gaussian not rejected, well-separated bimodal rejected",
          d1["p_value"] > 0.05 and d2["p_value"] < 0.02,
          f"Gaussian dip={d1['dip']:.4f} p={d1['p_value']:.2f}; bimodal dip={d2['dip']:.4f} p={d2['p_value']:.3f}")
    g1 = dg.gmm_bic(x1, rng=rng)
    g2 = dg.gmm_bic(x2, rng=rng)
    check("GMM/BIC prefers 1 component for Gaussian, 2 for bimodal", g1["preferred"] == 1 and g2["preferred"] == 2,
          f"dBIC = {g1['delta_bic']:+.1f} / {g2['delta_bic']:+.1f}; bimodal means {np.round(g2['two']['means'], 2).tolist()}")
    u1 = dg.binder(rng.standard_normal(200000))
    u2 = dg.binder(np.where(rng.random(1000) < 0.5, -1.0, 1.0))
    check("Binder: Gaussian ~ 0, two sharp peaks = 2/3", abs(u1) < 0.02 and abs(u2 - 2 / 3) < 1e-12,
          f"U_gauss = {u1:+.4f}, U_twopeak = {u2:.4f}")

    # 10. (optional) Q near integer after flow on 8^4
    if not args.quick:
        dims8 = (8, 8, 8, 8)
        U8 = hot_start(dims8, rng)
        up8 = Updater(dims8, 2.4, rng)
        for _ in range(100):
            up8.update(U8, 2)
        Qs = []
        for _ in range(3):
            for _ in range(20):
                up8.update(U8, 2)
            V8, _ = gradient_flow(U8, 2.0, 0.025, record=False)
            Qs.append(float(topological_charge_density(V8).sum()))
        dev = max(abs(q_ - round(q_)) for q_ in Qs)
        info("Q after flow to t=2 (sqrt(8t)=4) on 8^4, beta=2.4, 3 configs",
             f"Q = {[round(q_, 3) for q_ in Qs]}; max |Q - round(Q)| = {dev:.3f} "
             "(small lattice: integer-ness is a sanity indicator, not a pass/fail test)")

    n_fail = sum(1 for _, ok, _ in RESULTS if ok is False)
    n_pass = sum(1 for _, ok, _ in RESULTS if ok is True)
    print(f"\nselftest: {n_pass} passed, {n_fail} failed, {time.time() - t0:.1f}s")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
