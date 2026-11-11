"""Integrated autocorrelation time and frozen-topology warnings.

tau_int uses the automatic windowing of Madras & Sokal / Wolff
(window W = smallest W with W >= c * tau_int(W), c = 6).

Frozen topology fakes clusters: a chain stuck at integer charge Q has
q-bar = Q/V != 0, and independent stuck streams look like separate peaks. The
discriminator: frozen-sector peaks sit near +-Q/V and shrink like 1/V as the
volume grows; a genuine broken phase would give peaks at +-m that do not
shrink. warnings_for_streams and cross_volume_check implement these checks.
"""

import numpy as np


def tau_int(x, c=6.0):
    """Integrated autocorrelation time (tau_int = 1/2 + sum_t rho(t))."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        return float("nan"), 0
    y = x - x.mean()
    var = np.dot(y, y) / n
    if var == 0:
        return float("inf"), 0
    f = np.fft.rfft(y, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n] / (n * var)
    tau = 0.5
    for W in range(1, n):
        tau += acf[W]
        if W >= c * tau:
            return float(max(tau, 0.5)), W
    return float(max(tau, 0.5)), n - 1


def warnings_for_streams(Q_streams, V, min_eff=50):
    """Per-ensemble frozen-topology warnings.

    Q_streams: list of 1D arrays of flowed Q, one per independent Markov stream.
    Returns (list of warning strings, per-stream info dicts).
    """
    warns = []
    info = []
    for i, Q in enumerate(Q_streams):
        Q = np.asarray(Q, dtype=float)
        n = len(Q)
        Qr = np.rint(Q).astype(int)
        tau, W = tau_int(Q)
        sectors = sorted(set(Qr.tolist()))
        changes = int(np.sum(Qr[1:] != Qr[:-1])) if n > 1 else 0
        d = {"stream": i, "n": n, "tau_int_Q": tau, "window": W,
             "n_eff": float(n / (2 * tau)) if np.isfinite(tau) and tau > 0 else 0.0,
             "sectors_visited": sectors, "sector_changes": changes,
             "mean_Q": float(Q.mean()) if n else float("nan"),
             "mean_abs_Q_minus_round": float(np.mean(np.abs(Q - Qr))) if n else float("nan")}
        info.append(d)
        if n >= 2 and changes == 0 and Qr[0] != 0:
            warns.append(f"stream {i}: rounded Q never changed over {n} configs (stuck at Q={Qr[0]}) -> FROZEN; "
                         f"its q-bar sits at Q/V = {Qr[0] / V:.3e}, a fake nonzero peak that shrinks like 1/V")
        elif n >= 2 and changes == 0:
            warns.append(f"stream {i}: rounded Q stayed 0 over all {n} configs (no tunnelling seen): no fake "
                         "nonzero peak from this stream, but chi is biased low and sampling is unverified")
        elif n >= 4 and len(sectors) > 1 and (np.all(Qr >= 0) or np.all(Qr <= 0)) and np.any(Qr != 0):
            warns.append(f"stream {i}: Q never changed sign (sectors {sectors}); possible slow tunnelling")
        if np.isfinite(tau) and n >= 4 and n / (2 * tau) < min_eff:
            warns.append(f"stream {i}: tau_int(Q) = {tau:.1f} configs -> only ~{n / (2 * tau):.1f} "
                         f"effective samples (< {min_eff}); errors and peak shapes unreliable")
        if not np.isfinite(tau):
            warns.append(f"stream {i}: Q constant to round-off; tau_int undefined (frozen or Q~0 everywhere)")
        if n and d["mean_abs_Q_minus_round"] > 0.2:
            warns.append(f"stream {i}: <|Q - round(Q)|> = {d['mean_abs_Q_minus_round']:.2f} > 0.2; "
                         "flow too short or lattice too coarse/small for integer-like Q")
    # streams sitting in different nonzero sectors with no exchange
    stuck = [int(np.rint(d["mean_Q"])) for d in info if d["n"] >= 2 and d["sector_changes"] == 0]
    if len(stuck) >= 2 and len(set(stuck)) >= 2:
        warns.append(f"{len(stuck)} frozen streams sit in distinct sectors {sorted(set(int(s) for s in stuck))}: "
                     "they will look like separate clusters (fake mixture)")
    return warns, info


def cross_volume_check(peaks_by_volume):
    """peaks_by_volume: {V: |peak position of Q/V| from the 2-component fit}.

    Fits log|peak| = a + s log V. s ~ -1 means peaks track fixed integer Q
    (frozen-sector-like, shrinking like 1/V); s ~ 0 means peaks at constant
    +-m (candidate genuine breaking, still subject to all other checks).
    """
    vs = np.array(sorted(peaks_by_volume), dtype=float)
    pk = np.array([peaks_by_volume[v] for v in sorted(peaks_by_volume)], dtype=float)
    ok = (pk > 0) & np.isfinite(pk)
    if np.sum(ok) < 2:
        return {"status": "need >=2 volumes with a two-peak fit", "slope": None}
    s, a = np.polyfit(np.log(vs[ok]), np.log(pk[ok]), 1)
    if s < -0.6:
        verdict = "peaks shrink ~1/V: consistent with frozen topological sectors (fake clusters)"
    elif s > -0.3:
        verdict = "peaks roughly volume-independent: candidate +-m structure (check frozen-topology and finite-volume caveats)"
    else:
        verdict = "inconclusive scaling between 1/V and constant"
    return {"status": "ok", "slope": float(s), "volumes": vs[ok].tolist(),
            "peaks": pk[ok].tolist(), "verdict": verdict}
