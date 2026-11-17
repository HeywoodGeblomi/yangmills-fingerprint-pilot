"""Peak / plateau diagnostics: histograms, Hartigan dip test, 1- vs 2-component
Gaussian mixture by BIC, Binder cumulant, and the q-q two-point function.

All pure numpy/scipy. The dip statistic is a line-by-line port of the
Hartigan & Hartigan (1985) algorithm (AS 217) as maintained in the R package
`diptest` (Maechler's C code), with min.is.0 = FALSE. Its p-value here is by
simulation from the uniform null (the least favourable unimodal law).

Caveat: block averages from one configuration are spatially correlated, and
consecutive configurations are autocorrelated; the dip and BIC tests assume
i.i.d. data, so their p-values are optimistic. Use them as indicators only.
"""

import math

import numpy as np
from scipy import optimize


# ----------------------------------------------------------------- histograms

def histogram(x, bins=30):
    x = np.asarray(x, dtype=float)
    counts, edges = np.histogram(x, bins=bins)
    return counts, edges


def write_histogram_csv(path, x, bins=30):
    counts, edges = histogram(x, bins)
    with open(path, "w") as fh:
        fh.write("bin_lo,bin_hi,count\n")
        for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
            fh.write(f"{lo:.10g},{hi:.10g},{int(c)}\n")
    return counts, edges


def ascii_histogram(x, bins=20, width=50, label=""):
    counts, edges = histogram(x, bins)
    m = max(1, counts.max())
    lines = [f"  histogram {label} (n={len(np.asarray(x))})"]
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(round(width * c / m))
        lines.append(f"  [{lo:+.3e},{hi:+.3e}) {int(c):6d} {bar}")
    return "\n".join(lines)


# ----------------------------------------------------------------- dip test

def dip_statistic(x):
    """Hartigan's dip statistic D_n (port of AS 217 / R diptest, min.is.0=FALSE)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n < 2 or x[-1] == x[0]:
        return 1.0 / (2 * n) if n > 0 else float("nan")
    # 1-based arrays as in the reference code
    xs = np.concatenate(([0.0], x)).tolist()
    mn = [0] * (n + 1)
    mj = [0] * (n + 1)
    lcm = [0] * (n + 2)
    gcm = [0] * (n + 2)
    low, high = 1, n
    dip = 1.0
    # indices for the convex minorant
    mn[1] = 1
    for j in range(2, n + 1):
        mn[j] = j - 1
        while True:
            mnj = mn[j]
            mnmnj = mn[mnj]
            if mnj == 1 or (xs[j] - xs[mnj]) * (mnj - mnmnj) < (xs[mnj] - xs[mnmnj]) * (j - mnj):
                break
            mn[j] = mnmnj
    # indices for the concave majorant
    mj[n] = n
    for k in range(n - 1, 0, -1):
        mj[k] = k + 1
        while True:
            mjk = mj[k]
            mjmjk = mj[mjk]
            if mjk == n or (xs[k] - xs[mjk]) * (mjk - mjmjk) < (xs[mjk] - xs[mjmjk]) * (k - mjk):
                break
            mj[k] = mjmjk
    while True:
        # GCM change points from high to low
        ic = 1
        gcm[1] = high
        while gcm[ic] > low:
            i1 = gcm[ic]
            ic += 1
            gcm[ic] = mn[i1]
        l_gcm = ic
        # LCM change points from low to high
        ic = 1
        lcm[1] = low
        while lcm[ic] < high:
            i1 = lcm[ic]
            ic += 1
            lcm[ic] = mj[i1]
        l_lcm = ic
        ig = l_gcm
        ih = l_lcm
        ix = ig - 1
        iv = 2
        d = 0.0
        if l_gcm != 2 or l_lcm != 2:
            while True:
                gcm_ix = gcm[ix]
                lcm_iv = lcm[iv]
                if gcm_ix > lcm_iv:
                    gcm_i1 = gcm[ix + 1]
                    dx = (lcm_iv - gcm_i1 + 1) - (xs[lcm_iv] - xs[gcm_i1]) * (gcm_ix - gcm_i1) / (xs[gcm_ix] - xs[gcm_i1])
                    iv += 1
                    if dx >= d:
                        d = dx
                        ig = ix + 1
                        ih = iv - 1
                else:
                    lcm_i1 = lcm[iv - 1]
                    dx = (xs[gcm_ix] - xs[lcm_i1]) * (lcm_iv - lcm_i1) / (xs[lcm_iv] - xs[lcm_i1]) - (gcm_ix - lcm_i1 - 1)
                    ix -= 1
                    if dx >= d:
                        d = dx
                        ig = ix + 1
                        ih = iv
                if ix < 1:
                    ix = 1
                if iv > l_lcm:
                    iv = l_lcm
                if gcm[ix] == lcm[iv]:
                    break
        else:
            d = 1.0
        if d < dip:
            break
        # dip for the convex minorant
        dip_l = 0.0
        for j in range(ig, l_gcm):
            max_t = 1.0
            j_, jb = gcm[j], gcm[j + 1]
            if j_ - jb > 1 and xs[j_] != xs[jb]:
                C = (j_ - jb) / (xs[j_] - xs[jb])
                for jk in range(jb, j_ + 1):
                    t = (jk - jb + 1) - (xs[jk] - xs[jb]) * C
                    if max_t < t:
                        max_t = t
            if dip_l < max_t:
                dip_l = max_t
        # dip for the concave majorant
        dip_u = 0.0
        for j in range(ih, l_lcm):
            max_t = 1.0
            j_, jb = lcm[j], lcm[j + 1]
            if jb - j_ > 1 and xs[jb] != xs[j_]:
                C = (jb - j_) / (xs[jb] - xs[j_])
                for jk in range(j_, jb + 1):
                    t = (xs[jk] - xs[j_]) * C - (jk - j_ - 1)
                    if max_t < t:
                        max_t = t
            if dip_u < max_t:
                dip_u = max_t
        dipnew = max(dip_l, dip_u)
        if dip < dipnew:
            dip = dipnew
        if low == gcm[ig] and high == lcm[ih]:
            break
        low = gcm[ig]
        high = lcm[ih]
    return dip / (2 * n)


def dip_test(x, n_sim=200, rng=None, max_n=5000):
    """Dip statistic and simulated p-value under the uniform null.
    If len(x) > max_n, a random subsample of size max_n is used."""
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    if len(x) > max_n:
        x = rng.choice(x, size=max_n, replace=False)
    n = len(x)
    if n < 4:
        return {"dip": float("nan"), "p_value": float("nan"), "n": n, "n_sim": 0}
    d = dip_statistic(x)
    sims = np.array([dip_statistic(rng.random(n)) for _ in range(n_sim)]) if n_sim > 0 else np.array([])
    p = float((np.sum(sims >= d) + 1) / (len(sims) + 1)) if n_sim > 0 else float("nan")
    return {"dip": float(d), "p_value": p, "n": n, "n_sim": int(n_sim)}


# ----------------------------------------------------------------- Gaussian mixtures

def _norm_logpdf(x, mu, var):
    return -0.5 * (np.log(2 * np.pi * var) + (x - mu) ** 2 / var)


def gmm_bic(x, n_init=8, max_iter=500, tol=1e-10, rng=None, max_n=100000):
    """Fit 1- and 2-component 1D Gaussian mixtures (EM) and score by BIC.
    BIC = k ln n - 2 ln L; lower is better; k = 2 (one comp), 5 (two comps).
    If len(x) > max_n a random subsample of size max_n is used (EM cost)."""
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    if len(x) > max_n:
        x = rng.choice(x, size=max_n, replace=False)
    n = len(x)
    out = {"n": n}
    if n < 6 or np.var(x) == 0:
        out["status"] = "insufficient data"
        return out
    mu1, var1 = x.mean(), x.var()
    ll1 = float(np.sum(_norm_logpdf(x, mu1, var1)))
    bic1 = 2 * math.log(n) - 2 * ll1
    floor = 1e-6 * var1
    best = None
    for it in range(n_init):
        if it == 0:
            mus = np.array([np.percentile(x, 25), np.percentile(x, 75)])
        else:
            mus = rng.choice(x, 2, replace=False)
        vars_ = np.array([var1, var1])
        w = np.array([0.5, 0.5])
        prev = -np.inf
        for _ in range(max_iter):
            lp = np.stack([np.log(w[k]) + _norm_logpdf(x, mus[k], vars_[k]) for k in range(2)])
            m = lp.max(axis=0)
            lse = m + np.log(np.exp(lp - m).sum(axis=0))
            ll = float(lse.sum())
            r = np.exp(lp - lse)
            nk = r.sum(axis=1) + 1e-300
            w = nk / n
            mus = (r @ x) / nk
            vars_ = np.maximum((r * (x[None, :] - mus[:, None]) ** 2).sum(axis=1) / nk, floor)
            if abs(ll - prev) < tol * max(1.0, abs(ll)):
                break
            prev = ll
        if best is None or ll > best[0]:
            best = (ll, w.copy(), mus.copy(), vars_.copy())
    ll2, w, mus, vars_ = best
    bic2 = 5 * math.log(n) - 2 * ll2
    order = np.argsort(mus)
    out.update({
        "status": "ok",
        "one": {"mean": float(mu1), "var": float(var1), "loglik": ll1, "bic": float(bic1)},
        "two": {"weights": w[order].tolist(), "means": mus[order].tolist(),
                "vars": vars_[order].tolist(), "loglik": float(ll2), "bic": float(bic2)},
        "delta_bic": float(bic2 - bic1),  # negative favours two components
        "preferred": 2 if bic2 < bic1 else 1,
    })
    return out


# ----------------------------------------------------------------- Binder cumulant

def binder(m):
    """U = 1 - <m^4> / (3 <m^2>^2): ~0 for one Gaussian peak at 0,
    -> 2/3 for two sharp peaks at +-m0."""
    m = np.asarray(m, dtype=float)
    m2 = np.mean(m ** 2)
    if m2 == 0:
        return float("nan")
    return float(1.0 - np.mean(m ** 4) / (3.0 * m2 ** 2))


def binder_with_error(m, n_boot=200, rng=None, max_n=200000):
    """Binder cumulant with a naive (i.i.d.) bootstrap error. For len(m) > max_n
    the bootstrap runs on a random subsample and the error is rescaled by
    sqrt(max_n / n)."""
    rng = rng or np.random.default_rng(0)
    m = np.asarray(m, dtype=float)
    u = binder(m)
    if len(m) < 3:
        return u, float("nan")
    sub, scale = m, 1.0
    if len(m) > max_n:
        sub = rng.choice(m, size=max_n, replace=False)
        scale = np.sqrt(max_n / len(m))
    bs = [binder(sub[rng.integers(0, len(sub), len(sub))]) for _ in range(n_boot)]
    return u, float(np.nanstd(bs, ddof=1) * scale)


# ----------------------------------------------------------------- two-point function

def torus_r2(dims):
    """Squared minimal-image distance for each displacement on the torus."""
    grids = np.meshgrid(*[np.minimum(np.arange(d), d - np.arange(d)) for d in dims], indexing="ij")
    return sum(g.astype(np.int64) ** 2 for g in grids)


class TwoPoint:
    """C(r) = (1/V) sum_x q(x) q(x+r), via FFT on the torus; binned by |r|
    (radial) and along the 4 lattice axes (axial, averaged over axes)."""

    def __init__(self, dims):
        self.dims = tuple(dims)
        self.r2 = torus_r2(self.dims)
        self.r2_values, self.inv = np.unique(self.r2.ravel(), return_inverse=True)
        self.counts = np.bincount(self.inv)
        self.axial_len = min(self.dims) // 2 + 1

    def correlate(self, q):
        fq = np.fft.fftn(q)
        return np.fft.ifftn(fq * np.conj(fq)).real / q.size

    def radial(self, corr):
        return np.bincount(self.inv, weights=corr.ravel()) / self.counts

    def axial(self, corr):
        out = np.zeros(self.axial_len)
        for ax in range(4):
            idx = [0, 0, 0, 0]
            vals = []
            for d in range(self.axial_len):
                idx[ax] = d
                vals.append(corr[tuple(idx)])
            out += np.array(vals)
        return out / 4.0

    def measure(self, q):
        c = self.correlate(q)
        return self.radial(c), self.axial(c)


def jackknife(samples, func=np.mean):
    """Jackknife mean and error of func over axis 0 (samples: (n, ...))."""
    samples = np.asarray(samples, dtype=float)
    n = samples.shape[0]
    full = func(samples, axis=0)
    if n < 2:
        return full, np.full_like(np.asarray(full, dtype=float), np.nan)
    tot = samples.sum(axis=0)
    loo = (tot[None, ...] - samples) / (n - 1)
    err = np.sqrt((n - 1) / n * np.sum((loo - loo.mean(axis=0)) ** 2, axis=0))
    return full, err


def analyse_twopoint(r, C_samples, r_min, n_plateau_frac=0.5):
    """r: distances; C_samples: (n_cfg, n_r) per-config correlators.

    - sign check: C(r) < 0 somewhere at r >= r_min (expected for a pseudoscalar
      by reflection positivity once r is well beyond the flow smearing radius;
      at r <~ 2 sqrt(8t) the flowed correlator is smeared positive).
    - plateau: average of C(r) over r >= n_plateau_frac * r_max with jackknife
      error; a positive plateau many sigma from 0 is the mixture signal.
    - fits on r >= r_min: A exp(-m r) and A exp(-m r) + c (weighted by the
      jackknife errors; correlations between r values are ignored).
    """
    r = np.asarray(r, dtype=float)
    C, dC = jackknife(C_samples)
    out = {"r": r.tolist(), "C": C.tolist(), "dC": dC.tolist()}
    sel = r >= r_min
    out["r_min"] = float(r_min)
    out["negative_beyond_rmin"] = bool(np.any(C[sel] < 0)) if np.any(sel) else None
    if np.any(sel & (C < 0)):
        out["first_negative_r"] = float(r[sel & (C < 0)][0])
        out["most_negative"] = float(C[sel].min())
    rmax = r.max()
    psel = r >= n_plateau_frac * rmax
    if np.any(psel):
        plateau_samples = C_samples[:, psel].mean(axis=1)
        pm, pe = jackknife(plateau_samples[:, None])
        out["plateau"] = float(pm[0])
        out["plateau_err"] = float(pe[0])
        out["plateau_sigma"] = float(pm[0] / pe[0]) if pe[0] > 0 else float("nan")
        out["plateau_r_range"] = [float(n_plateau_frac * rmax), float(rmax)]
    fsel = sel & (r > 0) & np.isfinite(dC) & (dC > 0)
    out["fit_exp"] = None
    out["fit_exp_const"] = None
    if np.sum(fsel) >= 3:
        rr, cc, ee = r[fsel], C[fsel], dC[fsel]
        # m is bounded below by 1/r_max so the exponential actually decays
        # inside the window (otherwise A exp(-m r) + c is degenerate with c)
        m_min = 1.0 / rmax
        out["fit_m_min"] = m_min
        try:
            p, cov = optimize.curve_fit(lambda x, A, m: A * np.exp(-m * x), rr, cc,
                                        p0=[cc[0] if cc[0] != 0 else 1e-6, max(1.0, 2 * m_min)], sigma=ee,
                                        absolute_sigma=True, maxfev=20000,
                                        bounds=([-np.inf, m_min], [np.inf, np.inf]))
            chi2 = float(np.sum(((cc - p[0] * np.exp(-p[1] * rr)) / ee) ** 2))
            out["fit_exp"] = {"A": float(p[0]), "m": float(p[1]), "chi2": chi2, "dof": int(len(rr) - 2)}
        except Exception as exc:  # noqa: BLE001
            out["fit_exp"] = {"error": str(exc)}
        if np.sum(fsel) >= 4:
            try:
                p, cov = optimize.curve_fit(lambda x, A, m, c: A * np.exp(-m * x) + c, rr, cc,
                                            p0=[cc[0] if cc[0] != 0 else 1e-6, max(1.0, 2 * m_min), 0.0], sigma=ee,
                                            absolute_sigma=True, maxfev=20000,
                                            bounds=([-np.inf, m_min, -np.inf], [np.inf, np.inf, np.inf]))
                chi2 = float(np.sum(((cc - (p[0] * np.exp(-p[1] * rr) + p[2])) / ee) ** 2))
                cerr = float(np.sqrt(cov[2, 2])) if np.all(np.isfinite(cov)) else float("nan")
                out["fit_exp_const"] = {"A": float(p[0]), "m": float(p[1]), "c": float(p[2]),
                                        "c_err": cerr, "chi2": chi2, "dof": int(len(rr) - 3)}
            except Exception as exc:  # noqa: BLE001
                out["fit_exp_const"] = {"error": str(exc)}
    return out
