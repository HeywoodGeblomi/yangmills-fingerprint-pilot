"""Block averages of q over ell^4 hypercubes, variance split, and the
block-size scaling fit Var(qbar_ell) = Var_nu + chi/ell^4 (+ c/ell^5).

Logic (inference from the protocol, not a theorem): in a single clustering
phase with a mass gap, Var(qbar_ell) = chi_ell/ell^4 with chi_ell -> chi (the
topological susceptibility) plus an O(1/ell^5) surface correction. In a
mixture of phases with phase values <q>_xi, the variance levels off at
Var_nu = int <q>_xi^2 dnu - (int <q>_xi dnu)^2.
"""

import numpy as np


def valid_block_sizes(dims, requested=None):
    """Block sides ell that divide every lattice extent (hypercubic blocks)."""
    dmin = min(dims)
    cands = requested if requested else [l for l in range(1, dmin + 1)]
    out = sorted({int(l) for l in cands if l >= 1 and all(d % l == 0 for d in dims)})
    return out


def block_average(q, ell):
    """Mean of q over non-overlapping ell^4 hypercubes -> flat array of blocks."""
    dims = q.shape
    shp = []
    for d in dims:
        shp += [d // ell, ell]
    r = q.reshape(shp)
    return r.mean(axis=(1, 3, 5, 7)).ravel()


def variance_split(qbar):
    """qbar: (n_cfg, n_blocks). Returns dict with pooled variance, mean
    within-config block spread and between-config spread of config means."""
    qbar = np.asarray(qbar, dtype=float)
    n_cfg, n_blk = qbar.shape
    pooled = float(np.var(qbar, ddof=1)) if qbar.size > 1 else float("nan")
    within = float(np.mean(np.var(qbar, axis=1, ddof=1))) if n_blk > 1 else float("nan")
    cfg_means = qbar.mean(axis=1)
    between = float(np.var(cfg_means, ddof=1)) if n_cfg > 1 else float("nan")
    return {"pooled": pooled, "within": within, "between": between,
            "mean": float(qbar.mean()), "n_cfg": n_cfg, "n_blocks": n_blk}


def _design(v, surface, full):
    """Columns: 1, 1/v and (optionally) the surface term 1/v^(5/4), which is
    set to zero for the full periodic volume (a torus has no block boundary)."""
    cols = [np.ones_like(v), 1.0 / v]
    if surface:
        cols.append(np.where(full, 0.0, 1.0 / v ** 1.25))
    return np.vstack(cols).T


def _wls(v, y, w, surface, full):
    X = _design(v, surface, full)
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return coef


def scaling_fit(block_data, full_volume, n_boot=200, rng=None, surface="auto", min_v=1):
    """Fit Var(v) = Var_nu + chi/v [+ c/v^(5/4) for sub-blocks] with v = block volume.

    block_data: dict {v: qbar array (n_cfg, n_blocks)} (same configs for all v);
    the full volume (key full_volume) enters as one block per config, i.e. Q/V.
    Only sub-blocks with v >= min_v are fitted (blocks smaller than the flow
    smearing range are dominated by the smearing, not by chi); the full volume
    is always included. Configs are bootstrapped jointly across v, which keeps
    the correlations between points. Returns the central fit, bootstrap errors
    and the one-sided 95% bootstrap upper bound on Var_nu.
    """
    rng = rng or np.random.default_rng(0)
    all_vs = np.array(sorted(block_data), dtype=float)
    n_cfg = next(iter(block_data.values())).shape[0]
    out = {"v_all": all_vs.tolist(), "min_v": float(min_v)}
    vs = np.array([v for v in all_vs if v >= min_v or int(v) == int(full_volume)])
    full = np.array([int(v) == int(full_volume) for v in vs])
    n_sub = int(np.sum(~full))
    use_surface = (n_sub >= 2 and len(vs) >= 4) if surface == "auto" else (surface == "on")
    if use_surface and (n_sub < 1 or len(vs) < 3):
        use_surface = False
    out.update({"v": vs.tolist(), "surface_term": bool(use_surface)})

    def variances(idx, vv):
        return np.array([np.var(block_data[int(v)][idx], ddof=1) for v in vv])

    # table over all block sizes (for the v*Var vs v plot)
    if n_cfg >= 2:
        yall = variances(np.arange(n_cfg), all_vs)
        out["table"] = [(float(v), float(y), float(v * y)) for v, y in zip(all_vs, yall)]
    if len(vs) < 2 or n_cfg < 3:
        out["status"] = ("insufficient data: need >= 3 configs and >= 2 fitted volumes "
                         f"(sub-blocks with ell^4 >= {min_v:g} plus the full volume); "
                         "use a larger lattice, or lower --fit-min-ell")
        return out
    y = variances(np.arange(n_cfg), vs)
    boots = np.array([variances(rng.integers(0, n_cfg, n_cfg), vs) for _ in range(n_boot)])
    err = boots.std(axis=0, ddof=1)
    err = np.where(err > 0, err, np.max(err[err > 0]) if np.any(err > 0) else 1.0)
    w = 1.0 / err ** 2
    coef = _wls(vs, y, w, use_surface, full)
    bcoef = np.array([_wls(vs, b, w, use_surface, full) for b in boots])
    pred = _design(vs, use_surface, full) @ coef
    chi2 = float(np.sum(((y - pred) / err) ** 2))
    dof = len(vs) - len(coef)
    out.update({
        "status": "ok",
        "var": y.tolist(), "var_err": err.tolist(),
        "v_times_var": (vs * y).tolist(),
        "var_nu": float(coef[0]), "var_nu_err": float(bcoef[:, 0].std(ddof=1)),
        "var_nu_upper95": float(max(0.0, np.percentile(bcoef[:, 0], 95))),
        "chi": float(coef[1]), "chi_err": float(bcoef[:, 1].std(ddof=1)),
        "surface_coef": float(coef[2]) if use_surface else None,
        "chi2": chi2, "dof": dof,
    })
    return out


def cross_volume_fit(var_by_volume, n_boot=200, rng=None):
    """Across lattice volumes at fixed beta: Var(Q/V) = Var_nu + chi/V.

    var_by_volume: {V: array of Q/V per config}. Volumes are independent
    ensembles, so each is bootstrapped separately. This is the cleanest
    Var_nu estimate (no smearing or surface terms), but it needs >= 2 volumes.
    """
    rng = rng or np.random.default_rng(0)
    Vs = np.array(sorted(var_by_volume), dtype=float)
    if len(Vs) < 2:
        return {"status": "need >= 2 volumes"}
    data = [np.asarray(var_by_volume[int(v)]) for v in Vs]
    if min(len(d) for d in data) < 3:
        return {"status": "need >= 3 configs per volume"}
    y = np.array([np.var(d, ddof=1) for d in data])
    boots = np.array([[np.var(d[rng.integers(0, len(d), len(d))], ddof=1) for d in data] for _ in range(n_boot)])
    err = boots.std(axis=0, ddof=1)
    err = np.where(err > 0, err, 1.0)
    full = np.ones(len(Vs), dtype=bool)
    coef = _wls(Vs, y, 1 / err ** 2, False, full)
    bcoef = np.array([_wls(Vs, b, 1 / err ** 2, False, full) for b in boots])
    return {"status": "ok", "V": Vs.tolist(), "var_QV": y.tolist(), "var_err": err.tolist(),
            "var_nu": float(coef[0]), "var_nu_err": float(bcoef[:, 0].std(ddof=1)),
            "var_nu_upper95": float(max(0.0, np.percentile(bcoef[:, 0], 95))),
            "chi": float(coef[1]), "chi_err": float(bcoef[:, 1].std(ddof=1))}
