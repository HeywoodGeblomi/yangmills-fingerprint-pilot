"""Markov chain driver: thermalize, update, flow, measure one stream."""

import time

import numpy as np

from .lattice import hot_start, cold_start, mean_plaquette
from .heatbath import Updater
from .flow import gradient_flow, flow_time_from_radius
from .observables import measure_flowed
from .blocks import block_average
from .diagnostics import TwoPoint


def stream_rng(seed, dims, beta):
    ss = np.random.SeedSequence([int(seed), *[int(d) for d in dims], int(round(beta * 1e6))])
    return np.random.default_rng(ss)


def run_stream(dims, beta, seed, n_configs, n_therm, n_sweeps, n_or, flow_radius,
               flow_eps, block_sizes, start="hot", save_q_fields=False, log=None):
    """Generate one independent Markov stream and measure every saved config.

    Returns a dict of numpy arrays (one row per measured configuration).
    """
    rng = stream_rng(seed, dims, beta)
    U = hot_start(dims, rng) if start == "hot" else cold_start(dims)
    up = Updater(dims, beta, rng)
    t_flow = flow_time_from_radius(flow_radius)
    tp = TwoPoint(dims)
    vol = int(np.prod(dims))
    t0 = time.time()
    for i in range(n_therm):
        up.update(U, n_or)
    if log:
        log(f"    seed {seed}: thermalized {n_therm} sweeps in {time.time() - t0:.1f}s, "
            f"plaq = {mean_plaquette(U):.5f}")
    rec = {k: [] for k in ("plaq", "Q", "E_clover", "t2E_plaq", "q_abs_max")}
    blocks = {l: [] for l in block_sizes}
    radial, axial, hists, qfields = [], [], [], []
    for c in range(n_configs):
        for _ in range(n_sweeps):
            up.update(U, n_or)
        rec["plaq"].append(mean_plaquette(U))
        V, hist = gradient_flow(U, t_flow, flow_eps)
        q, E = measure_flowed(V)
        rec["Q"].append(float(q.sum()))
        rec["E_clover"].append(float(E.mean()))
        rec["t2E_plaq"].append(float(hist[-1, 2]))
        rec["q_abs_max"].append(float(np.abs(q).max()))
        for l in block_sizes:
            blocks[l].append(block_average(q, l))
        r, a = tp.measure(q)
        radial.append(r)
        axial.append(a)
        hists.append(hist)
        if save_q_fields:
            qfields.append(q.astype(np.float32))
        if log and (c + 1) % max(1, n_configs // 5) == 0:
            log(f"    seed {seed}: config {c + 1}/{n_configs}  plaq={rec['plaq'][-1]:.5f}  Q={rec['Q'][-1]:+.3f}")
    out = {k: np.array(v) for k, v in rec.items()}
    out["blocks"] = {l: np.array(v) for l, v in blocks.items()}
    out["radial"] = np.array(radial)
    out["axial"] = np.array(axial)
    out["r_radial"] = np.sqrt(tp.r2_values.astype(float))
    out["flow_history"] = np.mean(np.array(hists), axis=0)
    out["seed"] = np.full(n_configs, int(seed))
    out["vol"] = vol
    if save_q_fields:
        out["q_fields"] = np.array(qfields)
    out["elapsed_s"] = time.time() - t0
    return out
