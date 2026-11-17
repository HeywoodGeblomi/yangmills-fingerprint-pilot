"""Side check: does flowed Q approach integers at beta=2.3 if we flow longer?
8^4, 6 configs, Q(t) at t = 0.5, 1, 2, 3, 4 (sqrt(8t) = 2, 2.8, 4, 4.9, 5.7)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from fingerprint.lattice import hot_start
from fingerprint.heatbath import Updater
from fingerprint.flow import rk3_step
from fingerprint.observables import topological_charge_density
rng = np.random.default_rng(11)
dims = (8,) * 4
U = hot_start(dims, rng); up = Updater(dims, 2.3, rng)
for _ in range(200): up.update(U, 2)
ts = [0.5, 1.0, 2.0, 3.0, 4.0]; eps = 0.025
rows = []
for c in range(6):
    for _ in range(20): up.update(U, 2)
    V = U.copy(); t = 0.0; out = []
    for tt in ts:
        while t < tt - 1e-9:
            V = rk3_step(V, eps); t += eps
        out.append(float(topological_charge_density(V).sum()))
    rows.append(out)
    print("config", c, " ".join(f"t={tt}:{q:+.3f}" for tt, q in zip(ts, out)), flush=True)
R = np.array(rows)
for j, tt in enumerate(ts):
    print(f"t={tt}: mean|Q-round(Q)| = {np.mean(np.abs(R[:, j] - np.rint(R[:, j]))):.3f}")
