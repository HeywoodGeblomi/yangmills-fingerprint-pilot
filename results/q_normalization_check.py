"""Normalization check of the clover q(x): abelian constant-flux configuration on the
torus, U = exp(i theta sigma3), flux n1 through (0,1) planes and n2 through (2,3)
planes. Expected continuum value for this sigma3 embedding: Q = 2 n1 n2 (from
tr(sigma3^2) = 2 in q = (1/32 pi^2) eps tr(F F)); lattice artifacts O(phi^2),
phi = 2 pi n / L^2."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from fingerprint.observables import topological_charge_density
from fingerprint.flow import gradient_flow


def flux_config(L, n1, n2):
    th = np.zeros((4, L, L, L, L))
    x = np.indices((L, L, L, L))
    th[1] = 2 * np.pi * n1 * x[0] / L**2
    th[0] = np.where(x[0] == L - 1, -2 * np.pi * n1 * x[1] / L, 0.0)
    th[3] = 2 * np.pi * n2 * x[2] / L**2
    th[2] = np.where(x[2] == L - 1, -2 * np.pi * n2 * x[3] / L, 0.0)
    U = np.zeros((4, L, L, L, L, 4))
    U[..., 0] = np.cos(th)
    U[..., 3] = np.sin(th)
    return U


for L in (8, 12, 16):
    for n1, n2 in ((1, 1), (1, -1), (2, 1)):
        U = flux_config(L, n1, n2)
        Q = topological_charge_density(U).sum()
        Vf, _ = gradient_flow(U, 0.5, 0.025, record=False) if L <= 12 else (U, None)
        Qf = topological_charge_density(Vf).sum()
        print(f"L={L:2d} n1={n1:+d} n2={n2:+d}: Q(clover, unflowed) = {Q:+.4f}, after flow t=0.5: {Qf:+.4f}, "
              f"expected 2*n1*n2 = {2*n1*n2:+d}")
