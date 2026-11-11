"""Wilson gradient flow with Luscher's third-order Runge-Kutta integrator.

Flow equation (Luscher, JHEP 08 (2010) 071):
    dV_mu(x)/dt = Z_mu(x) V_mu(x),   Z_mu(x) = -P_ah[ V_mu(x) A_mu(x) ],
with A_mu the Wilson staple sum and P_ah the traceless anti-hermitian
projection. For SU(2) in the quaternion basis P_ah(b0 + i b.s) = i b.s, so Z
is stored as the real 3-vector z with Z = i z.s.

RK3 step (eps):
    W1 = exp(Z0/4) W0
    W2 = exp(8/9 Z1 - 17/36 Z0) W1
    W3 = exp(3/4 Z2 - 8/9 Z1 + 17/36 Z0) W2,     Zi = eps * Z(Wi).

The flow time is set from the smoothing radius r = sqrt(8 t) (lattice units).
"""

import math

import numpy as np

from .lattice import NDIM, staple_sum, mean_plaquette
from .su2 import qmul, expm_iv, normalize


def flow_force(U):
    Z = np.empty(U.shape[:-1] + (3,))
    for mu in range(NDIM):
        om = qmul(U[mu], staple_sum(U, mu))
        Z[mu] = -om[..., 1:]
    return Z


def _apply(z, U):
    return normalize(qmul(expm_iv(z), U))


def rk3_step(U, eps):
    Z0 = eps * flow_force(U)
    W1 = _apply(0.25 * Z0, U)
    Z1 = eps * flow_force(W1)
    W2 = _apply((8.0 / 9.0) * Z1 - (17.0 / 36.0) * Z0, W1)
    Z2 = eps * flow_force(W2)
    return _apply(0.75 * Z2 - (8.0 / 9.0) * Z1 + (17.0 / 36.0) * Z0, W2)


def flow_time_from_radius(radius):
    """t such that sqrt(8 t) = radius (lattice units)."""
    return radius * radius / 8.0


def energy_density_plaq(U):
    """Plaquette discretization E = 2 sum_{mu<nu} Re tr(1 - P_mu,nu), per site.
    For SU(2): Re tr(1 - P) = 2 (1 - p0), so E = 24 (1 - <p0>)."""
    return 24.0 * (1.0 - mean_plaquette(U))


def gradient_flow(U, t_final, eps=0.02, record=True):
    """Flow U to time t_final. Returns (V, history) with history rows
    (t, E_plaq, t^2 E_plaq). The input array is not modified."""
    V = U.copy()
    hist = []
    if t_final <= 0:
        e = energy_density_plaq(V)
        return V, np.array([[0.0, e, 0.0]])
    n = max(1, int(math.ceil(t_final / eps - 1e-12)))
    h = t_final / n
    if record:
        e = energy_density_plaq(V)
        hist.append((0.0, e, 0.0))
    for i in range(1, n + 1):
        V = rk3_step(V, h)
        if record:
            t = i * h
            e = energy_density_plaq(V)
            hist.append((t, e, t * t * e))
    return V, np.array(hist)
