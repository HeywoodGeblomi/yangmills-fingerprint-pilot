"""Gauge-invariant local observables: clover field strength, topological
charge density, energy density.

Clover: Q_mu,nu(x) = sum of the 4 plaquette leaves in the (mu,nu) plane that
start and end at x. The hermitian clover field strength is
    F_mu,nu = -(i/8) [ Q - Q^+ ]   (traceless part; automatic for SU(2)).
With Q = c0 + i c.s this is F = (c/4).s, i.e. F is stored as the real
3-vector f = c/4 and tr(F_a F_b) = 2 f_a . f_b.

Topological charge density (lattice units):
    q(x) = (1/32 pi^2) eps_{mu nu rho sigma} tr(F_mu,nu F_rho,sigma)
         = (1/4 pi^2) [ tr(F01 F23) - tr(F02 F13) + tr(F03 F12) ].
After enough gradient flow, Q = sum_x q(x) is close to an integer.

Energy density (clover): E = sum_{mu<nu} tr(F_mu,nu^2).
"""

import numpy as np

from .lattice import shift
from .su2 import qmul, dag


def clover_leaves_sum(U, mu, nu):
    """Q_mu,nu(x): sum of the 4 leaves (quaternion field)."""
    Umu, Unu = U[mu], U[nu]
    Umu_xpnu = shift(Umu, nu, 1)
    Unu_xpmu = shift(Unu, mu, 1)
    Umu_xmmu = shift(Umu, mu, -1)
    Unu_xmmu = shift(Unu, mu, -1)
    Umu_xmnu = shift(Umu, nu, -1)
    Unu_xmnu = shift(Unu, nu, -1)
    # L1 = U_mu(x) U_nu(x+mu) U_mu(x+nu)^+ U_nu(x)^+
    L1 = qmul(qmul(Umu, Unu_xpmu), qmul(dag(Umu_xpnu), dag(Unu)))
    # L2 = U_nu(x) U_mu(x+nu-mu)^+ U_nu(x-mu)^+ U_mu(x-mu)
    L2 = qmul(qmul(Unu, dag(shift(Umu_xmmu, nu, 1))), qmul(dag(Unu_xmmu), Umu_xmmu))
    # L3 = U_mu(x-mu)^+ U_nu(x-mu-nu)^+ U_mu(x-mu-nu) U_nu(x-nu)
    L3 = qmul(qmul(dag(Umu_xmmu), dag(shift(Unu_xmmu, nu, -1))),
              qmul(shift(Umu_xmmu, nu, -1), Unu_xmnu))
    # L4 = U_nu(x-nu)^+ U_mu(x-nu) U_nu(x-nu+mu) U_mu(x)^+
    L4 = qmul(qmul(dag(Unu_xmnu), Umu_xmnu), qmul(shift(Unu_xpmu, nu, -1), dag(Umu)))
    return L1 + L2 + L3 + L4


def clover_field_strength(U):
    """Dict {(mu,nu): f} with F_mu,nu = f.s (hermitian), f = (vector part of Q)/4."""
    F = {}
    for mu in range(4):
        for nu in range(mu + 1, 4):
            F[(mu, nu)] = 0.25 * clover_leaves_sum(U, mu, nu)[..., 1:]
    return F


def _trFF(fa, fb):
    return 2.0 * np.sum(fa * fb, axis=-1)


def topological_charge_density(U, F=None):
    """q(x) on every site (array of shape dims)."""
    if F is None:
        F = clover_field_strength(U)
    s = (_trFF(F[(0, 1)], F[(2, 3)]) - _trFF(F[(0, 2)], F[(1, 3)])
         + _trFF(F[(0, 3)], F[(1, 2)]))
    return s / (4.0 * np.pi ** 2)


def energy_density_clover(U, F=None):
    """E(x) = sum_{mu<nu} tr(F_mu,nu^2) on every site."""
    if F is None:
        F = clover_field_strength(U)
    return sum(_trFF(f, f) for f in F.values())


def measure_flowed(V):
    """Return (q field, E field) for a (flowed) configuration."""
    F = clover_field_strength(V)
    return topological_charge_density(V, F), energy_density_clover(V, F)
