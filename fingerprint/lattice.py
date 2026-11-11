"""Periodic 4D lattice helpers: shifts, checkerboard, staples, plaquettes,
gauge transformations and reflections.

Gauge field layout: U has shape (4, L0, L1, L2, L3, 4); U[mu] is the link
field in direction mu (axis mu of the site array), last axis = quaternion.
Axis 0 is treated as Euclidean time.
"""

import numpy as np

from .su2 import qmul, dag, identity, random_su2

NDIM = 4


def shift(f, mu, s=1):
    """g(x) = f(x + s*mu_hat) on the periodic torus (site axes first)."""
    return np.roll(f, -s, axis=mu)


def checkerboard(dims):
    """Integer array (sum of coordinates) mod 2."""
    grids = np.meshgrid(*[np.arange(n) for n in dims], indexing="ij")
    return (sum(grids) % 2).astype(np.int8)


def cold_start(dims):
    return identity((NDIM,) + tuple(dims))


def hot_start(dims, rng):
    return random_su2(rng, (NDIM,) + tuple(dims))


def staple_sum(U, mu):
    """A_mu(x) = sum_{nu != mu} [ U_nu(x+mu) U_mu(x+nu)^+ U_nu(x)^+
                                 + U_nu(x+mu-nu)^+ U_mu(x-nu)^+ U_nu(x-nu) ]
    so that Re tr(U_mu(x) A_mu(x)) is the sum of the 6 plaquettes containing
    the link (x, mu)."""
    A = np.zeros(U.shape[1:])
    for nu in range(NDIM):
        if nu == mu:
            continue
        unu_xpmu = shift(U[nu], mu, 1)
        A += qmul(qmul(unu_xpmu, dag(shift(U[mu], nu, 1))), dag(U[nu]))
        A += qmul(qmul(dag(shift(unu_xpmu, nu, -1)), dag(shift(U[mu], nu, -1))),
                  shift(U[nu], nu, -1))
    return A


def plaquette_field(U, mu, nu):
    """P_mu,nu(x) = U_mu(x) U_nu(x+mu) U_mu(x+nu)^+ U_nu(x)^+."""
    return qmul(qmul(U[mu], shift(U[nu], mu, 1)),
                qmul(dag(shift(U[mu], nu, 1)), dag(U[nu])))


def plaquette_planes():
    return [(mu, nu) for mu in range(NDIM) for nu in range(mu + 1, NDIM)]


def mean_plaquette(U):
    """<(1/2) Re tr P> averaged over sites and the 6 planes."""
    return float(np.mean([plaquette_field(U, mu, nu)[..., 0].mean()
                          for mu, nu in plaquette_planes()]))


def wilson_action(U, beta):
    """S = beta * sum_p (1 - (1/2) Re tr U_p)."""
    vol = np.prod(U.shape[1:-1])
    return float(beta * 6 * vol * (1.0 - mean_plaquette(U)))


def gauge_transform(U, g):
    """U'_mu(x) = g(x) U_mu(x) g(x+mu)^+ for g of shape (L0..L3, 4)."""
    out = np.empty_like(U)
    for mu in range(NDIM):
        out[mu] = qmul(qmul(g, U[mu]), dag(shift(g, mu, 1)))
    return out


def reflect_sites(f, axes):
    """(Pf)(x) = f(Px) with Px_i = -x_i mod L_i for i in axes."""
    for ax in axes:
        f = np.roll(np.flip(f, axis=ax), 1, axis=ax)
    return f


def reflect_gauge_field(U, axes=(1, 2, 3)):
    """Reflection of the configuration through x_i -> -x_i, i in axes.

    Default axes (1, 2, 3) is spatial parity with axis 0 as time.
    Links along a reflected axis become U_mu(Px - mu)^+ ; others U_mu(Px).
    """
    out = np.empty_like(U)
    for mu in range(NDIM):
        if mu in axes:
            out[mu] = reflect_sites(dag(shift(U[mu], mu, -1)), axes)
        else:
            out[mu] = reflect_sites(U[mu], axes)
    return out


def random_gauge(dims, rng):
    return random_su2(rng, tuple(dims))
