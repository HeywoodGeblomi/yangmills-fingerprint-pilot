"""Kennedy-Pendleton SU(2) heatbath and overrelaxation for the Wilson action.

Action: S = beta * sum_p (1 - (1/2) Re tr U_p). For a link U = U_mu(x) with
staple sum A (see lattice.staple_sum), the local weight is
    exp((beta/2) Re tr(U A)) = exp(beta * k * w0),  W = U V,
with k = sqrt(det A), V = A / k in SU(2) and w0 = (1/2) tr W. The heatbath
draws W from sqrt(1 - w0^2) exp(alpha w0) dw0 x (uniform direction),
alpha = beta * k, and sets U = W V^+.

Links of one direction mu are updated on one checkerboard parity at a time:
the staple of an even-site mu-link only contains mu-links on odd sites (and
links in other directions), so a same-parity, same-direction update is exact.
The staple is recomputed before each half-update.
"""

import numpy as np

from .lattice import NDIM, staple_sum, checkerboard
from .su2 import qmul, dag

_MAX_KP_ITER = 10000


def kp_sample_w0(alpha, rng):
    """Kennedy-Pendleton draw of w0 in [-1, 1] with density
    ~ sqrt(1 - w0^2) exp(alpha w0), vectorized over alpha (alpha > 0)."""
    alpha = np.asarray(alpha, dtype=np.float64)
    out = np.empty_like(alpha)
    todo = np.arange(alpha.size)
    a_flat = alpha.ravel()
    o_flat = out.reshape(-1)
    for _ in range(_MAX_KP_ITER):
        if todo.size == 0:
            break
        a = a_flat[todo]
        r1 = 1.0 - rng.random(todo.size)  # (0, 1]
        r2 = rng.random(todo.size)
        r3 = 1.0 - rng.random(todo.size)
        r4 = rng.random(todo.size)
        lam2 = -(np.log(r1) + np.cos(2.0 * np.pi * r2) ** 2 * np.log(r3)) / (2.0 * a)
        acc = r4 * r4 <= 1.0 - lam2
        o_flat[todo[acc]] = 1.0 - 2.0 * lam2[acc]
        todo = todo[~acc]
    if todo.size:
        raise RuntimeError("Kennedy-Pendleton heatbath failed to converge")
    return out


def random_unit_vectors(n, rng):
    v = rng.standard_normal((n, 3))
    return v / np.linalg.norm(v, axis=1)[:, None]


def heatbath_su2(A, beta, rng):
    """New links U distributed as exp((beta/2) Re tr(U A)) for staples A (n, 4)."""
    k = np.sqrt(np.sum(A * A, axis=-1))
    k = np.maximum(k, 1e-300)
    V = A / k[:, None]
    w0 = kp_sample_w0(beta * k, rng)
    W = np.empty_like(A)
    W[:, 0] = w0
    W[:, 1:] = np.sqrt(np.clip(1.0 - w0 * w0, 0.0, None))[:, None] * random_unit_vectors(len(w0), rng)
    return qmul(W, dag(V))


def overrelax_su2(U, A):
    """Microcanonical reflection U -> V^+ U^+ V^+ (V = A/|A|): preserves Re tr(U A)."""
    k = np.sqrt(np.sum(A * A, axis=-1))
    V = A / np.maximum(k, 1e-300)[:, None]
    Vd = dag(V)
    return qmul(qmul(Vd, dag(U)), Vd)


class Updater:
    """Checkerboard heatbath / overrelaxation sweeps on a gauge field in place."""

    def __init__(self, dims, beta, rng):
        self.beta = float(beta)
        self.rng = rng
        cb = checkerboard(dims)
        self.masks = [cb == 0, cb == 1]

    def _half(self, U, mu, mask, kind):
        A = staple_sum(U, mu)[mask]
        Uold = U[mu][mask]
        if kind == "hb":
            new = heatbath_su2(A, self.beta, self.rng)
        else:
            new = overrelax_su2(Uold, A)
        # reunitarize to stop round-off drift
        new /= np.sqrt(np.sum(new * new, axis=-1))[:, None]
        U[mu][mask] = new

    def sweep(self, U, kind="hb"):
        for mu in range(NDIM):
            for mask in self.masks:
                self._half(U, mu, mask, kind)
        return U

    def update(self, U, n_or=0):
        """One compound update: 1 heatbath sweep followed by n_or overrelaxation sweeps."""
        self.sweep(U, "hb")
        for _ in range(n_or):
            self.sweep(U, "or")
        return U
