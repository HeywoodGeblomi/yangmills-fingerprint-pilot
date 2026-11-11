"""SU(2) algebra in the quaternion (Pauli-matrix) basis, vectorized over arrays.

A quaternion a = (a0, a1, a2, a3) (last array axis) represents the 2x2 matrix
    M(a) = a0*1 + i*(a1*s1 + a2*s2 + a3*s3),
with s_k the Pauli matrices. M(a) is in SU(2) iff |a| = 1. Sums of SU(2)
matrices (staples, clover leaves) stay in this real 4-dim algebra, so
everything here is real arithmetic. tr M(a) = 2*a0, det M(a) = |a|^2.
"""

import numpy as np


def qmul(a, b):
    """Matrix product M(a) M(b) in quaternion components."""
    a0, a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    b0, b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    out = np.empty(np.broadcast_shapes(a.shape, b.shape), dtype=np.float64)
    # (a0 + i a.s)(b0 + i b.s) = a0 b0 - a.b + i (a0 b + b0 a - a x b).s
    out[..., 0] = a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3
    out[..., 1] = a0 * b1 + b0 * a1 - (a2 * b3 - a3 * b2)
    out[..., 2] = a0 * b2 + b0 * a2 - (a3 * b1 - a1 * b3)
    out[..., 3] = a0 * b3 + b0 * a3 - (a1 * b2 - a2 * b1)
    return out


def dag(a):
    """Hermitian conjugate M(a)^dagger."""
    out = -a
    out[..., 0] = a[..., 0]
    return out


def norm(a):
    """sqrt(det M(a))."""
    return np.sqrt(np.sum(a * a, axis=-1))


def normalize(a):
    """Project a nonzero quaternion back onto SU(2) (reunitarization)."""
    return a / norm(a)[..., None]


def identity(shape):
    out = np.zeros(tuple(shape) + (4,))
    out[..., 0] = 1.0
    return out


def random_su2(rng, shape):
    """Haar-random SU(2) elements (uniform on the 3-sphere)."""
    x = rng.standard_normal(tuple(shape) + (4,))
    return normalize(x)


def expm_iv(v):
    """exp(i v.s) for a real 3-vector field v (last axis 3) -> quaternion."""
    th = np.sqrt(np.sum(v * v, axis=-1))
    out = np.empty(v.shape[:-1] + (4,))
    out[..., 0] = np.cos(th)
    # sin(th)/th with a safe small-angle limit
    s = np.where(th > 1e-12, np.sin(th) / np.where(th > 1e-12, th, 1.0), 1.0 - th * th / 6.0)
    out[..., 1:] = v * s[..., None]
    return out


def to_matrix(a):
    """Explicit 2x2 complex matrices (for tests / interoperability)."""
    a0, a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    m = np.empty(a.shape[:-1] + (2, 2), dtype=np.complex128)
    m[..., 0, 0] = a0 + 1j * a3
    m[..., 0, 1] = a2 + 1j * a1
    m[..., 1, 0] = -a2 + 1j * a1
    m[..., 1, 1] = a0 - 1j * a3
    return m
