"""SU(2) Wilson lattice gauge theory pilot for the Yang-Mills phase-fingerprint protocol.

Pure numpy/scipy. SU(2) link variables are stored as real unit quaternions
(a0, a1, a2, a3) <-> U = a0*1 + i*(a1*sigma1 + a2*sigma2 + a3*sigma3),
which is a 2x2 unitary matrix with det U = a0^2 + |a|^2 = 1.
"""

__version__ = "0.1.0"
