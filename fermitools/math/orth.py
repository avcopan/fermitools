import numpy
import scipy.linalg


def orthogonalize(a, against=None, tol=None):
    m, n = a.shape
    b = numpy.zeros((m, 0)) if against is None else against
    tol = max(m, n) * numpy.finfo(float).eps if tol is None else tol

    a_proj = a - numpy.linalg.multi_dot([b, b.T, a])
    a_orth, svals, _ = scipy.linalg.svd(a_proj)
    nkeep = numpy.sum(svals > tol, dtype=int)
    return a_orth[:, :nkeep]
