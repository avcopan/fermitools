import numpy
import scipy
import scipy.linalg


def diag(a, dim):
    """diagonal matrix elements of a linear operator

    :param a: linear operator
    :type a: typing.Callable
    :param dim: the dimension of a (if rectangular, use the smaller one)
    :type dim: int
    """
    a_diag = tuple(a(ei)[i] for i, ei in enumerate(numpy.eye(dim)))
    return numpy.array(a_diag)


def evec_guess(ad, nvec):
    """approximate the lowest eigenvectors of a diagonally dominant matrix

    :param ad: matrix diagonal
    :type ad: numpy.ndarray
    :param nvec: the number of eigenvectors
    :type nvec: int
    """
    dim = numpy.size(ad)
    srt = numpy.argsort(ad)
    vals = numpy.ones(nvec)
    keys = (srt[:nvec], range(nvec))
    return scipy.sparse.coo_matrix((vals, keys), shape=(dim, nvec)).toarray()
