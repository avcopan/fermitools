import numpy
import scipy.sparse


def evec_guess(ad, nvec, bd=None, highest=False):
    """approximate the lowest eigenvectors of a diagonally dominant matrix

    :param ad: matrix diagonal
    :type ad: numpy.ndarray
    :param nvec: the number of eigenvectors
    :type nvec: int
    :param bd: metric matrix diagonal
    :type bd: numpy.ndarray
    :param highest: estimate the highest eigenvectors, instead of the lowest?
    :type highest: bool

    :returns: guess vectors
    :rtype: numpy.ndarray
    """
    dim = numpy.size(ad)
    slc = slice(None, nvec) if not highest else slice(None, -nvec-1, -1)
    bd = 1. if bd is None else bd
    srt = numpy.argsort(ad / bd)
    vals = numpy.ones(nvec)
    keys = (srt[slc], range(nvec))
    return scipy.sparse.coo_matrix((vals, keys), shape=(dim, nvec)).toarray()
