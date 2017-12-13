"""Provides broadcasting operations."""
import numpy
import itertools as it
import more_itertools as mit


# Public
def broadcast_sum(vecd):
    """broadcast several vectors together

    :param vecd: a dictionary of arrays, keyed by axis
    :type vecd: numpy.ndarray

    :rtype: numpy.ndarray
    """
    axkeys, arrays = zip(*vecd.items())
    axtups = tuple(map(tuple, map(mit.always_iterable, axkeys)))
    ndim = max(it.chain(*axtups)) + 1
    return sum(_expand(a, axes, ndim) for a, axes in zip(arrays, axtups))


# Private
def _expand(a, axes, ndim):
    """expand an array over n dimensions for broadcasting

    :param axis: axes along which to broadcast the array
    :type axis: tuple
    :param ndim: number of dimensions
    :type ndim: int
    :param a: array
    :type a: numpy.ndarray

    :rtype: numpy.ndarray
    """
    atrans = numpy.transpose(numpy.atleast_1d(a), numpy.argsort(axes))
    ix = tuple(numpy.newaxis if ax not in axes else slice(None,)
               for ax in range(ndim))
    return atrans[ix]
