import numpy
import functools as ft
import itertools as it


# Public
def broadcast_sum(vecd):
    """broadcast several vectors together

    :param vecd: a dictionary of arrays, keyed by axis
    :type vecd: numpy.ndarray

    :rtype: numpy.ndarray
    """
    ndim = max(vecd) + 1
    expand = ft.partial(_expand, ndim)
    return sum(it.starmap(expand, vecd.items()))


# Private
def _expand(ndim, axis, v):
    """expand a vector for n-dimensional broadcasting along an axis

    :param axis: axis
    :type axis: int
    :param ndim: number of dimensions
    :type ndim: int
    :param v: vector
    :type v: numpy.ndarray

    :rtype: numpy.ndarray
    """
    ix = tuple(numpy.newaxis if not ax == axis else slice(None,)
               for ax in range(ndim))
    return numpy.array(v)[ix]


if __name__ == '__main__':
    a = [1, 2, 3]
    b = [-3, -2, -1]
    c = [1, 1, 1]
    z = broadcast_sum({0: a, 1: b, 2: c})
    print(z)
