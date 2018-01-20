import numpy


# Public
def sort(a, order, axes):
    """reorder several axes of an array

    :param a: array
    :type a: numpy.ndarray
    :param order: sort order
    :type order: tuple
    :param axes: axes to be sorted
    :type axes: tuple

    :rtype: numpy.ndarray
    """
    args = tuple(order if ax in axes else range(a.shape[ax])
                 for ax in range(a.ndim))
    ix = numpy.ix_(*args)
    return a[ix]


def ab2ov(dim, na, nb):
    """sort vector for occupied/virtual blocking

    :param dim: number of spatial functions
    :type dim: int
    :param na: number of alpha electrons
    :type na: int
    :param nb: number of beta electrons
    :type nb: int

    :rtype: tuple
    """
    ao = tuple(range(0, na))
    av = tuple(range(na, dim))
    bo = tuple(range(dim, dim+nb))
    bv = tuple(range(dim+nb, 2*dim))
    return ao + bo + av + bv


def ov2ab(dim, na, nb):
    """sort vector for alpha/beta blocking

    :param dim: number of spatial functions
    :type dim: int
    :param na: number of alpha electrons
    :type na: int
    :param nb: number of beta electrons
    :type nb: int

    :rtype: tuple
    """
    oa = tuple(range(0, na))
    ob = tuple(range(na, na+nb))
    va = tuple(range(na+nb, dim+nb))
    vb = tuple(range(dim+nb, 2*dim))
    return oa + va + ob + vb
