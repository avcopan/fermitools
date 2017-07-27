import numpy


# Public
def ab2ov(dim, na, nb):
    """sort vector for occupied/virtual blocking wrt alpha/beta blocking

    Assumes alpha block comes first and that occupied indices precede virtual
    ones within each block, so that occupied/virtual blocking is achieved by
    swapping the two inner blocks (beta-occ and alpha-vir).  As a single block
    transposition, this permutation is self-inverse and also can be used to
    convert back to alpha/beta blocking.

    :param dim: number of spatial functions
    :type dim: int
    :param na: number of alpha electrons
    :type na: int
    :param nb: number of beta electrons
    :type nb: int

    :rtype: tuple
    """
    a_occ = tuple(range(0, na))
    a_vir = tuple(range(na, dim))
    b_occ = tuple(range(dim, dim + nb))
    b_vir = tuple(range(dim + nb, 2 * dim))
    return a_occ + b_occ + a_vir + b_vir


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
    args = (order if ax in axes else slice(None) for ax in range(a.ndim))
    ix = numpy.ix_(*args)
    return a[ix]
