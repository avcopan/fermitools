import numpy
import itertools
import scipy.special
from toolz import functoolz
from .op import antisymmetrizer


# Public
def unravel(a, packd):
    """unravel antisymmetric axes with compound indices

    :param a: array
    :type a: numpy.ndarray
    :param packd: unraveled axis destinations, keyed by the compound axes
    :type packd: dict

    :rtype: numpy.ndarray
    """
    unravf = unraveler(packd)
    return unravf(a)


def unraveler(packd):
    """unravels antisymmetric axes with compound indices

    {rax1: (uax11, uax12, ...), rax2: ...}

    :param packd: unraveled axis destinations, keyed by the compound axes
    :type packd: dict

    :rtype: typing.Callable
    """
    rav_axes = packd.keys()
    unrav_axes = packd.values()

    def preorder(a):
        source = rav_axes
        dest = tuple(range(len(source)))
        return numpy.moveaxis(a, source, dest)

    def unravel(a):
        pack_sizes = map(len, unrav_axes)
        unravfs = reversed(tuple(map(_unraveler, pack_sizes)))
        unravf = functoolz.compose(*unravfs)
        return unravf(a)

    def reorder(a):
        nunrav = sum(map(len, unrav_axes))
        source = tuple(range(-nunrav, 0))
        dest = sum(unrav_axes, ())
        return numpy.moveaxis(a, source, dest)

    return functoolz.compose(reorder, unravel, preorder)


# Private
def _inverse_choose(n_choose_k, k):

    def lower_bound(n):
        return scipy.special.binom(n, k) < n_choose_k

    n = next(itertools.filterfalse(lower_bound, itertools.count()), 0.)

    assert scipy.special.binom(n, k) == n_choose_k

    return n


def _unraveler(ndim):
    """unravels the first dimension of an array and moves those axes to the end

    the compound index has the form (ijk...) where i<j<k<..., so for
    antisymmetric arrays this will contain all unique values

    :param shape: the shape of the unraveled dimensions
    :type shape: tuple

    :rtype: typing.Callable
    """

    def _unravel(a):
        rav_dim = a.shape[0]
        dim = _inverse_choose(rav_dim, ndim)
        shape = (dim,) * ndim
        ix = itertools.combinations(range(dim), ndim)
        a_unrav_tril = numpy.zeros(shape + a.shape[1:])
        a_unrav_tril[list(zip(*ix))] = a
        a_unrav = antisymmetrizer(range(ndim))(a_unrav_tril)
        return numpy.moveaxis(a_unrav, range(ndim), range(-ndim, 0))

    return _unravel
