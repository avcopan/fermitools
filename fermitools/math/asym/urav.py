import numpy
import scipy
import itertools
from toolz import functoolz
from . import antisymmetrizer
from ..urav import unraveler as ordinary_unraveler
from .._ravhelper import presorter
from .._ravhelper import resorter
from .._ravhelper import reverse_starmap
from .._ravhelper import dict_values
from .._ravhelper import dict_keys
from .._ravhelper import dict_items
from ...iter import split


# Public
def megaunravel(a, d):
    """unravel antisymmetric axes, then unravel the result

    :param a: array
    :type a: numpy.ndarray
    :param d: {rax1: {(uax111, uax112, ...): dim11, ...}, rax2: ...}
    :type d: dict

    :rtype: typing.Callable
    """
    uravf = megaunraveler(d)
    return uravf(a)


def megaunraveler(d):
    """does an ordinary unravel, followed by an antisymmetric unravel

    :param d: {rax1: {(uax111, uax112, ...): dim11, ...}, rax2: ...}
    :type d: dict

    :rtype: typing.Callable
    """
    raxes1 = dict_keys(d)
    iter_uaxes2, iter_udims2 = zip(*sum(map(dict_items, dict_values(d)), ()))
    iter_nuaxes2 = map(len, iter_uaxes2)
    iter_nuaxes1 = map(len, dict_values(d))
    iter_udims1 = map(int, itertools.starmap(scipy.special.binom,
                                             zip(iter_udims2, iter_nuaxes2)))
    iter_uaxes1 = map(dict,
                      split(i=enumerate(iter_udims1), sizes=iter_nuaxes1))

    d2 = dict(enumerate(zip(iter_uaxes2, iter_udims2)))
    uravf1 = ordinary_unraveler(dict(zip(raxes1, iter_uaxes1)))
    uravf2 = unraveler(d2)
    return functoolz.compose(uravf2, uravf1)


def unravel(a, d):
    """unravel antisymmetric axes with compound indices

    :param a: array
    :type a: numpy.ndarray
    :param d: {rax1: (uax11, uax12, ...), rax2: ...}
    :type d: dict

    :rtype: numpy.ndarray
    """
    uravf = unraveler(d)
    return uravf(a)


def unraveler(d):
    """unravels antisymmetric axes of an array

    :param d: {rax1: ((uax11, uax12, ...), udim1), rax2: ...}
    :type d: dict

    :rtype: typing.Callable
    """
    raxes = dict_keys(d)
    iter_uaxes, iter_udim = zip(*dict_values(d))
    iter_nuaxes = map(len, iter_uaxes)
    uaxes = sum(iter_uaxes, ())
    unravelers = reverse_starmap(_unraveler, zip(iter_nuaxes, iter_udim))

    presortf = presorter(src=raxes)
    uravf = functoolz.compose(*unravelers)
    resortf = resorter(dst=uaxes)
    return functoolz.compose(resortf, uravf, presortf)


# Private
__all__ = ['unravel', 'unraveler']


def _unraveler(nuaxes, udim):

    def _unravel(a):
        (rdim,), odims = numpy.split(a.shape, (1,))
        udims = (udim,) * nuaxes
        b = numpy.zeros(numpy.concatenate((udims, odims)))
        ix = tuple(zip(*itertools.combinations(range(udim), r=nuaxes)))
        b[ix] = a
        c = antisymmetrizer(range(nuaxes))(b)
        return numpy.moveaxis(c, range(nuaxes), range(-nuaxes, 0))

    return _unravel
