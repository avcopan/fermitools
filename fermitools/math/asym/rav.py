import numpy
import itertools
from toolz import functoolz
from .._ravhelper import presorter
from .._ravhelper import resorter
from .._ravhelper import reverse_map
from .._ravhelper import dict_values
from .._ravhelper import dict_keys

# Extra imports
from ..rav import raveler as ordinary_raveler
from ...iter import split


# Public
def megaraveler(d):
    """does an antisymmetric ravel, followed by an ordinary ravel

    :param d: {rax1: ((uax111, uax112, ...), ...), rax2: ...}
    :type d: dict

    :rtype: typing.Callable
    """
    # Antisymmetric ravel
    raxes1, iter_uaxes1 = zip(*enumerate(sum(dict_values(d), ())))
    ravf1 = raveler(dict(zip(raxes1, iter_uaxes1)))

    # Ordinary ravel
    raxes2 = dict_keys(d)
    iter_nuaxes2 = map(len, dict_values(d))
    iter_uaxes2 = split(i=raxes1, sizes=iter_nuaxes2)
    ravf2 = ordinary_raveler(dict(zip(raxes2, iter_uaxes2)))

    return functoolz.compose(ravf2, ravf1)


def ravel(a, d):
    """ravel axes of an array

    :param a: array
    :type a: numpy.ndarray
    :param d: {rax1: (uax11, uax12, ...), rax2: ...}
    :type d: dict

    :rtype: typing.Callable
    """
    ravf = raveler(d)
    return ravf(a)


def raveler(d):
    """ravels axes of an array

    :param d: {rax1: (uax11, uax12, ...), rax2: ...}
    :type d: dict

    :rtype: typing.Callable
    """
    raxes = dict_keys(d)
    uaxes = sum(dict_values(d), ())
    iter_nuaxes = map(len, dict_values(d))
    ravelers = reverse_map(_raveler, iter_nuaxes)

    presortf = presorter(src=uaxes)
    ravf = functoolz.compose(*ravelers)
    resortf = resorter(dst=raxes)
    return functoolz.compose(resortf, ravf, presortf)


# Private
__all__ = ['ravel', 'raveler']


def _raveler(nuaxes):
    """ravels the first n dimensions of an array and moves them to the end

    :param nuaxes: the number of axes to ravel
    :type nuaxes: int

    :rtype: typing.Callable
    """

    def _ravel(a):
        udims, odims = numpy.split(a.shape, (nuaxes,))
        udim = udims[0]
        assert all(numpy.equal(udims, udim))

        ix = tuple(zip(*itertools.combinations(range(udim), r=nuaxes)))
        b = a[ix]
        return numpy.moveaxis(b, 0, -1)

    return _ravel
