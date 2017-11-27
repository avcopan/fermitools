import numpy
import itertools
from toolz import functoolz
from .._ravhelper import presorter
from .._ravhelper import resorter
from .._ravhelper import reverse_map
from .._ravhelper import dict_values
from .._ravhelper import dict_keys

# Extra imports
import more_itertools
from ..rav import raveler as nsym_raveler


# Public
def megaraveler(mravd):
    """ravel antisymmetric axes with compound indices, then ravel the result

    :param mravd: tuples of tuples of axes to compound, keyed by target axis
    :type mravd: dict

    :rtype: typing.Callable
    """
    final_axes = mravd.keys()
    mega_packs = mravd.values()
    stops = tuple(more_itertools.accumulate(map(len, mega_packs)))
    starts = (0,) + stops[:-1]
    intrm_packs = tuple(
            map(tuple, itertools.starmap(range, zip(starts, stops))))

    packs = sum(mega_packs, ())
    intrm_axes = sum(intrm_packs, ())
    packd = dict(zip(intrm_axes, packs))
    asym_ravf = raveler(packd)

    ravd = dict(zip(final_axes, intrm_packs))
    nsym_ravf = nsym_raveler(ravd)

    return functoolz.compose(nsym_ravf, asym_ravf)


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

        ix = itertools.combinations(range(udim), r=nuaxes)
        b = a[tuple(zip(*ix))]
        return numpy.moveaxis(b, 0, -1)

    return _ravel
