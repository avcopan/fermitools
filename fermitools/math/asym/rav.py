import numpy
import itertools
import more_itertools
from toolz import functoolz
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


def ravel(a, packd):
    """ravel antisymmetric axes with compound indices

    :param a: array
    :type a: numpy.ndarray
    :param packd: axes to compound, keyed by the position of the target axis
    :type packd: dict

    :rtype: numpy.ndarray
    """
    ravf = raveler(packd)
    return ravf(a)


def raveler(packd):
    """ravels antisymmetric axes with compound indices

    :param packd: axes to compound, keyed by the position of the target axis
    :type packd: dict

    :rtype: typing.Callable
    """
    npacks = len(packd)
    packs = packd.values()
    orig_axes = sum(packs, ())
    comp_axes = packd.keys()

    def _preorder(a):
        source = orig_axes
        dest = tuple(range(len(source)))
        return numpy.moveaxis(a, source, dest)

    def _ravel(a):
        pack_sizes = map(len, packs)
        ravfs = reversed(tuple(map(_raveler, pack_sizes)))
        ravf = functoolz.compose(*ravfs)
        return ravf(a)

    def _reorder(a):
        source = tuple(range(a.ndim - npacks, a.ndim))
        dest = comp_axes
        return numpy.moveaxis(a, source, dest)

    return functoolz.compose(_reorder, _ravel, _preorder)


# Private
def _raveler(ndim):
    """compounds the first n dimensions of an array and moves them to the end

    the compound index has the form (ijk...) where i<j<k<..., so for
    antisymmetric arrays this will retain all unique values

    :param ndim: the number of dimensions to compound
    :type ndim: int

    :rtype: typing.Callable
    """

    def _ravel(a):
        assert len(set(a.shape[:ndim])) == 1
        ix = itertools.combinations(range(a.shape[0]), ndim)
        a_comp = a[list(zip(*ix))]
        return numpy.moveaxis(a_comp, 0, -1)

    return _ravel
