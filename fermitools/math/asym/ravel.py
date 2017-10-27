import numpy
import itertools
from toolz import functoolz


# Public
def compound_index(a, packd):
    """ravel antisymmetric axes with a compound index

    :param a: array
    :type a: numpy.ndarray
    :param packd: axes to compound, keyed by the position of the compound axis
    :type packd: dict

    :rtype: numpy.ndarray
    """
    compf = compound_indexer(packd)
    return compf(a)


def compound_indexer(packd):
    """ravels antisymmetric axes with a compound index

    :param packd: axes to compound, keyed by the position of the compound axis
    :type packd: dict

    :rtype: typing.Callable
    """
    npacks = len(packd)
    packs = packd.values()
    orig_axes = sum(packs, ())
    comp_axes = packd.keys()

    def preorder(a):
        source = orig_axes
        dest = tuple(range(len(source)))
        return numpy.moveaxis(a, source, dest)

    def compound(a):
        pack_sizes = map(len, packs)
        compounders = reversed(tuple(map(_compounder, pack_sizes)))
        compounder = functoolz.compose(*compounders)
        return compounder(a)

    def reorder(a):
        source = tuple(range(a.ndim - npacks, a.ndim))
        dest = comp_axes
        return numpy.moveaxis(a, source, dest)

    return functoolz.compose(reorder, compound, preorder)


# Private
def _compounder(ndim):
    """compounds the first n dimensions of an array and moves them to the end

    the compound index has the form (ijk...) where i<j<k<..., so for
    antisymmetric arrays this will retain all unique values

    :param ndim: the number of dimensions to compound
    :type ndim: int

    :rtype: typing.Callable
    """

    def compound(a):
        assert all(dim == a.shape[0] for dim in a.shape[:ndim])
        ix = itertools.combinations(range(a.shape[0]), ndim)
        a_comp = a[list(zip(*ix))]
        return numpy.moveaxis(a_comp, 0, -1)

    return compound
