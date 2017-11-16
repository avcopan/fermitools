import numpy
from toolz import functoolz


# Public
def ravel(a, ravd):
    """ravel axes

    :param a: array
    :type a: numpy.ndarray
    :param ravd: axes to ravel, keyed by the position of the compound axis
    :type ravd: dict

    :rtype: numpy.ndarray
    """
    compf = raveler(ravd)
    return compf(a)


def raveler(ravd):
    """ravels antisymmetric axes with ravel indices

    :param ravd: axes to ravel, keyed by the position of the compound axis
    :type ravd: dict

    :rtype: typing.Callable
    """
    npacks = len(ravd)
    packs = ravd.values()
    orig_axes = sum(packs, ())
    rav_axes = ravd.keys()

    def _preorder(a):
        source = orig_axes
        dest = tuple(range(len(source)))
        return numpy.moveaxis(a, source, dest)

    def _ravel(a):
        pack_sizes = map(len, packs)
        ravs_ = reversed(tuple(map(_raveler, pack_sizes)))
        rav_ = functoolz.compose(*ravs_)
        return rav_(a)

    def _reorder(a):
        source = tuple(range(a.ndim - npacks, a.ndim))
        dest = rav_axes
        return numpy.moveaxis(a, source, dest)

    return functoolz.compose(_reorder, _ravel, _preorder)


# Private
def _raveler(ndim):
    """ravels the first n dimensions of an array and moves them to the end

    :param ndim: the number of dimensions to ravel
    :type ndim: int

    :rtype: typing.Callable
    """

    def _ravel(a):
        shape = (numpy.product(a.shape[:ndim]),) + a.shape[ndim:]
        a_rav = numpy.reshape(a, shape)
        return numpy.moveaxis(a_rav, 0, -1)

    return _ravel
