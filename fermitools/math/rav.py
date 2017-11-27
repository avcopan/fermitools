import numpy
from toolz import functoolz
from ._ravhelper import presorter
from ._ravhelper import resorter
from ._ravhelper import reverse_map
from ._ravhelper import dict_values
from ._ravhelper import dict_keys


# Public
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
        rdims = (numpy.product(udims),)
        rshape = numpy.concatenate((rdims, odims))

        b = numpy.reshape(a, rshape)
        return numpy.moveaxis(b, 0, -1)

    return _ravel
