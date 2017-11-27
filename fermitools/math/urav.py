import numpy
from toolz import functoolz
from ._ravhelper import presorter
from ._ravhelper import resorter
from ._ravhelper import reverse_map
from ._ravhelper import dict_values
from ._ravhelper import dict_keys


# Public
def unravel(a, d):
    uravf = unraveler(d)
    return uravf(a)


def unraveler(d):
    """unravels axes of an array

    :param d: {rax1: {uax11: udim11, uax12: udim12, ...}, rax2: ...}
    :type packd: dict

    :rtype: typing.Callable
    """
    raxes = dict_keys(d)
    uaxes = sum(map(dict_keys, dict_values(d)), ())
    iter_udims = map(dict_values, dict_values(d))
    unravelers = reverse_map(_unraveler, iter_udims)

    presortf = presorter(src=raxes)
    uravf = functoolz.compose(*unravelers)
    resortf = resorter(dst=uaxes)
    return functoolz.compose(resortf, uravf, presortf)


# Private
__all__ = ['unravel', 'unraveler']


def _unraveler(udims):
    """unravels the first axis an array and moves the new axes to the end

    :param udims: the dimensions of the unraveled axes
    :type udims: tuple

    :rtype: typing.Callable
    """

    def _unravel(a):
        rdims, odims = numpy.split(a.shape, (1,))
        nuaxes = len(udims)
        ushape = numpy.concatenate((udims, odims))

        b = numpy.reshape(a, ushape)
        return numpy.moveaxis(b, range(nuaxes), range(-nuaxes, 0))

    return _unravel
