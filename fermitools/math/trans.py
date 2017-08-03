import numpy
import itertools as it
import toolz.functoolz as ftz


# Public
def transform(a, transd):
    """transform an array

    :param a: array
    :type a: numpy.ndarray
    :param transd: dictionary of transformation matrices, keyed by axis
    :type transd: dict

    :rtype: numpy.ndarray
    """
    transf = transformer(transd)
    return transf(a)


def transformer(transd):
    """transforms arrays

    :param transd: dictionary of transformation matrices, keyed by axis
    :type transd: dict

    :rtype: typing.Callable
    """
    transformers = it.starmap(_axis_transformer, transd.items())
    return ftz.compose(*transformers)


# Private
def _axis_transformer(ax, t):
    """transforms one axis

    :param ax: axis
    :type ax: int
    :param t: transformation matrix
    :type t: numpy.ndarray

    :rtype: typing.Callable
    """
    def transform(a):
        return numpy.tensordot(a, t, axes=(ax, 0))

    def reorder(a):
        return numpy.moveaxis(a, -1, ax)

    return ftz.compose(reorder, transform)
