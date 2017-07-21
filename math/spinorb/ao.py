import numpy
import toolz.functoolz as ftz


# Public
def expand(i, brakets):
    """expand an integral array in terms of spin-orbitals

    :param i: integral array
    :type i: numpy.ndarray
    :param brakets: the 1e-integrals in `i`, as bra/ket axis pairs
    :type brakets: tuple[tuple[int, int], ...]

    :rtype: numpy.ndarray
    """
    exp = expander(brakets)
    return exp(i)


def expander(brakets):
    """expands integral arrays in terms of spin-orbitals

    :param brakets: 1e-integrals, as bra/ket axis pairs
    :type brakets: tuple[tuple[int, int], ...]

    :rtype: typing.Callable
    """
    braket_expanders = map(_braket_expander, brakets)
    return ftz.compose(*braket_expanders)


# Private
def _braket_expander(braket):
    """expands a one-electron integral in terms of spin-orbitals

    :param bra: bra/ket axis pair
    :type bra: tuple[int, int]

    :rtype: typing.Callable
    """

    def transpose(i):
        return numpy.moveaxis(i, braket, (-2, -1))

    def expand(i):
        return numpy.kron(numpy.eye(2), i)

    def undo_transpose(i):
        return numpy.moveaxis(i, (-2, -1), braket)

    return ftz.compose(undo_transpose, expand, transpose)
