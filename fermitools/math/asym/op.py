import numpy
import functools as ft
import more_itertools as mit
import toolz.functoolz as ftz

from ..combinatorics import riffle_shuffles, signature, permuter


# Public
def antisymmetrizer_product(bs):
    args = _process_bartlett_string(bs)
    return ftz.compose(*map(antisymmetrizer, args))


def symmetrizer_product(bs):
    args = _process_bartlett_string(bs)
    return ftz.compose(*map(symmetrizer, args))


def antisymmetrizer(axes):
    """antisymmetrizes an array over permutations or riffle-shuffles of axes

    :param a: array
    :type a: numpy.ndarray
    :param axes: axes
    :type axes: tuple[int or tuple[int], ...]

    :rtype: numpy.ndarray
    """
    return ft.partial(antisymmetrize, axes=axes)


def symmetrizer(axes):
    """symmetrizes an array over permutations or riffle-shuffles of axes

    :param a: array
    :type a: numpy.ndarray
    :param axes: axes
    :type axes: tuple[int or tuple[int], ...]

    :rtype: numpy.ndarray
    """
    return ft.partial(symmetrize, axes=axes)


def antisymmetrize(a, axes):
    """antisymmetrize an array over permutations or riffle-shuffles of axes

    Antisymmetrizing over ((0, 1), (2, 3), 4) antisymmetrizes the first five
    axes assuming that the array is already antisymmetric with respect to
    permutations of 0 and 1 as well as 2 and 3.  Antisymmetrizing first over
    (0, 1), then over (2, 3), and finally over ((0, 1), (2, 3), 4) is
    equivalent to directly antisymmetrizing over (0, 1, 2, 3, 4).

    :param a: array
    :type a: numpy.ndarray
    :param axes: axes
    :type axes: tuple[int or tuple[int], ...]

    :rtype: numpy.ndarray
    """
    groups = tuple(map(tuple, map(mit.always_iterable, axes)))
    i = sum(groups, ())

    ksizes = tuple(len(g) for g in groups)

    sgn = ft.partial(signature, i=i)
    per = ft.partial(permuter, i=i)

    allaxes = tuple(range(a.ndim))

    return sum(sgn(p) * numpy.transpose(a, per(p)(allaxes)) for p in
               riffle_shuffles(i=i, ksizes=ksizes))


def symmetrize(a, axes):
    """symmetrize an array over permutations or riffle-shuffles of axes

    :param a: array
    :type a: numpy.ndarray
    :param axes: axes
    :type axes: tuple[int or tuple[int], ...]

    :rtype: numpy.ndarray
    """
    groups = tuple(map(tuple, map(mit.always_iterable, axes)))
    i = sum(groups, ())

    ksizes = tuple(len(g) for g in groups)

    per = ft.partial(permuter, i=i)

    allaxes = tuple(range(a.ndim))

    return sum(numpy.transpose(a, per(p)(allaxes)) for p in
               riffle_shuffles(i=i, ksizes=ksizes))


# Private
def _process_bartlett_string(bs):
    return tuple(tuple(tuple(
                 int(no)
                 for no in str.split(cs, ','))
                 for cs in str.split(ss, '/'))
                 for ss in str.split(bs, '|'))
