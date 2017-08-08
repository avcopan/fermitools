import numpy
import functools as ft
import more_itertools as mit
from . import shuffle
from . import sign


# Public
def antisymmetrizer(eqaxes):
    return ft.partial(antisymmetrize, eqaxes=eqaxes)


def antisymmetrize(a, eqaxes):
    classes = tuple(tuple(c) for c in map(mit.always_iterable, eqaxes))
    lens = tuple(len(c) for c in classes)
    axes = sum(classes, ())

    sgn = ft.partial(sign.signature, elems=axes)
    transp = ft.partial(_permute, iterable=range(a.ndim), elems=axes)

    return sum(sgn(p=p) * numpy.transpose(a, transp(p=p)) for p in
               shuffle.riffle_shuffles(elems=axes, dsizes=lens))


# Private
def _permute(iterable, elems, p):
    return tuple(x if x not in elems else p[elems.index(x)]
                 for x in iterable)


if __name__ == '__main__':
    import itertools as it
    from numpy.testing import assert_almost_equal

    a = numpy.random.rand(5, 5, 5, 5)
    c_ref = sum(sign.signature(p, elems=range(4)) * a.transpose(p)
                for p in it.permutations(range(4)))

    b = antisymmetrize(a, eqaxes=(1, 2))
    c = antisymmetrize(b, eqaxes=(0, (1, 2), 3))
    assert_almost_equal(c, c_ref)
    print(numpy.linalg.norm(c))
