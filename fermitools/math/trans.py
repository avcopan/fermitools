import numpy
from functools import partial
from functools import reduce


def transform(a, transformers):
    ax = numpy.ndim(a) - len(transformers)
    tdot = partial(numpy.tensordot, axes=(ax, 0))
    return reduce(tdot, transformers, a)


if __name__ == '__main__':
    a = numpy.random.random((5, 5, 5, 5))
    c1 = numpy.random.random((5, 1))
    c2 = numpy.random.random((5, 2))
    c3 = numpy.random.random((5, 3))
    c4 = numpy.random.random((5, 4))

    B = numpy.einsum('ijkl,iI,jJ,kK,lL->IJKL', a, c1, c2, c3, c4)
    b = transform(a, transformers=(c1, c2, c3, c4))
    print(b.shape)

    from numpy.testing import assert_almost_equal
    assert_almost_equal(b, B, decimal=14)
