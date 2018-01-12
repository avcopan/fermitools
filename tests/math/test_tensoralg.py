import fermitools.math.tensoralg as tensoralg

import numpy
from numpy.testing import assert_almost_equal


def test__einsum():
    a = numpy.random.random((3, 4))
    b = numpy.random.random((4, 5))
    c = numpy.random.random((5, 6))
    D = numpy.einsum('ik,kl,lj->ji', a, b, c)
    d = tensoralg.einsum('ik,kl,lj->ji', a, b, c)
    print(d-D)
    assert_almost_equal(d, D)
    E = numpy.einsum('ik,kl,lj', a, b, c)
    e = tensoralg.einsum('ik,kl,lj', a, b, c)
    print(e-E)
    assert_almost_equal(e, E)


if __name__ == '__main__':
    test__einsum()
