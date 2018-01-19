import fermitools.math.ot as ot

import numpy


def test__orth():
    numpy.random.seed(0)

    t1 = 1.
    t2 = 1e-3
    t3 = 1e-5
    t4 = 1e-8

    a = numpy.eye(100, 3)
    r = numpy.random.rand(100, 3)
    b1 = a + t1 * r
    b2 = a + t2 * r
    b3 = a + t3 * r
    b4 = a + t4 * r

    c1 = ot.orth(b1, against=a, tol=1e-4)
    c2 = ot.orth(b2, against=a, tol=1e-4)
    c3 = ot.orth(b3, against=a, tol=1e-4)
    c4 = ot.orth(b4, against=a, tol=1e-4)

    assert c1.shape == (100, 3)
    assert c2.shape == (100, 3)
    assert c3.shape == (100, 0)
    assert c4.shape == (100, 0)
