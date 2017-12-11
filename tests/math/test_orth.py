import fermitools.math.orth as orth

import numpy


def test__orthogonalize():
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

    c1 = orth.orthogonalize(b1, against=a, tol=1e-4)
    c2 = orth.orthogonalize(b2, against=a, tol=1e-4)
    c3 = orth.orthogonalize(b3, against=a, tol=1e-4)
    c4 = orth.orthogonalize(b4, against=a, tol=1e-4)

    assert c1.shape == (100, 3)
    assert c2.shape == (100, 3)
    assert c3.shape == (100, 0)
    assert c4.shape == (100, 0)
