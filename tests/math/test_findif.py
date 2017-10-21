import fermitools.math.findif as findif

from numpy.testing import assert_almost_equal


def test__central_difference():

    def f(x):
        return x - x**2 - x**3 + x**4

    c1 = [findif.central_difference(f, 0., nder=k) for k in range(1, 6)]
    c2 = [findif.central_difference(f, (0.,), nder=k) for k in range(1, 6)]

    assert_almost_equal(c1, [1., -2., -6., 24., 0.], decimal=4)
    assert_almost_equal(c2, [(1.,), (-2.,), (-6.,), (24.,), (0.,)], decimal=4)
