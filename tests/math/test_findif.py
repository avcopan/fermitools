import fermitools.math.findif as findif

from numpy.testing import assert_almost_equal


def test__central_difference():
    import numpy

    def f(x):
        return numpy.vdot(x, x)

    def df(x):
        return findif.central_difference(f, x, nder=1)

    def g(x):
        return [2. * numpy.vdot(x, x), 3. * numpy.vdot(x, x)]

    def h(x):
        return numpy.power(x, 2) + numpy.power(x, 3)

    assert_almost_equal(findif.central_difference(f, 0., nder=1), 0.)
    assert_almost_equal(findif.central_difference(f, 0., nder=2), 2.)
    assert_almost_equal(findif.central_difference(df, 0., nder=1), 2.)
    assert_almost_equal(findif.central_difference(f, [0., 0.], nder=2),
                        [2., 2.])
    assert_almost_equal(findif.central_difference(g, 0., nder=2), [4., 6.])
    assert_almost_equal(findif.central_difference(h, [0., 1.], nder=2),
                        [[2., 0.],
                         [0., 8.]])
