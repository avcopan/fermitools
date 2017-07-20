from simplehf.math import trans

import numpy
from numpy.testing import assert_array_equal


def test__transform():
    a = numpy.ones((2, 2, 2))
    t0 = numpy.array([[1, 0], [0, -1]])
    t1 = numpy.array([[2, 0], [0, 3]])
    t2 = numpy.array([[5, 0], [0, 7]])

    b = trans.transform(a, {0: t0})
    assert_array_equal(b, [[[+1, +1],
                            [+1, +1]],
                           [[-1, -1],
                            [-1, -1]]])

    c = trans.transform(a, {1: t0})
    assert_array_equal(c, [[[+1, +1],
                            [-1, -1]],
                           [[+1, +1],
                            [-1, -1]]])

    d = trans.transform(a, {2: t0})
    assert_array_equal(d, [[[+1, -1],
                            [+1, -1]],
                           [[+1, -1],
                            [+1, -1]]])

    e = trans.transform(a, {0: t0, 1: t1, 2: t2})
    assert_array_equal(e, [[[+2 * 5, +2 * 7],
                            [+3 * 5, +3 * 7]],
                           [[-2 * 5, -2 * 7],
                            [-3 * 5, -3 * 7]]])
