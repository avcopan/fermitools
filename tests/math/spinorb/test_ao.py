from simplehf.math.spinorb import ao

import numpy
from numpy.testing import assert_array_equal


def test__expand():
    a = numpy.ones((2, 2, 2, 2, 2, 2, 2, 2))

    b = ao.expand(a, brakets=((0, 7),))
    assert b.shape == (4, 2, 2, 2, 2, 2, 2, 4)

    c = ao.expand(a, brakets=((3, 5),))
    assert c.shape == (2, 2, 2, 4, 2, 4, 2, 2)

    d = ao.expand(a, brakets=((0, 7), (3, 5)))
    assert d.shape == (4, 2, 2, 4, 2, 4, 2, 4)

    e = ao.expand(a, brakets=((3, 5), (0, 7)))
    assert e.shape == (4, 2, 2, 4, 2, 4, 2, 4)
    assert_array_equal(e[:, 0, 0, 0, 0, 0, 0, :], [[1, 1, 0, 0],
                                                   [1, 1, 0, 0],
                                                   [0, 0, 1, 1],
                                                   [0, 0, 1, 1]])
    assert_array_equal(e[0, 0, 0, :, 0, :, 0, 0], [[1, 1, 0, 0],
                                                   [1, 1, 0, 0],
                                                   [0, 0, 1, 1],
                                                   [0, 0, 1, 1]])
