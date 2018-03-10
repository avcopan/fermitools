import fermitools.math.spinorb as spinorb

import numpy
from numpy.testing import assert_array_equal


def test__ab2ov():
    assert spinorb.ab2ov(dim=4, na=2, nb=2) == (0, 1, 4, 5, 2, 3, 6, 7)
    assert spinorb.ab2ov(dim=5, na=3, nb=2) == (0, 1, 2, 5, 6, 3, 4, 7, 8, 9)


def test__sort():
    a = numpy.array([[1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [0, 0, 1, 1],
                     [0, 0, 1, 1]])

    b = spinorb.sort(a, order=(0, 2, 1, 3), axes=(0,))
    assert_array_equal(b, [[1, 1, 0, 0],
                           [0, 0, 1, 1],
                           [1, 1, 0, 0],
                           [0, 0, 1, 1]])

    c = spinorb.sort(a, order=(0, 2, 1, 3), axes=(1,))
    assert_array_equal(c, [[1, 0, 1, 0],
                           [1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 1, 0, 1]])

    d = spinorb.sort(a, order=(0, 2, 1, 3), axes=(0, 1))
    assert_array_equal(d, [[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [1, 0, 1, 0],
                           [0, 1, 0, 1]])
