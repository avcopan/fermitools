import fermitools.math.spinorb as spinorb

import numpy
from numpy.testing import assert_array_equal


def test__expand():
    a = numpy.ones((2, 2, 2, 2, 2, 2, 2, 2))

    b = spinorb.expand(a, brakets=((0, 7),))
    assert b.shape == (4, 2, 2, 2, 2, 2, 2, 4)

    c = spinorb.expand(a, brakets=((3, 5),))
    assert c.shape == (2, 2, 2, 4, 2, 4, 2, 2)

    d = spinorb.expand(a, brakets=((0, 7), (3, 5)))
    assert d.shape == (4, 2, 2, 4, 2, 4, 2, 4)

    e = spinorb.expand(a, brakets=((3, 5), (0, 7)))
    assert e.shape == (4, 2, 2, 4, 2, 4, 2, 4)
    assert_array_equal(e[:, 0, 0, 0, 0, 0, 0, :], [[1, 1, 0, 0],
                                                   [1, 1, 0, 0],
                                                   [0, 0, 1, 1],
                                                   [0, 0, 1, 1]])
    assert_array_equal(e[0, 0, 0, :, 0, :, 0, 0], [[1, 1, 0, 0],
                                                   [1, 1, 0, 0],
                                                   [0, 0, 1, 1],
                                                   [0, 0, 1, 1]])


def test__ab2ov():
    assert spinorb.ab2ov(dim=4, na=2, nb=2) == (0, 1, 4, 5, 2, 3, 6, 7)
    assert spinorb.ab2ov(dim=5, na=3, nb=2) == (0, 1, 2, 5, 6, 3, 4, 7, 8, 9)
