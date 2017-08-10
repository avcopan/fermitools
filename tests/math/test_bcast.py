import fermitools.math.bcast as bcast

import numpy
from numpy.testing import assert_equal


def test__broadcast_sum():
    a = [1, 2, 3]
    b = [-2, -1, 0]
    c = [1, 1, 1]

    z = bcast.broadcast_sum({0: a, 1: b, 2: c})

    assert_equal(z, numpy.array([[[0, 0, 0],
                                  [1, 1, 1],
                                  [2, 2, 2]],
                                 [[1, 1, 1],
                                  [2, 2, 2],
                                  [3, 3, 3]],
                                 [[2, 2, 2],
                                  [3, 3, 3],
                                  [4, 4, 4]]]))
