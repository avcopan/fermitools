import fermitools.math.ix as ix

import numpy
from numpy.testing import assert_equal


def test__cast():
    a = [1, 2, 3]
    b = [-2, -1, 0]
    c = [1, 1, 1]

    z = ix.cast(a, 0, 3) + ix.cast(b, 1, 3) + ix.cast(c, 2, 3)

    assert_equal(z, numpy.array([[[0, 0, 0],
                                  [1, 1, 1],
                                  [2, 2, 2]],
                                 [[1, 1, 1],
                                  [2, 2, 2],
                                  [3, 3, 3]],
                                 [[2, 2, 2],
                                  [3, 3, 3],
                                  [4, 4, 4]]]))
