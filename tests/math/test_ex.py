import fermitools.math.ex as ex

import numpy
import scipy
from numpy.testing import assert_almost_equal


def test__expm():
    a = numpy.random.random((10, 10))
    B = scipy.linalg.expm(a)
    b = ex.expm(a)
    assert_almost_equal(b, B, decimal=13)
