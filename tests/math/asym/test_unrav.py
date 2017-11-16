from fermitools.math.asym import unrav

import os
import numpy
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
U = numpy.load(os.path.join(data_path, 'unraveled.npy'))
R = numpy.load(os.path.join(data_path, 'raveled.npy'))


def test__unravel():
    u = unrav.unravel(R, {0: (0, 4), 1: (1, 5), 2: (2, 3, 6)})
    assert_almost_equal(u, U, decimal=10)
