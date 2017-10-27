from fermitools.math.asym import unravel

import os
import numpy
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
UNRAV = numpy.load(os.path.join(data_path, 'unraveled.npy'))
RAV = numpy.load(os.path.join(data_path, 'raveled.npy'))


def test__unravel_compound_index():
    unrav = unravel.unravel_compound_index(RAV, {0: (0, 4),
                                                 1: (1, 5),
                                                 2: (2, 3, 6)})
    assert_almost_equal(unrav, UNRAV, decimal=10)
