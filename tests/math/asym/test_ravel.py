from fermitools.math.asym import ravel

import os
import numpy
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
UNRAV = numpy.load(os.path.join(data_path, 'unraveled.npy'))
RAV = numpy.load(os.path.join(data_path, 'raveled.npy'))


def test__compound_index():
    rav = ravel.compound_index(UNRAV, {0: (0, 4), 1: (1, 5), 2: (2, 3, 6)})
    assert_almost_equal(rav, RAV, decimal=10)

    # Make sure it also works for other orderings
    a = numpy.random.rand(3, 5, 6, 6, 3, 5, 6)
    b0 = ravel.compound_index(a, {0: (0, 4), 1: (1, 5), 2: (2, 3, 6)})
    b1 = ravel.compound_index(a, {0: (0, 4), 2: (1, 5), 1: (2, 3, 6)})
    b2 = ravel.compound_index(a, {1: (0, 4), 0: (1, 5), 2: (2, 3, 6)})
    b3 = ravel.compound_index(a, {1: (0, 4), 2: (1, 5), 0: (2, 3, 6)})
    b4 = ravel.compound_index(a, {2: (0, 4), 0: (1, 5), 1: (2, 3, 6)})
    b5 = ravel.compound_index(a, {2: (0, 4), 1: (1, 5), 0: (2, 3, 6)})
    assert b0.shape == (3, 10, 20)
    assert b1.shape == (3, 20, 10)
    assert b2.shape == (10, 3, 20)
    assert b3.shape == (20, 3, 10)
    assert b4.shape == (10, 20, 3)
    assert b5.shape == (20, 10, 3)
