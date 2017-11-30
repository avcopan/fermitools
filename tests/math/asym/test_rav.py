from fermitools.math.asym import rav

import os
import numpy
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
U = numpy.load(os.path.join(data_path, 'u.npy'))
R = numpy.load(os.path.join(data_path, 'r.npy'))
MU = numpy.load(os.path.join(data_path, 'mu.npy'))
MR = numpy.load(os.path.join(data_path, 'mr.npy'))


def test__ravel():
    r = rav.ravel(U, {0: (0, 4), 1: (1, 5), 2: (2, 3, 6)})
    assert_almost_equal(r, R, decimal=10)

    # Make sure it also works for other orderings
    a = numpy.random.rand(3, 5, 6, 6, 3, 5, 6)
    b0 = rav.ravel(a, {0: (0, 4), 1: (1, 5), 2: (2, 3, 6)})
    b1 = rav.ravel(a, {0: (0, 4), 2: (1, 5), 1: (2, 3, 6)})
    b2 = rav.ravel(a, {1: (0, 4), 0: (1, 5), 2: (2, 3, 6)})
    b3 = rav.ravel(a, {1: (0, 4), 2: (1, 5), 0: (2, 3, 6)})
    b4 = rav.ravel(a, {2: (0, 4), 0: (1, 5), 1: (2, 3, 6)})
    b5 = rav.ravel(a, {2: (0, 4), 1: (1, 5), 0: (2, 3, 6)})
    assert b0.shape == (3, 10, 20)
    assert b1.shape == (3, 20, 10)
    assert b2.shape == (10, 3, 20)
    assert b3.shape == (20, 3, 10)
    assert b4.shape == (10, 20, 3)
    assert b5.shape == (20, 10, 3)


def test__megaravel():
    mr = rav.megaravel(MU, {0: ((0, 2, 4), (6, 8), (10,)),
                            1: ((1, 3), (5, 7)),
                            2: ((9,),)})
    assert mr.shape == (12, 9, 1)

    assert_almost_equal(mr, MR, decimal=10)
