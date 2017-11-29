from fermitools.math.asym import urav

import os
import numpy
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
U = numpy.load(os.path.join(data_path, 'u.npy'))
R = numpy.load(os.path.join(data_path, 'r.npy'))
MU = numpy.load(os.path.join(data_path, 'mu.npy'))
MR = numpy.load(os.path.join(data_path, 'mr.npy'))


def test__unravel():
    u = urav.unravel(R, {0: ((0, 4), 3), 1: ((1, 5), 5), 2: ((2, 3, 6), 6)})
    assert_almost_equal(u, U, decimal=10)


def test__megaunraveler():
    uravf = urav.megaunraveler({0: {(0, 2, 4): 4, (6, 8): 3, (10,): 1},
                                1: {(1, 3): 3, (5, 7): 3},
                                2: {(9,): 1}})
    mu = uravf(MR)
    assert mu.shape == (4, 3, 4, 3, 4, 3, 3, 3, 3, 1, 1)

    assert_almost_equal(mu, MU, decimal=10)
