import fermitools.math.rav as rav

import os
import numpy
from numpy.testing import assert_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
R = numpy.load(os.path.join(data_path, 'rav/r.npy'))
U = numpy.load(os.path.join(data_path, 'rav/u.npy'))


def test__ravel():
    r = rav.ravel(U, {1: (1, 5), 3: (0, 6)})
    assert_equal(r, R)
