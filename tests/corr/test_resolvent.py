import fermitools.corr as corr

import os
import numpy
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
AE_O = numpy.load(os.path.join(data_path, 'ae_o.npy'))
BE_O = numpy.load(os.path.join(data_path, 'be_o.npy'))
AE_V = numpy.load(os.path.join(data_path, 'ae_v.npy'))
BE_V = numpy.load(os.path.join(data_path, 'be_v.npy'))
R = numpy.load(os.path.join(data_path, 'r.npy'))


def test__resolvent():
    r = corr.resolvent((AE_O, BE_O), (AE_V, BE_V))
    assert_almost_equal(r, R, decimal=10)
