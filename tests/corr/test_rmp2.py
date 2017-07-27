import fermitools.corr.rmp2 as rmp2

import os
import numpy
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
N = numpy.load(os.path.join(data_path, 'rmp2/n.npy'))
G = numpy.load(os.path.join(data_path, 'rmp2/g.npy'))
E = numpy.load(os.path.join(data_path, 'rmp2/e.npy'))
CORR_ENERGY = numpy.load(os.path.join(data_path, 'rmp2/corr_energy.npy'))


def test__correlation_energy():
    corr_energy = rmp2.correlation_energy(n=N, g=G, e=E)
    assert_almost_equal(corr_energy, CORR_ENERGY, decimal=10)
